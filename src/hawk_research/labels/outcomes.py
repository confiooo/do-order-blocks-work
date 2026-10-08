"""Forward labels on 15m bars.

The decision is taken at the close of bar ``i``; a label may only read bars
``i+1`` onwards.

Two kinds of label:

- per-bar labels: forward returns, first-touch barrier offsets and forward
  realised volatility for every bar of every symbol. They are computed for all
  bars, not only zone bars, because the null tests compare zones against the
  other bars.
- zone lifecycle: first valid retest, ``hypothetical_entry_price`` and MFE/MAE
  at 1h/4h/8h/24h, computed with the live ``ai/outcome_tracker`` functions so
  the sign conventions match the live book.

Gaps: halts and delistings leave holes, so ``i+k`` is only ``k`` bars later when
the forward window is contiguous. Every label is NaN unless enough consecutive
bars at one bar-interval spacing follow.
"""

from __future__ import annotations

import bisect
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from hawk_research import config as research_config
from hawk_research.constants import BAR_MS
from hawk_research.labels.retest import candles_after_fill, compute_excursions, entry_price, find_first_retest

_LABELS = research_config.get().labels

# Forward-return horizons, in bars. 15m/1h/4h/8h/24h/48h.
RETURN_HORIZONS: Dict[str, int] = dict(_LABELS.return_horizons)

# Barrier levels, as fractions. First-touch offsets are stored per level; the
# label for any (target, stop) pair is then derived without rescanning bars.
BARRIER_LEVELS: Tuple[float, ...] = tuple(_LABELS.barrier_levels)

# (up_pct, down_pct) pairs that research asked for, symmetric and asymmetric.
_REQUESTED_PAIRS: Tuple[Tuple[float, float], ...] = (
    (0.005, 0.005),
    (0.01, 0.01),
    (0.02, 0.02),
    (0.03, 0.03),
    (0.02, 0.01),
    (0.01, 0.02),
    (0.03, 0.015),
    (0.015, 0.03),
)
# Only pairs whose legs are both configured barrier levels can be derived.
BARRIER_PAIRS: Tuple[Tuple[float, float], ...] = tuple(
    (up, dn) for up, dn in _REQUESTED_PAIRS if up in BARRIER_LEVELS and dn in BARRIER_LEVELS
)

BARRIER_HORIZONS: Dict[str, int] = dict(_LABELS.barrier_horizons)

MAX_HORIZON_BARS = max(max(RETURN_HORIZONS.values()), max(BARRIER_HORIZONS.values()))

FORWARD_VOL_HORIZONS: Dict[str, int] = dict(_LABELS.forward_vol_horizons)

# Live lifecycle constants (config trade_lifecycle).
RETEST_TIMEOUT_HOURS = int(_LABELS.retest_timeout_hours)
RETEST_TIMEOUT_BARS = int(RETEST_TIMEOUT_HOURS * 4)
MFE_MAE_HORIZONS = tuple(_LABELS.mfe_mae_horizons_hours)

_CHUNK = 4096


def _level_tag(level: float) -> str:
    return f"{level * 100:g}".replace(".", "p")


def contiguous_forward_bars(open_time: np.ndarray, bar_ms: int = BAR_MS) -> np.ndarray:
    """``out[i]`` = how many consecutive evenly spaced bars follow bar ``i``.

    A label needing ``k`` forward bars is only valid where ``out[i] >= k``.
    """
    n = len(open_time)
    out = np.zeros(n, dtype=np.int32)
    if n < 2:
        return out
    step_ok = np.diff(open_time) == bar_ms
    run = 0
    for i in range(n - 2, -1, -1):
        run = run + 1 if step_ok[i] else 0
        out[i] = run
    return out


def forward_returns(close: np.ndarray, runs: np.ndarray) -> Dict[str, np.ndarray]:
    """``close[i+k]/close[i] - 1`` for each horizon, masked across gaps."""
    n = len(close)
    out: Dict[str, np.ndarray] = {}
    for name, k in RETURN_HORIZONS.items():
        vals = np.full(n, np.nan, dtype=np.float64)
        if n > k:
            fwd = close[k:]
            base = close[:-k]
            with np.errstate(divide="ignore", invalid="ignore"):
                vals[:-k] = np.where(base != 0, fwd / base - 1.0, np.nan)
        vals[runs < k] = np.nan
        out[name] = vals
    return out


def forward_realized_vol(close: np.ndarray, runs: np.ndarray) -> Dict[str, np.ndarray]:
    """Std of the 15m log returns over the next k bars. Uses only bars > i."""
    n = len(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.full(n, np.nan)
        logret[1:] = np.log(close[1:] / close[:-1])
    out: Dict[str, np.ndarray] = {}
    for name, k in FORWARD_VOL_HORIZONS.items():
        vals = np.full(n, np.nan, dtype=np.float64)
        for start in range(0, n, _CHUNK):
            stop = min(start + _CHUNK, n)
            idx = np.arange(start, stop)
            # Forward returns of bars i+1 .. i+k.
            cols = idx[:, None] + np.arange(1, k + 1)[None, :]
            valid = cols < n
            cols = np.clip(cols, 0, n - 1)
            block = np.where(valid, logret[cols], np.nan)
            with warnings.catch_warnings():
                # Rows at the end of the series have no forward bars; NaN is the answer.
                warnings.simplefilter("ignore", RuntimeWarning)
                vals[start:stop] = np.nanstd(block, axis=1)
        vals[runs < k] = np.nan
        out[name] = vals
    return out


def first_touch_offsets(
    high: np.ndarray,
    low: np.ndarray,
    ref: np.ndarray,
    runs: np.ndarray,
    levels: Sequence[float] = BARRIER_LEVELS,
    horizon: int = MAX_HORIZON_BARS,
) -> Dict[str, np.ndarray]:
    """Bars until price first trades ``level`` above / below ``ref[i]``.

    Returns 1-based offsets (1 = the very next bar) or ``0`` for "not touched
    inside ``horizon``". Only bars ``i+1 .. i+horizon`` are inspected, so the
    decision bar's own high/low can never create a touch.
    """
    n = len(high)
    out: Dict[str, np.ndarray] = {}
    for level in levels:
        out[f"first_up_{_level_tag(level)}"] = np.zeros(n, dtype=np.int16)
        out[f"first_dn_{_level_tag(level)}"] = np.zeros(n, dtype=np.int16)

    offsets = np.arange(1, horizon + 1)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        idx = np.arange(start, stop)
        cols = idx[:, None] + offsets[None, :]
        valid = cols < n
        cols_c = np.clip(cols, 0, n - 1)
        # Bars beyond the contiguous run are unusable, not merely absent.
        valid &= offsets[None, :] <= runs[idx][:, None]
        h_block = np.where(valid, high[cols_c], -np.inf)
        l_block = np.where(valid, low[cols_c], np.inf)
        r = ref[idx][:, None]
        for level in levels:
            tag = _level_tag(level)
            up_hit = h_block >= r * (1.0 + level)
            dn_hit = l_block <= r * (1.0 - level)
            up_any = up_hit.any(axis=1)
            dn_any = dn_hit.any(axis=1)
            up_off = np.where(up_any, up_hit.argmax(axis=1) + 1, 0)
            dn_off = np.where(dn_any, dn_hit.argmax(axis=1) + 1, 0)
            out[f"first_up_{tag}"][start:stop] = up_off.astype(np.int16)
            out[f"first_dn_{tag}"][start:stop] = dn_off.astype(np.int16)
    return out


def barrier_label(
    up_offset: np.ndarray,
    dn_offset: np.ndarray,
    horizon_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Which barrier is hit first: ``+1`` up, ``-1`` down, ``0`` neither.

    When both barriers fall inside the same 15m bar's high-low range, 15m data
    cannot tell which came first. That case is flagged ``ambiguous`` and resolved
    as **the down barrier**. For a long that is the stop (pessimistic); for a short
    the down barrier is the target, so the same rule books the short as a win
    (optimistic). Barrier results report ``ambiguous_fraction`` and a sensitivity
    for this (``evaluation.strategies.ambiguity_sensitivity``; methodology §9).
    Missing offsets (NaN) count as "not touched".
    """
    never = np.int32(np.iinfo(np.int32).max)
    # A missing offset (NaN: label unavailable) counts as no touch. Map it to 0 explicitly
    # rather than relying on the platform-defined result of casting NaN to an integer.
    up_offset = np.nan_to_num(np.asarray(up_offset, dtype=np.float64), nan=0.0).astype(np.int32)
    dn_offset = np.nan_to_num(np.asarray(dn_offset, dtype=np.float64), nan=0.0).astype(np.int32)
    up = np.where((up_offset > 0) & (up_offset <= horizon_bars), up_offset, never)
    dn = np.where((dn_offset > 0) & (dn_offset <= horizon_bars), dn_offset, never)
    neither = (up == never) & (dn == never)
    ambiguous = (up == dn) & ~neither
    label = np.where(up < dn, 1, -1).astype(np.int8)
    label[ambiguous] = -1
    label[neither] = 0
    return label, ambiguous


def add_barrier_pair_labels(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Materialise the requested (target, stop) x horizon first-touch labels."""
    out = {}
    for up, dn in BARRIER_PAIRS:
        up_col = f"{prefix}first_up_{_level_tag(up)}"
        dn_col = f"{prefix}first_dn_{_level_tag(dn)}"
        if up_col not in df.columns or dn_col not in df.columns:
            continue
        u = df[up_col].to_numpy()
        d = df[dn_col].to_numpy()
        for hname, hbars in BARRIER_HORIZONS.items():
            lab, amb = barrier_label(u, d, hbars)
            tag = f"{prefix}bar_{_level_tag(up)}_{_level_tag(dn)}_{hname}"
            out[tag] = lab
            out[f"{tag}_ambiguous"] = amb
    # One concat rather than an insert per column (pandas warns about fragmentation).
    # Existing columns are overwritten in place, as ``assign`` would; new ones are appended.
    existing = [c for c in out if c in df.columns]
    base = df.copy() if existing else df
    for c in existing:
        base[c] = out.pop(c)
    return pd.concat([base, pd.DataFrame(out, index=df.index)], axis=1)


def per_bar_labels(df: pd.DataFrame) -> pd.DataFrame:
    """All causal forward labels for one symbol's full bar series."""
    open_time = df["open_time"].to_numpy(dtype=np.int64)
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    runs = contiguous_forward_bars(open_time)

    cols: Dict[str, np.ndarray] = {"open_time": open_time, "fwd_bars_available": runs}
    for name, vals in forward_returns(close, runs).items():
        cols[f"fwd_ret_{name}"] = vals.astype(np.float32)
    for name, vals in forward_realized_vol(close, runs).items():
        cols[f"fwd_rv_{name}"] = vals.astype(np.float32)
    cols.update(first_touch_offsets(high, low, close, runs))
    return pd.DataFrame(cols)


def _candles(df: pd.DataFrame) -> List[Tuple[int, float, float, float, float, float]]:
    return list(
        zip(
            df["open_time"].astype("int64").tolist(),
            df["open"].astype(float).tolist(),
            df["high"].astype(float).tolist(),
            df["low"].astype(float).tolist(),
            df["close"].astype(float).tolist(),
            df["volume"].astype(float).tolist(),
        )
    )


def zone_lifecycle(
    bars: pd.DataFrame,
    zones: pd.DataFrame,
    retest_timeout_bars: int = RETEST_TIMEOUT_BARS,
) -> pd.DataFrame:
    """Replicate the live retest / entry / MFE / MAE lifecycle for replayed zones.

    The live evaluator re-runs every 15 minutes and marks a zone ``no_retest``
    once ``now - source_close >= retest_timeout_hours``. A retest bar is
    therefore only actionable if it closes before that deadline, which means its
    open time is at most ``source_open + 24h``. The search window here is capped
    accordingly instead of scanning the whole series, which is what
    ``find_first_retest`` would otherwise do.

    ``find_first_retest``, ``entry_price``, ``candles_after_fill`` and
    ``compute_excursions`` are the live functions, vendored unchanged, so the
    MAE sign convention (positive = adverse) cannot drift from production.
    """
    if zones.empty:
        return pd.DataFrame()
    candles = _candles(bars)
    # Bars are in strictly increasing open-time order (data/validation.py), so the
    # bars after a fill are found by bisection rather than by scanning the series.
    open_times = [c[0] for c in candles]
    pos_by_open = {t: i for i, t in enumerate(open_times)}
    max_excursion_bars = max(MFE_MAE_HORIZONS) * 4

    records: List[Dict] = []
    for row in zones.itertuples(index=False):
        src = int(row.source_open_time)
        i = pos_by_open.get(src)
        rec: Dict[str, object] = {
            "symbol": row.symbol,
            "source_open_time": src,
            "retested": False,
            "retest_open_time": None,
            "hypothetical_entry_price": None,
            "retest_lag_bars": None,
            "lifecycle_status": "no_retest",
        }
        for h in MFE_MAE_HORIZONS:
            rec[f"mfe_{h}h"] = None
            rec[f"mae_{h}h"] = None
        if i is None:
            rec["lifecycle_status"] = "missing_bar"
            records.append(rec)
            continue

        # Only bars inside the live retest deadline are eligible.
        hi = min(len(candles), i + 1 + retest_timeout_bars)
        window = candles[max(0, i - 1) : hi]
        retest = find_first_retest(window, src, row.direction, row.zone_high, row.zone_low)
        if retest is None:
            records.append(rec)
            continue

        fill = int(retest[0])
        entry = entry_price(row.direction, row.zone_high, row.zone_low)
        rec["retested"] = True
        rec["retest_open_time"] = fill
        rec["hypothetical_entry_price"] = float(entry)
        rec["retest_lag_bars"] = int((fill - src) // BAR_MS)
        rec["lifecycle_status"] = "entered"

        k = bisect.bisect_right(open_times, fill + 1000)
        after = candles_after_fill(candles[k : k + max_excursion_bars], fill)
        for h in MFE_MAE_HORIZONS:
            needed = h * 4
            if len(after) < needed:
                continue
            seg = after[:needed]
            # Excursions must not silently span a data gap.
            if seg[-1][0] - fill != needed * BAR_MS:
                continue
            mfe, mae = compute_excursions(row.direction, entry, seg)
            rec[f"mfe_{h}h"] = mfe
            rec[f"mae_{h}h"] = mae
        if rec["mfe_24h"] is not None and rec["mae_24h"] is not None:
            rec["lifecycle_status"] = "complete"
        records.append(rec)

    return pd.DataFrame.from_records(records)


def entry_anchored_labels(
    bars: pd.DataFrame,
    lifecycle: pd.DataFrame,
) -> pd.DataFrame:
    """Forward returns and barrier first-touch measured from the retest entry.

    The reference price is ``hypothetical_entry_price`` and the clock starts on
    the bar *after* the fill bar — the same exclusion the live tracker applies,
    so the fill bar's own wick cannot manufacture an excursion.
    """
    entered = lifecycle[lifecycle["retested"]].copy()
    if entered.empty:
        return pd.DataFrame()

    open_time = bars["open_time"].to_numpy(dtype=np.int64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    runs = contiguous_forward_bars(open_time)
    pos = {int(t): i for i, t in enumerate(open_time.tolist())}

    idx = np.array([pos.get(int(t), -1) for t in entered["retest_open_time"]], dtype=np.int64)
    ok = idx >= 0
    entered = entered[ok].copy()
    idx = idx[ok]
    ref = entered["hypothetical_entry_price"].to_numpy(dtype=np.float64)

    n = len(open_time)
    out = pd.DataFrame(
        {
            "symbol": entered["symbol"].to_numpy(),
            "source_open_time": entered["source_open_time"].to_numpy(),
        }
    )
    avail = runs[idx]
    out["entry_fwd_bars_available"] = avail
    for name, k in RETURN_HORIZONS.items():
        tgt = idx + k
        vals = np.where((tgt < n) & (avail >= k), close[np.clip(tgt, 0, n - 1)] / ref - 1.0, np.nan)
        out[f"entry_fwd_ret_{name}"] = vals.astype(np.float32)

    offsets = np.arange(1, MAX_HORIZON_BARS + 1)
    cols = idx[:, None] + offsets[None, :]
    valid = (cols < n) & (offsets[None, :] <= avail[:, None])
    cols_c = np.clip(cols, 0, n - 1)
    h_block = np.where(valid, high[cols_c], -np.inf)
    l_block = np.where(valid, low[cols_c], np.inf)
    r = ref[:, None]
    for level in BARRIER_LEVELS:
        tag = _level_tag(level)
        up_hit = h_block >= r * (1.0 + level)
        dn_hit = l_block <= r * (1.0 - level)
        out[f"entry_first_up_{tag}"] = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1) + 1, 0).astype(
            np.int16
        )
        out[f"entry_first_dn_{tag}"] = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1) + 1, 0).astype(
            np.int16
        )
    return out
