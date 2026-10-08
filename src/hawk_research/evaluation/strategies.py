"""Turn zone rows into the harness's trade table.

Entries:

- ``close``: at the close of the zone bar.
- ``retest``: at ``hypothetical_entry_price`` on the first valid retest, as the
  live system proposed. About 78% of zones fill, and the filled ones are a
  selected subset.

Exits:

- ``time``: after a fixed number of bars.
- ``barrier``: at whichever of take-profit or stop is touched first, else the
  time exit. Fills are priced at the barrier, which is optimistic for a stop in a
  fast market.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.labels import outcomes as lab


@dataclass(frozen=True)
class TradeSpec:
    """One fully specified way of trading a zone population."""

    side_rule: str = "direction"  # 'direction' | 'fade' | 'long' | 'short'
    entry: str = "close"  # 'close' | 'retest'
    exit: str = "time"  # 'time' | 'barrier'
    horizon: str = "4h"
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    label: str = ""

    def describe(self) -> Dict[str, object]:
        return {
            "side_rule": self.side_rule,
            "entry": self.entry,
            "exit": self.exit,
            "horizon": self.horizon,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
        }


def _sides(ds: pd.DataFrame, rule: str) -> np.ndarray:
    nominal = np.where(ds["direction"].to_numpy() == "bullish", 1, -1)
    if rule == "direction":
        return nominal
    if rule == "fade":
        return -nominal
    if rule == "long":
        return np.ones(len(ds), dtype=int)
    if rule == "short":
        return -np.ones(len(ds), dtype=int)
    raise ValueError(f"unknown side_rule {rule!r}")


def _time_exit(ds: pd.DataFrame, spec: TradeSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gross return before side, entry_time_ms, exit_time_ms) for a held position."""
    bars = lab.RETURN_HORIZONS[spec.horizon]
    if spec.entry == "close":
        ret = ds[f"fwd_ret_{spec.horizon}"].to_numpy(dtype=float)
        entry_ms = ds["source_open_time"].to_numpy(dtype=np.int64) + BAR_MS
    else:
        ret = ds[f"entry_fwd_ret_{spec.horizon}"].to_numpy(dtype=float)
        # The live tracker excludes the fill bar, so the position is live from
        # the close of the bar after the retest bar.
        entry_ms = ds["retest_open_time"].to_numpy(dtype="float64")
        entry_ms = np.where(np.isnan(entry_ms), np.nan, entry_ms + BAR_MS)
    exit_ms = entry_ms + bars * BAR_MS
    return ret, entry_ms, exit_ms


def _barrier_exit(
    ds: pd.DataFrame, spec: TradeSpec, side: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed return, entry/exit time and ambiguity flag under first-touch TP / SL.

    The up and down barriers are fixed distances from the entry price. Which one
    is the target and which the stop depends on the side, but the *price levels*
    do not, so a single pair of first-touch offsets serves both.

    Same-bar ambiguity (both barriers inside one bar) resolves as "down barrier
    first" (``labels.outcomes.barrier_label``): the stop for a long, but the
    *target* for a short. The rule is kept as published; the returned flag lets
    callers report how much a result depends on it (``ambiguity_sensitivity``).
    """
    if spec.take_profit is None or spec.stop_loss is None:
        raise ValueError("barrier exit requires take_profit and stop_loss")
    prefix = "" if spec.entry == "close" else "entry_"
    long_up, long_dn = spec.take_profit, spec.stop_loss

    up_col = f"{prefix}first_up_{lab._level_tag(long_up)}"
    dn_col = f"{prefix}first_dn_{lab._level_tag(long_dn)}"
    # A short's target sits below and its stop above, so it reads the mirrored
    # pair of levels.
    s_up_col = f"{prefix}first_up_{lab._level_tag(spec.stop_loss)}"
    s_dn_col = f"{prefix}first_dn_{lab._level_tag(spec.take_profit)}"

    horizon_bars = lab.BARRIER_HORIZONS.get(spec.horizon, lab.RETURN_HORIZONS[spec.horizon])
    is_long = side > 0

    up_off = np.where(is_long, ds[up_col].to_numpy(), ds[s_up_col].to_numpy())
    dn_off = np.where(is_long, ds[dn_col].to_numpy(), ds[s_dn_col].to_numpy())
    label, ambiguous = lab.barrier_label(up_off, dn_off, horizon_bars)

    tp = np.where(is_long, spec.take_profit, spec.take_profit)
    sl = np.where(is_long, spec.stop_loss, spec.stop_loss)

    # label +1 = up barrier first, -1 = down barrier first, 0 = neither.
    win = np.where(is_long, label == 1, label == -1)
    loss = np.where(is_long, label == -1, label == 1)

    if spec.entry == "close":
        base_ret = ds[f"fwd_ret_{spec.horizon}"].to_numpy(dtype=float)
        entry_ms = ds["source_open_time"].to_numpy(dtype=np.int64) + BAR_MS
    else:
        base_ret = ds[f"entry_fwd_ret_{spec.horizon}"].to_numpy(dtype=float)
        entry_ms = ds["retest_open_time"].to_numpy(dtype="float64") + BAR_MS

    signed = np.where(win, tp, np.where(loss, -sl, side * base_ret))
    touched = np.where(
        win | loss,
        np.where(win, np.where(is_long, up_off, dn_off), np.where(is_long, dn_off, up_off)),
        horizon_bars,
    )
    exit_ms = entry_ms + np.asarray(touched, dtype=float) * BAR_MS
    return signed, entry_ms, exit_ms, np.asarray(ambiguous, dtype=bool)


def build_trades(
    ds: pd.DataFrame,
    spec: TradeSpec,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Materialise the harness trade table for one specification.

    Rows whose outcome is unknown (no retest when entering on retest, or a
    forward window truncated by a data gap or the end of the sample) are
    dropped rather than filled with a guess.
    """
    frame = ds if mask is None else ds[mask]
    frame = frame.reset_index(drop=True)
    if spec.entry == "retest":
        frame = frame[frame["retested"].fillna(False).to_numpy().astype(bool)].reset_index(drop=True)
    if frame.empty:
        return _empty_trades()

    side = _sides(frame, spec.side_rule)
    ambiguous = None
    if spec.exit == "time":
        raw, entry_ms, exit_ms = _time_exit(frame, spec)
        ret_gross = side * raw
    else:
        ret_gross, entry_ms, exit_ms, ambiguous = _barrier_exit(frame, spec, side)

    out = pd.DataFrame(
        {
            "zone_uid": frame["zone_uid"].to_numpy(),
            "symbol": frame["symbol"].to_numpy(),
            "decision_time_ms": frame["source_open_time"].to_numpy(dtype=np.int64) + BAR_MS,
            "entry_time_ms": entry_ms,
            "exit_time_ms": exit_ms,
            "side": side,
            "ret_gross": ret_gross,
        }
    )
    for col in ("liquidity_tier", "pit_rank_prev_1d", "universe_rank"):
        if col in frame.columns:
            out[col] = frame[col].to_numpy()
    if ambiguous is not None:
        out["barrier_ambiguous"] = ambiguous
    out = out.dropna(subset=["ret_gross", "entry_time_ms", "exit_time_ms"])
    out["entry_time_ms"] = out["entry_time_ms"].astype("int64")
    out["exit_time_ms"] = out["exit_time_ms"].astype("int64")
    out["weight"] = 1.0
    return out.sort_values(["entry_time_ms", "zone_uid"], kind="mergesort").reset_index(drop=True)


def ambiguity_sensitivity(
    trades: pd.DataFrame, spec: TradeSpec, mean_ret_net: Optional[float]
) -> Dict[str, object]:
    """How much a barrier result leans on the same-bar rule.

    An ambiguous short is booked as a win of ``take_profit``. Booking it as a loss of
    ``stop_loss`` instead moves that trade by ``-(take_profit + stop_loss)``; costs do
    not change, so the net mean moves by the same amount per trade.
    """
    if spec.exit != "barrier" or "barrier_ambiguous" not in trades.columns or trades.empty:
        return {}
    assert spec.take_profit is not None and spec.stop_loss is not None  # a barrier spec has both
    amb = trades["barrier_ambiguous"].to_numpy(dtype=bool)
    short = trades["side"].to_numpy() < 0
    frac_short = float((amb & short).mean())
    shift = -(float(spec.take_profit) + float(spec.stop_loss)) * frac_short
    return {
        "ambiguous_fraction": float(amb.mean()),
        "ambiguous_short_fraction": frac_short,
        "mean_ret_net_if_ambiguous_shorts_lose": None
        if mean_ret_net is None
        else float(mean_ret_net) + shift,
    }


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_uid": pd.Series(dtype="string"),
            "symbol": pd.Series(dtype="string"),
            "decision_time_ms": pd.Series(dtype="int64"),
            "entry_time_ms": pd.Series(dtype="int64"),
            "exit_time_ms": pd.Series(dtype="int64"),
            "side": pd.Series(dtype="int64"),
            "ret_gross": pd.Series(dtype="float64"),
            "weight": pd.Series(dtype="float64"),
        }
    )


def liquidity_tier(ds: pd.DataFrame, col: str = "pit_rank_prev_1d") -> pd.Series:
    """Tier by the point-in-time volume rank, not by today's ranking."""
    if col not in ds.columns:
        return pd.Series(["unknown"] * len(ds), index=ds.index, dtype="object")
    rank = ds[col]
    return pd.Series(
        np.select(
            [rank <= 20, rank <= 50, rank <= 100, rank.notna()],
            ["top20", "21-50", "51-100", "100+"],
            default="unknown",
        ),
        index=ds.index,
        dtype="object",
    )


def illiquidity_score(ds: pd.DataFrame, col: str = "pit_rank_prev_1d") -> pd.Series:
    """Non-negative score feeding the slippage model. 0 = most liquid.

    Uses the point-in-time volume rank so slippage on a coin that was thin in
    2024 is not modelled using its 2026 liquidity.
    """
    if col not in ds.columns or ds[col].isna().all():
        return pd.Series(np.zeros(len(ds)), index=ds.index)
    rank = ds[col].astype(float)
    # Rank 1 -> 0, rank 100 -> ~1, rank 400 -> ~2. Log so the penalty grows
    # quickly across the liquid head and slowly across the illiquid tail.
    score = np.log10(np.clip(rank.to_numpy(), 1.0, None)) / 2.0
    return pd.Series(np.nan_to_num(score, nan=1.0), index=ds.index)
