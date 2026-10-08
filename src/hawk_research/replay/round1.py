"""Point-in-time replay of the live round-1 detector over the full history.

``detect_order_blocks`` takes its lookback from the length of the list it is
given (``int(len(candles) * lookback_ratio)``). Live, the collector passed 400
bars with a ratio of 0.15, i.e. a 60-bar lookback; passing 70,000 bars would give
a 10,500-bar lookback and a different detector. The replay therefore slides a
400-bar window and only takes candidates at offset >= 60 inside it, so each bar
is judged against the same 60 bars the live collector used. Windows step by
400 - 60, so every bar is evaluated once. The detector itself is imported
unchanged.

Live gating: ``engine.run_loop`` dropped detections while the BTC 4h range was
at or above the stability threshold, so every live row has ``btc_stable = 1``.
``signal_gate`` and ``min_signal_interval_per_pair`` only decided whether an
alert was sent. All three are stored as flags, so both the pre-gate and the
live-equivalent populations are available.

Each replayed row has the detector's fields, the trailing 1h/4h/24h returns the
live engine stored (``symbol_return_*``, backward-looking despite the name), the
live confidence score and the gate flags: the columns of the live ``zones``
table, which s3 compares field by field. Forward outcomes are added in s5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hawk_research import config as research_config
from hawk_research.config import ResearchConfig
from hawk_research.constants import BAR_MS
from hawk_research.detectors.order_blocks import detect_order_blocks
from hawk_research.replay.live_scoring import passes_signal_gate, score_order_block

Candle = Tuple[int, float, float, float, float, float]

_REPLAY = research_config.get().replay

# Live collector geometry. Changing these changes the detector (see module docstring).
LIVE_WINDOW = _REPLAY.live_window  # 400
LOOKBACK_RATIO = _REPLAY.lookback_ratio  # 0.15
EFFECTIVE_LOOKBACK = int(LIVE_WINDOW * LOOKBACK_RATIO)  # == 60

# Bars of causal history required before a replayed bar may produce a zone.
# The live collector always held a 400-bar archive, so matching that keeps the
# replayed and live populations comparable.
WARMUP_BARS = _REPLAY.warmup_bars

RETURN_BARS = {"1h": 4, "4h": 16, "24h": 96}
BTC_RANGE_BARS = _REPLAY.btc_range_bars  # 4h of 15m bars


@dataclass(frozen=True)
class DetectorParams:
    strength_threshold: float
    wick_to_body_ratio_min: float
    min_distance_from_high_low: float
    btc_volatility_threshold: float
    min_volume_ratio: float
    max_distance_to_price: float
    min_signal_interval_minutes: int

    @classmethod
    def from_research_config(cls, cfg: Optional[ResearchConfig] = None) -> "DetectorParams":
        """Detector thresholds and live gates from the research configuration."""
        cfg = cfg or research_config.get()
        if cfg.replay.live_window != LIVE_WINDOW or cfg.replay.lookback_ratio != LOOKBACK_RATIO:
            raise ValueError("replay geometry changed after import; activate the config first")
        return cls(
            strength_threshold=cfg.detector.strength_threshold,
            wick_to_body_ratio_min=cfg.detector.wick_to_body_ratio_min,
            min_distance_from_high_low=cfg.detector.min_distance_from_high_low,
            btc_volatility_threshold=cfg.live_gates.btc_volatility_threshold,
            min_volume_ratio=cfg.live_gates.min_volume_ratio,
            max_distance_to_price=cfg.live_gates.max_distance_to_price,
            min_signal_interval_minutes=cfg.live_gates.min_signal_interval_minutes,
        )


def rolling_windows(n_bars: int, warmup: int = WARMUP_BARS) -> List[Tuple[int, int, int]]:
    """``(window_start, window_end_exclusive, first_candidate_index)`` triples.

    Every window is exactly ``LIVE_WINDOW`` long so the detector's derived
    lookback is always 60. Together the triples cover each bar index in
    ``[warmup, n_bars)`` exactly once.
    """
    if n_bars < LIVE_WINDOW:
        return []
    out: List[Tuple[int, int, int]] = []
    step = LIVE_WINDOW - EFFECTIVE_LOOKBACK
    next_candidate = max(warmup, EFFECTIVE_LOOKBACK)
    start = max(0, next_candidate - EFFECTIVE_LOOKBACK)
    while start + LIVE_WINDOW <= n_bars:
        end = start + LIVE_WINDOW
        out.append((start, end, next_candidate))
        next_candidate = end
        if next_candidate >= n_bars:
            return out
        start += step
    # Tail: re-anchor the last full window on the end of the series so the
    # remaining bars are still judged inside a 400-bar window.
    start = n_bars - LIVE_WINDOW
    if next_candidate < n_bars and next_candidate >= start + EFFECTIVE_LOOKBACK:
        out.append((start, n_bars, next_candidate))
    return out


def btc_range_series(btc_candles: Sequence[Candle], bars: int = BTC_RANGE_BARS) -> Dict[int, float]:
    """``open_time -> `` trailing 4h BTC range, matching ``btc_range_at`` exactly.

    ``btc_range_at`` keeps bars with ``open_time <= as_of``, takes the last
    ``bars`` of them, and returns ``(max high - min low) / mean close``. That is
    a trailing window *inclusive* of the as-of bar, so the rolling window below
    must not be shifted.
    """
    if not btc_candles:
        return {}
    arr = np.asarray(btc_candles, dtype=object)
    times = np.array([int(c[0]) for c in btc_candles], dtype=np.int64)
    high = pd.Series([float(c[2]) for c in btc_candles], dtype=float)
    low = pd.Series([float(c[3]) for c in btc_candles], dtype=float)
    close = pd.Series([float(c[4]) for c in btc_candles], dtype=float)
    max_high = high.rolling(bars, min_periods=1).max()
    min_low = low.rolling(bars, min_periods=1).min()
    mean_close = close.rolling(bars, min_periods=1).mean()
    rng = np.where(
        mean_close.to_numpy() != 0,
        (max_high.to_numpy() - min_low.to_numpy()) / mean_close.to_numpy(),
        np.inf,
    )
    del arr
    return dict(zip(times.tolist(), rng.tolist()))


def _verify_btc_range_series(btc_candles: Sequence[Candle], sample: int = 200, seed: int = 7) -> int:
    """Spot-check the vectorised BTC range against the live scalar helper."""
    from hawk_research.replay.btc_gate import btc_range_at

    series = btc_range_series(btc_candles)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(btc_candles), size=min(sample, len(btc_candles)), replace=False)
    bad = 0
    for i in idx:
        as_of = int(btc_candles[int(i)][0])
        expected = btc_range_at(btc_candles, as_of, BTC_RANGE_BARS)
        got = series.get(as_of)
        if got is None or not np.isclose(expected, got, rtol=1e-12, atol=0):
            bad += 1
    return bad


def checked_btc_range_series(btc_candles: Sequence[Candle]) -> Dict[int, float]:
    """``btc_range_series``, after spot-checking it against the live scalar helper."""
    bad = _verify_btc_range_series(btc_candles)
    if bad:
        raise SystemExit(f"vectorised BTC range does not reproduce btc_range_at ({bad} mismatches)")
    return btc_range_series(btc_candles)


def _closed_return(closes: np.ndarray, idx: int, bars: int) -> Optional[float]:
    """Mirror of ``engine._closed_return_at``: close[idx] vs close[idx-bars]."""
    if idx < bars:
        return None
    past = closes[idx - bars]
    if past == 0:
        return None
    return float((closes[idx] - past) / past)


def replay_symbol(
    symbol: str,
    candles: Sequence[Candle],
    params: DetectorParams,
    btc_range: Dict[int, float],
    warmup: int = WARMUP_BARS,
) -> pd.DataFrame:
    """Every detector hit on one symbol, with the live gating flags attached."""
    n = len(candles)
    windows = rolling_windows(n, warmup=warmup)
    if not windows:
        return pd.DataFrame()

    index_by_open = {int(c[0]): i for i, c in enumerate(candles)}
    closes = np.array([c[4] for c in candles], dtype=float)
    rows: List[Dict] = []

    for start, end, first_candidate in windows:
        window = list(candles[start:end])
        candidate_opens = [int(c[0]) for c in candles[first_candidate:end]]
        if not candidate_opens:
            continue
        blocks = detect_order_blocks(
            window,
            lookback_ratio=LOOKBACK_RATIO,
            strength_threshold=params.strength_threshold,
            wick_to_body_ratio_min=params.wick_to_body_ratio_min,
            min_distance_from_high_low=params.min_distance_from_high_low,
            candidate_open_times=candidate_opens,
        )
        for b in blocks:
            idx = index_by_open[int(b.source_open_time)]
            ret_1h = _closed_return(closes, idx, RETURN_BARS["1h"])
            ret_4h = _closed_return(closes, idx, RETURN_BARS["4h"])
            ret_24h = _closed_return(closes, idx, RETURN_BARS["24h"])
            brange = btc_range.get(int(b.source_open_time))
            btc_stable = bool(brange is not None and brange < params.btc_volatility_threshold)
            rows.append(
                {
                    "symbol": symbol,
                    "source_open_time": int(b.source_open_time),
                    "bar_index": idx,
                    "direction": b.direction,
                    "zone_high": b.zone_high,
                    "zone_low": b.zone_low,
                    "zone_midpoint": b.zone_midpoint,
                    "zone_size_pct": b.zone_size_pct,
                    "size": b.size,
                    "volume_ratio": b.volume_ratio,
                    "wick_ratio": b.wick_ratio,
                    "distance_to_swing": b.distance_to_swing,
                    "distance_to_price": b.distance_to_price,
                    "symbol_return_1h": ret_1h,
                    "symbol_return_4h": ret_4h,
                    "symbol_return_24h": ret_24h,
                    "btc_range_4h_pct": None if brange is None or not np.isfinite(brange) else float(brange),
                    "btc_stable": btc_stable,
                    "confidence": score_order_block(b.volume_ratio, b.direction, ret_4h, b.distance_to_price),
                    "passes_signal_gate": passes_signal_gate(
                        b.volume_ratio,
                        b.distance_to_price,
                        min_volume_ratio=params.min_volume_ratio,
                        max_distance_to_price=params.max_distance_to_price,
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("source_open_time", kind="mergesort").reset_index(drop=True)

    # Live-equivalent population: what engine.run_loop would have inserted.
    df["live_equivalent"] = df["btc_stable"]

    # Alert eligibility chain, applied over the live-equivalent rows only, in
    # time order, exactly as signal_control does on the alerts table.
    eligible = df["live_equivalent"] & df["passes_signal_gate"]
    min_gap_ms = params.min_signal_interval_minutes * 60 * 1000
    would_alert = np.zeros(len(df), dtype=bool)
    last_alert_ms: Optional[int] = None
    for i in np.flatnonzero(eligible.to_numpy()):
        # Alert time is the bar's close, i.e. the moment the collector saw it.
        t = int(df["source_open_time"].iat[int(i)]) + BAR_MS
        if last_alert_ms is None or (t - last_alert_ms) >= min_gap_ms:
            would_alert[int(i)] = True
            last_alert_ms = t
    df["would_alert"] = would_alert
    return df
