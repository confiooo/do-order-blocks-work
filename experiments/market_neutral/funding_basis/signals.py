"""Causal funding signals and the empirical entry thresholds.

A value at a bar uses only settlements whose timestamp is at or before that
bar's close.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from experiments.market_neutral.calendar import development_bounds
from experiments.market_neutral.data.bars import HOUR_MS


def latest_funding_index(close_time_ms: np.ndarray, funding_time_ms: np.ndarray) -> np.ndarray:
    """Index of the latest settlement at or before each close, or -1."""
    close = np.asarray(close_time_ms, dtype=np.int64)
    funding = np.asarray(funding_time_ms, dtype=np.int64)
    if funding.size == 0:
        return np.full(close.shape, -1, dtype=np.int64)
    return np.searchsorted(funding, close, side="right").astype(np.int64) - 1


def rolling_funding_mean(rates: np.ndarray, latest: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` settlements known at each bar."""
    if window < 1:
        raise ValueError("window must be positive")
    rate = np.asarray(rates, dtype=float)
    idx = np.asarray(latest, dtype=np.int64)
    out = np.full(idx.shape, np.nan, dtype=float)
    if rate.size == 0:
        return out
    cumulative = np.cumsum(np.where(np.isfinite(rate), rate, np.nan))
    valid = (idx + 1 >= window) & (idx < rate.size)
    chosen = idx[valid]
    left = chosen - window
    total = cumulative[chosen].copy()
    has_left = left >= 0
    total[has_left] = total[has_left] - cumulative[left[has_left]]
    out[valid] = total / float(window)
    return out


def perp_price_at_funding(
    open_time_ms: np.ndarray,
    perp_close: np.ndarray,
    funding_time_ms: np.ndarray,
) -> np.ndarray:
    """Last completed perp close at or before each settlement. No later bar."""
    close_time = np.asarray(open_time_ms, dtype=np.int64) + HOUR_MS
    funding = np.asarray(funding_time_ms, dtype=np.int64)
    price = np.full(funding.shape, np.nan, dtype=float)
    if close_time.size == 0 or funding.size == 0:
        return price
    bar = np.searchsorted(close_time, funding, side="right").astype(np.int64) - 1
    ok = bar >= 0
    price[ok] = np.asarray(perp_close, dtype=float)[bar[ok]]
    return price


def development_funding_sample(frames: Iterable[pd.DataFrame]) -> np.ndarray:
    """Settlement rates inside tier D, concatenated across names."""
    start, end = development_bounds()
    parts: List[np.ndarray] = []
    for frame in frames:
        time = frame["funding_time_ms"].to_numpy(dtype=np.int64)
        rate = frame["funding_rate"].to_numpy(dtype=float)
        keep = (time >= start) & (time < end) & np.isfinite(rate)
        parts.append(rate[keep])
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


def distinct_thresholds(pooled: np.ndarray) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """Three strictly increasing entry levels taken from the empirical distribution.

    The reported 75th and 90th percentiles are kept even when they fall on the
    same point mass. A repeated level is not used twice: the third level is
    then the 97th percentile, the first percentile above that mass.
    """
    values = np.asarray(pooled, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise RuntimeError("not enough tier-D funding settlements to set thresholds")
    report = {
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p97": float(np.quantile(values, 0.97)),
    }
    if np.isclose(report["p75"], report["p90"]):
        levels = [
            ("p50", report["p50"]),
            ("p75_p90", report["p75"]),
            ("p97", report["p97"]),
        ]
        report["collapse"] = "p75 equals p90; third level is p97 so the grid does not repeat a cell"
    else:
        levels = [
            ("p50", report["p50"]),
            ("p75", report["p75"]),
            ("p90", report["p90"]),
        ]
        report["collapse"] = None
    ordered = [level[1] for level in levels]
    if not (ordered[0] < ordered[1] < ordered[2]):
        raise RuntimeError(f"thresholds are not strictly increasing: {ordered}")
    return levels, report


def per_name_percentiles(frame: pd.DataFrame) -> Dict[str, float]:
    start, end = development_bounds()
    time = frame["funding_time_ms"].to_numpy(dtype=np.int64)
    rate = frame["funding_rate"].to_numpy(dtype=float)
    sample = rate[(time >= start) & (time < end) & np.isfinite(rate)]
    if sample.size == 0:
        return {"n": 0, "p50": None, "p75": None, "p90": None}
    q50, q75, q90 = np.quantile(sample, [0.50, 0.75, 0.90])
    return {"n": int(sample.size), "p50": float(q50), "p75": float(q75), "p90": float(q90)}
