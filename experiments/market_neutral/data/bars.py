"""Load Binance spot 15m bars and resample them to completed 1h bars.

Only tier D is returned. Firewall, holdout, and the post-holdout September
slice are dropped inside the parquet read, before any feature sees them.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from experiments.market_neutral.calendar import development_bounds
from hawk_research.constants import MS_PER_HOUR
from hawk_research.data import paths

HOUR_MS = MS_PER_HOUR
STEP_15_MS = 900_000
_OFFSETS = np.array([0, STEP_15_MS, 2 * STEP_15_MS, 3 * STEP_15_MS], dtype=np.int64)


def development_15m(symbol: str) -> pd.DataFrame:
    """15m bars whose open time is inside tier D. Later bars are not loaded."""
    start_ms, end_ms = development_bounds()
    path = paths.ohlcv_path(symbol, "15m")
    if not path.exists():
        raise FileNotFoundError(f"no 15m OHLCV for {symbol} at {path}")
    frame = pd.read_parquet(
        path,
        columns=["open_time", "open", "high", "low", "close", "volume"],
        filters=[("open_time", ">=", start_ms), ("open_time", "<", end_ms)],
    )
    frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        return frame
    times = frame["open_time"].to_numpy(dtype=np.int64)
    if int(times.min()) < start_ms or int(times.max()) >= end_ms:
        raise RuntimeError(f"{symbol} 15m read leaked outside tier D")
    return frame


def resample_15m_to_1h(frame: pd.DataFrame) -> pd.DataFrame:
    """Completed clock hours only: four consecutive 15m bars, open at the hour."""
    if frame.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    ordered = frame.sort_values("open_time", kind="mergesort")
    times = ordered["open_time"].to_numpy(dtype=np.int64)
    hour = (times // HOUR_MS) * HOUR_MS
    offset = times - hour
    keep = np.isin(offset, _OFFSETS)
    ordered = ordered.loc[keep].copy()
    hour = hour[keep]
    offset = offset[keep]
    if ordered.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    bucket = pd.DataFrame(
        {
            "open_time": hour,
            "offset": offset,
            "open": ordered["open"].to_numpy(dtype=float),
            "high": ordered["high"].to_numpy(dtype=float),
            "low": ordered["low"].to_numpy(dtype=float),
            "close": ordered["close"].to_numpy(dtype=float),
            "volume": ordered["volume"].to_numpy(dtype=float),
        }
    )
    counts = bucket.groupby("open_time", sort=True)["offset"].nunique()
    complete = counts[counts == 4].index
    bucket = bucket[bucket["open_time"].isin(complete)].sort_values(["open_time", "offset"], kind="mergesort")
    grouped = bucket.groupby("open_time", sort=True)
    out = pd.DataFrame(
        {
            "open_time": grouped["open_time"].first().to_numpy(dtype=np.int64),
            "open": grouped["open"].first().to_numpy(dtype=float),
            "high": grouped["high"].max().to_numpy(dtype=float),
            "low": grouped["low"].min().to_numpy(dtype=float),
            "close": grouped["close"].last().to_numpy(dtype=float),
            "volume": grouped["volume"].sum().to_numpy(dtype=float),
        }
    )
    return out.reset_index(drop=True)


def development_1h(symbol: str) -> pd.DataFrame:
    """1h bars for tier D, built from 15m. Close of hour H is known at H+1h."""
    bars = resample_15m_to_1h(development_15m(symbol))
    _, end_ms = development_bounds()
    if bars.empty:
        return bars
    # The hour that opens at end_ms - 1h closes exactly at end_ms, the firewall boundary.
    close_known = bars["open_time"].to_numpy(dtype=np.int64) + HOUR_MS
    return bars.loc[close_known <= end_ms].reset_index(drop=True)


def align_closes(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Inner join on hour open. Hours missing on either leg are dropped, not filled."""
    a = left[["open_time", "close"]].rename(columns={"close": "close_a"})
    b = right[["open_time", "close"]].rename(columns={"close": "close_b"})
    merged = a.merge(b, on="open_time", how="inner")
    return merged.sort_values("open_time", kind="mergesort").reset_index(drop=True)


def load_closes(symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    return {symbol: development_1h(symbol) for symbol in symbols}
