"""BTC stability gate, vendored from the live bot.

The live bot dropped a detection when BTC's recent high-low range was too wide.
The replay reproduces that gate as a flag (``btc_stable``), using these functions
unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Candle = Tuple[int, float, float, float, float, float]


def compute_volatility(candles: Sequence[Candle]) -> float:
    """(max high - min low) / mean close over ``candles``."""
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    max_price = max(highs)
    min_price = min(lows)
    mean_price = sum(closes) / len(closes)
    return (max_price - min_price) / mean_price if mean_price != 0 else math.inf


def btc_range_at(
    candles: Sequence[Candle],
    as_of_open_time: int,
    window_bars: int,
) -> float:
    """BTC high-low range using only bars at or before ``as_of_open_time``."""
    if window_bars < 1 or not candles:
        return math.inf
    window = [c for c in candles if c[0] <= as_of_open_time]
    window = window[-window_bars:]
    if not window:
        return math.inf
    return compute_volatility(window)
