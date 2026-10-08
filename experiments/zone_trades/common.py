"""What the zone-trade scripts share: per-symbol 15m bars, the forward window after an
alert, and a worker pool that keeps results in the order a serial loop would give.

A window starts at the bar after the alert bar and runs while bars are contiguous, so
no simulated path crosses a data gap.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.data import loader
from hawk_research.parallel import process_pool

OHLC = ["open_time", "open", "high", "low", "close"]
# Results do not depend on the pool size (map keeps the serial order).
WORKERS = min(8, os.cpu_count() or 1)


def symbol_bars(symbol: str) -> Optional[Dict[str, np.ndarray]]:
    """The symbol's 15m bars as arrays, or ``None`` if there are none."""
    bars = loader.load_ohlcv(symbol, columns=OHLC, missing_ok=True)
    if bars.empty:
        return None
    bars = bars.sort_values("open_time")
    out = {c: bars[c].to_numpy(dtype=np.float64) for c in OHLC[1:]}
    out["open_time"] = bars["open_time"].to_numpy(dtype=np.int64)
    return out


def forward_window(
    opens: np.ndarray, source_open_time: int, max_bars: int, min_bars: int = 16
) -> Optional[slice]:
    """Bars after the alert bar, up to ``max_bars`` and only while contiguous.

    ``None`` when the next bar is missing or fewer than ``min_bars`` contiguous bars follow.
    """
    start = int(source_open_time) + BAR_MS
    i0 = int(np.searchsorted(opens, start, side="left"))
    i1 = min(i0 + max_bars, len(opens))
    if i0 >= len(opens) or i1 <= i0 or int(opens[i0]) != start:
        return None
    gap = np.flatnonzero(np.diff(opens[i0:i1]) != BAR_MS)
    n = int(gap[0] + 1) if len(gap) else i1 - i0
    return slice(i0, i0 + n) if n >= min_bars else None


def map_symbols(fn: Callable[..., Any], events: pd.DataFrame, **kwargs: Any) -> List[Any]:
    """``fn(symbol, rows, **kwargs)`` per symbol, in first-appearance order, across worker processes."""
    groups = list(events.groupby("symbol", sort=False))
    if not groups:
        return []
    symbols, frames = zip(*groups)
    with process_pool(WORKERS) as pool:
        return list(pool.map(partial(fn, **kwargs), symbols, frames))
