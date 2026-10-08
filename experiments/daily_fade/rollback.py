"""How far price comes back through the daily close over the next 24 hours.

Rollback is measured from the candle close, the last price of the signal day.
For a gainer the rollback is the deepest print below that close. For a loser
it is the deepest print above that close. The figure kept for each day is
that maximum, and the study averages those maxima.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from experiments.daily_fade.sessions import bar_index

# Bars whose open is strictly before each mark. The mark itself is the next open.
_WINDOWS = (
    ("by_london_open", bar_index(8 * 60)),
    ("by_new_york_open", bar_index(13 * 60 + 30)),
    ("by_new_york_close", bar_index(21 * 60)),
    ("by_next_close", None),
)


def _rollback_path(forward: pd.DataFrame, close: float, side: int) -> np.ndarray:
    """Rollback of each 15m bar versus the signal close. Negative means no giveback yet."""
    if side < 0:
        extreme = forward["low"].to_numpy(dtype=float)
        return (float(close) - extreme) / float(close)
    extreme = forward["high"].to_numpy(dtype=float)
    return (extreme - float(close)) / float(close)


def maximum_rollback(
    forward: pd.DataFrame, close: float, side: int, prior_close: float
) -> Dict[str, Optional[float]]:
    """Deepest rollback after the close, and how much of the 24h candle that is."""
    path = _rollback_path(forward, close, side)
    index = int(np.argmax(path))
    raw = float(path[index])
    rolled = raw > 0.0
    impulse = abs(float(close) - float(prior_close))
    price = max(raw, 0.0) * float(close)
    out: Dict[str, Optional[float]] = {
        "max_rollback": max(raw, 0.0),
        "rollback_bar": float(index) if rolled else None,
        "rollback_hours": (index * 0.25) if rolled else None,
        "retrace_of_candle": (price / impulse) if rolled and impulse > 0.0 else 0.0,
    }
    for name, end in _WINDOWS:
        window = path if end is None else path[:end]
        out[name] = float(max(0.0, window.max())) if window.size else 0.0
    if side < 0:
        continuation = (float(forward["high"].max()) - float(close)) / float(close)
    else:
        continuation = (float(close) - float(forward["low"].min())) / float(close)
    out["max_continuation"] = max(float(continuation), 0.0)
    return out
