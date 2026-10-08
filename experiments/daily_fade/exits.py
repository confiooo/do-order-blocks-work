"""Exit ideas for a short taken at the next UTC open after a up-day.

Each rule is reported on its own. Nothing here searches a new threshold after
seeing results. Costs are a flat spot round trip, subtracted once at the end.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from experiments.daily_fade.sessions import BARS_PER_DAY, SESSION_MARKS, SESSIONS, bar_index

# Binance spot taker 10 bps per side, plus the 4 bp top-of-book spread from
# config/costs.json. One round trip, not a swept cost.
ROUND_TRIP_COST = 0.0024


def short_return(entry: float, exit_px: float) -> float:
    return float(entry - exit_px) / float(entry)


def session_path(forward: pd.DataFrame, entry: float) -> Dict[str, float]:
    """Short return from the entry open to each session boundary, and within each session."""
    opens = forward["open"].to_numpy(dtype=float)
    closes = forward["close"].to_numpy(dtype=float)
    out: Dict[str, float] = {}
    for name, minutes in SESSION_MARKS:
        index = bar_index(minutes)
        if name == "next_daily_close":
            price = float(closes[-1])
        else:
            price = float(opens[index])
        out[name] = short_return(entry, price)
    for name, start_min, end_min in SESSIONS:
        start = bar_index(start_min)
        end = bar_index(end_min)
        start_px = float(opens[start]) if start > 0 else float(entry)
        end_px = float(closes[-1]) if end >= BARS_PER_DAY else float(opens[end])
        out[f"during_{name}"] = short_return(start_px, end_px)
    return out


def _first_touch(
    high: np.ndarray, low: np.ndarray, stop: Optional[float], target: Optional[float]
) -> Optional[tuple]:
    """First bar where the short's stop or target is touched. Stop wins if both print."""
    for i in range(high.size):
        stopped = stop is not None and float(high[i]) >= stop
        targeted = target is not None and float(low[i]) <= target
        if stopped:
            return i, float(stop), "stop"
        if targeted:
            return i, float(target), "target"
    return None


def evaluate_exit(
    forward: pd.DataFrame,
    entry: float,
    prior_close: float,
    day_range: float,
    rule: str,
) -> Dict[str, object]:
    opens = forward["open"].to_numpy(dtype=float)
    highs = forward["high"].to_numpy(dtype=float)
    lows = forward["low"].to_numpy(dtype=float)
    closes = forward["close"].to_numpy(dtype=float)
    gain = float(entry - prior_close)
    if rule == "time_london":
        index = bar_index(8 * 60)
        exit_px, reason = float(opens[index]), "london_open"
    elif rule == "time_new_york":
        index = bar_index(13 * 60 + 30)
        exit_px, reason = float(opens[index]), "new_york_open"
    elif rule == "time_new_york_close":
        index = bar_index(21 * 60)
        exit_px, reason = float(opens[index]), "new_york_close"
    elif rule == "time_24h":
        index = BARS_PER_DAY - 1
        exit_px, reason = float(closes[index]), "next_daily_close"
    elif rule == "giveback_half":
        target = entry - 0.5 * gain
        hit = _first_touch(highs, lows, None, target)
        if hit is None:
            index, exit_px, reason = BARS_PER_DAY - 1, float(closes[-1]), "time_fallback"
        else:
            index, exit_px, reason = hit
    elif rule == "giveback_full":
        hit = _first_touch(highs, lows, None, float(prior_close))
        if hit is None:
            index, exit_px, reason = BARS_PER_DAY - 1, float(closes[-1]), "time_fallback"
        else:
            index, exit_px, reason = hit
    elif rule == "continuation_stop":
        stop = entry + 0.5 * float(day_range)
        hit = _first_touch(highs, lows, stop, None)
        if hit is None:
            index, exit_px, reason = BARS_PER_DAY - 1, float(closes[-1]), "time_fallback"
        else:
            index, exit_px, reason = hit
    elif rule == "half_or_stop_or_24h":
        target = entry - 0.5 * gain
        stop = entry + 0.5 * float(day_range)
        hit = _first_touch(highs, lows, stop, target)
        if hit is None:
            index, exit_px, reason = BARS_PER_DAY - 1, float(closes[-1]), "time_fallback"
        else:
            index, exit_px, reason = hit
    else:
        raise ValueError(f"unknown exit rule {rule}")
    gross = short_return(entry, float(exit_px))
    return {
        "rule": rule,
        "exit_bar": int(index),
        "exit_price": float(exit_px),
        "exit_reason": reason,
        "ret_gross": gross,
        "ret_net": gross - ROUND_TRIP_COST,
        "hold_hours": hold_hours(rule, int(index)),
    }


def hold_hours(rule: str, exit_bar: int) -> float:
    """Time from the entry open to the exit price's timestamp."""
    if rule in {"time_london", "time_new_york", "time_new_york_close"}:
        return exit_bar * 0.25
    return (exit_bar + 1) * 0.25


EXIT_RULES = (
    "time_london",
    "time_new_york",
    "time_new_york_close",
    "time_24h",
    "giveback_half",
    "giveback_full",
    "continuation_stop",
    "half_or_stop_or_24h",
)
