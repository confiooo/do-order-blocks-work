"""Retest, entry and excursion rules, vendored from the live bot's outcome tracker.

These are the exact functions the live system used to decide whether a zone was
retested, where the hypothetical entry sat, and how far price moved afterwards.
The research labels import them unchanged so the replayed lifecycle cannot drift
from what the live book recorded. Only the pieces the research uses are kept.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Candle = Tuple[int, float, float, float, float, float]


def intersects_zone(high: float, low: float, zone_high: float, zone_low: float) -> bool:
    return low <= zone_high and high >= zone_low


def is_valid_retest(
    direction: str,
    zone_high: float,
    zone_low: float,
    prev_close: float,
    candle: Candle,
) -> bool:
    """First return to the zone from the correct side, not a smash-through.

    Bullish: previous close is above the zone, this candle intersects it,
    and it does not close below zone_low.
    Bearish: previous close is below the zone, this candle intersects it,
    and it does not close above zone_high.
    """
    _open_time, _o, high, low, close, _v = candle
    intersects = intersects_zone(high, low, zone_high, zone_low)
    if direction == "bullish":
        established_above = prev_close > zone_high
        did_not_close_through = close >= zone_low
        return established_above and intersects and did_not_close_through
    established_below = prev_close < zone_low
    did_not_close_through = close <= zone_high
    return established_below and intersects and did_not_close_through


def entry_price(direction: str, zone_high: float, zone_low: float) -> float:
    """Resting limit at the near edge: the top of a bullish zone, the bottom of a bearish one."""
    return zone_high if direction == "bullish" else zone_low


def find_first_retest(
    candles: Sequence[Candle],
    source_open_time: int,
    direction: str,
    zone_high: float,
    zone_low: float,
) -> Optional[Candle]:
    """Return the first later closed candle that retests from the correct side."""
    later = [c for c in candles if c[0] > source_open_time]
    if not later:
        return None
    prior = [c for c in candles if c[0] <= source_open_time]
    series = ([prior[-1]] + later) if prior else later
    if len(series) < 2:
        return None

    for i in range(1, len(series)):
        prev_close = series[i - 1][4]
        current = series[i]
        if current[0] <= source_open_time:
            continue
        if is_valid_retest(direction, zone_high, zone_low, prev_close, current):
            return current
    return None


def candles_after_fill(candles: Sequence[Candle], fill_open_ms: int) -> List[Candle]:
    """Closed bars strictly after the retest/fill bar.

    The fill bar is excluded: its own high/low can mechanically print both MFE
    and MAE against an entry at the zone edge. Performance is the subsequent path.
    +1000 ms of slack absorbs ISO-roundtrip jitter without pulling the fill bar
    back in.
    """
    return [c for c in candles if c[0] > fill_open_ms + 1000]


def compute_excursions(
    direction: str,
    entry: float,
    window: Sequence[Candle],
) -> Tuple[float, float]:
    """(MFE, MAE) as fractions of ``entry``, rounded to 6 places as the live tracker stored them."""
    max_high = max(c[2] for c in window)
    min_low = min(c[3] for c in window)
    if entry == 0:
        return 0.0, 0.0
    if direction == "bullish":
        mfe = (max_high - entry) / entry
        mae = (entry - min_low) / entry
    else:
        mfe = (entry - min_low) / entry
        mae = (max_high - entry) / entry
    return round(mfe, 6), round(mae, 6)
