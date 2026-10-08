"""Textbook (SMC/ICT) order blocks, detected point in time.

Round 1's ``detect_order_blocks`` flags high-volume, long-wick candles and takes
the direction from the candle's colour. This module implements the usual
textbook definition instead:

1. Swing points are 3-bar fractals: a high above the 3 highs on each side
   (mirrored for lows). A fractal at ``j`` is known at the close of ``j + 3``.
2. Break of structure (BOS): a candle closes beyond the latest confirmed,
   unbroken swing high (bullish) or swing low (bearish).
3. Order block: the last opposite-coloured candle at or before the extreme of
   the impulse leg that made the BOS. The zone is that candle's high-low range;
   the direction is the BOS direction.
4. Fair value gap: flagged when the leg from the OB to the BOS bar left a
   3-candle gap in the break direction.
5. Unmitigated: if price left the zone and came back into it before the BOS
   bar closed, the OB is dropped.

The event time is the BOS bar (``source_open_time``); nothing depends on later
bars. The rule constants below are fixed, not tuned.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from hawk_research import config as research_config

Candle = Tuple[int, float, float, float, float, float]

_SMC = research_config.get().smc
FRACTAL_BARS = int(_SMC.fractal_bars)
# How far back from the BOS bar the OB candle may sit (default 8h of 15m bars).
MAX_OB_AGE_BARS = int(_SMC.max_ob_age_bars)
# How far back from the leg extreme to look for an opposite-coloured candle.
MAX_OB_SEARCH_BARS = int(_SMC.max_ob_search_bars)
VOLUME_LOOKBACK = int(_SMC.volume_lookback)


@dataclass(frozen=True)
class SmcZone:
    source_open_time: int  # BOS bar, the decision bar
    bar_index: int
    ob_open_time: int
    ob_index: int
    zone_high: float
    zone_low: float
    zone_midpoint: float
    zone_size_pct: float
    direction: str
    has_fvg: bool
    swing_price: float
    swing_open_time: int
    volume_ratio: float
    wick_ratio: float


@dataclass
class _Swing:
    index: int
    price: float
    broken: bool = False


def _is_fractal_high(highs: Sequence[float], j: int) -> bool:
    h = highs[j]
    return all(h > highs[k] for k in range(j - FRACTAL_BARS, j + FRACTAL_BARS + 1) if k != j)


def _is_fractal_low(lows: Sequence[float], j: int) -> bool:
    lo = lows[j]
    return all(lo < lows[k] for k in range(j - FRACTAL_BARS, j + FRACTAL_BARS + 1) if k != j)


def _has_fvg(candles: Sequence[Candle], start: int, end: int, bullish: bool) -> bool:
    """Any 3-candle gap with its middle candle in ``(start, end)``."""
    for k in range(start + 1, end):
        if bullish and candles[k + 1][3] > candles[k - 1][2]:
            return True
        if not bullish and candles[k + 1][2] < candles[k - 1][3]:
            return True
    return False


def _mitigated_before(
    candles: Sequence[Candle], ob: int, end: int, zone_high: float, zone_low: float, bullish: bool
) -> bool:
    """Did price leave the zone and come back into it within ``(ob, end]``?"""
    departed = False
    for k in range(ob + 1, end + 1):
        _t, _o, h, l, _c, _v = candles[k]
        if bullish:
            if departed and l <= zone_high:
                return True
            if l > zone_high:
                departed = True
        else:
            if departed and h >= zone_low:
                return True
            if h < zone_low:
                departed = True
    return False


def _find_ob(candles: Sequence[Candle], swing: _Swing, bos: int, bullish: bool) -> Optional[int]:
    lo_bound = max(swing.index + 1, bos - MAX_OB_AGE_BARS)
    if lo_bound >= bos:
        return None
    # Extreme of the impulse leg: the lowest low (bullish) / highest high
    # (bearish) between the broken swing and the BOS bar.
    leg = range(lo_bound, bos)
    if bullish:
        extreme = min(leg, key=lambda k: (candles[k][3], -k))
    else:
        extreme = max(leg, key=lambda k: (candles[k][2], k))
    for k in range(extreme, max(lo_bound, extreme - MAX_OB_SEARCH_BARS) - 1, -1):
        _t, o, _h, _l, c, _v = candles[k]
        if (bullish and c < o) or (not bullish and c > o):
            return k
    return None


def _build_zone(
    candles: Sequence[Candle], swing: _Swing, bos: int, ob: int, bullish: bool
) -> Optional[SmcZone]:
    ob_t, o, h, l, c, v = candles[ob]
    if _mitigated_before(candles, ob, bos, h, l, bullish):
        return None
    prior = candles[max(0, ob - VOLUME_LOOKBACK) : ob]
    avg_volume = sum(p[5] for p in prior) / len(prior) if prior else 0.0
    body = abs(c - o)
    wick = max(abs(h - max(o, c)), abs(min(o, c) - l))
    bos_close = candles[bos][4]
    return SmcZone(
        source_open_time=int(candles[bos][0]),
        bar_index=bos,
        ob_open_time=int(ob_t),
        ob_index=ob,
        zone_high=h,
        zone_low=l,
        zone_midpoint=(h + l) / 2,
        zone_size_pct=(h - l) / bos_close * 100 if bos_close else 0.0,
        direction="bullish" if bullish else "bearish",
        has_fvg=_has_fvg(candles, ob, bos, bullish),
        swing_price=swing.price,
        swing_open_time=int(candles[swing.index][0]),
        volume_ratio=v / avg_volume if avg_volume > 0 else 0.0,
        wick_ratio=wick / body if body > 0 else 0.0,
    )


def detect_smc_order_blocks(candles: List[Candle]) -> List[SmcZone]:
    """Every order block confirmed by a BOS, in BOS-bar order.

    Bars are processed strictly in time order and each step uses only bars up
    to and including the current one, so truncating the input after any bar
    leaves every earlier detection unchanged.
    """
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    swing_high: Optional[_Swing] = None
    swing_low: Optional[_Swing] = None
    zones: List[SmcZone] = []

    for i in range(len(candles)):
        # Confirm the fractal whose right-hand side completes on this bar.
        j = i - FRACTAL_BARS
        if j >= FRACTAL_BARS:
            if _is_fractal_high(highs, j):
                swing_high = _Swing(j, highs[j])
            if _is_fractal_low(lows, j):
                swing_low = _Swing(j, lows[j])

        close = candles[i][4]
        for swing, bullish in ((swing_high, True), (swing_low, False)):
            if swing is None or swing.broken or swing.index >= i:
                continue
            if (bullish and close > swing.price) or (not bullish and close < swing.price):
                swing.broken = True
                ob = _find_ob(candles, swing, i, bullish)
                if ob is None:
                    continue
                zone = _build_zone(candles, swing, i, ob, bullish)
                if zone is not None:
                    zones.append(zone)
    return zones
