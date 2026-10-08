"""Creation-time confidence score and alert gate, copied from the live bot.

Replayed zones carry them as columns (``confidence``, ``passes_signal_gate``);
they never filter the research population.

The score estimates P(+1% is touched before -1% in the 4h after a retest fill).
The live base rate is about 0.50. ``score_order_block`` was only displayed;
``passes_signal_gate`` decided whether the bot sent an alert:

- ``volume_ratio >= 1.8``
- ``distance_to_price <= 0.45`` (zone midpoint close to that bar's close)

On the live book this was the only creation-time rule whose favourable-minus-stop
spread stayed positive on both the early 60% and the late 40% of zones, including
the lag-0 alerts. Each condition alone did not hold up, and a long/short split
reversed on the holdout.
"""

from __future__ import annotations

from typing import Optional

# 4h +1% before -1% among resolved fills on the live book (~0.52).
BASE_RATE = 0.50
SCORE_MIN = 0.28
SCORE_MAX = 0.68

MIN_VOLUME_RATIO = 1.8
MAX_DISTANCE_TO_PRICE = 0.45


def _f(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def aligned_return(direction: str, symbol_return: Optional[float]) -> Optional[float]:
    """Signed return in the setup's favor. Bullish +r, bearish -r."""
    ret = _f(symbol_return)
    if ret is None:
        return None
    if (direction or "").lower() == "bearish":
        return -ret
    return ret


def _volume_delta(volume_ratio: Optional[float]) -> float:
    vr = _f(volume_ratio)
    if vr is None:
        return 0.0
    if vr < 1.55:
        return -0.07
    if vr < 1.80:
        return -0.03
    if vr >= 3.20:
        return 0.06
    if vr >= 2.40:
        return 0.04
    return 0.0


def _fade_delta(direction: str, symbol_return_4h: Optional[float]) -> float:
    """Penalize setups the 4h tape has already moved against.

    A 'with-trend' bonus is omitted: it did not hold on the holdout week.
    The against-trend penalty did.
    """
    aligned = aligned_return(direction, symbol_return_4h)
    if aligned is None:
        return 0.0
    if aligned <= -0.025:
        return -0.08
    if aligned <= -0.012:
        return -0.05
    return 0.0


def _distance_delta(distance_to_price: Optional[float]) -> float:
    dist = _f(distance_to_price)
    if dist is None:
        return 0.0
    if dist > MAX_DISTANCE_TO_PRICE:
        return -0.08
    if dist > 0.30:
        return -0.04
    return 0.0


def passes_signal_gate(
    volume_ratio: Optional[float],
    distance_to_price: Optional[float],
    min_volume_ratio: float = MIN_VOLUME_RATIO,
    max_distance_to_price: float = MAX_DISTANCE_TO_PRICE,
) -> bool:
    """True if this zone is allowed to page / trade.

    Missing volume or distance fails closed — do not page a bar we cannot score.
    """
    vr = _f(volume_ratio)
    dist = _f(distance_to_price)
    if vr is None or dist is None:
        return False
    return vr >= float(min_volume_ratio) and dist <= float(max_distance_to_price)


def score_order_block(
    volume_ratio: float,
    direction: str,
    symbol_return_4h: Optional[float] = None,
    distance_to_price: Optional[float] = None,
) -> float:
    """Return a 0-1 confidence for display. Does not decide whether to alert."""
    raw = (
        BASE_RATE
        + _volume_delta(volume_ratio)
        + _fade_delta(direction, symbol_return_4h)
        + _distance_delta(distance_to_price)
    )
    return round(min(SCORE_MAX, max(SCORE_MIN, raw)), 3)
