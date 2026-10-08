from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Candle = Tuple[int, float, float, float, float, float]


@dataclass(frozen=True)
class DetectedZone:
    source_open_time: int
    zone_high: float
    zone_low: float
    zone_midpoint: float
    zone_size_pct: float
    size: float
    volume_ratio: float
    wick_ratio: float
    distance_to_swing: float
    distance_to_price: float
    direction: str


def detect_order_blocks(
    candles: List[Candle],
    lookback_ratio: float,
    strength_threshold: float,
    wick_to_body_ratio_min: float,
    min_distance_from_high_low: float,
    candidate_open_times: Optional[Sequence[int]] = None,
) -> List[DetectedZone]:
    """Detect order blocks using only information known at each candidate's close.

    Each candidate at index i is judged against bars ``[i-lookback, i)`` — never
    against later closes, highs, lows, or volume, and never against a later
    "current" price. That makes detection point-in-time: a bar either qualifies
    when it closes, or it does not.

    ``candidate_open_times`` restricts evaluation to those opens (the live
    collector passes newly closed bars). ``None`` evaluates every bar with
    enough causal history (tests / replay).
    """
    if not candles:
        return []

    lookback = int(len(candles) * lookback_ratio)
    if lookback < 1:
        return []

    wanted = None if candidate_open_times is None else set(candidate_open_times)
    blocks: List[DetectedZone] = []

    for i in range(lookback, len(candles)):
        open_time, o, h, l, c, v = candles[i]
        if c == 0:
            continue
        if wanted is not None and open_time not in wanted:
            continue

        prior = candles[i - lookback : i]
        if not prior:
            continue

        avg_volume = sum(p[5] for p in prior) / len(prior)
        recent_high = max(p[2] for p in prior)
        recent_low = min(p[3] for p in prior)

        body = abs(c - o)
        wick = max(abs(h - max(o, c)), abs(min(o, c) - l))
        wick_ratio = wick / body if body > 0 else 0.0
        distance_to_swing = min(abs(h - recent_high), abs(l - recent_low)) / c * 100
        midpoint = (h + l) / 2
        # Geometry of this bar vs its own close — not vs a later market price.
        distance_to_price = abs(midpoint - c) / c * 100
        volume_ratio = v / avg_volume if avg_volume > 0 else 0.0

        if (
            v >= avg_volume * strength_threshold
            and wick_ratio >= wick_to_body_ratio_min
            and distance_to_swing >= min_distance_from_high_low
        ):
            blocks.append(
                DetectedZone(
                    source_open_time=open_time,
                    zone_high=h,
                    zone_low=l,
                    zone_midpoint=midpoint,
                    zone_size_pct=(h - l) / c * 100,
                    size=v,
                    volume_ratio=volume_ratio,
                    wick_ratio=wick_ratio,
                    distance_to_swing=distance_to_swing,
                    distance_to_price=distance_to_price,
                    direction="bullish" if c > o else "bearish",
                )
            )

    return blocks
