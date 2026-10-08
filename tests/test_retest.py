"""Directional retest edge cases: smash-through is not an entry."""

from hawk_research.labels.retest import find_first_retest

# open_time, open, high, low, close, volume
SOURCE = 1000
ZONE_HIGH = 110.0
ZONE_LOW = 100.0


def _c(open_time, high, low, close):
    return (open_time, close, high, low, close, 1.0)


def test_bullish_valid_return_from_above():
    candles = [
        _c(1000, 108, 101, 105),  # source, inside zone
        _c(2000, 116, 111, 114),  # established above
        _c(3000, 113, 107, 109),  # intersects, closes inside
    ]
    hit = find_first_retest(candles, SOURCE, "bullish", ZONE_HIGH, ZONE_LOW)
    assert hit is not None and hit[0] == 3000


def test_bullish_close_below_zone_low_is_not_retest():
    candles = [
        _c(1000, 108, 101, 105),
        _c(2000, 116, 111, 114),
        _c(3000, 115, 90, 95),  # breakdown through the zone
    ]
    assert find_first_retest(candles, SOURCE, "bullish", ZONE_HIGH, ZONE_LOW) is None


def test_bullish_never_left_above_is_not_retest():
    candles = [
        _c(1000, 108, 101, 105),
        _c(2000, 107, 102, 104),  # still inside
        _c(3000, 106, 101, 103),
    ]
    assert find_first_retest(candles, SOURCE, "bullish", ZONE_HIGH, ZONE_LOW) is None


def test_bullish_later_valid_retest_after_rejected_smash():
    candles = [
        _c(1000, 108, 101, 105),
        _c(2000, 116, 111, 114),
        _c(3000, 115, 90, 95),  # smash: skip
        _c(4000, 98, 92, 96),  # still below
        _c(5000, 118, 112, 115),  # re-established above
        _c(6000, 113, 108, 110),  # valid first retest
    ]
    hit = find_first_retest(candles, SOURCE, "bullish", ZONE_HIGH, ZONE_LOW)
    assert hit is not None and hit[0] == 6000


def test_bearish_valid_return_from_below():
    candles = [
        _c(1000, 108, 101, 105),
        _c(2000, 99, 90, 95),  # established below
        _c(3000, 103, 94, 101),  # intersects, closes inside
    ]
    hit = find_first_retest(candles, SOURCE, "bearish", ZONE_HIGH, ZONE_LOW)
    assert hit is not None and hit[0] == 3000


def test_bearish_close_above_zone_high_is_not_retest():
    candles = [
        _c(1000, 108, 101, 105),
        _c(2000, 99, 90, 95),
        _c(3000, 120, 96, 115),  # breakout through the zone
    ]
    assert find_first_retest(candles, SOURCE, "bearish", ZONE_HIGH, ZONE_LOW) is None


if __name__ == "__main__":
    test_bullish_valid_return_from_above()
    test_bullish_close_below_zone_low_is_not_retest()
    test_bullish_never_left_above_is_not_retest()
    test_bullish_later_valid_retest_after_rejected_smash()
    test_bearish_valid_return_from_below()
    test_bearish_close_above_zone_high_is_not_retest()
    print("retest edge cases ok")
