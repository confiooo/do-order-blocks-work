"""Textbook order blocks: structure, direction, and point-in-time detection."""

from __future__ import annotations

from hawk_research.detectors.smc_order_blocks import detect_smc_order_blocks

STEP = 900_000


def _c(i, o, h, l, c, v=10.0):
    return (i * STEP, o, h, l, c, v)


def _bullish_bos_series():
    """Swing high at bar 5, pullback with a red OB at bar 10, break at bar 14."""
    bars = [
        _c(0, 100.0, 100.5, 99.5, 100.2),
        _c(1, 100.2, 100.8, 99.8, 100.6),
        _c(2, 100.6, 101.2, 100.2, 101.0),
        _c(3, 101.0, 101.6, 100.6, 101.4),
        _c(4, 101.4, 102.0, 101.0, 101.8),
        _c(5, 101.8, 103.0, 101.5, 102.2),  # swing high 103.0
        _c(6, 102.2, 102.4, 101.2, 101.4),
        _c(7, 101.4, 101.6, 100.4, 100.6),
        _c(8, 100.6, 100.8, 99.6, 99.8),
        _c(9, 99.8, 100.0, 99.0, 99.4),
        _c(10, 99.4, 99.6, 98.5, 98.8),  # last red candle: the OB
        _c(11, 98.8, 100.6, 98.7, 100.4),
        _c(12, 100.4, 101.9, 100.3, 101.7),  # low 100.3 > OB high 99.6: departed
        _c(13, 101.7, 102.9, 101.6, 102.7),
        _c(14, 102.7, 104.2, 102.6, 104.0),  # closes above 103.0: BOS
        _c(15, 104.0, 104.5, 103.5, 104.3),
        _c(16, 104.3, 104.8, 103.8, 104.6),
    ]
    return bars


def test_bullish_bos_finds_last_red_candle():
    zones = detect_smc_order_blocks(_bullish_bos_series())
    bull = [z for z in zones if z.direction == "bullish"]
    assert len(bull) == 1
    z = bull[0]
    assert z.ob_index == 10
    assert (z.zone_high, z.zone_low) == (99.6, 98.5)
    assert z.swing_price == 103.0
    assert z.has_fvg  # bar 12 low 100.3 > bar 10 high 99.6


def test_event_time_is_the_bos_bar_not_the_ob():
    z = [z for z in detect_smc_order_blocks(_bullish_bos_series()) if z.direction == "bullish"][0]
    assert z.bar_index == 14
    assert z.source_open_time == 14 * STEP


def test_no_break_no_zone():
    bars = _bullish_bos_series()[:14]  # stops before the break
    assert not [z for z in detect_smc_order_blocks(bars) if z.direction == "bullish"]


def test_direction_comes_from_the_break_not_the_candle_colour():
    z = [z for z in detect_smc_order_blocks(_bullish_bos_series()) if z.direction == "bullish"][0]
    _t, o, _h, _l, c, _v = _bullish_bos_series()[z.ob_index]
    assert c < o  # the OB candle is red, yet the zone is bullish


def test_mitigated_before_break_is_dropped():
    bars = _bullish_bos_series()
    # Bar 13 dips back into the OB after price had left it.
    bars[13] = _c(13, 101.7, 102.9, 99.5, 102.7)
    assert not [z for z in detect_smc_order_blocks(bars) if z.direction == "bullish"]


def test_truncation_never_changes_earlier_detections():
    bars = _bullish_bos_series() + [_c(17 + k, 104.6 - k, 104.8 - k, 103.6 - k, 103.8 - k) for k in range(20)]
    full = detect_smc_order_blocks(bars)
    for cut in range(1, len(bars) + 1):
        head = detect_smc_order_blocks(bars[:cut])
        assert head == [z for z in full if z.bar_index < cut]
