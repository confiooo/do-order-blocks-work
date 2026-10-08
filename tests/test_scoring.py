"""Confidence is a display estimate; the signal gate decides what to page."""

from hawk_research import config as research_config
from hawk_research.replay.live_scoring import (
    BASE_RATE,
    MAX_DISTANCE_TO_PRICE,
    MIN_VOLUME_RATIO,
    SCORE_MAX,
    SCORE_MIN,
    aligned_return,
    passes_signal_gate,
    score_order_block,
)


def test_base_case_is_coin_flip():
    assert score_order_block(1.95, "bullish", 0.0) == BASE_RATE
    assert score_order_block(1.95, "bearish", 0.0) == BASE_RATE


def test_direction_alone_does_not_change_score():
    assert score_order_block(2.0, "bullish", None) == score_order_block(2.0, "bearish", None)


def test_weak_volume_lowers_score():
    assert score_order_block(1.40, "bullish", 0.0) < BASE_RATE


def test_hot_volume_raises_score():
    assert score_order_block(2.80, "bearish", 0.0) > BASE_RATE


def test_against_trend_lowers_score():
    # Price already rallied 3% into a bearish zone.
    faded = score_order_block(1.95, "bearish", 0.03)
    flat = score_order_block(1.95, "bearish", 0.0)
    assert faded < flat


def test_with_trend_does_not_inflate():
    # Continuation bonus is omitted on purpose.
    assert score_order_block(1.95, "bullish", 0.04) == score_order_block(1.95, "bullish", 0.0)


def test_far_from_price_lowers_score():
    close = score_order_block(1.95, "bullish", 0.0, distance_to_price=0.10)
    far = score_order_block(1.95, "bullish", 0.0, distance_to_price=0.80)
    assert far < close


def test_aligned_return_sign():
    assert aligned_return("bullish", 0.02) == 0.02
    assert aligned_return("bearish", 0.02) == -0.02


def test_does_not_use_raw_size_and_stays_bounded():
    low = score_order_block(1.2, "bullish", 0.05)
    high = score_order_block(8.0, "bearish", -0.05)
    assert SCORE_MIN <= low <= SCORE_MAX
    assert SCORE_MIN <= high <= SCORE_MAX
    assert high < 0.90


def test_signal_gate_keeps_close_strong_volume():
    assert passes_signal_gate(2.1, 0.20) is True
    assert passes_signal_gate(MIN_VOLUME_RATIO, MAX_DISTANCE_TO_PRICE) is True


def test_signal_gate_rejects_far_or_weak():
    assert passes_signal_gate(2.1, 0.80) is False
    assert passes_signal_gate(1.4, 0.10) is False
    assert passes_signal_gate(None, 0.10) is False
    assert passes_signal_gate(2.1, None) is False


def test_research_config_gate_defaults_match_live_constants():
    gates = research_config.ResearchConfig().live_gates
    assert gates.min_volume_ratio == MIN_VOLUME_RATIO
    assert gates.max_distance_to_price == MAX_DISTANCE_TO_PRICE
    custom = research_config.load(
        overrides=["live_gates.min_volume_ratio=2.0", "live_gates.max_distance_to_price=0.3"]
    ).live_gates
    kwargs = {
        "min_volume_ratio": custom.min_volume_ratio,
        "max_distance_to_price": custom.max_distance_to_price,
    }
    assert passes_signal_gate(1.9, 0.20, **kwargs) is False
    assert passes_signal_gate(2.1, 0.20, **kwargs) is True


if __name__ == "__main__":
    test_base_case_is_coin_flip()
    test_direction_alone_does_not_change_score()
    test_weak_volume_lowers_score()
    test_hot_volume_raises_score()
    test_against_trend_lowers_score()
    test_with_trend_does_not_inflate()
    test_far_from_price_lowers_score()
    test_aligned_return_sign()
    test_does_not_use_raw_size_and_stays_bounded()
    test_signal_gate_keeps_close_strong_volume()
    test_signal_gate_rejects_far_or_weak()
    print("scoring tests ok")
