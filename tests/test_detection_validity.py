"""Point-in-time detection, causal features, and post-fill MFE/MAE."""

from __future__ import annotations

from hawk_research import config as research_config
from hawk_research.detectors.order_blocks import detect_order_blocks
from hawk_research.labels.retest import candles_after_fill, compute_excursions, entry_price, find_first_retest

STEP = 3_600_000
# The detector thresholds the published results used (config/research.yaml). The
# hand-built candles below are sized to clear, or fail, exactly these values.
_DET = research_config.ResearchConfig().detector
STRENGTH = _DET.strength_threshold
WICK_MIN = _DET.wick_to_body_ratio_min
SWING_MIN = _DET.min_distance_from_high_low
PRIOR_BARS = 30


def _c(i, o, h, l, c, v):
    return (i * STEP, o, h, l, c, v)


def _flat(n, start=0, price=100.0, vol=10.0):
    return [_c(start + i, price, price + 0.1, price - 0.1, price, vol) for i in range(n)]


def _ratio(n: int) -> float:
    return PRIOR_BARS / float(n)


def _prior_with_swings():
    prior = _flat(PRIOR_BARS, price=100.0)
    prior[10] = _c(10, 100.0, 105.0, 99.9, 104.0, 10.0)
    prior[15] = _c(15, 100.0, 100.1, 95.0, 96.0, 10.0)
    return prior


def _ob_at(index: int):
    return _c(index, 100.0, 100.3, 99.0, 100.2, 30.0)


def test_future_volume_does_not_leak_into_candidate():
    """A later volume spike must not change whether bar i qualified at its close."""
    candidate = _ob_at(PRIOR_BARS)
    causal_series = _prior_with_swings() + [candidate]
    with_future = causal_series + _flat(20, start=PRIOR_BARS + 1, vol=10_000.0)

    causal = detect_order_blocks(
        causal_series,
        lookback_ratio=_ratio(len(causal_series)),
        strength_threshold=STRENGTH,
        wick_to_body_ratio_min=WICK_MIN,
        min_distance_from_high_low=SWING_MIN,
        candidate_open_times=[candidate[0]],
    )
    leaked_if_buggy = detect_order_blocks(
        with_future,
        lookback_ratio=_ratio(len(with_future)),
        strength_threshold=STRENGTH,
        wick_to_body_ratio_min=WICK_MIN,
        min_distance_from_high_low=SWING_MIN,
        candidate_open_times=[candidate[0]],
    )
    assert len(causal) == 1, causal
    assert len(leaked_if_buggy) == 1
    assert abs(causal[0].volume_ratio - leaked_if_buggy[0].volume_ratio) < 1e-9


def test_later_price_does_not_select_or_reject_a_closed_bar():
    """Old detector required the candidate close to sit near the latest price."""
    candidate = _ob_at(PRIOR_BARS)
    candles = _prior_with_swings() + [candidate] + _flat(20, start=PRIOR_BARS + 1, price=150.0)
    hits = detect_order_blocks(
        candles,
        lookback_ratio=_ratio(len(candles)),
        strength_threshold=STRENGTH,
        wick_to_body_ratio_min=WICK_MIN,
        min_distance_from_high_low=SWING_MIN,
        candidate_open_times=[candidate[0]],
    )
    assert len(hits) == 1
    assert hits[0].source_open_time == candidate[0]


def test_only_requested_candidates_are_evaluated():
    first = _ob_at(PRIOR_BARS)
    gap = _flat(5, start=PRIOR_BARS + 1)
    second = _ob_at(PRIOR_BARS + 6)
    candles = _prior_with_swings() + [first] + gap + [second]
    hits = detect_order_blocks(
        candles,
        lookback_ratio=_ratio(len(candles)),
        strength_threshold=STRENGTH,
        wick_to_body_ratio_min=WICK_MIN,
        min_distance_from_high_low=SWING_MIN,
        candidate_open_times=[second[0]],
    )
    assert [h.source_open_time for h in hits] == [second[0]]


def test_mfe_mae_exclude_the_fill_bar():
    zone_high, zone_low = 110.0, 100.0
    source = (0, 105, 108, 101, 106, 1)
    above = (STEP, 112, 116, 111, 114, 1)
    fill = (2 * STEP, 115, 120, 105, 108, 1)
    after = (3 * STEP, 108, 111, 109, 110, 1)
    candles = [source, above, fill, after]

    retest = find_first_retest(candles, 0, "bullish", zone_high, zone_low)
    assert retest is not None and retest[0] == fill[0]

    entry = entry_price("bullish", zone_high, zone_low)
    assert entry == zone_high
    window = candles_after_fill(candles, fill[0])
    assert [c[0] for c in window] == [after[0]]

    mfe, mae = compute_excursions("bullish", entry, window)
    assert abs(mfe - round((111.0 - 110.0) / 110.0, 6)) < 1e-9
    assert abs(mae - round((110.0 - 109.0) / 110.0, 6)) < 1e-9
    included = compute_excursions("bullish", entry, [fill, after])
    assert included[0] > mfe


if __name__ == "__main__":
    test_future_volume_does_not_leak_into_candidate()
    test_later_price_does_not_select_or_reject_a_closed_bar()
    test_only_requested_candidates_are_evaluated()
    test_mfe_mae_exclude_the_fill_bar()
    print("detection validity tests ok")
