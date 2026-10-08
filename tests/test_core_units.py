"""Unit tests for the research core, with hand-computed expectations on tiny inputs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hawk_research import smoke
from hawk_research.audit import leakage
from hawk_research.constants import BAR_MS
from hawk_research.data import manifest
from hawk_research.data.backfill import _normalise_epoch
from hawk_research.data.validation import CorruptOHLCV, check_ohlcv, validate_ohlcv
from hawk_research.evaluation import strategies
from hawk_research.features import point_in_time as feat
from hawk_research.labels import outcomes as lab
from hawk_research.replay import btc_gate, round1
from hawk_research.stats import nulls

H = 3_600_000


def _bars(closes, opens=None, highs=None, lows=None, start=0, step=BAR_MS):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    highs = np.maximum(opens, closes) if highs is None else np.asarray(highs, dtype=float)
    lows = np.minimum(opens, closes) if lows is None else np.asarray(lows, dtype=float)
    n = len(closes)
    return pd.DataFrame(
        {
            "open_time": start + np.arange(n, dtype=np.int64) * step,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 10.0),
        }
    )


# --------------------------------------------------------------------------- labels


def test_contiguous_forward_bars_stops_at_gaps():
    ot = np.array([0, 1, 2, 4, 5], dtype=np.int64) * BAR_MS  # bar 3 is missing
    assert lab.contiguous_forward_bars(ot).tolist() == [2, 1, 0, 1, 0]


def test_forward_returns_are_masked_across_gaps():
    ot = np.array([0, 1, 2, 4, 5], dtype=np.int64) * BAR_MS
    close = np.array([100.0, 110.0, 121.0, 50.0, 55.0])
    fwd = lab.forward_returns(close, lab.contiguous_forward_bars(ot))
    r = fwd["15m"]
    assert r[0] == pytest.approx(0.1) and r[1] == pytest.approx(0.1) and r[3] == pytest.approx(0.1)
    # Bar 2 -> bar 4 spans the gap: no 15m return. The last bar has no future.
    assert math.isnan(r[2]) and math.isnan(r[4])
    assert np.isnan(fwd["1h"]).all()  # never four contiguous bars ahead


def test_first_touch_ignores_the_decision_bar_and_respects_runs():
    high = np.array([200.0, 101.0, 103.0])
    low = np.array([50.0, 99.0, 99.0])
    close = np.array([100.0, 100.0, 100.0])
    runs = lab.contiguous_forward_bars(np.arange(3, dtype=np.int64) * BAR_MS)
    out = lab.first_touch_offsets(high, low, close, runs, levels=(0.02,), horizon=2)
    # Bar 0's own high of 200 must not count; +2% is first reached on bar 2 (offset 2).
    assert out["first_up_2"].tolist() == [2, 1, 0]
    assert out["first_dn_2"].tolist() == [0, 0, 0]


def test_barrier_label_ties_resolve_to_the_down_barrier():
    up = np.array([1, 3, 2, 0, 5])
    dn = np.array([2, 1, 2, 0, 0])
    label, ambiguous = lab.barrier_label(up, dn, horizon_bars=4)
    assert label.tolist() == [1, -1, -1, 0, 0]  # last: up touch at 5 is beyond the horizon
    assert ambiguous.tolist() == [False, False, True, False, False]


def test_zone_lifecycle_finds_the_retest_and_measures_from_it():
    # Bullish zone 99..100 on bar 0; price leaves upward, returns on bar 2, then drifts.
    closes = [101.5, 103.0, 100.5, 101, 102, 101, 103, 104, 104, 104]
    lows = [99.0, 102.0, 99.5, 100.5, 101, 100.5, 102, 103, 103, 103]
    highs = [102.0, 103.5, 101.0, 101.5, 102.5, 101.5, 103.5, 104.5, 104.5, 104.5]
    bars = _bars(closes, opens=closes, highs=highs, lows=lows)
    zones = pd.DataFrame(
        {
            "symbol": ["X"],
            "source_open_time": [0],
            "direction": ["bullish"],
            "zone_high": [100.0],
            "zone_low": [99.0],
        }
    )
    out = lab.zone_lifecycle(bars, zones).iloc[0]
    assert out["retested"] and out["retest_open_time"] == 2 * BAR_MS
    assert out["hypothetical_entry_price"] == 100.0  # top edge of a bullish zone
    assert out["retest_lag_bars"] == 2
    # 1h excursion uses bars 3..6 (fill bar excluded): best high 103.5, lowest low 100.5,
    # as fractions of the 100 entry; MAE is positive when adverse, so a low above entry is negative.
    assert out["mfe_1h"] == pytest.approx(0.035) and out["mae_1h"] == pytest.approx(-0.005)
    assert out["lifecycle_status"] == "entered"  # not enough bars for the 24h excursion
    # A retest that only arrives after the deadline does not count.
    late = lab.zone_lifecycle(bars, zones, retest_timeout_bars=1).iloc[0]
    assert not late["retested"] and late["lifecycle_status"] == "no_retest"


# --------------------------------------------------------------------- build_trades


def _trade_ds():
    return pd.DataFrame(
        {
            "zone_uid": ["A", "B", "C", "D"],
            "symbol": ["X"] * 4,
            "direction": ["bullish", "bearish", "bullish", "bullish"],
            "source_open_time": np.array([0, 1, 2, 3], dtype=np.int64) * BAR_MS,
            "fwd_ret_4h": [0.01, 0.01, -0.03, 0.005],
            "fwd_ret_24h": [0.03, 0.01, -0.03, 0.004],
            "retested": [True, False, False, False],
            "retest_open_time": [1 * BAR_MS, np.nan, np.nan, np.nan],
            "entry_fwd_ret_4h": [0.02, np.nan, np.nan, np.nan],
            # first-touch offsets (bars) for +2% / -1% and the mirrored +1% / -2%
            "first_up_2": [5, 0, 4, 0],
            "first_dn_1": [0, 0, 4, 0],
            "first_up_1": [0, 3, 0, 0],
            "first_dn_2": [0, 3, 0, 0],
        }
    )


def test_build_trades_close_entry_time_exit():
    t = strategies.build_trades(_trade_ds(), strategies.TradeSpec("direction", "close", "time", "4h"))
    a, b = t.set_index("zone_uid").loc["A"], t.set_index("zone_uid").loc["B"]
    assert a["side"] == 1 and a["ret_gross"] == pytest.approx(0.01)
    assert b["side"] == -1 and b["ret_gross"] == pytest.approx(-0.01)  # short loses on a rise
    # Decision and entry at the close of the zone bar; held 16 bars.
    assert a["decision_time_ms"] == BAR_MS and a["entry_time_ms"] == BAR_MS
    assert a["exit_time_ms"] == BAR_MS + 16 * BAR_MS
    fade = strategies.build_trades(_trade_ds(), strategies.TradeSpec("fade", "close", "time", "4h"))
    assert fade.set_index("zone_uid").loc["A", "ret_gross"] == pytest.approx(-0.01)


def test_build_trades_retest_entry_drops_unfilled_zones():
    t = strategies.build_trades(_trade_ds(), strategies.TradeSpec("direction", "retest", "time", "4h"))
    assert t["zone_uid"].tolist() == ["A"]
    # The fill bar is excluded: live from the close of the bar after the retest bar.
    assert t.iloc[0]["entry_time_ms"] == 2 * BAR_MS
    assert t.iloc[0]["ret_gross"] == pytest.approx(0.02)


def test_build_trades_barrier_exits_and_same_bar_ambiguity():
    spec = strategies.TradeSpec("direction", "close", "barrier", "24h", take_profit=0.02, stop_loss=0.01)
    t = strategies.build_trades(_trade_ds(), spec).set_index("zone_uid")
    assert t.loc["A", "ret_gross"] == pytest.approx(0.02)  # long: +2% first, on bar 5
    assert t.loc["A", "exit_time_ms"] == BAR_MS + 5 * BAR_MS
    # Same-bar ambiguity resolves to "down barrier first": a loss for a long ...
    assert t.loc["C", "ret_gross"] == pytest.approx(-0.01)
    # ... and a WIN for a short (its target is the down barrier). Documented, not changed.
    assert t.loc["B", "side"] == -1 and t.loc["B", "ret_gross"] == pytest.approx(0.02)
    # Neither barrier inside 24h: the 24h close-to-close return, signed.
    assert t.loc["D", "ret_gross"] == pytest.approx(0.004)
    assert t.loc["D", "exit_time_ms"] == 4 * BAR_MS + 96 * BAR_MS  # zone bar 3, entry at its close


# ---------------------------------------------------------------- matched controls


def _control_frame():
    n = 60
    vr = np.full(n, 5.0)
    rv = np.full(n, 0.05)
    vr[30], rv[30] = 2.0, 0.010  # the zone bar
    for i in (25, 34):  # inside +/- 8 bars, matching
        vr[i], rv[i] = 2.1, 0.011
    vr[50], rv[50] = 2.0, 0.010  # matching but outside the window
    vr[27], rv[27] = 2.0, 0.020  # volume matches, volatility does not
    return pd.DataFrame(
        {
            "open_time": np.arange(n, dtype=np.int64) * BAR_MS,
            "det_volume_ratio": vr,
            "rv_24h": rv,
            "fwd_ret_4h": np.linspace(-0.01, 0.01, n),
        }
    )


def test_matched_controls_respect_window_and_tolerances():
    bl = _control_frame()
    out = nulls.sample_matched_controls(bl, [30 * BAR_MS], match_hours=2, n_controls=5, seed=1)
    picked = sorted((out["open_time"] // BAR_MS).tolist())
    assert picked == [25, 34]
    assert (out["matched_zone_open_time"] == 30 * BAR_MS).all()


def test_matched_controls_are_seed_deterministic_and_ignore_outcomes():
    bl = _control_frame()
    a = nulls.sample_matched_controls(bl, [30 * BAR_MS], match_hours=2, n_controls=1, seed=4)
    b = nulls.sample_matched_controls(bl, [30 * BAR_MS], match_hours=2, n_controls=1, seed=4)
    assert a["open_time"].tolist() == b["open_time"].tolist()
    # Rewriting every future outcome must not change which bars are chosen.
    shuffled = bl.assign(fwd_ret_4h=bl["fwd_ret_4h"].to_numpy()[::-1])
    c = nulls.sample_matched_controls(shuffled, [30 * BAR_MS], match_hours=2, n_controls=1, seed=4)
    assert a["open_time"].tolist() == c["open_time"].tolist()


# ------------------------------------------------------------------------ features


@pytest.fixture(scope="module")
def market():
    sym = smoke.synthetic_bars("X", 11, 50.0).iloc[:900].reset_index(drop=True)
    btc = smoke.synthetic_bars("BTC", 12, 100.0).iloc[:900].reset_index(drop=True)
    return sym, btc


def test_features_unchanged_when_the_future_is_cut(market):
    sym, btc = market
    full = feat.build_symbol_features(sym, btc)
    for t in (450, 700, 899):
        cut = feat.build_symbol_features(sym.iloc[: t + 1], btc.iloc[: t + 1]).iloc[-1]
        row = full.iloc[t]
        for col in full.columns:
            a, b = row[col], cut[col]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), (t, col)


def test_feature_alignment_by_hand(market):
    sym, btc = market
    f = feat.build_symbol_features(sym, btc)
    t = 500
    o, c = sym.loc[t, "open"], sym.loc[t, "close"]
    assert f.loc[t, "det_direction"] == (1.0 if c > o else -1.0)
    assert f.loc[t, "bar_return"] == pytest.approx(c / sym.loc[t - 1, "close"] - 1.0)
    assert f.loc[t, "btc_ret_4h"] == pytest.approx(btc.loc[t, "close"] / btc.loc[t - 16, "close"] - 1.0)


def test_missing_btc_bar_is_nan_not_forward_filled(market):
    sym, btc = market
    holed = btc.drop(index=600).reset_index(drop=True)
    f = feat.build_symbol_features(sym, holed)
    assert np.isnan(f.loc[600, "btc_ret_4h"])
    assert not np.isnan(f.loc[599, "btc_ret_4h"])


# -------------------------------------------------------------- data: epochs, checks


def test_normalise_epoch_converts_microseconds_only():
    v = pd.Series([1_725_148_800_000, 1_735_689_600_000_000, "1735689600900000"])
    assert _normalise_epoch(v).tolist() == [1_725_148_800_000, 1_735_689_600_000, 1_735_689_600_900]


def test_ohlcv_validation_counts_soft_issues_and_rejects_corruption():
    good = _bars([10.0, 10.5, 10.2, 10.4])
    good.loc[1, "volume"] = 0.0
    good = good.drop(index=2).reset_index(drop=True)  # one missing bar
    rep = validate_ohlcv(good, "15m")
    assert rep.zero_volume == 1 and rep.gaps == 1 and rep.missing_bars == 1 and not rep.hard_errors

    bad = _bars([10.0, 10.5, 10.2])
    bad.loc[1, ["high", "low"]] = [9.0, 11.0]
    assert check_ohlcv(bad, "15m").high_below_low == 1
    with pytest.raises(CorruptOHLCV):
        validate_ohlcv(bad, "15m", "BAD")

    for mutate in (
        lambda d: d.__setitem__("close", [10.0, -1.0, 10.2]),
        lambda d: d.__setitem__("open_time", [0, 0, 2 * BAR_MS]),
        lambda d: d.__setitem__("open", [10.0, 99.0, 10.2]),
    ):
        d = _bars([10.0, 10.5, 10.2])
        mutate(d)
        assert check_ohlcv(d, "15m").hard_errors


def test_manifest_hash_is_canonical():
    df = _bars([10.0, 10.5, 10.2]).assign(quote_volume=1.0, trades=3, taker_buy_quote=0.5)
    h = manifest.content_sha256(df)
    assert manifest.content_sha256(df.iloc[::-1]) == h  # row order does not matter
    assert manifest.content_sha256(df.assign(trades=df["trades"].astype(float))) == h  # nor dtype
    assert manifest.content_sha256(df.assign(close=[10.0, 10.5, 10.3])) != h
    local = manifest.describe("X", "15m", df)
    assert manifest.compare(local, dict(local)) == []
    diffs = manifest.compare(local, {**local, "rows": 4})
    assert [d["field"] for d in diffs] == ["rows"]


# ---------------------------------------------------------------------- replay


@pytest.mark.parametrize("n", [399, 400, 401, 1000, 1357])
def test_rolling_windows_cover_every_candidate_once(n):
    covered = []
    for start, end, first in round1.rolling_windows(n):
        assert end - start == round1.LIVE_WINDOW and start <= first <= end
        covered.extend(range(first, end))
    expected = list(range(round1.WARMUP_BARS, n)) if n >= round1.LIVE_WINDOW else []
    assert covered == expected


def test_btc_range_series_matches_the_live_gate(market):
    _, btc = market
    candles = list(
        btc[["open_time", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None)
    )
    series = round1.btc_range_series(candles)
    for i in (20, 300, 899):
        t = int(candles[i][0])
        assert series[t] == pytest.approx(btc_gate.btc_range_at(candles, t, round1.BTC_RANGE_BARS))


# ----------------------------------------------------------------------- audit


def _audit_dataset(sym, btc, times):
    f = feat.build_symbol_features(sym, btc).set_index("open_time").loc[times].reset_index()
    return f.rename(columns={"open_time": "source_open_time"}).assign(symbol="X")


def test_truncation_audit_passes_clean_features_and_catches_a_leak(market):
    sym, btc = market
    times = sym["open_time"].iloc[[300, 450, 600, 750]].tolist()
    ds = _audit_dataset(sym, btc, times)
    ok = leakage.truncation_test(ds, {"X": sym}, btc, n_samples=4, seed=0)
    assert ok["n_tested"] == 4 and ok["passed"]

    # Replace one feature with its next-bar value: the audit must flag exactly that column.
    full = feat.build_symbol_features(sym, btc).set_index("open_time")
    nxt = full["trend_ret_4h"].shift(-1)
    leaky = ds.assign(trend_ret_4h=nxt.loc[times].to_numpy())
    bad = leakage.truncation_test(leaky, {"X": sym}, btc, n_samples=4, seed=0)
    assert not bad["passed"] and set(bad["columns_with_mismatch"]) == {"trend_ret_4h"}


def test_label_truncation_audit(market):
    sym, _ = market
    ds = pd.DataFrame({"symbol": "X", "source_open_time": sym["open_time"].iloc[[100, 500]].tolist()})
    out = leakage.label_truncation_test(ds, {"X": sym}, n_samples=2, seed=0)
    assert out["n_tested"] == 2 and out["passed"]


def test_ambiguity_sensitivity_books_ambiguous_shorts_as_losses():
    spec = strategies.TradeSpec("direction", "close", "barrier", "24h", take_profit=0.02, stop_loss=0.01)
    t = strategies.build_trades(_trade_ds(), spec)
    # Rows: A long win, B short ambiguous (booked +2%), C long ambiguous (booked -1%), D timeout.
    assert t["barrier_ambiguous"].sum() == 2
    out = strategies.ambiguity_sensitivity(t, spec, mean_ret_net=0.001)
    assert out["ambiguous_fraction"] == pytest.approx(0.5)
    assert out["ambiguous_short_fraction"] == pytest.approx(0.25)
    # B flips from +2% to -1%: the mean moves by -3% / 4 trades.
    assert out["mean_ret_net_if_ambiguous_shorts_lose"] == pytest.approx(0.001 - 0.0075)
    time_spec = strategies.TradeSpec("direction", "close", "time", "4h")
    assert (
        strategies.ambiguity_sensitivity(strategies.build_trades(_trade_ds(), time_spec), time_spec, 0.0)
        == {}
    )


def test_matched_controls_cache_key_tracks_its_inputs():
    from hawk_research.stages import s8_null_baselines as s8

    ds = pd.DataFrame({"symbol": ["A", "B"], "source_open_time": [BAR_MS, 2 * BAR_MS]})
    key = s8.controls_cache_key(ds, seed=3)
    assert key == s8.controls_cache_key(ds.iloc[::-1], seed=3)  # row order does not matter
    assert key != s8.controls_cache_key(ds, seed=4)
    assert key != s8.controls_cache_key(ds.assign(source_open_time=[BAR_MS, 3 * BAR_MS]), seed=3)
    assert key["match_params"]["match_hours"] == 24.0 and key["match_params"]["n_controls"] == 2


def test_barrier_label_treats_missing_offsets_as_no_touch():
    # NaN offset = label unavailable (gap). It must count as "never touched", explicitly,
    # not through the platform-defined result of casting NaN to an integer.
    up = np.array([3.0, np.nan, np.nan, 2.0])
    dn = np.array([np.nan, 4.0, np.nan, 2.0])
    label, ambiguous = lab.barrier_label(up, dn, horizon_bars=16)
    assert label.tolist() == [1, -1, 0, -1]
    assert ambiguous.tolist() == [False, False, False, True]
