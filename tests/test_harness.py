"""Harness correctness tests.

The two controls at the top are the reason this file exists. A harness that always
answers "no edge" is useless but looks rigorous, and a harness that always answers
"edge" is worse. So both directions are pinned: synthetic trades with zero true edge
must produce an interval containing zero, and synthetic trades with a real 20 bps
edge must produce an interval excluding zero. The remaining tests are hand-computed
arithmetic and leakage checks on the individual pieces.

Seeds are fixed everywhere, so a failure here is a real regression, not variance.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

from hawk_research.evaluation import metrics, portfolio, regimes
from hawk_research.evaluation.costs import CostModel, apply_costs
from hawk_research.evaluation.harness import (
    HarnessConfig,
    evaluate,
    purged_split,
    walk_forward_folds,
)
from hawk_research.ledger import TrialLedger
from hawk_research.stats import resampling
from hawk_research.stats.multiple_testing import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    pbo_cscv,
    probabilistic_sharpe_ratio,
)

BAR_MS = 900_000
HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS
T0 = 1_700_000_000_000 // BAR_MS * BAR_MS


def _synthetic_trades(
    n: int,
    mean_ret: float,
    std: float,
    seed: int,
    n_symbols: int = 40,
    span_days: int = 180,
    holding_hours: int = 4,
) -> pd.DataFrame:
    """Trade table with a known true mean. Entries land on the 15m grid."""
    rng = np.random.default_rng(seed)
    bars = np.sort(rng.integers(0, span_days * 96, size=n))
    entry = T0 + bars.astype(np.int64) * BAR_MS
    side = np.where(rng.random(n) < 0.5, 1, -1).astype(int)
    return pd.DataFrame(
        {
            "zone_uid": [f"z{i:06d}" for i in range(n)],
            "symbol": [f"SYM{i % n_symbols}USDT" for i in range(n)],
            "entry_time_ms": entry,
            "exit_time_ms": entry + holding_hours * HOUR_MS,
            "side": side,
            "ret_gross": rng.normal(mean_ret, std, size=n),
            "decision_time_ms": entry,
        }
    )


def _test_config(**overrides) -> HarnessConfig:
    base = dict(bootstrap_samples=300, bootstrap_block_hours=24, max_concurrent=5, seed=7)
    base.update(overrides)
    return HarnessConfig(**base)


class NegativeControlTest(unittest.TestCase):
    """Zero true edge must not be reported as an edge."""

    def test_no_edge_interval_contains_zero(self):
        trades = _synthetic_trades(n=1500, mean_ret=0.0, std=0.01, seed=11)
        result = evaluate(trades, CostModel.zero(), _test_config(), label="negative_control")

        for name in ("block", "cross_sectional"):
            ci = result["bootstrap"]["net"][name]
            self.assertIsNotNone(ci["ci_low"])
            self.assertLessEqual(ci["ci_low"], 0.0, f"{name} lower bound above zero on noise")
            self.assertGreaterEqual(ci["ci_high"], 0.0, f"{name} upper bound below zero on noise")
        self.assertFalse(result["significant_net"])

        net = result["trade_metrics"]["net"]
        sr = net["sharpe_per_trade"]
        n_obs = net["n_trades"]
        t_stat = abs(sr) * np.sqrt(n_obs)
        self.assertLess(t_stat, 2.0, "noise produced a significant t-statistic")

        values = trades["ret_gross"].to_numpy()
        psr = probabilistic_sharpe_ratio(
            sr, 0.0, n_obs, float(sp_skew(values)), float(sp_kurtosis(values, fisher=False))
        )
        self.assertLess(psr, 0.95)

    def test_costs_make_a_zero_edge_negative(self):
        trades = _synthetic_trades(n=800, mean_ret=0.0, std=0.01, seed=12)
        result = evaluate(trades, CostModel.binance_spot_taker(), _test_config())
        self.assertLess(result["trade_metrics"]["net"]["expectancy"], 0.0)
        self.assertAlmostEqual(
            result["trade_metrics"]["gross"]["expectancy"] - result["trade_metrics"]["net"]["expectancy"],
            0.002 + 0.0004,
            places=9,
        )


class KnownEdgeControlTest(unittest.TestCase):
    """A real 20 bps edge must be detected, otherwise the harness is a rubber stamp."""

    def test_real_edge_interval_excludes_zero(self):
        trades = _synthetic_trades(n=1500, mean_ret=0.002, std=0.01, seed=21)
        result = evaluate(trades, CostModel.zero(), _test_config(), label="known_edge")

        self.assertTrue(result["significant_net"])
        for name in ("block", "cross_sectional"):
            ci = result["bootstrap"]["net"][name]
            self.assertGreater(ci["ci_low"], 0.0, f"{name} failed to detect a 20bps edge")
        self.assertGreater(result["trade_metrics"]["net"]["expectancy"], 0.0015)
        self.assertGreater(result["equity_metrics"]["net"]["end_equity"], 1.0)

    def test_edge_smaller_than_costs_does_not_survive(self):
        trades = _synthetic_trades(n=1500, mean_ret=0.002, std=0.01, seed=21)
        result = evaluate(trades, CostModel.binance_spot_taker(), _test_config())
        self.assertGreater(result["trade_metrics"]["gross"]["expectancy"], 0.0)
        self.assertLess(result["trade_metrics"]["net"]["expectancy"], 0.0)


class PurgedSplitTest(unittest.TestCase):
    def test_hand_checked_purge_and_embargo(self):
        rows = [
            # decision, exit, expected partition
            ("keep_early", 0, 1 * DAY_MS, "train"),
            ("purge_late_train", 9 * DAY_MS, 9 * DAY_MS + 12 * HOUR_MS, "purged"),
            ("keep_mid", 7 * DAY_MS, 7 * DAY_MS + 12 * HOUR_MS, "train"),
            ("embargoed", 10 * DAY_MS + 12 * HOUR_MS, 11 * DAY_MS, "embargoed"),
            ("test_trade", 12 * DAY_MS, 12 * DAY_MS + 4 * HOUR_MS, "test"),
        ]
        trades = pd.DataFrame(
            {
                "zone_uid": [r[0] for r in rows],
                "symbol": ["AAAUSDT"] * len(rows),
                "entry_time_ms": [T0 + r[1] for r in rows],
                "exit_time_ms": [T0 + r[2] for r in rows],
                "side": [1] * len(rows),
                "ret_gross": [0.01] * len(rows),
                "decision_time_ms": [T0 + r[1] for r in rows],
            }
        )
        split = purged_split(
            trades,
            train_start_ms=T0,
            train_end_ms=T0 + 10 * DAY_MS,
            test_start_ms=T0 + 10 * DAY_MS,
            test_end_ms=T0 + 20 * DAY_MS,
            embargo_ms=DAY_MS,
            max_outcome_ms=48 * HOUR_MS,
        )
        self.assertEqual(split["n_train"], 2)
        self.assertEqual(split["n_purged"], 1)
        self.assertEqual(split["n_embargoed"], 1)
        self.assertEqual(split["n_test"], 1)
        self.assertEqual(set(split["train"]["zone_uid"]), {"keep_early", "keep_mid"})
        self.assertEqual(set(split["purged"]["zone_uid"]), {"purge_late_train"})
        self.assertEqual(set(split["embargoed"]["zone_uid"]), {"embargoed"})
        self.assertEqual(set(split["test"]["zone_uid"]), {"test_trade"})

    def test_partitions_are_disjoint_and_input_unchanged(self):
        trades = _synthetic_trades(n=300, mean_ret=0.0, std=0.01, seed=31)
        before = trades.copy()
        split = purged_split(
            trades,
            train_start_ms=int(trades["decision_time_ms"].min()),
            train_end_ms=int(trades["decision_time_ms"].min()) + 60 * DAY_MS,
            test_start_ms=int(trades["decision_time_ms"].min()) + 60 * DAY_MS,
            test_end_ms=int(trades["decision_time_ms"].max()) + 1,
            embargo_ms=24 * HOUR_MS,
            max_outcome_ms=48 * HOUR_MS,
        )
        train_ids = set(split["train"]["zone_uid"])
        test_ids = set(split["test"]["zone_uid"])
        purged_ids = set(split["purged"]["zone_uid"])
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(train_ids & purged_ids, set())
        pd.testing.assert_frame_equal(trades, before)

    def test_walk_forward_folds_are_chronological(self):
        trades = _synthetic_trades(n=2000, mean_ret=0.0, std=0.01, seed=33, span_days=400)
        folds = walk_forward_folds(trades, _test_config())
        self.assertGreater(len(folds), 1)
        for fold in folds:
            self.assertLess(fold["train_end_ms"], fold["test_start_ms"])
            if fold["n_train"] and fold["n_test"]:
                self.assertLess(
                    int(fold["train"]["decision_time_ms"].max()),
                    int(fold["test"]["decision_time_ms"].min()),
                )


class PortfolioTest(unittest.TestCase):
    def _burst(self, n: int, same_symbol: bool = False) -> pd.DataFrame:
        entry = T0 + np.arange(n, dtype=np.int64) * BAR_MS
        return pd.DataFrame(
            {
                "zone_uid": [f"b{i}" for i in range(n)],
                "symbol": ["AAAUSDT"] * n if same_symbol else [f"S{i}USDT" for i in range(n)],
                "entry_time_ms": entry,
                "exit_time_ms": entry + 24 * HOUR_MS,
                "side": [1] * n,
                "ret_gross": [0.01] * n,
                "ret_net": [0.01] * n,
                "decision_time_ms": entry,
            }
        )

    def test_max_concurrent_is_never_exceeded(self):
        trades = self._burst(20)
        sim = portfolio.build_equity_curve(trades, max_concurrent=3, one_position_per_symbol=True)
        self.assertLessEqual(int(sim["equity"]["n_open"].max()), 3)
        self.assertEqual(sim["accepted"], 3)
        self.assertEqual(sim["rejected"], 17)
        self.assertAlmostEqual(sim["rejection_rate"], 17 / 20)
        self.assertEqual(sim["rejected_symbol"], 0)

    def test_one_position_per_symbol(self):
        trades = self._burst(5, same_symbol=True)
        blocked = portfolio.build_equity_curve(trades, max_concurrent=5, one_position_per_symbol=True)
        self.assertEqual(blocked["accepted"], 1)
        self.assertEqual(blocked["rejected_symbol"], 4)
        allowed = portfolio.build_equity_curve(trades, max_concurrent=5, one_position_per_symbol=False)
        self.assertEqual(allowed["accepted"], 5)
        self.assertLessEqual(int(allowed["equity"]["n_open"].max()), 5)

    def test_capital_compounds_on_realized_exits_only(self):
        entry = np.array([T0, T0 + 48 * HOUR_MS], dtype=np.int64)
        trades = pd.DataFrame(
            {
                "zone_uid": ["a", "b"],
                "symbol": ["AAAUSDT", "BBBUSDT"],
                "entry_time_ms": entry,
                "exit_time_ms": entry + 24 * HOUR_MS,
                "side": [1, 1],
                "ret_gross": [0.10, 0.10],
                "ret_net": [0.10, 0.10],
                "decision_time_ms": entry,
            }
        )
        sim = portfolio.build_equity_curve(trades, max_concurrent=1, capital=1.0, risk_fraction=1.0)
        self.assertEqual(sim["accepted"], 2)
        self.assertAlmostEqual(sim["final_equity"], 1.21, places=9)
        self.assertAlmostEqual(float(sim["equity"]["equity"].iloc[0]), 1.0, places=9)

    def test_max_drawdown_and_equity_metrics(self):
        equity = np.array([1.0, 1.2, 0.9, 1.1])
        self.assertAlmostEqual(metrics.max_drawdown(equity), 0.25, places=9)
        curve = pd.DataFrame({"equity": equity}, index=[T0 + i * DAY_MS for i in range(4)])
        stats = metrics.equity_metrics(curve)
        self.assertAlmostEqual(stats["max_drawdown"], 0.25, places=9)
        self.assertAlmostEqual(stats["total_return"], 0.1, places=9)

    def test_degenerate_metrics_return_none(self):
        self.assertIsNone(metrics.expectancy([]))
        self.assertIsNone(metrics.profit_factor([0.01, 0.02]))
        self.assertIsNone(metrics.sharpe([0.01, 0.01, 0.01], 365))
        self.assertIsNone(metrics.max_drawdown([]))
        self.assertEqual(metrics.trade_metrics(pd.DataFrame())["n_trades"], 0)


class CostsTest(unittest.TestCase):
    def test_round_trip_arithmetic_hand_computed(self):
        trades = pd.DataFrame(
            {
                "zone_uid": ["long", "short"],
                "symbol": ["AAAUSDT", "BBBUSDT"],
                "entry_time_ms": [T0, T0],
                "exit_time_ms": [T0 + 16 * HOUR_MS, T0 + 16 * HOUR_MS],
                "side": [1, -1],
                "ret_gross": [0.01, 0.01],
                "decision_time_ms": [T0, T0],
            }
        )
        model = CostModel(
            taker_fee=0.001,
            maker_fee=0.001,
            fee_discount=0.25,
            slippage_bps_base=2.0,
            slippage_illiquidity_coef=1.0,
            funding_rate_8h=0.0001,
            venue="unit_test",
        )
        illiquidity = pd.Series([0.5, 0.5], index=trades["zone_uid"])
        priced = apply_costs(trades, model, illiquidity=illiquidity)

        # fees: 2 * 0.001 * (1 - 0.25) = 0.0015
        self.assertAlmostEqual(float(priced["fee_cost"].iloc[0]), 0.0015, places=12)
        # slippage: 2 * 2bps * (1 + 1 * 0.5) = 0.0006
        self.assertAlmostEqual(float(priced["slippage_cost"].iloc[0]), 0.0006, places=12)
        # funding: 0.0001 * 16h / 8h = 0.0002, charged long, credited short
        self.assertAlmostEqual(float(priced["funding_cost"].iloc[0]), 0.0002, places=12)
        self.assertAlmostEqual(float(priced["funding_cost"].iloc[1]), -0.0002, places=12)
        self.assertAlmostEqual(float(priced["total_cost"].iloc[0]), 0.0023, places=12)
        self.assertAlmostEqual(float(priced["ret_net"].iloc[0]), 0.0077, places=12)
        self.assertAlmostEqual(float(priced["ret_net"].iloc[1]), 0.0081, places=12)
        self.assertNotIn("ret_net", trades.columns)

    def test_named_constructors(self):
        self.assertAlmostEqual(CostModel.binance_spot_taker().round_trip_fee(), 0.002, places=12)
        self.assertAlmostEqual(CostModel.binance_spot_taker_bnb().round_trip_fee(), 0.0015, places=12)
        self.assertAlmostEqual(CostModel.mexc_futures_taker().round_trip_fee(), 0.0008, places=12)
        self.assertAlmostEqual(CostModel.mexc_futures_taker(fee=0.0002).round_trip_fee(), 0.0004, places=12)
        self.assertEqual(CostModel.zero().round_trip_fee(), 0.0)

    def test_zero_cost_model_is_a_passthrough(self):
        trades = _synthetic_trades(n=50, mean_ret=0.0, std=0.01, seed=41)
        priced = apply_costs(trades, CostModel.zero())
        np.testing.assert_allclose(priced["ret_net"], priced["ret_gross"])


class MultipleTestingTest(unittest.TestCase):
    def test_benjamini_hochberg_textbook_example(self):
        # Benjamini & Hochberg (1995), 15 hypotheses: 4 rejections at alpha = 0.05.
        pvalues = np.array(
            [
                0.0001,
                0.0004,
                0.0019,
                0.0095,
                0.0201,
                0.0278,
                0.0298,
                0.0344,
                0.0459,
                0.3240,
                0.4262,
                0.5719,
                0.6528,
                0.7590,
                1.0000,
            ]
        )
        rejected, qvalues = benjamini_hochberg(pvalues, alpha=0.05)
        self.assertEqual(int(rejected.sum()), 4)
        self.assertTrue(rejected[:4].all())
        self.assertFalse(rejected[4:].any())
        self.assertAlmostEqual(qvalues[0], 0.0015, places=6)
        self.assertAlmostEqual(qvalues[3], 0.0095 * 15 / 4, places=6)
        self.assertTrue(np.all(np.diff(qvalues) >= -1e-12), "q-values must be monotone in p")
        self.assertLessEqual(float(qvalues.max()), 1.0)

    def test_benjamini_hochberg_rejects_nothing_on_noise(self):
        rejected, qvalues = benjamini_hochberg(np.linspace(0.2, 0.9, 20), alpha=0.05)
        self.assertEqual(int(rejected.sum()), 0)
        self.assertTrue(np.all(qvalues > 0.05))

    def test_deflated_sharpe_penalizes_many_trials(self):
        one = deflated_sharpe_ratio(0.12, n_trials=1, sr_variance=0.01, n_obs=500)
        many = deflated_sharpe_ratio(0.12, n_trials=500, sr_variance=0.01, n_obs=500)
        self.assertEqual(one["expected_max_sr"], 0.0)
        self.assertGreater(many["expected_max_sr"], 0.0)
        self.assertLess(many["dsr"], one["dsr"])
        self.assertIsNotNone(many["sr_std_error"])

    def test_pbo_is_half_on_noise_and_low_for_a_dominant_config(self):
        rng = np.random.default_rng(101)
        noise = rng.normal(0.0, 0.01, size=(240, 20))
        noisy = pbo_cscv(noise, n_splits=16)
        self.assertEqual(noisy["n_combinations"], 12870)
        self.assertGreater(noisy["pbo"], 0.3)
        self.assertLess(noisy["pbo"], 0.7)

        dominant = rng.normal(0.0, 0.01, size=(240, 20))
        dominant[:, 7] += 0.02
        strong = pbo_cscv(dominant, n_splits=16)
        self.assertLess(strong["pbo"], 0.05)
        self.assertGreater(strong["median_logit"], 0.0)
        self.assertEqual(len(strong["logit_distribution"]), strong["n_combinations"])

    def test_pbo_handles_degenerate_input(self):
        self.assertIsNone(pbo_cscv(np.zeros((2, 1)))["pbo"])


class ResamplingTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_given_seed(self):
        trades = _synthetic_trades(n=400, mean_ret=0.001, std=0.01, seed=61)
        values = trades["ret_gross"].to_numpy()
        times = trades["entry_time_ms"].to_numpy()
        a = resampling.block_bootstrap_ci(values, times, 24, 200, seed=3)
        b = resampling.block_bootstrap_ci(values, times, 24, 200, seed=3)
        self.assertEqual(a["ci_low"], b["ci_low"])
        self.assertEqual(a["ci_high"], b["ci_high"])

    def test_cross_sectional_ci_is_wider_when_trades_are_clustered(self):
        # One burst per day, 40 correlated symbols sharing the same daily shock.
        rng = np.random.default_rng(71)
        n_days, n_symbols = 120, 40
        shocks = rng.normal(0.0, 0.01, size=n_days)
        values, times = [], []
        for d in range(n_days):
            values.extend(shocks[d] + rng.normal(0.0, 0.0005, size=n_symbols))
            times.extend([T0 + d * DAY_MS] * n_symbols)
        values = np.asarray(values)
        times = np.asarray(times, dtype=np.int64)
        clustered = resampling.cross_sectional_bootstrap_ci(values, times, 24, 300, seed=9)
        naive_width = 2 * 1.96 * values.std(ddof=1) / np.sqrt(values.size)
        width = clustered["ci_high"] - clustered["ci_low"]
        self.assertGreater(width, 3 * naive_width)
        self.assertEqual(clustered["n_blocks"], n_days)

    def test_paired_diff_ci_detects_a_real_difference(self):
        a = _synthetic_trades(n=600, mean_ret=0.004, std=0.01, seed=81)
        b = _synthetic_trades(n=600, mean_ret=0.000, std=0.01, seed=82)
        diff = resampling.paired_block_bootstrap_diff_ci(
            a["ret_gross"].to_numpy(),
            a["entry_time_ms"].to_numpy(),
            b["ret_gross"].to_numpy(),
            b["entry_time_ms"].to_numpy(),
            block_hours=24,
            n_samples=300,
            seed=5,
        )
        self.assertGreater(diff["ci_low"], 0.0)
        self.assertGreater(diff["diff"], 0.003)


class RegimesTest(unittest.TestCase):
    def _btc(self, n_bars: int = 96 * 40) -> pd.DataFrame:
        rng = np.random.default_rng(91)
        steps = rng.normal(0.0005, 0.004, size=n_bars)
        close = 30000 * np.exp(np.cumsum(steps))
        open_time = T0 + np.arange(n_bars, dtype=np.int64) * BAR_MS
        return pd.DataFrame(
            {
                "open_time": open_time,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(n_bars, 100.0),
            }
        )

    def test_labels_are_causal(self):
        btc = self._btc()
        table = regimes.btc_regime_labels(btc, trend_lookback_bars=96, vol_lookback_bars=96)
        self.assertEqual(len(table), len(btc))
        self.assertTrue(set(table["btc_trend"].unique()) <= {"up", "down", "sideways", "unknown"})
        self.assertTrue(set(table["btc_vol_regime"].unique()) <= {"low", "high", "unknown"})
        # Truncating the future must not change any already-published label.
        cut = len(btc) - 96 * 5
        truncated = regimes.btc_regime_labels(btc.iloc[:cut], trend_lookback_bars=96, vol_lookback_bars=96)
        pd.testing.assert_frame_equal(truncated, table.iloc[:cut])

    def test_assign_regimes_never_looks_forward(self):
        btc = self._btc()
        table = regimes.btc_regime_labels(btc, trend_lookback_bars=96, vol_lookback_bars=96)
        trades = _synthetic_trades(n=200, mean_ret=0.0, std=0.01, seed=92, span_days=30)
        tagged = regimes.assign_regimes(trades, table)
        self.assertIn("btc_trend", tagged.columns)
        self.assertNotIn("btc_trend", trades.columns)
        # A trade decided before the first published label gets 'unknown', never a future one.
        early = trades.copy()
        early["decision_time_ms"] = int(table.index[0]) - DAY_MS
        early["entry_time_ms"] = early["decision_time_ms"]
        early["exit_time_ms"] = early["entry_time_ms"] + HOUR_MS
        self.assertTrue((regimes.assign_regimes(early, table)["btc_trend"] == "unknown").all())

    def test_regime_breakdown_cells(self):
        btc = self._btc()
        table = regimes.btc_regime_labels(btc, trend_lookback_bars=96, vol_lookback_bars=96)
        trades = _synthetic_trades(n=600, mean_ret=0.001, std=0.01, seed=93, span_days=35)
        priced = apply_costs(trades, CostModel.zero())
        tagged = regimes.assign_regimes(priced, table)
        breakdown = regimes.regime_breakdown(tagged, ret_col="ret_net", n_samples=100)
        self.assertGreater(len(breakdown), 0)
        self.assertEqual(int(breakdown["n"].sum()), len(tagged))
        self.assertTrue(set(["n", "mean", "ci_low", "ci_high"]).issubset(breakdown.columns))


class HarnessIntegrationTest(unittest.TestCase):
    def test_evaluate_does_not_mutate_input(self):
        trades = _synthetic_trades(n=300, mean_ret=0.001, std=0.01, seed=51)
        before = trades.copy()
        evaluate(trades, CostModel.binance_spot_taker(), _test_config(bootstrap_samples=50))
        pd.testing.assert_frame_equal(trades, before)

    def test_evaluate_reports_gross_and_net(self):
        trades = _synthetic_trades(n=300, mean_ret=0.003, std=0.01, seed=52)
        result = evaluate(trades, CostModel.mexc_futures_taker(), _test_config(bootstrap_samples=50))
        self.assertIn("gross", result["trade_metrics"])
        self.assertIn("net", result["trade_metrics"])
        self.assertIn("net", result["equity_metrics"])
        self.assertLess(
            result["trade_metrics"]["net"]["expectancy"],
            result["trade_metrics"]["gross"]["expectancy"],
        )
        self.assertLessEqual(
            int(result["portfolio"]["net"]["equity"]["n_open"].max()),
            result["config"]["max_concurrent"],
        )
        # The daily resample is the Sharpe to quote; the 15m grid is mostly flat bars.
        daily = result["equity_metrics_daily"]["net"]
        self.assertIsNotNone(daily["sharpe"])
        self.assertLess(daily["n_periods"], result["equity_metrics"]["net"]["n_periods"])
        self.assertAlmostEqual(daily["end_equity"], result["equity_metrics"]["net"]["end_equity"], places=12)

    def test_evaluate_handles_empty_table(self):
        empty = _synthetic_trades(n=0, mean_ret=0.0, std=0.01, seed=53)
        result = evaluate(empty, CostModel.zero(), _test_config(bootstrap_samples=10))
        self.assertEqual(result["n_trades"], 0)
        self.assertIn("note", result)

    def test_evaluate_adds_regime_breakdown_when_a_regime_table_is_given(self):
        n_bars = 96 * 40
        rng = np.random.default_rng(55)
        close = 30000 * np.exp(np.cumsum(rng.normal(0.0005, 0.004, size=n_bars)))
        btc = pd.DataFrame(
            {
                "open_time": T0 + np.arange(n_bars, dtype=np.int64) * BAR_MS,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.full(n_bars, 100.0),
            }
        )
        table = regimes.btc_regime_labels(btc, trend_lookback_bars=96, vol_lookback_bars=96)
        trades = _synthetic_trades(n=400, mean_ret=0.002, std=0.01, seed=56, span_days=35)
        result = evaluate(
            trades,
            CostModel.zero(),
            _test_config(bootstrap_samples=50),
            regime_table=table,
        )
        breakdown = result["regime_breakdown"]
        self.assertGreater(len(breakdown), 0)
        self.assertEqual(int(breakdown["n"].sum()), result["n_trades"])
        self.assertIn("btc_trend", result["trades"].columns)


class TrialLedgerTest(unittest.TestCase):
    """The trial count that deflates the Sharpe must be an audited database fact."""

    def test_n_trials_is_read_back_from_the_database(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db, jsonl = Path(tmp) / "ledger.sqlite", Path(tmp) / "ledger.jsonl"
            ledger = TrialLedger("scan", db_path=db, jsonl_path=jsonl)
            self.assertEqual(ledger.n_trials, 0)
            ledger.log_trial({"threshold": 1.35}, {"mean_ret_net": 0.001})
            ledger.log_trial({"threshold": np.float64(1.5)}, {"mean_ret_net": np.float64(0.002)})
            self.assertEqual(ledger.n_trials, 2)
            # A new session on the same prefix still counts the earlier trials.
            self.assertEqual(TrialLedger("scan", db_path=db, jsonl_path=jsonl).n_trials, 2)
            recorded = [t["parameters"]["threshold"] for t in ledger.trials()]
            self.assertEqual(recorded, [1.35, 1.5])


if __name__ == "__main__":
    unittest.main()
