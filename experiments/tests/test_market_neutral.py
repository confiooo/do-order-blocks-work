"""Market-neutral node tests, including the shared apply_costs regression."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.market_neutral.backtest.pair_costs import friction_on_capital
from experiments.market_neutral.backtest.simulate import simulate_pair
from experiments.market_neutral.calendar import (
    UNIVERSE_BIAS_NOTE,
    development_bounds,
    holdout_bounds,
    walk_forward_folds,
)
from experiments.market_neutral.data.bars import HOUR_MS, development_1h
from experiments.market_neutral.experiments.registry import (
    log_experiment,
    open_ledger,
    validate_record,
)
from experiments.market_neutral.features.spread import pair_zscore, rolling_ols_beta, rolling_zscore
from experiments.market_neutral.funding_basis.loaders import (
    MissingMarketData,
    load_funding,
    load_perpetual,
)
from experiments.market_neutral.holdout import HoldoutSealed, unseal_holdout
from experiments.market_neutral.strategies.relative_value import cells_per_pair, matrix_size
from experiments.market_neutral.validation.criteria import judge
from hawk_research.evaluation import metrics, portfolio
from hawk_research.evaluation.costs import CostModel, apply_costs
from hawk_research.studies.family_a.rules import d_bounds
from hawk_research.studies.family_a.statistics import rolling_folds

BAR_MS = 900_000
T0 = 1_700_000_000_000 // BAR_MS * BAR_MS


def _scalar_reference(trades: pd.DataFrame, model: CostModel, illiquidity: pd.Series) -> pd.DataFrame:
    """Independent copy of the pre-override scalar cost path."""
    out = trades.copy()
    keys = trades["zone_uid"]
    aligned = illiquidity.reindex(keys).to_numpy(dtype=float)
    illiq = np.nan_to_num(np.clip(aligned, 0.0, None), nan=0.0)
    fee = np.full(len(out), model.round_trip_fee(), dtype=float)
    slippage = 2.0 * (model.slippage_bps_base / 10_000.0) * (1.0 + model.slippage_illiquidity_coef * illiq)
    holding_hours = (
        out["exit_time_ms"].to_numpy(dtype=np.int64) - out["entry_time_ms"].to_numpy(dtype=np.int64)
    ) / 3_600_000
    side = out["side"].to_numpy(dtype=float)
    funding = side * model.funding_rate_8h * (holding_hours / 8.0)
    out["fee_cost"] = fee
    out["slippage_cost"] = slippage
    out["funding_cost"] = funding
    out["total_cost"] = fee + slippage + funding
    out["ret_net"] = out["ret_gross"].to_numpy(dtype=float) - out["total_cost"].to_numpy()
    return out


def _family_a_table() -> pd.DataFrame:
    """Columns a Family A filled-trade row carries into the harness."""
    return pd.DataFrame(
        {
            "zone_uid": ["fa-long", "fa-short"],
            "symbol": ["ETHUSDT", "SOLUSDT"],
            "entry_time_ms": [T0, T0 + 8 * 3_600_000],
            "exit_time_ms": [T0 + 16 * 3_600_000, T0 + 32 * 3_600_000],
            "side": [1, -1],
            "ret_gross": [0.012, -0.004],
            "decision_time_ms": [T0, T0 + 8 * 3_600_000],
            "ret_resid": [0.009, -0.002],
            "cost_bps": [34.0, 40.0],
            "ret_net": [0.012 - 0.0034, -0.004 - 0.0040],
            "fill_status": ["filled", "filled"],
            "n_funding": [2, 3],
            "weight": [1.0, 1.0],
        }
    )


class ApplyCostsRegressionTest(unittest.TestCase):
    def test_family_a_table_matches_scalar_path_byte_for_byte(self):
        trades = _family_a_table()
        original_net = trades["ret_net"].to_numpy(dtype=float).copy()
        model = CostModel(
            taker_fee=0.001,
            maker_fee=0.001,
            fee_discount=0.25,
            slippage_bps_base=2.0,
            slippage_illiquidity_coef=1.0,
            funding_rate_8h=0.0001,
            venue="family_a_regression",
        )
        illiquidity = pd.Series([0.5, 1.25], index=trades["zone_uid"])
        priced = apply_costs(trades, model, illiquidity=illiquidity)
        reference = _scalar_reference(trades, model, illiquidity)
        columns = ["fee_cost", "slippage_cost", "funding_cost", "total_cost", "ret_net"]
        pd.testing.assert_frame_equal(priced[columns], reference[columns], check_exact=True)
        self.assertEqual(
            priced[columns].to_csv(index=False).encode("utf-8"),
            reference[columns].to_csv(index=False).encode("utf-8"),
        )
        for column in ("ret_resid", "cost_bps", "fill_status", "n_funding"):
            pd.testing.assert_series_equal(priced[column], trades[column], check_names=False)
        np.testing.assert_array_equal(trades["ret_net"].to_numpy(dtype=float), original_net)
        self.assertFalse(np.allclose(priced["ret_net"].to_numpy(dtype=float), original_net))

    def test_override_replaces_the_scalar_model(self):
        trades = _family_a_table()
        trades["fee_cost"] = [0.01, 0.02]
        trades["slippage_cost"] = [0.003, 0.004]
        trades["funding_cost"] = [-0.005, 0.001]
        model = CostModel(taker_fee=0.05, funding_rate_8h=0.01, slippage_bps_base=50.0)
        priced = apply_costs(trades, model)
        np.testing.assert_allclose(priced["fee_cost"], [0.01, 0.02])
        np.testing.assert_allclose(priced["total_cost"], [0.008, 0.025])
        np.testing.assert_allclose(priced["ret_net"], [0.004, -0.029])

    def test_non_finite_override_is_rejected(self):
        trades = _family_a_table()
        trades["fee_cost"] = [0.01, np.nan]
        trades["slippage_cost"] = [0.0, 0.0]
        trades["funding_cost"] = [0.0, 0.0]
        with self.assertRaises(ValueError):
            apply_costs(trades, CostModel.zero())


class CompositeSymbolTest(unittest.TestCase):
    def _trades(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "zone_uid": ["p1", "p2", "p3"],
                "symbol": ["ETHUSDT|BTCUSDT", "ETHUSDT|BTCUSDT", "SOLUSDT|BTCUSDT"],
                "entry_time_ms": [T0, T0, T0],
                "exit_time_ms": [T0 + HOUR_MS, T0 + HOUR_MS, T0 + HOUR_MS],
                "side": [1, 1, -1],
                "ret_gross": [0.01, 0.02, 0.03],
                "ret_net": [0.01, 0.02, 0.03],
                "decision_time_ms": [T0, T0, T0],
            }
        )

    def test_portfolio_treats_the_composite_string_as_one_symbol(self):
        trades = self._trades()
        sim = portfolio.build_equity_curve(
            trades,
            ret_col="ret_net",
            max_concurrent=5,
            one_position_per_symbol=True,
            bar_ms=HOUR_MS,
        )
        self.assertEqual(sim["accepted"], 2)
        self.assertEqual(sim["rejected_symbol"], 1)
        self.assertEqual(sim["rejected_capacity"], 0)
        self.assertEqual(
            list(sim["trades"]["symbol"]),
            ["ETHUSDT|BTCUSDT", "SOLUSDT|BTCUSDT"],
        )
        self.assertLessEqual(int(sim["equity"]["n_open"].max()), 2)
        self.assertEqual(metrics.trade_metrics(sim["trades"], "ret_net")["n_trades"], 2)

        one_leg_looking = trades.iloc[[0]].reset_index(drop=True)
        single = portfolio.build_equity_curve(
            one_leg_looking,
            ret_col="ret_net",
            max_concurrent=1,
            one_position_per_symbol=True,
            bar_ms=HOUR_MS,
        )
        self.assertEqual(single["accepted"], 1)
        self.assertEqual(int(single["equity"]["n_open"].max()), 1)

        distinct = trades.iloc[[0, 2]].reset_index(drop=True)
        capped = portfolio.build_equity_curve(
            distinct,
            ret_col="ret_net",
            max_concurrent=1,
            one_position_per_symbol=True,
            bar_ms=HOUR_MS,
        )
        self.assertEqual(capped["accepted"], 1)
        self.assertEqual(capped["rejected_capacity"], 1)
        self.assertEqual(capped["rejected_symbol"], 0)


class SpreadFeatureTest(unittest.TestCase):
    def test_rolling_beta_recovers_a_known_slope(self):
        log_x = np.linspace(0.2, 1.4, 80)
        log_y = 2.0 * log_x + 0.3
        beta = rolling_ols_beta(log_y, log_x, 24)
        self.assertTrue(np.isnan(beta[:23]).all())
        np.testing.assert_allclose(beta[23:], 2.0, atol=1e-8)

    def test_zscore_matches_a_hand_window(self):
        values = np.arange(10, dtype=float)
        z = rolling_zscore(values, 5)
        window = values[-5:]
        expected = (window[-1] - window.mean()) / window.std(ddof=1)
        self.assertAlmostEqual(float(z[-1]), float(expected), places=12)

    def test_future_bar_does_not_change_past_z_or_beta(self):
        rng = np.random.default_rng(0)
        n = 200
        times = np.arange(n, dtype=np.int64) * HOUR_MS
        a = np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        b = np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        for mode in ("fixed", "rolling_ols"):
            z1, beta1 = pair_zscore(times, a, b, 24, mode)
            shocked = a.copy()
            shocked[-1] *= 1.5
            z2, beta2 = pair_zscore(times, shocked, b, 24, mode)
            np.testing.assert_allclose(z1[:-1], z2[:-1], equal_nan=True)
            np.testing.assert_allclose(beta1[:-1], beta2[:-1], equal_nan=True)

    def test_a_gap_invalidates_the_window_that_contains_it(self):
        times = np.arange(40, dtype=np.int64) * HOUR_MS
        times[30:] += 5 * HOUR_MS
        a = np.full(40, 100.0)
        b = np.linspace(50.0, 70.0, 40)
        z, _beta = pair_zscore(times, a, b, 10, "fixed")
        self.assertTrue(np.isnan(z[30]))
        self.assertTrue(np.isfinite(z[29]))

    def test_warmup_nan_does_not_blank_the_rest_of_the_rolling_zscore(self):
        rng = np.random.default_rng(3)
        n = 200
        times = np.arange(n, dtype=np.int64) * HOUR_MS
        b = np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        a = np.exp(np.log(b) * 1.2 + rng.normal(0.0, 0.02, n))
        z, beta = pair_zscore(times, a, b, 24, "rolling_ols")
        self.assertGreater(np.isfinite(beta).sum(), n - 30)
        self.assertGreater(np.isfinite(z).sum(), n // 2)
        self.assertGreater(int((np.abs(z[np.isfinite(z)]) > 1.5).sum()), 0)

    def test_beta_warmup_is_nan_and_then_finite(self):
        rng = np.random.default_rng(4)
        n = 80
        log_x = np.cumsum(rng.normal(0.0, 0.01, n))
        log_y = 0.3 + 1.4 * log_x + rng.normal(0.0, 0.01, n)
        beta = rolling_ols_beta(log_y, log_x, 24)
        self.assertTrue(np.isnan(beta[:23]).all())
        self.assertTrue(np.isfinite(beta[23:]).all())
        self.assertGreater(np.isfinite(beta).mean(), 0.7)

    def test_spread_is_nan_only_where_beta_is_nan(self):
        from experiments.market_neutral.features.spread import spread

        log_a = np.linspace(1.0, 2.0, 10)
        log_b = np.linspace(0.5, 1.0, 10)
        beta = np.array([np.nan, np.nan, 1.0, 1.2, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
        values = spread(log_a, log_b, beta)
        self.assertTrue(np.isnan(values[:2]).all())
        self.assertTrue(np.isfinite(values[2:]).all())

    def test_internal_gap_does_not_poison_later_rolling_windows(self):
        from experiments.market_neutral.features.spread import _rolling_sum

        values = np.arange(30, dtype=float)
        values[5] = np.nan
        rolled = _rolling_sum(values, 4)
        self.assertTrue(np.isnan(rolled[5:9]).all())
        self.assertTrue(np.isfinite(rolled[9]))
        self.assertAlmostEqual(float(rolled[9]), float(np.sum(values[6:10])))


class PairSimulationTest(unittest.TestCase):
    def _prices(self, n: int = 40):
        times = np.arange(n, dtype=np.int64) * HOUR_MS
        close_a = np.full(n, 100.0)
        close_b = np.full(n, 50.0)
        close_a[12:] = 110.0
        return times, close_a, close_b

    def test_same_spec_same_trades(self):
        times, close_a, close_b = self._prices()
        z = np.zeros(len(times))
        z[10] = 3.0
        z[11] = 1.0
        z[12] = 0.2
        beta = np.ones(len(times))
        kwargs = dict(
            entry=2.0,
            exit_band=0.5,
            max_hold_bars=24,
            symbol="ETHUSDT|BTCUSDT",
            beta_mode="fixed",
            window=24,
        )
        first = simulate_pair(times, close_a, close_b, z, beta, **kwargs)
        second = simulate_pair(times, close_a, close_b, z, beta, **kwargs)
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    def test_exit_bands_and_two_leg_sign(self):
        times, close_a, close_b = self._prices()
        z = np.zeros(len(times))
        z[10] = 3.0
        z[11] = 1.0
        z[12] = 0.2
        z[13] = -0.1
        beta = np.ones(len(times))
        inside = simulate_pair(
            times,
            close_a,
            close_b,
            z,
            beta,
            entry=2.0,
            exit_band=0.5,
            max_hold_bars=24,
            symbol="ETHUSDT|BTCUSDT",
            beta_mode="fixed",
            window=24,
        )
        through = simulate_pair(
            times,
            close_a,
            close_b,
            z,
            beta,
            entry=2.0,
            exit_band=0.0,
            max_hold_bars=24,
            symbol="ETHUSDT|BTCUSDT",
            beta_mode="fixed",
            window=24,
        )
        self.assertEqual(len(inside), 1)
        self.assertEqual(int(inside["side"].iloc[0]), -1)
        self.assertEqual(int(inside["side_b"].iloc[0]), 1)
        self.assertEqual(int(inside["hold_bars"].iloc[0]), 2)
        self.assertEqual(inside["exit_reason"].iloc[0], "z_exit")
        self.assertEqual(int(through["hold_bars"].iloc[0]), 3)
        # Short A from 100 to 110, long B unchanged: leg A loses 10%, capital return -5%.
        self.assertAlmostEqual(float(inside["ret_gross"].iloc[0]), -0.05, places=12)
        self.assertAlmostEqual(float(inside["net_exposure"].iloc[0]), 0.0, places=12)
        self.assertAlmostEqual(float(inside["gross_exposure"].iloc[0]), 1.0, places=12)

    def test_pair_friction_hand_values(self):
        fee, slip, funding = friction_on_capital(1.0, 8.0)
        self.assertAlmostEqual(fee, 0.002, places=12)
        self.assertAlmostEqual(slip, 1.25 * 4.0 / 10_000.0, places=12)
        self.assertEqual(funding, 0.0)
        _fee, wide, _fund = friction_on_capital(np.nan, 3.0)
        self.assertAlmostEqual(wide, 1.25 * 45.0 / 10_000.0, places=12)


class ProtocolTest(unittest.TestCase):
    def test_calendar_matches_the_directional_study_and_caps_holdout(self):
        d0, d1, h0 = d_bounds()
        self.assertEqual(development_bounds(), (d0, d1))
        self.assertEqual(holdout_bounds()[0], h0)
        self.assertEqual(
            holdout_bounds()[1],
            int(pd.Timestamp("2026-09-01", tz="UTC").timestamp() * 1000),
        )
        self.assertEqual(walk_forward_folds(), rolling_folds())
        self.assertEqual(cells_per_pair(), 144)
        self.assertEqual(matrix_size(12), 1728)

    def test_holdout_unseals_once(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "holdout_audit.jsonl"
            seen = []

            def loader():
                seen.append("read")
                return {"rows": 1}

            first = unseal_holdout("relative_value", "unit test", loader, audit_path=audit)
            self.assertEqual(first, {"rows": 1})
            self.assertEqual(seen, ["read"])
            with self.assertRaises(HoldoutSealed):
                unseal_holdout("relative_value", "second look", loader, audit_path=audit)
            self.assertEqual(seen, ["read"])

    def test_midnight_exit_credits_the_day_that_ended(self):
        from experiments.market_neutral.validation.score import compound_daily

        midnight = int(pd.Timestamp("2024-09-02", tz="UTC").timestamp() * 1000)
        previous = midnight - 24 * 3600 * 1000
        trades = pd.DataFrame({"exit_time_ms": [midnight], "zone_uid": ["z"], "ret_net": [0.01]})
        credited = compound_daily(trades, "ret_net", np.array([previous], dtype=np.int64))
        self.assertAlmostEqual(float(credited[0]), 0.01, places=12)

    def test_development_bars_stop_before_the_firewall(self):
        from hawk_research.data import paths

        if not paths.ohlcv_path("BTCUSDT", "15m").exists():
            self.skipTest("BTCUSDT 15m OHLCV has not been downloaded")
        bars = development_1h("BTCUSDT")
        _start, end = development_bounds()
        self.assertGreater(len(bars), 1000)
        self.assertLessEqual(int(bars["open_time"].max()) + HOUR_MS, end)

    def test_pass_bar(self):
        good = {
            "n_trades": 30,
            "dsr": 0.95,
            "pbo": 0.199999,
            "sharpe_net_ann": 0.8,
            "sharpe_ci_low": 0.05,
            "sharpe_ci_high": 1.4,
            "beta_btc": -0.19,
            "corr_btc": 0.19,
        }
        verdict = judge(good)
        self.assertEqual(verdict["status"], "pass")
        self.assertAlmostEqual(verdict["sharpe_ci_width"], 1.35, places=12)
        short = dict(good, n_trades=29)
        self.assertEqual(judge(short)["status"], "inconclusive")
        for broken in (
            dict(good, dsr=0.949),
            dict(good, pbo=0.2),
            dict(good, sharpe_ci_low=0.0),
            dict(good, beta_btc=0.2),
            dict(good, corr_btc=-0.2),
        ):
            self.assertEqual(judge(broken)["status"], "fail")

    def test_registry_requires_the_survivor_bias_note(self):
        spec = {
            "family": "relative_value",
            "strategy_name": "pair_zscore",
            "symbols": ["ETHUSDT", "BTCUSDT"],
            "timeframe": "1h",
            "parameters": {"entry": 2.0},
            "feature_definitions": {"spread": "log A - beta log B"},
            "entry_rules": "z > entry",
            "exit_rules": "z crosses 0",
            "cost_model": {"venue": "binance_spot"},
            "training_period": DEV_LABEL,
            "validation_period": "embargo",
            "test_period": "stitched oos",
            "universe_bias_note": UNIVERSE_BIAS_NOTE,
        }
        validate_record(spec)
        spec["universe_bias_note"] = "survivor effects were handled"
        with self.assertRaises(ValueError):
            validate_record(spec)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = open_ledger(
                db_path=root / "research.sqlite", jsonl_path=root / "trials.jsonl", name_prefix="mn_test"
            )
            spec["universe_bias_note"] = UNIVERSE_BIAS_NOTE
            experiment_id = log_experiment(ledger, spec, {"n_trades": 0, "status": "inconclusive"}, "0000")
            self.assertGreater(experiment_id, 0)
            text = (root / "trials.jsonl").read_text(encoding="utf-8")
            self.assertIn(UNIVERSE_BIAS_NOTE, text)

    def test_family_b_refuses_to_invent_missing_series(self):
        for loader in (load_funding, load_perpetual):
            with self.assertRaises(MissingMarketData) as ctx:
                loader("NOSUCHUSDT")
            self.assertIn("will not fabricate", str(ctx.exception))


class FamilyBIngestTest(unittest.TestCase):
    def test_funding_csv_keeps_published_timestamps(self):
        from experiments.market_neutral.funding_basis.ingest import parse_funding_csv

        raw = (
            b"calc_time,funding_interval_hours,last_funding_rate\n"
            b"1725148800000,8,0.00004113\n"
            b"1725177600001,8,-0.00003432\n"
        )
        frame = parse_funding_csv(raw)
        self.assertEqual(int(frame["funding_time_ms"].iloc[1]), 1725177600001)
        self.assertAlmostEqual(float(frame["funding_rate"].iloc[0]), 0.00004113, places=12)

    def test_short_history_is_excluded(self):
        from experiments.market_neutral.funding_basis.ingest import inclusion

        included, reason = inclusion(10.0, 10.0)
        self.assertFalse(included)
        self.assertIn("under 60", reason)
        included, reason = inclusion(400.0, 400.0)
        self.assertTrue(included)
        self.assertIsNone(reason)

    def test_row_count_guard(self):
        from experiments.market_neutral.funding_basis.ingest import counts_are_sane

        self.assertIsNone(counts_are_sane(28_000, 200_000))
        self.assertIsNotNone(counts_are_sane(5_000_000, 200_000))

    def test_binance_perp_fee_is_recorded_and_mexc_stays(self):
        import json

        from hawk_research.data import paths

        costs = json.loads((paths.CONFIG_DIR / "costs.json").read_text(encoding="utf-8"))
        perp = costs["venues"]["binance_usd_m"]
        self.assertEqual(perp["maker_fee_bps"], 2.0)
        self.assertEqual(perp["taker_fee_bps"], 5.0)
        self.assertIsNone(perp["funding_bps_per_settlement"])
        self.assertIn("mexc_perp_api", costs["venues"])
        self.assertEqual(costs["base_case"]["venue"], "mexc_perp_api")

    def test_development_read_stops_before_the_firewall(self):
        from experiments.market_neutral.calendar import development_bounds, holdout_bounds
        from experiments.market_neutral.funding_basis.loaders import (
            funding_path,
            load_funding,
            load_holdout_funding,
            load_perpetual,
        )
        from experiments.market_neutral.holdout import HoldoutSealed

        if not funding_path("BTCUSDT").exists():
            self.skipTest("BTCUSDT funding has not been ingested")

        funding = load_funding("BTCUSDT")
        perp = load_perpetual("BTCUSDT")
        _, dev_end = development_bounds()
        self.assertGreater(len(funding), 100)
        self.assertLess(int(funding["funding_time_ms"].max()), dev_end)
        self.assertLess(int(perp["open_time"].max()), dev_end)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "holdout_audit.jsonl"
            held = load_holdout_funding("BTCUSDT", "unit test of the seal", audit_path=audit)
            start, end = holdout_bounds()
            self.assertGreater(len(held), 10)
            self.assertGreaterEqual(int(held["funding_time_ms"].min()), start)
            self.assertLess(int(held["funding_time_ms"].max()), end)
            with self.assertRaises(HoldoutSealed):
                load_holdout_funding("ETHUSDT", "second touch", audit_path=audit)


DEV_LABEL = "2024-09-01/2026-02-28"


if __name__ == "__main__":
    unittest.main()
