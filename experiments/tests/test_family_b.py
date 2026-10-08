"""Family B phase-1 mechanics. No network and no holdout read."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.market_neutral.data.bars import HOUR_MS
from experiments.market_neutral.funding_basis.costs import bnb_net_adjustment, execution_costs
from experiments.market_neutral.funding_basis.criteria import judge_funding
from experiments.market_neutral.funding_basis.matrix import build_cells, expected_cells
from experiments.market_neutral.funding_basis.signals import (
    distinct_thresholds,
    latest_funding_index,
    perp_price_at_funding,
    rolling_funding_mean,
)
from experiments.market_neutral.funding_basis.simulate import simulate_basis, simulate_carry
from experiments.market_neutral.funding_basis.universe import FAMILY_B_NAMES


def _panel(n_hours: int = 80, rate: float = 0.001, every: int = 8):
    open_time = np.arange(n_hours, dtype=np.int64) * HOUR_MS
    spot = np.full(n_hours, 100.0)
    perp = np.full(n_hours, 100.0)
    fund_time = np.arange(every, n_hours, every, dtype=np.int64) * HOUR_MS
    fund_rate = np.full(fund_time.shape, rate)
    close = open_time + HOUR_MS
    latest = latest_funding_index(close, fund_time)
    perp_at = perp_price_at_funding(open_time, perp, fund_time)
    signal = rolling_funding_mean(fund_rate, latest, 1)
    return open_time, spot, perp, fund_time, fund_rate, perp_at, latest, signal


class FamilyBDesignTest(unittest.TestCase):
    def test_book_and_cell_count(self):
        self.assertEqual(len(FAMILY_B_NAMES), 11)
        self.assertNotIn("PEPEUSDT", FAMILY_B_NAMES)
        levels = [("p50", 0.00004), ("p75_p90", 0.0001), ("p97", 0.00014)]
        cells = build_cells(FAMILY_B_NAMES, levels)
        self.assertEqual(len(cells), 792)
        self.assertEqual(expected_cells(11, 3), 792)
        self.assertEqual(sum(cell["stage"] == "B2" for cell in cells), 66)
        self.assertEqual(sum(cell["stage"] == "B3" for cell in cells), 198)
        self.assertEqual(sum(cell["stage"] == "B4" for cell in cells), 528)

    def test_collapsed_percentiles_stay_three_distinct_levels(self):
        sample = np.array([0.00004] * 50 + [0.0001] * 45 + [0.0002] * 5)
        levels, report = distinct_thresholds(sample)
        self.assertEqual(len(levels), 3)
        self.assertLess(levels[0][1], levels[1][1])
        self.assertLess(levels[1][1], levels[2][1])
        self.assertTrue(np.isclose(report["p75"], report["p90"]))

    def test_execution_cost_hand_values(self):
        fee, slip, perp_fee = execution_costs(100.0, 100.0, 1.0)
        self.assertAlmostEqual(fee, 0.0015, places=12)
        self.assertAlmostEqual(slip, 1.25 * 0.0004, places=12)
        self.assertAlmostEqual(perp_fee, 0.0005, places=12)
        self.assertAlmostEqual(bnb_net_adjustment(perp_fee), 0.00005, places=12)

    def test_positive_funding_is_collected_by_the_short(self):
        open_time, spot, perp, fund_time, fund_rate, perp_at, latest, signal = _panel()
        trades = simulate_carry(
            open_time,
            spot,
            perp,
            fund_time,
            fund_rate,
            perp_at,
            latest,
            signal,
            symbol="BTCUSDT",
            threshold=0.0001,
            exit_mode="max_hold",
            rank=1.0,
            tag="t",
            max_settlements=3,
        )
        self.assertGreaterEqual(len(trades), 1)
        first = trades.iloc[0]
        self.assertEqual(first["exit_reason"], "max_hold")
        self.assertEqual(int(first["n_settlements"]), 3)
        self.assertAlmostEqual(float(first["ret_funding"]), 0.0015, places=12)
        self.assertAlmostEqual(float(first["ret_price"]), 0.0, places=12)
        self.assertAlmostEqual(float(first["net_coin_exposure"]), 0.0, places=12)

    def test_negative_funding_exits_before_the_cap(self):
        open_time, spot, perp, fund_time, fund_rate, perp_at, latest, signal = _panel()
        fund_rate = fund_rate.copy()
        fund_rate[1] = -0.0002
        signal = rolling_funding_mean(fund_rate, latest, 1)
        trades = simulate_carry(
            open_time,
            spot,
            perp,
            fund_time,
            fund_rate,
            perp_at,
            latest,
            signal,
            symbol="BTCUSDT",
            threshold=0.0001,
            exit_mode="funding_negative",
            rank=1.0,
            tag="t",
            max_settlements=21,
        )
        self.assertEqual(trades["exit_reason"].iloc[0], "funding_negative")
        self.assertLess(int(trades["n_settlements"].iloc[0]), 21)

    def test_max_hold_ignores_a_negative_print(self):
        open_time, spot, perp, fund_time, fund_rate, perp_at, latest, signal = _panel()
        fund_rate = fund_rate.copy()
        fund_rate[1] = -0.0002
        signal = rolling_funding_mean(fund_rate, latest, 1)
        trades = simulate_carry(
            open_time,
            spot,
            perp,
            fund_time,
            fund_rate,
            perp_at,
            latest,
            signal,
            symbol="BTCUSDT",
            threshold=0.0001,
            exit_mode="max_hold",
            rank=1.0,
            tag="t",
            max_settlements=3,
        )
        self.assertEqual(trades["exit_reason"].iloc[0], "max_hold")
        self.assertEqual(int(trades["n_settlements"].iloc[0]), 3)

    def test_future_funding_does_not_change_an_earlier_signal(self):
        _open, _spot, _perp, fund_time, fund_rate, _px, latest, signal = _panel()
        shocked = fund_rate.copy()
        shocked[-1] = 0.05
        later = rolling_funding_mean(shocked, latest, 1)
        # The last settlement is known only at closes at or after its timestamp.
        known_after = np.searchsorted(fund_time, _open + HOUR_MS, side="right") - 1
        untouched = known_after < len(fund_rate) - 1
        np.testing.assert_allclose(signal[untouched], later[untouched], equal_nan=True)

    def test_basis_does_not_enter_the_cheap_perp_side(self):
        open_time, spot, perp, fund_time, fund_rate, perp_at, latest, _signal = _panel(n_hours=30)
        z = np.full(len(open_time), -3.0)
        z[5] = 3.0
        z[6] = 2.5
        z[7] = 1.0
        z[8] = 0.0
        trades = simulate_basis(
            open_time,
            spot,
            perp,
            fund_time,
            fund_rate,
            perp_at,
            latest,
            z,
            symbol="ETHUSDT",
            entry=2.0,
            exit_band=0.0,
            funding_filter=False,
            rank=1.0,
            tag="b",
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(int(trades["hold_bars"].iloc[0]), 3)

    def test_pass_bar_uses_underlying_beta(self):
        good = {
            "n_trades": 40,
            "dsr": 0.96,
            "pbo": 0.1,
            "sharpe_net_ann": 1.0,
            "sharpe_ci_low": 0.2,
            "sharpe_ci_high": 1.5,
            "beta_underlying": 0.05,
            "corr_btc": -0.04,
        }
        self.assertEqual(judge_funding(good)["status"], "pass")
        self.assertEqual(judge_funding(dict(good, beta_underlying=0.2))["status"], "fail")
        self.assertEqual(judge_funding(dict(good, n_trades=10))["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
