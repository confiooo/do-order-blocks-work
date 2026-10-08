"""Walk-forward ADF gate: labels, causality, and which trades survive."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.market_neutral.cointegration import (
    adf_pvalue,
    evaluate_window,
    filter_trades,
    ols_residual,
)
from experiments.market_neutral.data.bars import HOUR_MS


class CointegrationFilterTest(unittest.TestCase):
    def test_white_noise_rejects_a_unit_root_and_a_random_walk_does_not(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 1.0, size=400)
        walk = np.cumsum(rng.normal(0.0, 1.0, size=400))
        self.assertLess(adf_pvalue(noise), 0.05)
        self.assertGreater(adf_pvalue(walk), 0.05)

    def test_fixed_spread_is_not_labeled_engle_granger(self):
        open_time = np.arange(200, dtype=np.int64) * HOUR_MS
        close = np.exp(np.linspace(0.0, 0.2, 200))
        fixed = evaluate_window(open_time, close, close, 0, 150 * HOUR_MS, "fixed")
        ols = evaluate_window(open_time, close * 1.01, close, 0, 150 * HOUR_MS, "rolling_ols")
        self.assertEqual(fixed["label"], "adf_fixed_spread")
        self.assertNotIn("Engle-Granger two-step", fixed["procedure"])
        self.assertEqual(ols["label"], "engle_granger_ols_residual")
        self.assertIn("Engle-Granger two-step", ols["procedure"])

    def test_ols_residual_is_orthogonal_to_the_regressor(self):
        rng = np.random.default_rng(1)
        log_x = np.linspace(0.0, 1.0, 80)
        log_y = 0.4 + 1.7 * log_x + rng.normal(0.0, 0.01, size=80)
        residual = ols_residual(log_y, log_x)
        self.assertAlmostEqual(float(residual @ np.ones(residual.size)), 0.0, places=8)
        self.assertAlmostEqual(float(residual @ log_x), 0.0, places=8)

    def test_prices_after_the_training_cut_do_not_change_the_pvalue(self):
        rng = np.random.default_rng(2)
        n = 300
        open_time = np.arange(n, dtype=np.int64) * HOUR_MS
        close_b = np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
        close_a = close_b * np.exp(rng.normal(0.0, 0.002, size=n))
        cut = int(open_time[180]) + HOUR_MS
        before = evaluate_window(open_time, close_a, close_b, 0, cut, "rolling_ols")
        shocked = close_a.copy()
        shocked[181:] *= 3.0
        after = evaluate_window(open_time, shocked, close_b, 0, cut, "rolling_ols")
        self.assertEqual(before["p_value"], after["p_value"])
        self.assertEqual(before["n_obs"], after["n_obs"])

    def test_only_passing_folds_keep_their_trades(self):
        trades = pd.DataFrame(
            {
                "decision_time_ms": [10, 25, 40],
                "zone_uid": ["a", "b", "c"],
                "ret_gross": [0.01, 0.02, 0.03],
            }
        )
        kept = filter_trades(trades, [(20, 30)])
        self.assertEqual(list(kept["zone_uid"]), ["b"])
        self.assertTrue(filter_trades(trades, []).empty)


if __name__ == "__main__":
    unittest.main()
