"""Causal daily-gainer selection and the fixed exit list."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.daily_fade.exits import evaluate_exit, session_path
from experiments.daily_fade.rollback import maximum_rollback
from experiments.daily_fade.select import daily_candles, rank_gainers, rank_losers
from experiments.daily_fade.sessions import BARS_PER_DAY, DAY_MS, STEP_15_MS
from experiments.daily_fade.zone_trade import behind_level, simulate, structural_level


def _day(start_ms: int, open_px: float, close_px: float, high: float, low: float) -> pd.DataFrame:
    times = start_ms + np.arange(BARS_PER_DAY, dtype=np.int64) * STEP_15_MS
    # Drift linearly so the day's first open and last close are the arguments.
    path = np.linspace(open_px, close_px, BARS_PER_DAY)
    return pd.DataFrame(
        {
            "open_time": times,
            "open": path,
            "high": np.maximum(path, high),
            "low": np.minimum(path, low),
            "close": path,
        }
    )


class DailyFadeTest(unittest.TestCase):
    def test_complete_day_uses_the_first_open_and_last_close(self):
        bars = _day(0, 100.0, 110.0, 112.0, 99.0)
        candles = daily_candles(bars)
        self.assertEqual(len(candles), 1)
        self.assertAlmostEqual(float(candles["open"].iloc[0]), 100.0)
        self.assertAlmostEqual(float(candles["close"].iloc[0]), 110.0)

    def test_biggest_gainer_is_short_and_future_days_do_not_change_it(self):
        def series(days):
            frames = [_day(i * DAY_MS, o, c, h, l) for i, (o, c, h, l) in enumerate(days)]
            return pd.concat(frames, ignore_index=True)

        up = series([(100.0, 100.0, 101.0, 99.0), (100.0, 130.0, 132.0, 99.0), (130.0, 131.0, 132.0, 129.0)])
        flat = series([(50.0, 50.0, 51.0, 49.0), (50.0, 51.0, 52.0, 49.0), (51.0, 200.0, 210.0, 50.0)])
        winners = rank_gainers(
            {
                "UPUSDT": daily_candles(up),
                "FLATUSDT": daily_candles(flat),
                "USDCUSDT": daily_candles(up),
            }
        )
        decision = winners[winners["day_ms"] == DAY_MS]
        self.assertEqual(decision["symbol"].iloc[0], "UPUSDT")
        self.assertEqual(int(decision["side"].iloc[0]), -1)
        self.assertNotIn("USDCUSDT", set(winners["symbol"]))

    def test_biggest_loser_is_faded_long(self):
        def series(days):
            frames = [_day(i * DAY_MS, o, c, h, l) for i, (o, c, h, l) in enumerate(days)]
            return pd.concat(frames, ignore_index=True)

        dumped = series([(100.0, 100.0, 101.0, 99.0), (100.0, 70.0, 101.0, 69.0), (70.0, 71.0, 72.0, 69.0)])
        mild = series([(50.0, 50.0, 51.0, 49.0), (50.0, 48.0, 51.0, 47.0), (48.0, 10.0, 49.0, 9.0)])
        losers = rank_losers(
            {
                "DUMPUSDT": daily_candles(dumped),
                "MILDUSDT": daily_candles(mild),
                "USDCUSDT": daily_candles(dumped),
            }
        )
        decision = losers[losers["day_ms"] == DAY_MS]
        self.assertEqual(decision["symbol"].iloc[0], "DUMPUSDT")
        self.assertEqual(int(decision["side"].iloc[0]), 1)
        self.assertNotIn("USDCUSDT", set(losers["symbol"]))

    def test_maximum_rollback_is_the_deepest_print_through_the_close(self):
        # Close 110. Next day trades down to 99, so the short's rollback is 11/110.
        forward = _day(DAY_MS, 110.0, 105.0, 112.0, 99.0)
        result = maximum_rollback(forward, close=110.0, side=-1, prior_close=100.0)
        self.assertAlmostEqual(float(result["max_rollback"]), 11.0 / 110.0)
        self.assertAlmostEqual(float(result["retrace_of_candle"]), 11.0 / 10.0)
        # A long fade of a loser uses the high above the close.
        bounced = maximum_rollback(forward, close=100.0, side=1, prior_close=120.0)
        self.assertAlmostEqual(float(bounced["max_rollback"]), 12.0 / 100.0)

    def test_no_trade_when_nothing_closed_up(self):
        down = pd.concat(
            [
                _day(0, 100.0, 100.0, 101.0, 99.0),
                _day(DAY_MS, 100.0, 90.0, 101.0, 89.0),
            ],
            ignore_index=True,
        )
        self.assertTrue(rank_gainers({"DOWNUSDT": daily_candles(down)}).empty)

    def test_london_mark_is_the_short_return_to_that_open(self):
        forward = _day(DAY_MS, 110.0, 100.0, 111.0, 99.0)
        path = session_path(forward, entry=110.0)
        london = float(forward["open"].iloc[8 * 4])
        self.assertAlmostEqual(path["london_open"], (110.0 - london) / 110.0)

    def test_half_giveback_exits_at_the_target_before_the_day_ends(self):
        # Entry 110, prior close 100, half giveback is 105. Price falls through it.
        forward = _day(DAY_MS, 110.0, 100.0, 110.0, 100.0)
        result = evaluate_exit(forward, 110.0, 100.0, day_range=12.0, rule="giveback_half")
        self.assertEqual(result["exit_reason"], "target")
        self.assertAlmostEqual(float(result["exit_price"]), 105.0)
        self.assertLess(int(result["exit_bar"]), 95)

    def test_limit_at_the_close_sells_the_open_when_the_open_is_above_it(self):
        forward = _day(DAY_MS, 112.0, 100.0, 112.0, 100.0)
        result = simulate(
            forward,
            close=110.0,
            prior=100.0,
            side=-1,
            day_high=132.0,
            day_low=99.0,
            retrace=0.5,
            max_rollback=0.10,
            entry_mode="limit_close",
            stop_mode="time_24h",
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["entry"], 112.0)
        self.assertEqual(result["reason"], "target")
        self.assertAlmostEqual(result["exit"], 105.0)

    def test_limit_is_skipped_when_the_rollback_never_trades_back_to_the_close(self):
        forward = _day(DAY_MS, 100.0, 90.0, 104.0, 90.0)
        result = simulate(
            forward,
            close=110.0,
            prior=100.0,
            side=-1,
            day_high=132.0,
            day_low=99.0,
            retrace=0.5,
            max_rollback=0.20,
            entry_mode="limit_close",
            stop_mode="time_24h",
        )
        self.assertIsNone(result)

    def test_swing_stop_sits_past_the_last_pivot_above_the_close(self):
        highs = [100.0, 101.0, 102.0, 108.0, 120.0, 109.0, 103.0, 102.0, 101.0, 100.0]
        day = pd.DataFrame(
            {"high": highs, "low": [v - 1.0 for v in highs], "close": [v - 0.5 for v in highs]}
        )
        level = structural_level(day, side=-1, kind="swing")
        self.assertAlmostEqual(level, 120.0)
        self.assertAlmostEqual(behind_level(level, side=-1, buffer=0.005), 120.6)

    def test_same_bar_stop_beats_the_rollback_target(self):
        forward = _day(DAY_MS, 110.0, 110.0, 140.0, 90.0)
        result = simulate(
            forward,
            close=110.0,
            prior=100.0,
            side=-1,
            day_high=120.0,
            day_low=99.0,
            retrace=0.5,
            max_rollback=0.20,
            entry_mode="market_open",
            stop_mode="day_extreme",
        )
        self.assertEqual(result["reason"], "stop")
        self.assertAlmostEqual(result["exit"], 120.0)

    def test_stop_is_taken_before_a_target_on_the_same_bar(self):
        forward = _day(DAY_MS, 110.0, 110.0, 130.0, 90.0)
        result = evaluate_exit(forward, 110.0, 100.0, day_range=20.0, rule="half_or_stop_or_24h")
        self.assertEqual(result["exit_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
