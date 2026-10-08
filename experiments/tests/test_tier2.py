"""Tier-2 universe filter: band, persistence floor, and the 15-name cap."""

from __future__ import annotations

import unittest

import pandas as pd

from experiments.market_neutral.data.tier2 import select_tier2_names


def _row(symbol: str, day: int, qv: float, in80: bool, in20: bool) -> dict:
    return {
        "date": pd.Timestamp("2024-09-01", tz="UTC") + pd.Timedelta(days=day),
        "symbol": symbol,
        "qv_30d": qv,
        "rank_qv_30d": 30.0,
        "in_top80_30d": in80,
        "in_top20_30d": in20,
    }


class Tier2SelectionTest(unittest.TestCase):
    def test_floor_excludes_brief_visitors_and_cap_keeps_the_highest_volume(self):
        rows = []
        for day in range(10):
            rows.append(_row("HIGHUSDT", day, 500.0, True, False))
            rows.append(_row("MIDUSDT", day, 100.0, True, False))
            rows.append(_row("BRIEFUSDT", day, 900.0, day < 2, False))
            rows.append(_row("BTCUSDT", day, 9_000.0, True, False))
            rows.append(_row("USDCUSDT", day, 8_000.0, True, False))
            rows.append(_row("TOP20USDT", day, 700.0, True, True))
        frame = pd.DataFrame(rows)
        selected = select_tier2_names(frame, n_days=10, cap=1, min_fraction=0.50)
        self.assertEqual(list(selected.index), ["HIGHUSDT"])

    def test_cap_is_not_relaxed_when_more_names_qualify(self):
        rows = []
        for day in range(10):
            for i, name in enumerate(("AUSDT", "BUSDT", "CUSDT")):
                rows.append(_row(name, day, 300.0 - i, True, False))
        selected = select_tier2_names(pd.DataFrame(rows), n_days=10, cap=2, min_fraction=0.50)
        self.assertEqual(list(selected.index), ["AUSDT", "BUSDT"])


if __name__ == "__main__":
    unittest.main()
