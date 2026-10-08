"""Development, firewall, and holdout bounds for this node.

The development and firewall dates match the directional-study protocol
(``family_a.d_bounds`` and its 180/5/60 walk-forward). The holdout is capped
at 2026-08-31. Bars from 2026-09-01 onward are not holdout and are not readable
in this phase.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

MS_DAY = 24 * 3600 * 1000

DEV_START = "2024-09-01"
DEV_END = "2026-03-01"  # exclusive: last development instant is 2026-02-28
FIREWALL_END = "2026-04-01"  # exclusive: March 2026 is the unread firewall
HOLDOUT_END = "2026-09-01"  # exclusive: holdout is 2026-04-01 through 2026-08-31

UNIVERSE_BIAS_NOTE = (
    "Phase-1 priced universe is the survivor subset of today's 80-symbol Binance "
    "spot OHLCV archive. pit_universe.parquet ranks are point-in-time, but names "
    "that were liquid historically and later delisted have no 15m prices on disk, "
    "so they cannot be traded in this pass. Results are not free of survivorship bias."
)


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def development_bounds() -> tuple[int, int]:
    """``[start, end)`` millisecond bounds of tier D."""
    return _ms(DEV_START), _ms(DEV_END)


def holdout_bounds() -> tuple[int, int]:
    """``[start, end)`` of the sealed holdout: 2026-04-01 through 2026-08-31."""
    return _ms(FIREWALL_END), _ms(HOLDOUT_END)


def walk_forward_folds() -> List[Dict[str, int]]:
    """Six rolling folds inside tier D. Train 180d, embargo 5d, test 60d, step 60d."""
    d0, d1 = development_bounds()
    folds: List[Dict[str, int]] = []
    for i in range(6):
        train_start = d0 + i * 60 * MS_DAY
        train_end = train_start + 180 * MS_DAY
        gap_end = train_end + 5 * MS_DAY
        test_end = gap_end + 60 * MS_DAY
        folds.append(
            {
                "fold": i + 1,
                "train_start_ms": int(train_start),
                "train_end_ms": int(train_end),
                "test_start_ms": int(gap_end),
                "test_end_ms": int(min(test_end, d1)),
                "scheme": "rolling",
            }
        )
    return folds


def test_window() -> tuple[int, int]:
    """Stitched out-of-sample window: first fold test start through last fold test end."""
    folds = walk_forward_folds()
    return int(folds[0]["test_start_ms"]), int(folds[-1]["test_end_ms"])
