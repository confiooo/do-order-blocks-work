"""OHLCV integrity checks.

Corruption raises; nothing is repaired. Things that happen in real exchange data
are counted instead: zero-volume bars (illiquid pairs; about 12.9k of the 15m
bars) and gaps (forward labels never span one). The counts go into the data
manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd

from hawk_research.data.backfill import interval_ms

PRICE_COLUMNS = ("open", "high", "low", "close")


class CorruptOHLCV(ValueError):
    """Bars that cannot be real exchange data."""


@dataclass(frozen=True)
class OHLCVReport:
    rows: int
    duplicate_timestamps: int
    non_monotonic: int
    high_below_low: int
    non_positive_price: int
    open_close_outside_range: int
    nan_price: int
    zero_volume: int
    gaps: int
    missing_bars: int

    @property
    def hard_errors(self) -> Dict[str, int]:
        hard = {
            "duplicate_timestamps": self.duplicate_timestamps,
            "non_monotonic": self.non_monotonic,
            "high_below_low": self.high_below_low,
            "non_positive_price": self.non_positive_price,
            "open_close_outside_range": self.open_close_outside_range,
            "nan_price": self.nan_price,
        }
        return {k: v for k, v in hard.items() if v}

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def check_ohlcv(df: pd.DataFrame, interval: str) -> OHLCVReport:
    """Count every integrity condition. Never modifies ``df``."""
    if df.empty:
        return OHLCVReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    ot = df["open_time"].to_numpy(dtype=np.int64)
    o, h, lo, c = (df[col].to_numpy(dtype=float) for col in PRICE_COLUMNS)
    step = interval_ms(interval)
    deltas = np.diff(ot)
    nan_price = int(np.isnan(np.column_stack([o, h, lo, c])).any(axis=1).sum())
    with np.errstate(invalid="ignore"):
        outside = (o > h) | (o < lo) | (c > h) | (c < lo)
        non_pos = (np.column_stack([o, h, lo, c]) <= 0).any(axis=1)
    span = int((ot.max() - ot.min()) // step) + 1
    unique = int(pd.Index(ot).nunique())
    return OHLCVReport(
        rows=int(len(df)),
        duplicate_timestamps=int(len(ot) - unique),
        non_monotonic=int((deltas <= 0).sum()),
        high_below_low=int(np.sum(h < lo)),
        non_positive_price=int(non_pos.sum()),
        open_close_outside_range=int(outside.sum()),
        nan_price=nan_price,
        zero_volume=int((df["volume"].to_numpy(dtype=float) == 0).sum()),
        gaps=int((deltas > step).sum()),
        missing_bars=max(0, span - unique),
    )


def validate_ohlcv(df: pd.DataFrame, interval: str, label: str = "") -> OHLCVReport:
    """:func:`check_ohlcv`, raising :class:`CorruptOHLCV` on any hard error."""
    report = check_ohlcv(df, interval)
    if report.hard_errors:
        raise CorruptOHLCV(f"{label or 'OHLCV'} {interval}: {report.hard_errors}")
    return report
