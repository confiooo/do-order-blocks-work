"""Shared constants. The single source of truth for time units.

Crypto trades 24/7, so a year is 365.25 days of 15-minute bars. Every Sharpe
or Sortino annualisation in this package uses these values.
"""

from __future__ import annotations

MS_PER_SECOND = 1_000
MS_PER_MINUTE = 60 * MS_PER_SECOND
MS_PER_HOUR = 60 * MS_PER_MINUTE
MS_PER_DAY = 24 * MS_PER_HOUR

BAR_MINUTES = 15
BAR_MS = BAR_MINUTES * MS_PER_MINUTE  # 900_000
BARS_PER_HOUR = 60 // BAR_MINUTES
BARS_PER_DAY = 24 * BARS_PER_HOUR

DAYS_PER_YEAR = 365.25
MS_PER_YEAR = DAYS_PER_YEAR * MS_PER_DAY
BARS_PER_YEAR_15M = int(DAYS_PER_YEAR * BARS_PER_DAY)  # 35_064
