"""UTC session clocks for the day after a daily-candle fade.

Entry is the 00:00 UTC open, which is the Asia boundary. London and New York
are the cash opens used as observation points, not as a second entry.
"""

from __future__ import annotations

from typing import Tuple

from hawk_research.constants import BARS_PER_DAY as BARS_PER_DAY  # re-exported for exits/select
from hawk_research.constants import MS_PER_DAY

DAY_MS = MS_PER_DAY
STEP_15_MS = 900_000

# Minutes from the 00:00 UTC entry to each mark. New York is 13:30 UTC.
SESSION_MARKS: Tuple[Tuple[str, int], ...] = (
    ("london_open", 8 * 60),
    ("new_york_open", 13 * 60 + 30),
    ("new_york_close", 21 * 60),
    ("next_daily_close", 24 * 60),
)

# Half-open sessions on the forward day, in minutes from entry.
SESSIONS: Tuple[Tuple[str, int, int], ...] = (
    ("asia", 0, 8 * 60),
    ("london", 8 * 60, 13 * 60 + 30),
    ("new_york", 13 * 60 + 30, 21 * 60),
    ("late", 21 * 60, 24 * 60),
)


def bar_index(minutes: int) -> int:
    """15m bar index whose open is exactly ``minutes`` after entry."""
    if minutes % 15 != 0:
        raise ValueError("session marks must fall on a 15-minute boundary")
    return minutes // 15
