"""Two-way harvest with symmetric remaining stop.

Same as the width sweep, except after one TP is hit the opposite stop is the
same percent beyond that TP as the TP was from entry (half the box), not 5%
of the box. Entry = midpoint. Sweep synthetic widths. 15m path rules unchanged.
"""

from __future__ import annotations

from typing import Any

from experiments.zone_trades.two_way_width_sweep import WIDTHS_PCT, run_sweep

# Buffer beyond each TP = half the box = TP distance from entry.
# Long TP at +W/2, short SL at +W/2 + W/2 = +W. Same on the downside.
BUFFER_FRAC = 0.5


def sweep() -> dict[str, Any]:
    half = (WIDTHS_PCT / 100.0) / 2.0
    return run_sweep(
        "sym-sl",
        BUFFER_FRAC,
        half.copy(),
        "same % as TP from entry: after long TP at +W/2, short SL at +W",
        "two_way_symmetric_sl.json",
    )


if __name__ == "__main__":
    sweep()
