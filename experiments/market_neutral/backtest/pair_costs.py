"""Two-leg spot costs for one pair trade, charged on posted capital.

Each leg is half of capital (dollar neutral). Both legs are entered and exited
as takers. The quoted-spread ladder is the worse leg's prior-day 30d volume
rank from ``config/costs.json``. Funding is zero: both legs are spot.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

TAKER_FEE = 0.001  # 10 bps, binance_spot taker, config/costs.json
LEG_WEIGHT = 0.5
SLIPPAGE_FRACTION_OF_SPREAD = 0.25

# rank band -> quoted spread in bps. Round-trip taker spread cost equals S.
_SPREAD_LADDER_BPS = (
    (10, 4.0),
    (25, 8.0),
    (45, 16.0),
    (65, 28.0),
    (10**9, 45.0),
)


def spread_bps(rank: float) -> float:
    if not np.isfinite(rank):
        return 45.0
    for upper, bps in _SPREAD_LADDER_BPS:
        if rank <= upper:
            return bps
    return 45.0


def friction_on_capital(rank_a: float, rank_b: float) -> Tuple[float, float, float]:
    """``(fee_cost, slippage_cost, funding_cost)`` as fractions of posted capital.

    Fee: 10 bps taker on each of 4 fills, each fill on half of capital → 20 bps.
    Slippage column: worse-leg quoted spread S plus 0.25·S, both round-trip,
    scaled onto capital (two half-weight legs) → 1.25·S bps.
    """
    if not (np.isfinite(rank_a) and np.isfinite(rank_b)):
        worse = float("inf")
    else:
        worse = max(float(rank_a), float(rank_b))
    s = spread_bps(float(worse)) / 10_000.0
    fee = 2.0 * (2.0 * TAKER_FEE) * LEG_WEIGHT
    slippage = 2.0 * (s + SLIPPAGE_FRACTION_OF_SPREAD * s) * LEG_WEIGHT
    return float(fee), float(slippage), 0.0


def cost_model_record() -> Dict[str, object]:
    return {
        "venue": "binance_spot",
        "legs": 2,
        "leg_weight": LEG_WEIGHT,
        "taker_fee": TAKER_FEE,
        "taker_fee_bps": TAKER_FEE * 10_000.0,
        "fills_per_leg": 2,
        "spread_ladder_bps": {str(k): v for k, v in _SPREAD_LADDER_BPS if k < 10**9} | {"66+": 45.0},
        "spread_note": "Round-trip taker spread cost equals S for the worse leg. Rank is prior-day PIT rank_qv_30d.",
        "slippage_fraction_of_spread": SLIPPAGE_FRACTION_OF_SPREAD,
        "funding": 0.0,
        "funding_note": "Spot versus spot. No funding stream is applied or invented.",
        "missing_rank_spread_bps": 45.0,
    }


def _day(ms: int) -> pd.Timestamp:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").normalize()


def attach_costs(
    trades: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    ranks: Mapping[tuple, float],
) -> pd.DataFrame:
    """Copy ``trades`` with finite per-trade cost columns for ``apply_costs``."""
    out = trades.copy()
    if out.empty:
        out["fee_cost"] = pd.Series(dtype=float)
        out["slippage_cost"] = pd.Series(dtype=float)
        out["funding_cost"] = pd.Series(dtype=float)
        return out
    fees = np.empty(len(out), dtype=float)
    slips = np.empty(len(out), dtype=float)
    funds = np.empty(len(out), dtype=float)
    entry = out["entry_time_ms"].to_numpy(dtype=np.int64)
    for i, ms in enumerate(entry.tolist()):
        day = _day(ms)
        fee, slip, fund = friction_on_capital(
            ranks.get((day, symbol_a), np.nan),
            ranks.get((day, symbol_b), np.nan),
        )
        fees[i] = fee
        slips[i] = slip
        funds[i] = fund
    out["fee_cost"] = fees
    out["slippage_cost"] = slips
    out["funding_cost"] = funds
    return out
