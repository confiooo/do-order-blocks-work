"""Execution costs for one long-spot / short-perp hedge.

Posted capital is the spot notional plus 1x perpetual margin (equal coin
quantity on each leg). Spot pays the Binance spot taker schedule. The
perpetual pays the standard USD-M taker rate, 5 bps, not the BNB discount.
Funding is not charged here; the backtest adds the historical settlement.
"""

from __future__ import annotations

from typing import Dict, Tuple

from experiments.market_neutral.backtest.pair_costs import spread_bps

SPOT_TAKER = 0.001
PERP_TAKER = 0.0005
PERP_TAKER_BNB = 0.00045
SLIPPAGE_FRACTION_OF_SPREAD = 0.25


def execution_costs(spot_px: float, perp_px: float, rank: float) -> Tuple[float, float, float]:
    """``(fee_cost, slippage_cost, perp_fee)`` as fractions of posted capital.

    Each leg is opened and closed as a taker. Quoted-spread cost S, plus
    0.25·S, is charged on both legs using the spot book's prior-day rank.
    """
    if spot_px <= 0.0 or perp_px <= 0.0:
        raise ValueError("prices must be positive")
    capital = float(spot_px) + float(perp_px)
    spot_w = float(spot_px) / capital
    perp_w = float(perp_px) / capital
    fee_spot = 2.0 * SPOT_TAKER * spot_w
    fee_perp = 2.0 * PERP_TAKER * perp_w
    spread = spread_bps(rank) / 10_000.0
    slip = (spread + SLIPPAGE_FRACTION_OF_SPREAD * spread) * (spot_w + perp_w)
    return float(fee_spot + fee_perp), float(slip), float(fee_perp)


def bnb_net_adjustment(perp_fee: float) -> float:
    """Return added back to ret_net when the perp taker rate is 4.5 bps instead of 5."""
    return float(perp_fee) * (1.0 - PERP_TAKER_BNB / PERP_TAKER)


def cost_model_record() -> Dict[str, object]:
    return {
        "structure": "long_spot_short_perp",
        "posted_capital": "spot notional + 1x perp margin, equal coin quantity",
        "spot_taker": SPOT_TAKER,
        "spot_taker_bps": SPOT_TAKER * 10_000.0,
        "perp_taker": PERP_TAKER,
        "perp_taker_bps": PERP_TAKER * 10_000.0,
        "perp_maker_bps": 2.0,
        "bnb_perp_taker": PERP_TAKER_BNB,
        "bnb_note": "4.5 bps perp taker is a footnote on the same trades, not a grid cell.",
        "funding": "historical settlement rate times perp price known at that settlement; credited to the short",
        "funding_placeholder": None,
        "spread": "config/costs.json ladder on both legs, round-trip taker spread S plus 0.25·S",
        "slippage_fraction_of_spread": SLIPPAGE_FRACTION_OF_SPREAD,
    }
