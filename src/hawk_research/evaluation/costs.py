"""Transaction costs for the trade table.

Results are always reported gross and net: a 20 bps edge is gone after one
Binance spot round trip (2 x 10 bps).

Trade table columns (times in epoch ms, UTC): ``zone_uid``, ``symbol``,
``entry_time_ms``, ``exit_time_ms``, ``side`` (+1/-1), ``ret_gross`` (signed
return on notional), ``decision_time_ms`` (when the signal was known, <= entry).
Optional: ``weight``, ``liquidity_tier``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from hawk_research.constants import MS_PER_HOUR

REQUIRED_COLUMNS: Sequence[str] = (
    "zone_uid",
    "symbol",
    "entry_time_ms",
    "exit_time_ms",
    "side",
    "ret_gross",
    "decision_time_ms",
)


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost parameters for one venue.

    Fees are per side and charged twice (entry and exit). ``fee_discount`` is the
    multiplicative rebate of a fee tier (BNB payment, VIP level): the effective fee
    is ``fee * (1 - fee_discount)``. Slippage is expressed in basis points of
    notional per side and scaled by a per-trade illiquidity score so that thin
    symbols are not credited with top-of-book execution. ``funding_rate_8h`` is
    non-zero only for the perpetual variant.
    """

    taker_fee: float = 0.001
    maker_fee: float = 0.001
    fee_discount: float = 0.0
    slippage_bps_base: float = 2.0
    slippage_illiquidity_coef: float = 1.0
    funding_rate_8h: float = 0.0
    venue: str = "binance_spot"

    @property
    def effective_taker_fee(self) -> float:
        return self.taker_fee * (1.0 - self.fee_discount)

    @property
    def effective_maker_fee(self) -> float:
        return self.maker_fee * (1.0 - self.fee_discount)

    def round_trip_fee(self, maker: bool = False) -> float:
        """Total fee for opening and closing one position."""
        side_fee = self.effective_maker_fee if maker else self.effective_taker_fee
        return 2.0 * side_fee

    def as_dict(self) -> Dict[str, object]:
        return {
            "venue": self.venue,
            "taker_fee": self.taker_fee,
            "maker_fee": self.maker_fee,
            "fee_discount": self.fee_discount,
            "effective_taker_fee": self.effective_taker_fee,
            "round_trip_fee": self.round_trip_fee(),
            "slippage_bps_base": self.slippage_bps_base,
            "slippage_illiquidity_coef": self.slippage_illiquidity_coef,
            "funding_rate_8h": self.funding_rate_8h,
        }

    @classmethod
    def binance_spot_taker(cls) -> "CostModel":
        return cls(venue="binance_spot_taker")

    @classmethod
    def binance_spot_taker_bnb(cls) -> "CostModel":
        """Binance spot taker with the 25% BNB fee discount."""
        return cls(fee_discount=0.25, venue="binance_spot_taker_bnb")

    @classmethod
    def mexc_futures_taker(cls, fee: float = 0.0004, funding_rate_8h: float = 0.0) -> "CostModel":
        """MEXC perpetual taker at 4 bps per side: the model behind every round-1/SMC net number.

        MEXC's published taker fee sits in 0.0002..0.0006 depending on region and
        account; see config/costs.json ``result_scenarios`` and
        :meth:`mexc_futures_taker_api` for the 6 bps API-rate scenario.
        """
        return cls(
            taker_fee=fee,
            maker_fee=fee,
            slippage_bps_base=3.0,
            funding_rate_8h=funding_rate_8h,
            venue="mexc_futures_taker",
        )

    @classmethod
    def mexc_futures_taker_api(cls, funding_rate_8h: float = 0.0) -> "CostModel":
        """Alternative scenario: MEXC bills API futures trades at 6 bps taker.

        Not used by any published round-1 or SMC number (those used
        :meth:`mexc_futures_taker` at 4 bps). Fees are linear, so this scenario lowers
        every net mean return by exactly 2 x 2 bps = 4 bps per round trip.
        """
        return cls(
            taker_fee=0.0006,
            maker_fee=0.0006,
            slippage_bps_base=3.0,
            funding_rate_8h=funding_rate_8h,
            venue="mexc_futures_taker_api",
        )

    @classmethod
    def zero(cls) -> "CostModel":
        """Frictionless reference. Only ever used to isolate gross signal content."""
        return cls(
            taker_fee=0.0,
            maker_fee=0.0,
            slippage_bps_base=0.0,
            slippage_illiquidity_coef=0.0,
            funding_rate_8h=0.0,
            venue="zero",
        )


def validate_trades(trades: pd.DataFrame) -> None:
    """Fail fast on a malformed trade table rather than silently producing metrics."""
    missing = [c for c in REQUIRED_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(f"trade table missing required columns: {missing}")
    if len(trades) == 0:
        return
    if (trades["exit_time_ms"].to_numpy() <= trades["entry_time_ms"].to_numpy()).any():
        raise ValueError("exit_time_ms must be strictly greater than entry_time_ms")
    if (trades["decision_time_ms"].to_numpy() > trades["entry_time_ms"].to_numpy()).any():
        raise ValueError("decision_time_ms must be at or before entry_time_ms (causality)")
    if not trades["zone_uid"].is_unique:
        raise ValueError("zone_uid must be unique")


def _illiquidity_vector(trades: pd.DataFrame, illiquidity: Optional[pd.Series]) -> np.ndarray:
    """Align an illiquidity score to trade rows. Missing means 'most liquid' (0)."""
    n = len(trades)
    if illiquidity is None:
        return np.zeros(n, dtype=float)
    if not isinstance(illiquidity, pd.Series):
        values = np.asarray(illiquidity, dtype=float)
        if values.shape[0] != n:
            raise ValueError("illiquidity length does not match the trade table")
        return np.nan_to_num(np.clip(values, 0.0, None), nan=0.0)
    keys = trades["zone_uid"]
    if illiquidity.index.isin(keys).any():
        aligned = illiquidity.reindex(keys).to_numpy(dtype=float)
    else:
        aligned = illiquidity.reindex(trades.index).to_numpy(dtype=float)
    return np.nan_to_num(np.clip(aligned, 0.0, None), nan=0.0)


_OVERRIDE_COLUMNS = ("fee_cost", "slippage_cost", "funding_cost")


def _per_trade_cost_override(
    frame: pd.DataFrame,
) -> Optional[tuple]:
    """Finite per-trade costs, or ``None`` when the override columns are absent.

    The override is taken only when all three columns are present. A table that
    does not carry them — including every Family A trade table — keeps the
    scalar ``CostModel`` path unchanged.
    """
    if not all(col in frame.columns for col in _OVERRIDE_COLUMNS):
        return None
    fee = frame["fee_cost"].to_numpy(dtype=float)
    slippage = frame["slippage_cost"].to_numpy(dtype=float)
    funding = frame["funding_cost"].to_numpy(dtype=float)
    if fee.size and not (
        np.isfinite(fee).all() and np.isfinite(slippage).all() and np.isfinite(funding).all()
    ):
        raise ValueError("per-trade cost override requires finite fee_cost, slippage_cost, and funding_cost")
    return fee, slippage, funding


def apply_costs(
    trades: pd.DataFrame,
    model: CostModel,
    illiquidity: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Charge round-trip costs and return a copy with net returns.

    Never mutates the input. Added columns: ``fee_cost``, ``slippage_cost``,
    ``funding_cost``, ``total_cost``, ``ret_net``.

    Fees and slippage are charged on both legs (2x), because a position that is
    opened must be closed. Funding accrues over the holding period at
    ``funding_rate_8h * holding_hours / 8`` and is a cost for longs, a credit for
    shorts, which is the sign convention of a positive perpetual funding rate.

    If the input already has finite ``fee_cost``, ``slippage_cost``, and
    ``funding_cost`` columns, those values are the costs (a two-leg or
    historical-funding charge computed by the caller). The scalar model is not
    applied on top. Absent those columns, the scalar path is unchanged.
    """
    validate_trades(trades)
    out = trades.copy()
    n = len(out)
    if n == 0:
        for col in ("fee_cost", "slippage_cost", "funding_cost", "total_cost", "ret_net"):
            out[col] = pd.Series(dtype=float)
        return out

    override = _per_trade_cost_override(out)
    if override is None:
        illiq = _illiquidity_vector(out, illiquidity)
        fee = np.full(n, model.round_trip_fee(), dtype=float)
        slippage = (
            2.0 * (model.slippage_bps_base / 10_000.0) * (1.0 + model.slippage_illiquidity_coef * illiq)
        )

        if model.funding_rate_8h != 0.0:
            holding_hours = (
                out["exit_time_ms"].to_numpy(dtype=np.int64) - out["entry_time_ms"].to_numpy(dtype=np.int64)
            ) / MS_PER_HOUR
            side = out["side"].to_numpy(dtype=float)
            funding = side * model.funding_rate_8h * (holding_hours / 8.0)
        else:
            funding = np.zeros(n, dtype=float)
    else:
        fee, slippage, funding = override

    out["fee_cost"] = fee
    out["slippage_cost"] = slippage
    out["funding_cost"] = funding
    out["total_cost"] = fee + slippage + funding
    out["ret_net"] = out["ret_gross"].to_numpy(dtype=float) - out["total_cost"].to_numpy()
    return out


def cost_summary(trades: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Per-trade average of each cost component, for the report header."""
    if len(trades) == 0 or "total_cost" not in trades.columns:
        return {
            "n_trades": 0,
            "mean_fee_cost": None,
            "mean_slippage_cost": None,
            "mean_funding_cost": None,
            "mean_total_cost": None,
        }
    return {
        "n_trades": int(len(trades)),
        "mean_fee_cost": float(trades["fee_cost"].mean()),
        "mean_slippage_cost": float(trades["slippage_cost"].mean()),
        "mean_funding_cost": float(trades["funding_cost"].mean()),
        "mean_total_cost": float(trades["total_cost"].mean()),
    }
