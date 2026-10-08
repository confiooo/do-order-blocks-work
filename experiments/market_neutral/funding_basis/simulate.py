"""Long-spot / short-perp path simulator.

Funding collected on a trade is every settlement strictly after the entry
close and at or before the exit close. The settlement that triggered the
entry is information, not a cashflow: the fill is at that close, so the
position did not earn it.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from experiments.market_neutral.data.bars import HOUR_MS
from experiments.market_neutral.funding_basis.costs import execution_costs
from experiments.market_neutral.funding_basis.matrix import MAX_SETTLEMENTS


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "zone_uid",
            "symbol",
            "decision_time_ms",
            "entry_time_ms",
            "exit_time_ms",
            "side",
            "ret_gross",
            "ret_price",
            "ret_funding",
            "fee_cost",
            "slippage_cost",
            "funding_cost",
            "perp_fee",
            "weight",
            "hold_bars",
            "n_settlements",
            "exit_reason",
            "gross_exposure",
            "net_coin_exposure",
            "dollar_imbalance",
        ]
    )


def _rank_at(rank, index: int) -> float:
    if isinstance(rank, np.ndarray):
        return float(rank[index])
    return float(rank)


def _trade_row(
    *,
    symbol: str,
    tag: str,
    entry_i: int,
    exit_i: int,
    open_time: np.ndarray,
    spot: np.ndarray,
    perp: np.ndarray,
    fund_time: np.ndarray,
    fund_rate: np.ndarray,
    perp_at_fund: np.ndarray,
    start_s: int,
    end_s: int,
    reason: str,
    rank: float,
) -> Optional[dict]:
    spot_in, spot_out = float(spot[entry_i]), float(spot[exit_i])
    perp_in, perp_out = float(perp[entry_i]), float(perp[exit_i])
    if min(spot_in, spot_out, perp_in, perp_out) <= 0.0:
        return None
    entry_close = int(open_time[entry_i]) + HOUR_MS
    exit_close = int(open_time[exit_i]) + HOUR_MS
    if exit_close <= entry_close:
        return None
    capital = spot_in + perp_in
    price = (spot_out - spot_in) - (perp_out - perp_in)
    funding_pnl = 0.0
    n_settlements = 0
    for s in range(start_s, end_s + 1):
        stamp = int(fund_time[s])
        if stamp <= entry_close or stamp > exit_close:
            continue
        px = float(perp_at_fund[s])
        rate = float(fund_rate[s])
        if not np.isfinite(px) or not np.isfinite(rate) or px <= 0.0:
            continue
        funding_pnl += rate * px
        n_settlements += 1
    fee, slip, perp_fee = execution_costs(spot_in, perp_in, rank)
    return {
        "zone_uid": f"{symbol}|{tag}|{entry_close}",
        "symbol": f"{symbol}|PERP",
        "decision_time_ms": entry_close,
        "entry_time_ms": entry_close,
        "exit_time_ms": exit_close,
        "side": 1,
        "ret_gross": float((price + funding_pnl) / capital),
        "ret_price": float(price / capital),
        "ret_funding": float(funding_pnl / capital),
        "fee_cost": fee,
        "slippage_cost": slip,
        "funding_cost": 0.0,
        "perp_fee": perp_fee,
        "weight": 1.0,
        "hold_bars": int(exit_i - entry_i),
        "n_settlements": int(n_settlements),
        "exit_reason": reason,
        "gross_exposure": 1.0,
        "net_coin_exposure": 0.0,
        "dollar_imbalance": float(abs(spot_in - perp_in) / capital),
    }


def _consume_until(
    k_close: int,
    entry_close: int,
    s: int,
    fund_time: np.ndarray,
    fund_rate: np.ndarray,
    exit_mode: str,
    max_settlements: int,
) -> tuple[int, int, Optional[str]]:
    """Advance through settlements known by ``k_close``. Return (s, count, reason)."""
    count = 0
    reason = None
    limit = len(fund_time)
    while s < limit and int(fund_time[s]) <= k_close:
        if int(fund_time[s]) > entry_close:
            count += 1
            if exit_mode == "funding_negative" and float(fund_rate[s]) < 0.0:
                return s, count, "funding_negative"
            if count >= max_settlements:
                return s, count, "max_hold"
        s += 1
    return s, count, reason


def simulate_carry(
    open_time: np.ndarray,
    spot: np.ndarray,
    perp: np.ndarray,
    fund_time: np.ndarray,
    fund_rate: np.ndarray,
    perp_at_fund: np.ndarray,
    latest: np.ndarray,
    signal: np.ndarray,
    *,
    symbol: str,
    threshold: float,
    exit_mode: str,
    rank: float,
    tag: str,
    max_settlements: int = MAX_SETTLEMENTS,
) -> pd.DataFrame:
    if exit_mode not in ("funding_negative", "max_hold"):
        raise ValueError(f"unknown exit_mode {exit_mode}")
    n = len(open_time)
    rows: List[dict] = []
    i = 0
    while i < n - 1:
        sig = float(signal[i]) if i < len(signal) else np.nan
        if not np.isfinite(sig) or sig <= threshold:
            i += 1
            continue
        entry_close = int(open_time[i]) + HOUR_MS
        s = int(latest[i]) + 1
        count = 0
        exit_i = None
        reason = None
        end_s = s - 1
        for k in range(i + 1, n):
            k_close = int(open_time[k]) + HOUR_MS
            s, step, hit = _consume_until(
                k_close,
                entry_close,
                s,
                fund_time,
                fund_rate,
                exit_mode,
                max_settlements - count,
            )
            count += step
            if hit is not None:
                exit_i = k
                reason = hit
                end_s = s
                break
            if not np.isfinite(spot[k]) or not np.isfinite(perp[k]):
                break
        if exit_i is None:
            break
        row = _trade_row(
            symbol=symbol,
            tag=tag,
            entry_i=i,
            exit_i=exit_i,
            open_time=open_time,
            spot=spot,
            perp=perp,
            fund_time=fund_time,
            fund_rate=fund_rate,
            perp_at_fund=perp_at_fund,
            start_s=int(latest[i]) + 1,
            end_s=end_s,
            reason=reason,
            rank=_rank_at(rank, i),
        )
        if row is not None:
            rows.append(row)
        i = exit_i + 1
    if not rows:
        return _empty()
    return pd.DataFrame.from_records(rows)


def simulate_basis(
    open_time: np.ndarray,
    spot: np.ndarray,
    perp: np.ndarray,
    fund_time: np.ndarray,
    fund_rate: np.ndarray,
    perp_at_fund: np.ndarray,
    latest: np.ndarray,
    z: np.ndarray,
    *,
    symbol: str,
    entry: float,
    exit_band: float,
    funding_filter: bool,
    rank: float,
    tag: str,
) -> pd.DataFrame:
    """Enter only when log-perp minus log-spot is rich (``z > entry``)."""
    n = len(open_time)
    rows: List[dict] = []
    i = 0
    while i < n - 1:
        zi = float(z[i]) if i < len(z) else np.nan
        if not np.isfinite(zi) or zi <= entry:
            i += 1
            continue
        if funding_filter:
            j = int(latest[i])
            rate = float(fund_rate[j]) if j >= 0 else np.nan
            if not np.isfinite(rate) or rate <= 0.0:
                i += 1
                continue
        exit_i = None
        for k in range(i + 1, n):
            zk = float(z[k])
            if not np.isfinite(zk):
                break
            converged = zk <= 0.0 if exit_band == 0.0 else abs(zk) <= exit_band
            if converged:
                exit_i = k
                break
        if exit_i is None:
            break
        row = _trade_row(
            symbol=symbol,
            tag=tag,
            entry_i=i,
            exit_i=exit_i,
            open_time=open_time,
            spot=spot,
            perp=perp,
            fund_time=fund_time,
            fund_rate=fund_rate,
            perp_at_fund=perp_at_fund,
            start_s=int(latest[i]) + 1,
            end_s=len(fund_time) - 1,
            reason="z_exit",
            rank=_rank_at(rank, i),
        )
        if row is not None:
            rows.append(row)
        i = exit_i + 1
    if not rows:
        return _empty()
    return pd.DataFrame.from_records(rows)
