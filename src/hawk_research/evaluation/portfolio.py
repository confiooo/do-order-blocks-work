"""Capacity-limited portfolio simulation over the trade table.

Zones fire on many correlated symbols within a few bars, so the mean per-trade
return assumes unlimited simultaneous capital. This module replays the trades in
time order under a concurrency cap; ``rejection_rate`` shows how much of the
signal a bounded account could actually take.

Acceptance depends only on positions already open at ``entry_time_ms`` and on
capital realised at or before that time, never on the trade's own outcome.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.evaluation.costs import validate_trades

BAR_MS_15M = BAR_MS


def _empty_result(
    ret_col: str,
    max_concurrent: int,
    capital: float,
    risk_fraction: float,
    one_position_per_symbol: bool,
) -> Dict[str, Any]:
    equity = pd.DataFrame({"equity": pd.Series(dtype=float), "n_open": pd.Series(dtype=int)})
    equity.index.name = "time_ms"
    return {
        "equity": equity,
        "trades": pd.DataFrame(),
        "accepted": 0,
        "rejected": 0,
        "rejected_capacity": 0,
        "rejected_symbol": 0,
        "rejection_rate": None,
        "capacity_utilization": None,
        "final_equity": capital,
        "total_return": None,
        "ret_col": ret_col,
        "max_concurrent": max_concurrent,
        "capital": capital,
        "risk_fraction": risk_fraction,
        "one_position_per_symbol": one_position_per_symbol,
        "n_bars": 0,
    }


def build_equity_curve(
    trades: pd.DataFrame,
    ret_col: str = "ret_net",
    max_concurrent: int = 5,
    capital: float = 1.0,
    risk_fraction: Optional[float] = None,
    one_position_per_symbol: bool = True,
    bar_ms: int = BAR_MS_15M,
    queue_by: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Replay trades under a concurrency cap and return the resulting equity curve.

    Sizing is ``equity_at_entry * risk_fraction`` (default ``1 / max_concurrent``, i.e.
    fully invested at full capacity), so position size compounds with realized equity
    only. Unrealized P&L is deliberately excluded from sizing: marking open positions
    to market would require intrabar paths this table does not carry.

    P&L is credited at ``exit_time_ms``. Ties at the same ``entry_time_ms`` are broken
    by ``queue_by`` (default ``entry_time_ms``, then ``zone_uid``) so the accept/reject
    decision is fully deterministic. Pre-registration §3.5 ranks by ascending
    ``pit_rank_prev_1d`` then ``zone_uid``.

    Returns a dict with ``equity`` (DataFrame on a regular ``bar_ms`` grid indexed by
    ``time_ms``, columns ``equity`` and ``n_open``), accept/reject counts,
    ``rejection_rate``, ``capacity_utilization`` and the accepted trade table.
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")
    if bar_ms <= 0:
        raise ValueError("bar_ms must be positive")
    if risk_fraction is None:
        risk_fraction = 1.0 / max_concurrent
    if risk_fraction <= 0:
        raise ValueError("risk_fraction must be positive")

    validate_trades(trades)
    if len(trades) == 0 or ret_col not in trades.columns:
        return _empty_result(ret_col, max_concurrent, capital, risk_fraction, one_position_per_symbol)

    if queue_by is None:
        queue_cols = ["entry_time_ms", "zone_uid"]
    else:
        queue_cols = [c for c in queue_by if c in trades.columns]
        if "entry_time_ms" not in queue_cols:
            queue_cols = ["entry_time_ms", *queue_cols]
        if "zone_uid" not in queue_cols:
            queue_cols.append("zone_uid")
    ordered = trades.sort_values(queue_cols, kind="mergesort")
    weights = (
        ordered["weight"].to_numpy(dtype=float)
        if "weight" in ordered.columns
        else np.ones(len(ordered), dtype=float)
    )
    entries = ordered["entry_time_ms"].to_numpy(dtype=np.int64)
    exits = ordered["exit_time_ms"].to_numpy(dtype=np.int64)
    rets = ordered[ret_col].to_numpy(dtype=float)
    symbols = ordered["symbol"].astype(str).to_numpy()
    uids = ordered["zone_uid"].astype(str).to_numpy()

    equity = float(capital)
    open_heap: List[tuple] = []  # (exit_time_ms, zone_uid, symbol, pnl)
    open_symbols: Dict[str, int] = {}
    realizations: List[tuple] = []  # (exit_time_ms, pnl)
    accepted_idx: List[int] = []
    accepted_notional: List[float] = []
    accepted_pnl: List[float] = []
    accepted_equity_at_entry: List[float] = []
    rejected_capacity = 0
    rejected_symbol = 0

    def _release(until_ms: int) -> None:
        nonlocal equity
        while open_heap and open_heap[0][0] <= until_ms:
            exit_ms, _uid, symbol, pnl = heapq.heappop(open_heap)
            equity += pnl
            realizations.append((exit_ms, pnl))
            remaining = open_symbols.get(symbol, 0) - 1
            if remaining <= 0:
                open_symbols.pop(symbol, None)
            else:
                open_symbols[symbol] = remaining

    for i in range(len(ordered)):
        entry_ms = int(entries[i])
        _release(entry_ms)
        if len(open_heap) >= max_concurrent:
            rejected_capacity += 1
            continue
        if one_position_per_symbol and symbols[i] in open_symbols:
            rejected_symbol += 1
            continue
        notional = max(equity, 0.0) * risk_fraction * float(weights[i])
        pnl = notional * float(rets[i])
        heapq.heappush(open_heap, (int(exits[i]), str(uids[i]), str(symbols[i]), pnl))
        open_symbols[symbols[i]] = open_symbols.get(symbols[i], 0) + 1
        accepted_idx.append(i)
        accepted_notional.append(notional)
        accepted_pnl.append(pnl)
        accepted_equity_at_entry.append(equity)

    _release(int(exits.max()))

    accepted = ordered.iloc[accepted_idx].copy()
    accepted["notional"] = accepted_notional
    accepted["pnl"] = accepted_pnl
    accepted["equity_at_entry"] = accepted_equity_at_entry

    grid_start = (int(entries.min()) // bar_ms) * bar_ms
    grid_end = ((int(exits.max()) // bar_ms) + 1) * bar_ms
    grid = np.arange(grid_start, grid_end + bar_ms, bar_ms, dtype=np.int64)

    equity_path = np.full(grid.size, float(capital), dtype=float)
    if realizations:
        realized = np.asarray(realizations, dtype=np.float64)
        slots = np.searchsorted(grid, realized[:, 0].astype(np.int64), side="left")
        deltas = np.zeros(grid.size, dtype=float)
        np.add.at(deltas, np.clip(slots, 0, grid.size - 1), realized[:, 1])
        equity_path = float(capital) + np.cumsum(deltas)

    open_delta = np.zeros(grid.size + 1, dtype=np.int64)
    if len(accepted):
        a_entry = accepted["entry_time_ms"].to_numpy(dtype=np.int64)
        a_exit = accepted["exit_time_ms"].to_numpy(dtype=np.int64)
        start_slots = np.clip(np.searchsorted(grid, a_entry, side="left"), 0, grid.size)
        end_slots = np.clip(np.searchsorted(grid, a_exit, side="left"), 0, grid.size)
        np.add.at(open_delta, start_slots, 1)
        np.add.at(open_delta, end_slots, -1)
    n_open = np.cumsum(open_delta)[: grid.size]

    curve = pd.DataFrame({"equity": equity_path, "n_open": n_open.astype(int)}, index=grid)
    curve.index.name = "time_ms"

    n_total = int(len(ordered))
    n_accepted = int(len(accepted))
    n_rejected = rejected_capacity + rejected_symbol
    return {
        "equity": curve,
        "trades": accepted,
        "accepted": n_accepted,
        "rejected": n_rejected,
        "rejected_capacity": int(rejected_capacity),
        "rejected_symbol": int(rejected_symbol),
        "rejection_rate": float(n_rejected / n_total) if n_total else None,
        "capacity_utilization": float(n_open.mean() / max_concurrent) if grid.size else None,
        "final_equity": float(equity),
        "total_return": float(equity / capital - 1.0) if capital > 0 else None,
        "ret_col": ret_col,
        "max_concurrent": int(max_concurrent),
        "capital": float(capital),
        "risk_fraction": float(risk_fraction),
        "one_position_per_symbol": bool(one_position_per_symbol),
        "n_bars": int(grid.size),
    }
