"""Performance metrics over returns, a trade table or an equity curve.

Functions are pure and return ``None`` for empty or degenerate input rather than
NaN, so a missing value cannot be averaged into a report by accident.

Trade-level metrics describe the signal; equity-level metrics describe what a
capacity-limited portfolio earned, which is smaller when zones fire in bursts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from hawk_research import constants
from hawk_research.constants import MS_PER_DAY, MS_PER_YEAR

BARS_PER_YEAR_15M = constants.BARS_PER_YEAR_15M


def _clean(returns: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(pd.Series(returns, dtype="float64").to_numpy(), dtype=float)
    return values[np.isfinite(values)]


def expectancy(returns) -> Optional[float]:
    """Mean return per trade. The only number that compounds."""
    values = _clean(returns)
    if values.size == 0:
        return None
    return float(values.mean())


def win_rate(returns) -> Optional[float]:
    values = _clean(returns)
    if values.size == 0:
        return None
    return float((values > 0).sum() / values.size)


def avg_win(returns) -> Optional[float]:
    values = _clean(returns)
    wins = values[values > 0]
    if wins.size == 0:
        return None
    return float(wins.mean())


def avg_loss(returns) -> Optional[float]:
    """Mean of losing trades, kept signed (negative) so the sign convention is explicit."""
    values = _clean(returns)
    losses = values[values < 0]
    if losses.size == 0:
        return None
    return float(losses.mean())


def payoff_ratio(returns) -> Optional[float]:
    w = avg_win(returns)
    l = avg_loss(returns)
    if w is None or l is None or l == 0.0:
        return None
    return float(w / abs(l))


def profit_factor(returns) -> Optional[float]:
    """Gross profit / gross loss. ``None`` when there are no losses (undefined, not infinite)."""
    values = _clean(returns)
    if values.size == 0:
        return None
    gross_loss = float(-values[values < 0].sum())
    if gross_loss <= 0.0:
        return None
    return float(values[values > 0].sum() / gross_loss)


def max_drawdown(equity: np.ndarray) -> Optional[float]:
    """Largest peak-to-trough fractional decline, returned positive."""
    values = _clean(equity)
    if values.size == 0:
        return None
    peaks = np.maximum.accumulate(values)
    valid = peaks > 0
    if not valid.any():
        return None
    dd = 1.0 - values[valid] / peaks[valid]
    return float(max(0.0, dd.max()))


def sharpe(returns, periods_per_year: float) -> Optional[float]:
    """Annualized Sharpe with zero risk-free rate. ``None`` if dispersion is zero."""
    values = _clean(returns)
    if values.size < 2 or periods_per_year <= 0:
        return None
    sd = float(values.std(ddof=1))
    if sd <= 0.0:
        return None
    return float(values.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns, periods_per_year: float) -> Optional[float]:
    """Annualized Sortino using downside deviation about zero."""
    values = _clean(returns)
    if values.size < 2 or periods_per_year <= 0:
        return None
    downside = values[values < 0]
    if downside.size == 0:
        return None
    dd = float(np.sqrt((downside**2).mean()))
    if dd <= 0.0:
        return None
    return float(values.mean() / dd * np.sqrt(periods_per_year))


def calmar(cagr: Optional[float], max_dd: Optional[float]) -> Optional[float]:
    if cagr is None or max_dd is None or max_dd <= 0.0:
        return None
    return float(cagr / max_dd)


def trades_per_day(trades: pd.DataFrame, time_col: str = "entry_time_ms") -> Optional[float]:
    if len(trades) == 0 or time_col not in trades.columns:
        return None
    times = trades[time_col].to_numpy(dtype=np.int64)
    span_ms = int(times.max() - times.min())
    if span_ms <= 0:
        return None
    return float(len(trades) / (span_ms / MS_PER_DAY))


def trade_metrics(trades: pd.DataFrame, ret_col: str = "ret_net") -> Dict[str, Any]:
    """Trade-level bundle. Independent of position sizing and capacity limits."""
    if len(trades) == 0 or ret_col not in trades.columns:
        return {
            "n_trades": 0,
            "ret_col": ret_col,
            "expectancy": None,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "payoff_ratio": None,
            "profit_factor": None,
            "total_return_sum": None,
            "std": None,
            "trades_per_day": None,
        }
    values = _clean(trades[ret_col])
    return {
        "n_trades": int(values.size),
        "ret_col": ret_col,
        "expectancy": expectancy(values),
        "win_rate": win_rate(values),
        "avg_win": avg_win(values),
        "avg_loss": avg_loss(values),
        "payoff_ratio": payoff_ratio(values),
        "profit_factor": profit_factor(values),
        "total_return_sum": float(values.sum()) if values.size else None,
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
        "trades_per_day": trades_per_day(trades),
    }


def resample_equity(equity_curve: pd.DataFrame, step_ms: int = MS_PER_DAY) -> pd.DataFrame:
    """Down-sample an equity curve by taking the last observation of each bucket.

    A 15m portfolio curve is mostly flat: P&L lands only on exit bars. Annualizing a
    Sharpe over 35064 such periods scales a sparse, lumpy series by sqrt(35064) and
    produces a figure that is not comparable to any published Sharpe and whose bias
    depends on how sparse the fills happen to be. Quote Sharpe on the daily resample;
    keep the 15m curve for drawdown, which a finer grid only measures more exactly.
    """
    if equity_curve is None or len(equity_curve) == 0:
        return equity_curve
    idx = np.asarray(equity_curve.index, dtype=np.int64)
    buckets = idx // max(1, int(step_ms))
    keep = np.ones(idx.size, dtype=bool)
    keep[:-1] = buckets[:-1] != buckets[1:]
    return equity_curve.loc[keep]


def equity_metrics(
    equity_curve: pd.DataFrame,
    equity_col: str = "equity",
    periods_per_year: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Portfolio-level bundle from a time-indexed equity curve.

    The index must be epoch ms on a regular grid; ``periods_per_year`` is inferred
    from the grid spacing when not given, so a 15m curve annualizes at 35064 bars.
    """
    empty: Dict[str, Optional[float]] = {
        "n_periods": 0,
        "start_equity": None,
        "end_equity": None,
        "total_return": None,
        "cagr": None,
        "sharpe": None,
        "sortino": None,
        "calmar": None,
        "max_drawdown": None,
        "periods_per_year": None,
        "years": None,
    }
    if equity_curve is None or len(equity_curve) < 2 or equity_col not in equity_curve.columns:
        return empty
    equity = _clean(equity_curve[equity_col])
    if equity.size < 2:
        return empty
    times = np.asarray(equity_curve.index, dtype=np.int64)
    span_ms = int(times[-1] - times[0])
    if periods_per_year is None:
        step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
        periods_per_year = MS_PER_YEAR / step if step > 0 else None
    years = span_ms / MS_PER_YEAR if span_ms > 0 else None

    start, end = float(equity[0]), float(equity[-1])
    total_return = (end / start - 1.0) if start > 0 else None
    if start > 0 and end > 0 and years and years > 0:
        cagr = float((end / start) ** (1.0 / years) - 1.0)
    else:
        cagr = None
    bar_returns = np.diff(equity) / np.where(equity[:-1] == 0.0, np.nan, equity[:-1])
    bar_returns = bar_returns[np.isfinite(bar_returns)]
    dd = max_drawdown(equity)
    return {
        "n_periods": int(equity.size),
        "start_equity": start,
        "end_equity": end,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe(bar_returns, periods_per_year) if periods_per_year else None,
        "sortino": sortino(bar_returns, periods_per_year) if periods_per_year else None,
        "calmar": calmar(cagr, dd),
        "max_drawdown": dd,
        "periods_per_year": float(periods_per_year) if periods_per_year else None,
        "years": float(years) if years else None,
    }
