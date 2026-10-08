"""Test-window metrics for one pair configuration.

Daily net Sharpe is computed on the stitched out-of-sample calendar, with a
zero return on days that have no exit. The bootstrap resamples contiguous
blocks of those daily returns. Trade count, expectancy, and drawdown are
reported beside it. Neutrality is the OLS beta and correlation of that daily
series against BTC, not the accounting net of a long/short pair.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from hawk_research import constants
from hawk_research.evaluation import metrics as metrics_mod

MS_DAY = 24 * 3600 * 1000
PERIODS_PER_YEAR = constants.DAYS_PER_YEAR


def oos_mask(trades: pd.DataFrame, test_start_ms: int, test_end_ms: int) -> pd.DataFrame:
    """Trades whose decision falls in the stitched test window and that exit inside it.

    An exit after ``test_end_ms`` would need firewall or holdout prices. Those
    trades are dropped rather than closed on a truncated bar.
    """
    if trades.empty:
        return trades.copy()
    decision = trades["decision_time_ms"].to_numpy(dtype=np.int64)
    exit_ms = trades["exit_time_ms"].to_numpy(dtype=np.int64)
    keep = (decision >= test_start_ms) & (decision < test_end_ms) & (exit_ms <= test_end_ms)
    return trades.loc[keep].reset_index(drop=True)


def _utc_day(ms: np.ndarray | int) -> np.ndarray:
    """Calendar day of a timestamp. An exact UTC midnight belongs to the day that ended."""
    values = np.asarray(ms, dtype=np.int64)
    return ((values - 1) // MS_DAY) * MS_DAY


def daily_calendar(start_ms: int, end_ms: int) -> np.ndarray:
    """UTC day starts covering ``[start, end)``."""
    start = (int(start_ms) // MS_DAY) * MS_DAY
    return np.arange(start, int(end_ms), MS_DAY, dtype=np.int64)


def compound_daily(trades: pd.DataFrame, ret_col: str, days: np.ndarray) -> np.ndarray:
    """Capital return credited on the exit day. Exits on the same day compound in order."""
    out = np.zeros(days.size, dtype=float)
    if trades.empty or days.size == 0 or ret_col not in trades.columns:
        return out
    index = {int(day): i for i, day in enumerate(days.tolist())}
    ordered = trades.sort_values(["exit_time_ms", "zone_uid"], kind="mergesort")
    exit_day = _utc_day(ordered["exit_time_ms"].to_numpy(dtype=np.int64))
    rets = ordered[ret_col].to_numpy(dtype=float)
    buckets: list[list[float]] = [[] for _ in range(days.size)]
    for day, ret in zip(exit_day.tolist(), rets.tolist()):
        slot = index.get(int(day))
        if slot is None or not np.isfinite(ret):
            continue
        buckets[slot].append(float(ret))
    for i, group in enumerate(buckets):
        if not group:
            continue
        equity = 1.0
        for ret in group:
            equity *= 1.0 + ret
        out[i] = equity - 1.0
    return out


def daily_close_returns(bars: pd.DataFrame) -> pd.Series:
    """Last 1h close of each UTC day, simple return versus the previous day. No fill."""
    if bars.empty:
        return pd.Series(dtype=float)
    frame = bars[["open_time", "close"]].copy()
    close_known = frame["open_time"].to_numpy(dtype=np.int64) + 3_600_000
    frame["day"] = _utc_day(close_known)
    last = frame.sort_values("open_time", kind="mergesort").groupby("day", sort=True)["close"].last()
    prev = last.shift(1)
    ret = last / prev - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    ret.index = ret.index.astype(np.int64)
    return ret


def basket_returns(closes: Dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight daily return of the names that have a close that day. No price fill."""
    series = [daily_close_returns(bars).rename(symbol) for symbol, bars in closes.items()]
    if not series:
        return pd.Series(dtype=float)
    panel = pd.concat(series, axis=1, join="outer").sort_index()
    return panel.mean(axis=1, skipna=True)


def ols_beta_corr(y: np.ndarray, x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    mask = np.isfinite(y) & np.isfinite(x)
    yy = np.asarray(y, dtype=float)[mask]
    xx = np.asarray(x, dtype=float)[mask]
    if yy.size < 10:
        return None, None
    xc = xx - xx.mean()
    yc = yy - yy.mean()
    var = float(np.dot(xc, xc))
    if var <= 0.0:
        return None, None
    beta = float(np.dot(xc, yc) / var)
    sx = float(yy.std(ddof=1))
    sy = float(xx.std(ddof=1))
    if sx <= 0.0 or sy <= 0.0:
        return beta, None
    corr = float(np.dot(yc, xc) / ((yy.size - 1) * sx * sy))
    return beta, corr


def bootstrap_sharpe_ci(
    daily: np.ndarray,
    *,
    block: int = 5,
    n_samples: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> Dict[str, Optional[float]]:
    """Moving-block bootstrap of the annualized daily Sharpe.

    Blocks are ``block`` consecutive calendar days and wrap, matching the
    stationary-bootstrap idea of a circular sample without a Python loop per
    observation. The point estimate is the full-sample Sharpe, not the
    bootstrap mean.
    """
    values = np.asarray(daily, dtype=float)
    point = metrics_mod.sharpe(values, PERIODS_PER_YEAR)
    if values.size < block + 1 or point is None:
        return {
            "sharpe_net_ann": point,
            "sharpe_ci_low": None,
            "sharpe_ci_high": None,
            "sharpe_ci_width": None,
            "n_samples": 0,
        }
    n = values.size
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_samples, n_blocks))
    offset = np.arange(block, dtype=np.int64)
    index = (starts[..., None] + offset) % n
    sampled = values[index].reshape(n_samples, -1)[:, :n]
    mean = sampled.mean(axis=1)
    std = sampled.std(axis=1, ddof=1)
    ann = np.full(n_samples, np.nan)
    good = std > 0.0
    ann[good] = mean[good] / std[good] * np.sqrt(PERIODS_PER_YEAR)
    finite = ann[np.isfinite(ann)]
    if finite.size < 10:
        low = high = None
    else:
        low = float(np.quantile(finite, alpha / 2.0))
        high = float(np.quantile(finite, 1.0 - alpha / 2.0))
    width = None if low is None or high is None else high - low
    return {
        "sharpe_net_ann": point,
        "sharpe_ci_low": low,
        "sharpe_ci_high": high,
        "sharpe_ci_width": width,
        "n_samples": int(finite.size),
    }


def per_period_sharpe(daily: np.ndarray) -> Optional[float]:
    """Mean/std of daily returns, the unit the deflated Sharpe expects."""
    values = np.asarray(daily, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    std = float(values.std(ddof=1))
    if std <= 0.0:
        return None
    return float(values.mean() / std)


def score_configuration(
    trades: pd.DataFrame,
    days: np.ndarray,
    btc_daily: pd.Series,
    market_daily: pd.Series,
    *,
    block: int = 5,
    n_samples: int = 1000,
    seed: int = 7,
) -> Dict[str, object]:
    """Gross and net trade stats plus test-window neutrality of the daily book."""
    net_daily = compound_daily(trades, "ret_net", days)
    gross_daily = compound_daily(trades, "ret_gross", days)
    trade_net = (
        metrics_mod.trade_metrics(trades, "ret_net")
        if "ret_net" in trades.columns
        else metrics_mod.trade_metrics(trades, "ret_gross")
    )
    trade_gross = metrics_mod.trade_metrics(trades, "ret_gross")
    boot = bootstrap_sharpe_ci(net_daily, block=block, n_samples=n_samples, seed=seed)
    aligned_btc = _align(days, btc_daily)
    aligned_mkt = _align(days, market_daily)
    beta, corr = ols_beta_corr(net_daily, aligned_btc)
    _, corr_mkt = ols_beta_corr(net_daily, aligned_mkt)
    equity = np.cumprod(np.concatenate([[1.0], 1.0 + np.nan_to_num(net_daily, nan=0.0)]))
    hold = trades["hold_bars"] if "hold_bars" in trades.columns and len(trades) else pd.Series(dtype=float)
    gross_exp = (
        float(trades["gross_exposure"].mean()) if len(trades) and "gross_exposure" in trades.columns else None
    )
    net_exp = (
        float(trades["net_exposure"].mean()) if len(trades) and "net_exposure" in trades.columns else None
    )
    return {
        "n_trades": int(len(trades)),
        "win_rate_net": trade_net.get("win_rate"),
        "expectancy_net": trade_net.get("expectancy"),
        "expectancy_gross": trade_gross.get("expectancy"),
        "avg_win_net": trade_net.get("avg_win"),
        "avg_loss_net": trade_net.get("avg_loss"),
        "profit_factor_net": trade_net.get("profit_factor"),
        "total_ret_net_sum": trade_net.get("total_return_sum"),
        "total_ret_gross_sum": trade_gross.get("total_return_sum"),
        "sharpe_gross_ann": metrics_mod.sharpe(gross_daily, PERIODS_PER_YEAR),
        "sharpe_daily_sr": per_period_sharpe(net_daily),
        "sortino_net_ann": metrics_mod.sortino(net_daily, PERIODS_PER_YEAR),
        "max_drawdown_net": metrics_mod.max_drawdown(equity),
        "volatility_ann": _vol(net_daily),
        "mean_hold_hours": float(hold.mean()) if len(hold) else None,
        "turnover_capital": float(2.0 * len(trades)) if len(trades) else 0.0,
        "gross_exposure": gross_exp,
        "net_exposure": net_exp,
        "beta_btc": beta,
        "corr_btc": corr,
        "corr_market": corr_mkt,
        "n_days": int(days.size),
        **boot,
        "daily_net": net_daily,
    }


def _align(days: np.ndarray, series: pd.Series) -> np.ndarray:
    if series.empty:
        return np.full(days.size, np.nan)
    mapped = series.reindex(days)
    return mapped.to_numpy(dtype=float)


def _vol(daily: np.ndarray) -> Optional[float]:
    values = np.asarray(daily, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    return float(values.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
