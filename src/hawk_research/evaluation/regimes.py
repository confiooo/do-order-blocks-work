"""Point-in-time BTC regime labels and per-regime breakdowns.

Labels use only bars at or before the labelled bar's close: trailing returns,
trailing volatility and an expanding (not full-sample) median for the volatility
split. A full-sample median or a centred window would leak later bars into
earlier labels.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from hawk_research import constants
from hawk_research.constants import BAR_MS
from hawk_research.stats.resampling import block_bootstrap_ci

BAR_MS_15M = BAR_MS
BARS_PER_YEAR_15M = float(constants.BARS_PER_YEAR_15M)
UNKNOWN = "unknown"


def btc_regime_labels(
    btc_15m: pd.DataFrame,
    trend_lookback_bars: int = 96 * 7,
    vol_lookback_bars: int = 96 * 7,
    flat_threshold: float = 0.02,
) -> pd.DataFrame:
    """Causal regime table indexed by ``open_time``.

    Columns:
    - ``btc_ret_trailing``: return over the trailing ``trend_lookback_bars``, ending
      at this bar's close.
    - ``btc_trend``: 'up' / 'down' / 'sideways' by comparing that return to
      +/- ``flat_threshold``.
    - ``btc_realized_vol``: annualized trailing standard deviation of 15m log returns.
    - ``btc_vol_regime``: 'low' / 'high' split at the EXPANDING median of realized
      volatility up to and including this bar. The threshold moves through time, which
      is the price of a point-in-time label; a full-sample median would be lookahead.

    Bars with insufficient history are labelled ``unknown`` rather than dropped, so
    joins never silently lose trades.
    """
    required = {"open_time", "close"}
    missing = required - set(btc_15m.columns)
    if missing:
        raise ValueError(f"btc_15m missing columns: {sorted(missing)}")

    frame = btc_15m.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    open_time = frame["open_time"].to_numpy(dtype=np.int64)
    close = frame["close"].astype(float)

    trailing = close / close.shift(int(trend_lookback_bars)) - 1.0
    log_ret = np.log(close).diff()
    realized_vol = log_ret.rolling(
        int(vol_lookback_bars), min_periods=int(vol_lookback_bars)
    ).std() * np.sqrt(BARS_PER_YEAR_15M)
    expanding_median = realized_vol.expanding(min_periods=1).median()

    trend = pd.Series(UNKNOWN, index=frame.index, dtype=object)
    known = trailing.notna()
    trend[known & (trailing > flat_threshold)] = "up"
    trend[known & (trailing < -flat_threshold)] = "down"
    trend[known & (trailing.abs() <= flat_threshold)] = "sideways"

    vol_regime = pd.Series(UNKNOWN, index=frame.index, dtype=object)
    vol_known = realized_vol.notna() & expanding_median.notna()
    vol_regime[vol_known & (realized_vol > expanding_median)] = "high"
    vol_regime[vol_known & (realized_vol <= expanding_median)] = "low"

    out = pd.DataFrame(
        {
            "btc_trend": trend.to_numpy(),
            "btc_ret_trailing": trailing.to_numpy(dtype=float),
            "btc_realized_vol": realized_vol.to_numpy(dtype=float),
            "btc_vol_median_expanding": expanding_median.to_numpy(dtype=float),
            "btc_vol_regime": vol_regime.to_numpy(),
        },
        index=pd.Index(open_time, name="open_time"),
    )
    return out


def assign_regimes(
    trades: pd.DataFrame,
    regime_table: pd.DataFrame,
    time_col: str = "decision_time_ms",
    bar_ms: int = BAR_MS_15M,
    regime_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Backward as-of join of regime labels onto trades. Never forward.

    A regime row indexed at ``open_time`` is only knowable at ``open_time + bar_ms``
    (the bar's close), so that is the timestamp matched against ``time_col``. The
    join direction is strictly backward: a trade takes the most recent label already
    published at its decision instant, and unlabelled trades get ``unknown``.

    Returns a copy; the input is not mutated.
    """
    out = trades.copy()
    if len(out) == 0 or regime_table is None or len(regime_table) == 0:
        return out
    if time_col not in out.columns:
        raise ValueError(f"trades missing {time_col}")

    cols = list(regime_cols) if regime_cols else list(regime_table.columns)
    right = regime_table[cols].copy()
    right["available_at_ms"] = np.asarray(regime_table.index, dtype=np.int64) + int(bar_ms)
    right = right.sort_values("available_at_ms", kind="mergesort")

    left = out[[time_col]].copy()
    left["_row"] = np.arange(len(out))
    left = left.sort_values(time_col, kind="mergesort")
    left[time_col] = left[time_col].astype(np.int64)

    merged = pd.merge_asof(
        left,
        right,
        left_on=time_col,
        right_on="available_at_ms",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row")

    for col in cols:
        series = pd.Series(merged[col].to_numpy(), index=out.index)
        if not pd.api.types.is_numeric_dtype(regime_table[col]):
            series = series.astype(object).where(series.notna(), UNKNOWN)
        out[col] = series
    return out


def regime_breakdown(
    trades: pd.DataFrame,
    ret_col: str = "ret_net",
    regime_cols: Sequence[str] = ("btc_trend", "btc_vol_regime"),
    time_col: str = "entry_time_ms",
    block_hours: int = 24,
    n_samples: int = 400,
    alpha: float = 0.05,
    seed: int = 7,
) -> pd.DataFrame:
    """Per-regime-cell n, mean and block-bootstrap CI.

    Cells are the observed cartesian product of ``regime_cols``. Intervals come from
    the time-block bootstrap, not from IID standard errors, because trades inside a
    regime cell are clustered in time by construction.
    """
    present = [c for c in regime_cols if c in trades.columns]
    if len(trades) == 0 or ret_col not in trades.columns or not present:
        return pd.DataFrame(
            columns=[*regime_cols, "n", "mean", "ci_low", "ci_high", "win_rate", "excludes_zero"]
        )

    rows: list[Dict[str, Any]] = []
    for key, group in trades.groupby(present, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        values = group[ret_col].to_numpy(dtype=float)
        times = group[time_col].to_numpy(dtype=np.int64)
        ci = block_bootstrap_ci(
            values,
            times,
            block_hours=block_hours,
            n_samples=n_samples,
            alpha=alpha,
            seed=seed,
        )
        low, high = ci["ci_low"], ci["ci_high"]
        rows.append(
            {
                **{col: keys[i] for i, col in enumerate(present)},
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else None,
                "ci_low": low,
                "ci_high": high,
                "win_rate": float((values > 0).mean()) if values.size else None,
                "excludes_zero": bool(low > 0 or high < 0) if low is not None and high is not None else None,
            }
        )
    return pd.DataFrame(rows).sort_values(present).reset_index(drop=True)
