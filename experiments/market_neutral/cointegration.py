"""Walk-forward spread filter for the phase-1 relative-value pairs.

Fixed beta = 1 is an ADF stationarity test on ``log(A) - log(B)``. It is not
Engle-Granger, because that procedure estimates the cointegrating vector.
Rolling OLS is Engle-Granger's two-step: OLS of ``log(A)`` on ``log(B)`` over
the fold's training window, then ADF on the residual. Eligibility is
``p < 0.05``. The p-value is the ADF p-value on that series, not a
MacKinnon cointegration critical-value adjustment.

The test uses only bar closes inside the fold's training window. A pair is
traded in that fold's test window only when the training window passed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from experiments.market_neutral.calendar import walk_forward_folds
from experiments.market_neutral.data.bars import HOUR_MS

ALPHA = 0.05
MIN_OBS = 100
Window = Tuple[int, int]


def adf_pvalue(series: np.ndarray) -> Optional[float]:
    """ADF p-value with a constant and AIC lag. ``None`` when the test cannot run."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < MIN_OBS or float(np.std(values)) == 0.0:
        return None
    try:
        result = adfuller(values, regression="c", autolag="AIC", result_object=False)
    except (ValueError, np.linalg.LinAlgError):
        return None
    p_value = float(result[1])
    if not np.isfinite(p_value):
        return None
    return p_value


def ols_residual(log_y: np.ndarray, log_x: np.ndarray) -> np.ndarray:
    """Residual from ``log_y = a + b log_x`` over the rows that are finite."""
    y = np.asarray(log_y, dtype=float)
    x = np.asarray(log_x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    design = np.column_stack([np.ones(y.size), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _training_logs(
    open_time: np.ndarray,
    close_a: np.ndarray,
    close_b: np.ndarray,
    train_start_ms: int,
    train_end_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Log prices whose close is inside ``[train_start, train_end)``."""
    close_time = np.asarray(open_time, dtype=np.int64) + HOUR_MS
    keep = (close_time >= int(train_start_ms)) & (close_time < int(train_end_ms))
    a = np.asarray(close_a, dtype=float)[keep]
    b = np.asarray(close_b, dtype=float)[keep]
    ok = (a > 0.0) & (b > 0.0) & np.isfinite(a) & np.isfinite(b)
    return np.log(a[ok]), np.log(b[ok])


def evaluate_window(
    open_time: np.ndarray,
    close_a: np.ndarray,
    close_b: np.ndarray,
    train_start_ms: int,
    train_end_ms: int,
    beta_mode: str,
) -> Dict[str, object]:
    log_a, log_b = _training_logs(open_time, close_a, close_b, train_start_ms, train_end_ms)
    if beta_mode == "fixed":
        series = log_a - log_b
        label = "adf_fixed_spread"
        procedure = "ADF stationarity test on log(A) - log(B). Not Engle-Granger."
    elif beta_mode == "rolling_ols":
        series = ols_residual(log_a, log_b) if log_a.size else np.zeros(0)
        label = "engle_granger_ols_residual"
        procedure = (
            "Engle-Granger two-step: OLS of log(A) on log(B) over the training window, "
            "then ADF on the residual. The p-value is the ADF p-value, not a "
            "MacKinnon cointegration adjustment."
        )
    else:
        raise ValueError(f"unknown beta_mode {beta_mode}")
    p_value = adf_pvalue(series)
    return {
        "beta_mode": beta_mode,
        "label": label,
        "procedure": procedure,
        "p_value": p_value,
        "passed": bool(p_value is not None and p_value < ALPHA),
        "n_obs": int(np.isfinite(series).sum()) if series.size else 0,
        "alpha": ALPHA,
    }


def fold_results(
    open_time: np.ndarray,
    close_a: np.ndarray,
    close_b: np.ndarray,
    beta_mode: str,
) -> List[Dict[str, object]]:
    rows = []
    for fold in walk_forward_folds():
        result = evaluate_window(
            open_time,
            close_a,
            close_b,
            int(fold["train_start_ms"]),
            int(fold["train_end_ms"]),
            beta_mode,
        )
        rows.append(
            {
                **result,
                "fold": int(fold["fold"]),
                "test_start_ms": int(fold["test_start_ms"]),
                "test_end_ms": int(fold["test_end_ms"]),
            }
        )
    return rows


def passing_windows(rows: Sequence[Dict[str, object]]) -> List[Window]:
    return [(int(row["test_start_ms"]), int(row["test_end_ms"])) for row in rows if row["passed"]]


def filter_trades(trades: pd.DataFrame, windows: Sequence[Window]) -> pd.DataFrame:
    """Keep trades whose decision falls in a fold that passed. Empty stays empty."""
    if trades.empty or not windows:
        return trades.iloc[0:0].copy()
    decision = trades["decision_time_ms"].to_numpy(dtype=np.int64)
    keep = np.zeros(len(trades), dtype=bool)
    for start, end in windows:
        keep |= (decision >= int(start)) & (decision < int(end))
    return trades.loc[keep].reset_index(drop=True)


def pass_fraction(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    return {
        "folds_passed": int(passed),
        "folds_total": int(total),
        "fraction": float(passed / total) if total else None,
        "label": rows[0]["label"] if rows else None,
        "p_values": [row["p_value"] for row in rows],
    }
