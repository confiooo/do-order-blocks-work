"""Causal spread, beta, and z-score. A value at bar t uses bars at or before t."""

from __future__ import annotations

import numpy as np

from experiments.market_neutral.data.bars import HOUR_MS


def window_is_contiguous(times_ms: np.ndarray, window: int) -> np.ndarray:
    """True at t when the trailing ``window`` bars are exactly one hour apart."""
    times = np.asarray(times_ms, dtype=np.int64)
    n = times.size
    ok = np.zeros(n, dtype=bool)
    if window < 2 or n < window:
        if window == 1:
            ok[:] = True
        return ok
    diffs = np.diff(times)
    bad = np.zeros(n, dtype=bool)
    bad[1:] = diffs != HOUR_MS
    # A gap at index j (bad[j]) contaminates every window that includes that step,
    # i.e. endpoints j .. j+window-2.
    c = np.cumsum(bad.astype(np.int32))
    # sum of bad[i-window+2 : i+1] == c[i] - c[i-window+1]
    end = np.arange(window - 1, n)
    start = end - (window - 1)
    gaps = c[end] - c[start]
    ok[window - 1 :] = gaps == 0
    return ok


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum. A window containing a non-finite value is NaN; later windows are not."""
    v = np.asarray(values, dtype=float)
    n = v.size
    out = np.full(n, np.nan, dtype=float)
    if window < 1 or n < window:
        return out
    finite = np.isfinite(v)
    cumulative = np.cumsum(np.where(finite, v, 0.0))
    counted = np.cumsum(finite.astype(np.int32))
    idx = np.arange(window - 1, n)
    total = cumulative[idx].astype(float)
    count = counted[idx].astype(int)
    prev = idx - window
    has_prev = prev >= 0
    total[has_prev] -= cumulative[prev[has_prev]]
    count[has_prev] -= counted[prev[has_prev]]
    good = count == window
    out[idx[good]] = total[good]
    return out


def rolling_ols_beta(log_y: np.ndarray, log_x: np.ndarray, window: int) -> np.ndarray:
    """Trailing OLS slope of ``log_y`` on ``log_x`` with an intercept. No future bars."""
    y = np.asarray(log_y, dtype=float)
    x = np.asarray(log_x, dtype=float)
    n = float(window)
    sx = _rolling_sum(x, window)
    sy = _rolling_sum(y, window)
    sxx = _rolling_sum(x * x, window)
    sxy = _rolling_sum(x * y, window)
    var = n * sxx - sx * sx
    cov = n * sxy - sx * sy
    beta = np.full(y.shape, np.nan, dtype=float)
    usable = np.isfinite(var) & (var > 1e-18) & np.isfinite(cov)
    beta[usable] = cov[usable] / var[usable]
    return beta


def spread(log_a: np.ndarray, log_b: np.ndarray, beta: np.ndarray | float) -> np.ndarray:
    return np.asarray(log_a, dtype=float) - np.asarray(beta, dtype=float) * np.asarray(log_b, dtype=float)


def rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing z-score. Mean and sample std use ``window`` points ending at t, inclusive."""
    v = np.asarray(values, dtype=float)
    if window < 2:
        raise ValueError("z-score window must be at least 2")
    total = _rolling_sum(v, window)
    total_sq = _rolling_sum(v * v, window)
    mean = total / float(window)
    var = (total_sq - (total * total) / float(window)) / float(window - 1)
    std = np.sqrt(np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan))
    z = np.full(v.shape, np.nan, dtype=float)
    usable = np.isfinite(std) & (std > 1e-12) & np.isfinite(v) & np.isfinite(mean)
    z[usable] = (v[usable] - mean[usable]) / std[usable]
    return z


def pair_zscore(
    times_ms: np.ndarray,
    close_a: np.ndarray,
    close_b: np.ndarray,
    window: int,
    beta_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(z, beta)`` aligned to ``times_ms``.

    ``beta_mode`` is ``fixed`` (beta = 1) or ``rolling_ols``. Invalid prices,
    non-contiguous hours, and an unusable beta are NaN so the simulator cannot
    trade them.
    """
    times = np.asarray(times_ms, dtype=np.int64)
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)
    if not (times.shape == a.shape == b.shape):
        raise ValueError("times, close_a, and close_b must share a shape")
    if beta_mode not in ("fixed", "rolling_ols"):
        raise ValueError(f"unknown beta_mode {beta_mode!r}")
    positive = (a > 0.0) & (b > 0.0) & np.isfinite(a) & np.isfinite(b)
    log_a = np.full(a.shape, np.nan)
    log_b = np.full(b.shape, np.nan)
    log_a[positive] = np.log(a[positive])
    log_b[positive] = np.log(b[positive])
    if beta_mode == "fixed":
        beta = np.ones(a.shape, dtype=float)
        beta[~positive] = np.nan
        raw = spread(log_a, log_b, beta)
        z = rolling_zscore(raw, window)
        z[~window_is_contiguous(times, window)] = np.nan
        return z, beta
    beta = rolling_ols_beta(log_a, log_b, window)
    beta[~window_is_contiguous(times, window)] = np.nan
    raw = spread(log_a, log_b, beta)
    z = rolling_zscore(raw, window)
    # The z window of spreads each used a beta window, so the causal span is longer.
    z[~window_is_contiguous(times, window * 2 - 1)] = np.nan
    z[~np.isfinite(beta)] = np.nan
    return z, beta
