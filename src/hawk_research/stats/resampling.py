"""Bootstrap confidence intervals that respect the dependence in the trade table.

Two distinct dependencies break IID standard errors here:

1. Serial dependence. Zones cluster in time; volatility regimes persist for days.
   The moving-block bootstrap preserves short-range serial structure by resampling
   contiguous time blocks instead of single trades.
2. Cross-sectional dependence. The same pattern fires on dozens of correlated
   symbols in the same bar. ``cross_sectional_bootstrap_ci`` resamples whole time
   blocks with every symbol inside them, so simultaneous trades stay together and
   the interval reflects the much smaller number of independent observations.

Every function takes an arbitrary statistic and is deterministic given ``seed``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from hawk_research.constants import MS_PER_HOUR

Statistic = Callable[[np.ndarray], float]


def _prepare(values, times_ms) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    t = np.asarray(times_ms, dtype=np.int64)
    if v.shape[0] != t.shape[0]:
        raise ValueError("values and times_ms must have the same length")
    keep = np.isfinite(v)
    v, t = v[keep], t[keep]
    order = np.argsort(t, kind="stable")
    return v[order], t[order]


def _quantile_ci(samples: np.ndarray, alpha: float) -> tuple[Optional[float], Optional[float]]:
    if samples.size == 0:
        return None, None
    return (
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    )


def _fixed_time_blocks(times_ms: np.ndarray, block_ms: int) -> List[np.ndarray]:
    """Non-overlapping calendar blocks. The block, not the trade, is the sampling unit."""
    if times_ms.size == 0:
        return []
    t0 = int(times_ms[0])
    block_id = (times_ms - t0) // max(1, block_ms)
    blocks: List[np.ndarray] = []
    _, starts = np.unique(block_id, return_index=True)
    starts = np.sort(starts)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < starts.size else times_ms.size
        blocks.append(np.arange(s, e, dtype=np.int64))
    return blocks


def _result(
    values: np.ndarray,
    samples: np.ndarray,
    stat: Statistic,
    alpha: float,
    n_samples: int,
    method: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    low, high = _quantile_ci(samples, alpha)
    out: Dict[str, Any] = {
        "n": int(values.size),
        "stat": float(stat(values)) if values.size else None,
        "ci_low": low,
        "ci_high": high,
        "se": float(samples.std(ddof=1)) if samples.size > 1 else None,
        "alpha": alpha,
        "n_samples": int(n_samples),
        "method": method,
    }
    if extra:
        out.update(extra)
    return out


def _degenerate(
    alpha: float, n_samples: int, method: str, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": 0,
        "stat": None,
        "ci_low": None,
        "ci_high": None,
        "se": None,
        "alpha": alpha,
        "n_samples": int(n_samples),
        "method": method,
    }
    if extra:
        out.update(extra)
    return out


def block_bootstrap_ci(
    values,
    times_ms,
    block_hours: int = 24,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    stat: Statistic = np.mean,
) -> Dict[str, Any]:
    """Moving-block bootstrap CI for an arbitrary statistic.

    Blocks are time-defined, not count-defined: a block starting at observation ``i``
    contains every observation within ``block_hours`` of it. Any observation may start
    a block (hence 'moving'), which avoids the arbitrary phase of fixed calendar edges.
    """
    v, t = _prepare(values, times_ms)
    method = "moving-block bootstrap (time-defined blocks; serial dependence preserved)"
    if v.size == 0:
        return _degenerate(alpha, n_samples, method, {"block_hours": block_hours})
    block_ms = max(1, int(block_hours) * MS_PER_HOUR)
    ends = np.searchsorted(t, t + block_ms, side="left")
    ends = np.maximum(ends, np.arange(t.size) + 1)
    rng = np.random.default_rng(seed)
    n = v.size
    samples = np.empty(max(1, int(n_samples)), dtype=float)
    for s in range(samples.size):
        chunks: List[np.ndarray] = []
        total = 0
        while total < n:
            i = int(rng.integers(0, n))
            chunk = v[i : ends[i]]
            chunks.append(chunk)
            total += chunk.size
        resampled = np.concatenate(chunks)[:n]
        samples[s] = float(stat(resampled))
    return _result(
        v,
        samples,
        stat,
        alpha,
        samples.size,
        method,
        {"block_hours": int(block_hours), "mean_block_size": float(np.mean(ends - np.arange(n)))},
    )


def cross_sectional_bootstrap_ci(
    values,
    times_ms,
    block_hours: int = 24,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    stat: Statistic = np.mean,
) -> Dict[str, Any]:
    """CI that treats a whole time block (all symbols inside it) as one observation.

    This is the interval every stage quotes: the same pattern fires across ~80
    correlated symbols at once, so the number of independent draws is the number of
    blocks, not the number of trades. The resampled series has a variable length
    because blocks hold different numbers of simultaneous trades.
    """
    v, t = _prepare(values, times_ms)
    method = "cross-sectional block bootstrap (whole time blocks; contemporaneous cluster kept intact)"
    if v.size == 0:
        return _degenerate(alpha, n_samples, method, {"block_hours": block_hours, "n_blocks": 0})
    block_ms = max(1, int(block_hours) * MS_PER_HOUR)
    blocks = _fixed_time_blocks(t, block_ms)
    if not blocks:
        blocks = [np.arange(v.size, dtype=np.int64)]
    block_values = [v[b] for b in blocks]
    rng = np.random.default_rng(seed)
    n_blocks = len(block_values)
    samples: List[float] = []
    for _ in range(max(1, int(n_samples))):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        resampled = np.concatenate([block_values[int(p)] for p in picks])
        if resampled.size:
            samples.append(float(stat(resampled)))
    arr = np.asarray(samples, dtype=float)
    return _result(
        v,
        arr,
        stat,
        alpha,
        arr.size,
        method,
        {
            "block_hours": int(block_hours),
            "n_blocks": n_blocks,
            "mean_trades_per_block": float(v.size / n_blocks),
        },
    )


def paired_block_bootstrap_diff_ci(
    a_values,
    a_times,
    b_values,
    b_times,
    block_hours: int = 24,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    stat: Statistic = np.mean,
) -> Dict[str, Any]:
    """CI for ``stat(a) - stat(b)`` with both populations resampled on the same blocks.

    Used for zone vs matched-baseline comparisons. Both series are cut on one shared
    calendar grid and a drawn block contributes its A rows and its B rows together,
    so market-wide moves that lift or sink both populations cancel in the difference
    instead of inflating its variance.
    """
    av, at = _prepare(a_values, a_times)
    bv, bt = _prepare(b_values, b_times)
    method = "paired time-block bootstrap of the difference (shared blocks; common market moves cancel)"
    if av.size == 0 or bv.size == 0:
        return _degenerate(
            alpha, n_samples, method, {"block_hours": block_hours, "n_a": int(av.size), "n_b": int(bv.size)}
        )
    block_ms = max(1, int(block_hours) * MS_PER_HOUR)
    t0 = int(min(at[0], bt[0]))
    a_block = (at - t0) // block_ms
    b_block = (bt - t0) // block_ms
    shared = np.intersect1d(np.unique(a_block), np.unique(b_block))
    if shared.size == 0:
        return _degenerate(
            alpha,
            n_samples,
            method,
            {"block_hours": block_hours, "n_a": int(av.size), "n_b": int(bv.size), "n_blocks": 0},
        )
    a_by_block = [av[a_block == k] for k in shared]
    b_by_block = [bv[b_block == k] for k in shared]
    rng = np.random.default_rng(seed)
    n_blocks = shared.size
    samples: List[float] = []
    for _ in range(max(1, int(n_samples))):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        a_res = np.concatenate([a_by_block[int(p)] for p in picks])
        b_res = np.concatenate([b_by_block[int(p)] for p in picks])
        if a_res.size and b_res.size:
            samples.append(float(stat(a_res)) - float(stat(b_res)))
    arr = np.asarray(samples, dtype=float)
    low, high = _quantile_ci(arr, alpha)
    observed = float(stat(av)) - float(stat(bv))
    return {
        "n_a": int(av.size),
        "n_b": int(bv.size),
        "stat_a": float(stat(av)),
        "stat_b": float(stat(bv)),
        "diff": observed,
        "stat": observed,
        "ci_low": low,
        "ci_high": high,
        "se": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "alpha": alpha,
        "n_samples": int(arr.size),
        "n_blocks": int(n_blocks),
        "block_hours": int(block_hours),
        "method": method,
    }


def ci_excludes_zero(result: Dict[str, Any]) -> Optional[bool]:
    """True when the interval is entirely on one side of zero. ``None`` if undefined."""
    low, high = result.get("ci_low"), result.get("ci_high")
    if low is None or high is None:
        return None
    return bool(low > 0.0 or high < 0.0)


def effective_sample_size(times_ms: Sequence[int], block_hours: int = 24) -> int:
    """Number of distinct time blocks. The sample size that matters for significance."""
    t = np.sort(np.asarray(times_ms, dtype=np.int64))
    if t.size == 0:
        return 0
    return len(_fixed_time_blocks(t, max(1, int(block_hours) * MS_PER_HOUR)))
