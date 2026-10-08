"""Multiple-testing and overfitting controls.

- ``benjamini_hochberg``: false discovery rate across many hypotheses.
- ``deflated_sharpe_ratio``: the observed Sharpe against the expected maximum
  under the null, given the number of trials and their dispersion (Bailey and
  Lopez de Prado, 2014).
- ``probabilistic_sharpe_ratio``: adjusts for skew and fat tails.
- ``pbo_cscv``: probability that the in-sample best configuration is below the
  median out of sample.

The trial counts come from ``hawk_research.ledger``.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Returns ``(rejected, qvalues)`` in the order of the input. q-values are the
    step-up adjusted p-values with the monotonicity correction applied, so they can
    be compared against any alpha after the fact.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=float)
    raw_q = ranked * n / ranks
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    qvalues = np.empty(n, dtype=float)
    qvalues[order] = q_sorted
    rejected = np.zeros(n, dtype=bool)
    below = np.nonzero(ranked <= alpha * ranks / n)[0]
    if below.size:
        cutoff = below.max()
        rejected[order[: cutoff + 1]] = True
    return rejected, qvalues


def _sr_variance_term(observed_sr: float, skew: float, kurtosis: float) -> float:
    """Variance multiplier of the Sharpe estimator under non-normal returns.

    ``kurtosis`` is the raw fourth moment ratio (3.0 for a normal), not excess.
    """
    return 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2


def sharpe_standard_error(
    observed_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Optional[float]:
    """Standard error of a Sharpe estimate with skew and kurtosis adjustment."""
    if n_obs is None or n_obs < 2:
        return None
    term = _sr_variance_term(observed_sr, skew, kurtosis)
    if term <= 0.0:
        return None
    return float(math.sqrt(term / (n_obs - 1)))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Optional[float]:
    """Probability that the true Sharpe exceeds ``benchmark_sr``.

    ``observed_sr`` and ``benchmark_sr`` must be in the same frequency units as
    ``n_obs`` (per-observation, not annualized), otherwise the skew/kurtosis terms
    are scaled inconsistently and the probability is meaningless.
    """
    se = sharpe_standard_error(observed_sr, n_obs, skew, kurtosis)
    if se is None or se <= 0.0:
        return None
    return float(norm.cdf((observed_sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

    Bailey & Lopez de Prado's extreme-value approximation. With one trial there is
    no selection bias, so the benchmark is zero.
    """
    if n_trials is None or n_trials <= 1 or sr_variance <= 0.0:
        return 0.0
    sd = math.sqrt(sr_variance)
    n = float(n_trials)
    return float(
        sd
        * (
            (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n)
            + EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n * math.e))
        )
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    sr_variance: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Dict[str, Optional[float]]:
    """Deflated Sharpe: PSR against the selection-adjusted benchmark.

    ``sr_variance`` is the variance of the Sharpe ratios observed across the trials
    of the scan (it measures how much of the spread is selection noise).
    ``n_trials`` should come from ``hawk_research.ledger.TrialLedger.n_trials``.

    A ``dsr`` below 0.95 means the result is not distinguishable from the best of
    ``n_trials`` coin flips.
    """
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return {
        "observed_sr": float(observed_sr),
        "n_trials": int(n_trials) if n_trials is not None else None,
        "expected_max_sr": benchmark,
        "sr_std_error": sharpe_standard_error(observed_sr, n_obs, skew, kurtosis),
        "dsr": probabilistic_sharpe_ratio(observed_sr, benchmark, n_obs, skew, kurtosis),
    }


def pbo_cscv(performance_matrix: np.ndarray, n_splits: int = 16) -> Dict[str, Any]:
    """Probability of backtest overfitting via combinatorially symmetric CV.

    ``performance_matrix`` is ``(n_observations x n_configurations)`` of per-period
    performance (e.g. daily net return of each configuration). The observations are
    cut into ``n_splits`` contiguous chunks; every balanced split of those chunks into
    in-sample and out-of-sample halves is evaluated, the in-sample-best configuration
    is selected, and its out-of-sample relative rank is recorded. PBO is the fraction
    of splits in which that choice lands below the out-of-sample median.

    ~0.5 means selection carries no information (pure overfitting); values near 0
    mean the in-sample winner keeps winning. Chunks stay contiguous so serial
    structure inside a chunk is preserved.
    """
    matrix = np.asarray(performance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 2:
        return {"pbo": None, "n_combinations": 0, "logit_distribution": [], "median_logit": None}

    n_obs, n_cfg = matrix.shape
    splits = int(min(n_splits, n_obs))
    if splits % 2 == 1:
        splits -= 1
    if splits < 2:
        return {"pbo": None, "n_combinations": 0, "logit_distribution": [], "median_logit": None}

    bounds = np.linspace(0, n_obs, splits + 1).astype(int)
    sums = np.empty((splits, n_cfg), dtype=float)
    counts = np.empty(splits, dtype=float)
    for s in range(splits):
        chunk = matrix[bounds[s] : bounds[s + 1]]
        sums[s] = np.nansum(chunk, axis=0)
        counts[s] = max(1, chunk.shape[0])

    combos = list(combinations(range(splits), splits // 2))
    masks = np.zeros((len(combos), splits), dtype=float)
    for i, combo in enumerate(combos):
        masks[i, list(combo)] = 1.0

    is_mean = (masks @ sums) / (masks @ counts)[:, None]
    oos_mean = ((1.0 - masks) @ sums) / ((1.0 - masks) @ counts)[:, None]

    best = np.argmax(is_mean, axis=1)
    chosen_oos = oos_mean[np.arange(oos_mean.shape[0]), best]
    ranks = (oos_mean <= chosen_oos[:, None]).sum(axis=1).astype(float)
    omega = ranks / (n_cfg + 1.0)
    logits = np.log(omega / (1.0 - omega))
    return {
        "pbo": float((logits < 0.0).mean()),
        "n_combinations": int(len(combos)),
        "n_splits": int(splits),
        "n_configurations": int(n_cfg),
        "logit_distribution": [float(x) for x in logits],
        "median_logit": float(np.median(logits)),
    }
