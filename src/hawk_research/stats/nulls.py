"""Null comparisons and feature screening for the zone population.

Zones are compared with three populations:

(a) all same-symbol bars (weak: zones may just sit on volatile bars);
(b) volume- and volatility-matched non-zone bars, same symbol, within ±24h, the
    comparison the conclusions rest on;
(c) the unconditional base rate over the whole panel.

Zones fire on up to 80 correlated symbols in the same bar, so intervals come
from block or cross-sectional bootstraps over time blocks, never from IID
standard errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from hawk_research.stats.multiple_testing import benjamini_hochberg
from hawk_research.stats.resampling import cross_sectional_bootstrap_ci, paired_block_bootstrap_diff_ci


def sample_matched_controls(
    bar_labels: pd.DataFrame,
    zone_open_times: Sequence[int],
    match_hours: float = 24.0,
    vol_tol: float = 0.15,
    rv_tol: float = 0.25,
    n_controls: int = 2,
    seed: int = 3,
) -> pd.DataFrame:
    """Non-zone bars matched to each zone on volume ratio and realized vol.

    Matching uses only values known at the control bar's own close; no outcome,
    of the zone or the control, is used to pick the match.
    """
    rng = np.random.default_rng(seed)
    ot = bar_labels["open_time"].to_numpy(dtype=np.int64)
    pos_of = {int(t): i for i, t in enumerate(ot.tolist())}
    is_zone = np.zeros(len(ot), dtype=bool)
    for t in zone_open_times:
        p = pos_of.get(int(t))
        if p is not None:
            is_zone[p] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        log_vr = np.log(bar_labels["det_volume_ratio"].to_numpy(dtype=float))
        log_rv = np.log(bar_labels["rv_24h"].to_numpy(dtype=float))
    finite = np.isfinite(log_vr) & np.isfinite(log_rv)

    half = int(match_hours * 4)
    picks: List[int] = []
    matched_to: List[int] = []
    for t in zone_open_times:
        i = pos_of.get(int(t))
        if i is None or not finite[i]:
            continue
        lo, hi = max(0, i - half), min(len(ot), i + half + 1)
        window = np.arange(lo, hi)
        cand = window[
            ~is_zone[window]
            & finite[window]
            & (np.abs(log_vr[window] - log_vr[i]) <= vol_tol)
            & (np.abs(log_rv[window] - log_rv[i]) <= rv_tol)
        ]
        if cand.size == 0:
            continue
        take = rng.choice(cand, size=min(n_controls, cand.size), replace=False)
        picks.extend(take.tolist())
        matched_to.extend([int(t)] * len(take))

    if not picks:
        return bar_labels.iloc[[]].copy()
    out = bar_labels.iloc[picks].copy()
    out["matched_zone_open_time"] = matched_to
    return out


def compare_means(
    zone_values: Sequence[float],
    zone_times: Sequence[int],
    control_values: Sequence[float],
    control_times: Sequence[int],
    block_hours: int = 24,
    n_samples: int = 1000,
    seed: int = 5,
) -> Dict[str, object]:
    """Block-bootstrap difference in means between zone and control bars."""
    z = np.asarray(zone_values, dtype=float)
    c = np.asarray(control_values, dtype=float)
    zt = np.asarray(zone_times, dtype=np.int64)
    ct = np.asarray(control_times, dtype=np.int64)
    zm, cm = np.isfinite(z), np.isfinite(c)
    if zm.sum() < 2 or cm.sum() < 2:
        return {"n_zone": int(zm.sum()), "n_control": int(cm.sum()), "diff": None}
    res = paired_block_bootstrap_diff_ci(
        z[zm],
        zt[zm],
        c[cm],
        ct[cm],
        block_hours=block_hours,
        n_samples=n_samples,
        seed=seed,
    )
    res.update(
        {
            "n_zone": int(zm.sum()),
            "n_control": int(cm.sum()),
            "zone_mean": float(np.mean(z[zm])),
            "control_mean": float(np.mean(c[cm])),
        }
    )
    return res


def proportion_test(successes: int, n: int, p0: float = 0.5) -> Dict[str, Any]:
    """Two-sided binomial test with a Wilson interval.

    Reported alongside the bootstrap, not instead of it: the binomial assumes
    independent trials, which simultaneous zones on correlated symbols are not.
    The Wilson interval is therefore an optimistic lower bound on the true
    uncertainty and is labelled as such.
    """
    if n <= 0:
        return {"n": 0, "rate": None, "p_value": None, "ci_low": None, "ci_high": None}
    phat = successes / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    halfwidth = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    p = float(sps.binomtest(int(successes), int(n), p0).pvalue)
    return {
        "n": int(n),
        "successes": int(successes),
        "rate": float(phat),
        "excess_over_p0": float(phat - p0),
        "p_value": p,
        "ci_low": float(centre - halfwidth),
        "ci_high": float(centre + halfwidth),
        "interval_note": "Wilson interval assumes IID trials; simultaneous zones are not IID.",
    }


def spearman_ic_series(
    values: pd.DataFrame,
    feature: str,
    target: str,
    time_col: str,
    block_ms: int,
) -> pd.Series:
    """Rank IC computed inside each time block, one observation per block.

    Pooling all rows into one Spearman would treat 80 simultaneous, correlated
    observations as 80 independent ones; the block series is the sample instead.
    """
    sub = values[[feature, target, time_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    block = (sub[time_col].to_numpy(dtype=np.int64) // block_ms).astype(np.int64)
    sub = sub.assign(_block=block)
    out = {}
    for b, grp in sub.groupby("_block", sort=True):
        if len(grp) < 20:
            continue
        f = grp[feature].to_numpy(dtype=float)
        t = grp[target].to_numpy(dtype=float)
        if np.all(f == f[0]) or np.all(t == t[0]):
            continue
        rho = sps.spearmanr(f, t).statistic
        if np.isfinite(rho):
            out[int(b)] = float(rho)
    return pd.Series(out, dtype=float)


def _newey_west_tstat(x: np.ndarray, lags: int = 4) -> Tuple[float, float]:
    """t-stat of the mean with a HAC variance, so serial correlation is priced in."""
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(np.dot(e[k:], e[:-k]) / n)
        var += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    var = max(var, 1e-18)
    se = np.sqrt(var / n)
    t = mu / se
    p = 2.0 * (1.0 - sps.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def rank_auc(scores: np.ndarray, binary: np.ndarray) -> Optional[float]:
    """AUC via the Mann-Whitney statistic. Ties handled by average ranks."""
    m = np.isfinite(scores) & np.isfinite(binary)
    s, y = scores[m], binary[m]
    pos = y > 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = sps.rankdata(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def screen_features(
    ds: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    binary_col: Optional[str],
    time_col: str = "decision_time_ms",
    block_hours: int = 24 * 7,
    alpha: float = 0.05,
    min_blocks: int = 8,
) -> pd.DataFrame:
    """Univariate association of every feature with a forward outcome.

    Reports the mean rank IC over time blocks with a Newey-West t-stat, the AUC
    against a barrier-direction label, and a Benjamini-Hochberg q-value across
    the whole feature set. Screening one feature at a time on 100+ features and
    then quoting the best raw p-value is how spurious "signals" are manufactured;
    the q-value is the number to read.
    """
    block_ms = int(block_hours * 3600 * 1000)
    rows: List[Dict[str, object]] = []
    for col in feature_cols:
        if col not in ds.columns:
            continue
        series = ds[col]
        if series.notna().sum() < 500 or series.nunique(dropna=True) < 5:
            continue
        ic = spearman_ic_series(ds, col, target_col, time_col, block_ms)
        if len(ic) < min_blocks:
            continue
        t, p = _newey_west_tstat(ic.to_numpy())
        auc = None
        if binary_col and binary_col in ds.columns:
            sub = ds[[col, binary_col]].dropna()
            sub = sub[sub[binary_col] != 0]
            if len(sub) > 200:
                auc = rank_auc(
                    sub[col].to_numpy(dtype=float),
                    (sub[binary_col].to_numpy() > 0).astype(float),
                )
        rows.append(
            {
                "feature": col,
                "n_obs": int(series.notna().sum()),
                "n_blocks": int(len(ic)),
                "mean_ic": float(ic.mean()),
                "std_ic": float(ic.std()),
                "ic_t_stat": t,
                "ic_p_value": p,
                "ic_hit_rate": float((np.sign(ic) == np.sign(ic.mean())).mean()),
                "auc": auc,
                "auc_edge": None if auc is None else abs(auc - 0.5),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    rejected, q = benjamini_hochberg(out["ic_p_value"].to_numpy(), alpha=alpha)
    out["ic_q_value"] = q
    out["survives_fdr"] = rejected
    return out.sort_values("ic_p_value", kind="mergesort").reset_index(drop=True)


def screen_is_oos(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    binary_col: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run the screen in-sample and out-of-sample and join them side by side.

    ``survives_both`` is the column to read. Features that fail out of sample are
    kept in the output, not dropped.
    """
    is_res = screen_features(development, feature_cols, target_col, binary_col, **kwargs)
    oos_res = screen_features(holdout, feature_cols, target_col, binary_col, **kwargs)
    if is_res.empty or oos_res.empty:
        return is_res
    merged = is_res.merge(oos_res, on="feature", suffixes=("_is", "_oos"), how="outer")
    merged["sign_agrees"] = np.sign(merged["mean_ic_is"]) == np.sign(merged["mean_ic_oos"])
    merged["survives_both"] = (
        merged["survives_fdr_is"].fillna(False)
        & merged["survives_fdr_oos"].fillna(False)
        & merged["sign_agrees"]
    )
    merged["ic_decay"] = merged["mean_ic_oos"] - merged["mean_ic_is"]
    return merged.sort_values("ic_p_value_is", kind="mergesort").reset_index(drop=True)


def directional_summary(
    ds: pd.DataFrame,
    horizon: str,
    side: np.ndarray,
    cost_round_trip: float,
    time_col: str = "decision_time_ms",
    block_hours: int = 24,
    n_samples: int = 1000,
    seed: int = 9,
) -> Dict[str, object]:
    """Mean signed forward return for a side rule, gross and net, with a CI."""
    ret = ds[f"fwd_ret_{horizon}"].to_numpy(dtype=float)
    signed = side * ret
    t = ds[time_col].to_numpy(dtype=np.int64)
    m = np.isfinite(signed)
    if m.sum() < 50:
        return {"n": int(m.sum())}
    gross = cross_sectional_bootstrap_ci(
        signed[m], t[m], block_hours=block_hours, n_samples=n_samples, seed=seed
    )
    net_vals = signed[m] - cost_round_trip
    net = cross_sectional_bootstrap_ci(
        net_vals, t[m], block_hours=block_hours, n_samples=n_samples, seed=seed
    )
    wins = int((signed[m] > 0).sum())
    return {
        "horizon": horizon,
        "n": int(m.sum()),
        "gross_mean": float(np.mean(signed[m])),
        "gross_ci": [gross.get("ci_low"), gross.get("ci_high")],
        "net_mean": float(np.mean(net_vals)),
        "net_ci": [net.get("ci_low"), net.get("ci_high")],
        "cost_round_trip": float(cost_round_trip),
        "win_rate": float(wins / m.sum()),
        "win_rate_test": proportion_test(wins, int(m.sum())),
        "std": float(np.std(signed[m], ddof=1)),
        "t_like": float(np.mean(signed[m]) / (np.std(signed[m], ddof=1) / np.sqrt(m.sum()))),
        "effective_n_blocks": gross.get("n_blocks"),
    }
