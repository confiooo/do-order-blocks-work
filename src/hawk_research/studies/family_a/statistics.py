"""Inference for Family A: uniqueness and n_eff, folds, intervals, Sharpe, CPCV, walk-forward."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS, DAYS_PER_YEAR
from hawk_research.evaluation import metrics as metrics_mod
from hawk_research.evaluation import portfolio as portfolio_mod
from hawk_research.studies.family_a.rules import MS_DAY, P0, P_STAR, QUEUE_BY, d_bounds


def uniqueness(trades: pd.DataFrame) -> Dict[str, float]:
    """AFML §4.4 average uniqueness on the 15m grid of live labels."""
    if trades.empty:
        return {"n_raw": 0, "sum_u": 0.0, "mean_u": None, "avg_concurrency": None}
    t0 = int(trades["entry_time_ms"].min())
    t1 = int(trades["exit_time_ms"].max())
    n_bins = int((t1 - t0) // BAR_MS) + 2
    delta = np.zeros(n_bins + 1, dtype=np.int32)
    e = ((trades["entry_time_ms"].to_numpy(dtype=np.int64) - t0) // BAR_MS).clip(0, n_bins)
    x = ((trades["exit_time_ms"].to_numpy(dtype=np.int64) - t0) // BAR_MS).clip(0, n_bins)
    np.add.at(delta, e, 1)
    np.add.at(delta, x, -1)
    c = np.cumsum(delta)[:n_bins]
    c = np.maximum(c, 1)
    u = np.zeros(len(trades), dtype=float)
    for i, (a, b) in enumerate(zip(e.tolist(), x.tolist())):
        if b <= a:
            u[i] = 1.0
        else:
            u[i] = float(np.mean(1.0 / c[a:b]))
    n_raw = float(len(trades))
    sum_u = float(u.sum())
    span_h = max(1.0, (t1 - t0) / (3600 * 1000))
    # concurrency ≈ n_raw * 16h / calendar hours, but realised from the curve
    avg_c = float(c.mean())
    return {
        "n_raw": int(n_raw),
        "sum_u": sum_u,
        "mean_u": float(u.mean()),
        "avg_concurrency": avg_c,
        "span_hours": span_h,
    }


def n_eff_from(u: Dict[str, float], rho_bar: float) -> float:
    conc = max(1.0, float(u.get("avg_concurrency") or 1.0))
    sum_u = float(u.get("sum_u") or 0.0)
    breadth = conc / (1.0 + (conc - 1.0) * rho_bar)
    return float(sum_u * breadth)


def measure_rho_bar(trades: pd.DataFrame) -> float:
    """Mean off-diagonal correlation of per-symbol daily signed residual (or net) returns."""
    if trades.empty or "symbol" not in trades.columns:
        return 0.50
    col = "ret_resid_net" if "ret_resid_net" in trades.columns else "ret_net"
    tmp = trades[["symbol", "entry_time_ms", col]].dropna()
    if tmp.empty:
        return 0.50
    tmp["day"] = tmp["entry_time_ms"] // MS_DAY
    panel = tmp.pivot_table(index="day", columns="symbol", values=col, aggfunc="mean")
    if panel.shape[1] < 3 or panel.shape[0] < 10:
        return 0.50
    corr = panel.corr()
    v = corr.to_numpy(dtype=float)
    mask = ~np.eye(v.shape[0], dtype=bool)
    vals = v[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.50
    return float(np.mean(vals))


def rolling_folds() -> List[Dict[str, int]]:
    d0, d1, _ = d_bounds()
    day = MS_DAY
    folds = []
    for i in range(6):
        train_start = d0 + i * 60 * day
        train_end = train_start + 180 * day
        gap_end = train_end + 5 * day
        test_end = gap_end + 60 * day
        folds.append(
            {
                "fold": i + 1,
                "train_start_ms": int(train_start),
                "train_end_ms": int(train_end),
                "test_start_ms": int(gap_end),
                "test_end_ms": int(min(test_end, d1)),
                "scheme": "rolling",
            }
        )
    return folds


def anchored_folds() -> List[Dict[str, int]]:
    d0, d1, _ = d_bounds()
    out = []
    for f in rolling_folds():
        out.append({**f, "train_start_ms": int(d0), "scheme": "anchored"})
    return out


def assert_no_leak(train: pd.DataFrame, test: pd.DataFrame, purge_h: int = 16, embargo_h: int = 40) -> bool:
    if train.empty or test.empty:
        return True
    t0 = train.copy()
    if "t0" not in t0.columns:
        t0["t0"] = t0["decision_time_ms"]
        t0["t1"] = t0["exit_time_ms"]
    te = test.copy()
    if "t0" not in te.columns:
        te["t0"] = te["decision_time_ms"]
        te["t1"] = te["exit_time_ms"]
    lo, hi = int(te["t0"].min()), int(te["t1"].max())
    before = t0[t0["t1"] < lo]
    after = t0[t0["t0"] > hi]
    if len(before) + len(after) != len(t0):
        return False
    if len(before) and (lo - int(before["t1"].max())) < purge_h * 3600 * 1000:
        return False
    if len(after) and (int(after["t0"].min()) - hi) < embargo_h * 3600 * 1000:
        return False
    return True


def fold_split(trades: pd.DataFrame, fold: Dict[str, int], purge_h: int = 16) -> Dict[str, pd.DataFrame]:
    dec = trades["decision_time_ms"].to_numpy(dtype=np.int64)
    ex = trades["exit_time_ms"].to_numpy(dtype=np.int64)
    in_train = (dec >= fold["train_start_ms"]) & (dec < fold["train_end_ms"])
    in_test = (dec >= fold["test_start_ms"]) & (dec < fold["test_end_ms"])
    overlaps = ex >= fold["test_start_ms"]
    train = trades.loc[in_train & ~overlaps].copy()
    test = trades.loc[in_test].copy()
    purged = trades.loc[in_train & overlaps].copy()
    return {
        "train": train,
        "test": test,
        "purged": purged,
        "leak_ok": assert_no_leak(train, test, purge_h, 40),
    }


def studentised_mean_ci(
    values: np.ndarray,
    times_ms: np.ndarray,
    block_days: int = 10,
    n_samples: int = 10_000,
    seed: int = 7,
) -> Dict[str, Any]:
    """Stationary-bootstrap percentile-t CI for a mean, on a daily (or any) series."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(times_ms, dtype=np.int64)
    m = np.isfinite(v)
    v, t = v[m], t[m]
    if v.size < 5:
        return {"n": int(v.size), "stat": None, "ci_low": None, "ci_high": None, "method": "percentile-t"}
    order = np.argsort(t, kind="stable")
    v, t = v[order], t[order]
    xbar = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(v.size))
    span = max(1, int(t[-1] - t[0]))
    mean_block = max(2.0, v.size * (block_days * MS_DAY) / span)
    p = min(1.0, 1.0 / float(mean_block))
    rng = np.random.default_rng(seed)
    n = v.size
    tstats = np.empty(n_samples, dtype=float)
    for s in range(n_samples):
        pieces: List[np.ndarray] = []
        total = 0
        while total < n:
            length = int(rng.geometric(p))
            length = min(length, n - total)
            start = int(rng.integers(0, n))
            pieces.append((np.arange(start, start + length, dtype=np.int64)) % n)
            total += length
        xs = v[np.concatenate(pieces)[:n]]
        mstar = float(xs.mean())
        sstar = float(xs.std(ddof=1) / math.sqrt(n))
        tstats[s] = (mstar - xbar) / sstar if sstar > 0 else 0.0
    q_lo, q_hi = np.quantile(tstats, [0.025, 0.975])
    return {
        "n": int(v.size),
        "stat": xbar,
        "ci_low": xbar - float(q_hi) * se,
        "ci_high": xbar - float(q_lo) * se,
        "se": se,
        "block_days": block_days,
        "n_samples": n_samples,
        "method": "stationary bootstrap percentile-t (Politis-Romano / Goncalves time-slice)",
    }


def daily_returns(equity: pd.DataFrame) -> pd.Series:
    daily = metrics_mod.resample_equity(equity, MS_DAY)
    if daily is None or len(daily) < 3:
        return pd.Series(dtype=float)
    r = daily["equity"].pct_change()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def portfolio_bundle(
    trades: pd.DataFrame,
    max_concurrent: int = 5,
    unconstrained: bool = False,
) -> Dict[str, Any]:
    k = 10**6 if unconstrained else int(max_concurrent)
    rf = 0.2 if not unconstrained else 1.0 / k
    sim = portfolio_mod.build_equity_curve(
        trades,
        ret_col="ret_net",
        max_concurrent=k,
        capital=1.0,
        risk_fraction=rf if not unconstrained else 0.2,
        one_position_per_symbol=True,
        queue_by=QUEUE_BY,
    )
    eq = sim["equity"]
    daily = daily_returns(eq)
    sharpe = metrics_mod.sharpe(daily.to_numpy(), DAYS_PER_YEAR) if len(daily) else None
    eqm = metrics_mod.equity_metrics(eq)
    eqd = metrics_mod.equity_metrics(metrics_mod.resample_equity(eq, MS_DAY), periods_per_year=DAYS_PER_YEAR)
    return {
        "sim": sim,
        "daily": daily,
        "sharpe_ann": sharpe,
        "equity_metrics": eqm,
        "equity_metrics_daily": eqd,
        "rejection_rate": sim.get("rejection_rate"),
        "capacity_utilization": sim.get("capacity_utilization"),
        "accepted": sim.get("accepted"),
        "rejected": sim.get("rejected"),
    }


def widest_ci(values, times, blocks=(4, 10, 20, 40), n_samples=10_000, seed=7) -> Dict[str, Any]:
    primary = studentised_mean_ci(values, times, 10, n_samples, seed)
    all_ci = {str(b): studentised_mean_ci(values, times, b, n_samples, seed + b) for b in blocks}
    width = []
    for b, r in all_ci.items():
        if r["ci_low"] is None:
            continue
        width.append((float(r["ci_high"]) - float(r["ci_low"]), r, b))
    headline = max(width, key=lambda x: x[0])[1] if width else primary
    return {"primary_b10": primary, "by_block": all_ci, "widest": headline}


def hit_rate_bundle(filled: pd.DataFrame, n_eff: float) -> Dict[str, Any]:
    if filled.empty or "exit_reason" not in filled.columns:
        return {"n": 0}
    n = len(filled)
    n_tgt = int((filled["exit_reason"] == "target").sum())
    n_stp = int((filled["exit_reason"] == "stop").sum())
    n_to = int((filled["exit_reason"] == "timeout").sum())
    p = n_tgt / n if n else None
    amb = float(filled["ambiguous"].mean()) if "ambiguous" in filled.columns else None
    # Wilson on n_eff, not n_raw
    ne = max(1.0, float(n_eff))
    if p is None:
        wilson = None
    else:
        z = 1.959963984540054
        denom = 1 + z * z / ne
        centre = (p + z * z / (2 * ne)) / denom
        half = z * math.sqrt(p * (1 - p) / ne + z * z / (4 * ne * ne)) / denom
        wilson = {"low": centre - half, "high": centre + half, "n_eff": ne}
    return {
        "n": n,
        "n_target": n_tgt,
        "n_stop": n_stp,
        "n_timeout": n_to,
        "hit_rate": p,
        "p0": P0,
        "p_star": P_STAR,
        "wilson_n_eff": wilson,
        "ambiguous_frac": amb,
    }


def lo_sharpe_se(daily: np.ndarray) -> Optional[float]:
    x = np.asarray(daily, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T < 10:
        return None
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    if sd <= 0:
        return None
    q = int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    # Newey-West of the mean, then Lo (2002) iid-adj as fallback scale
    xc = x - mu
    gamma0 = float(np.dot(xc, xc) / T)
    acc = gamma0
    for k in range(1, q + 1):
        w = 1.0 - k / (q + 1.0)
        acc += 2.0 * w * float(np.dot(xc[k:], xc[:-k]) / T)
    se_mu = math.sqrt(max(acc, 0.0) / T)
    # d(SR)/d(mu) ~ 1/sd ; ignore sd uncertainty (Lo's first-order)
    return float(se_mu / sd * math.sqrt(DAYS_PER_YEAR))


def cpcv_paths(trades: pd.DataFrame, n_blocks: int = 12) -> Dict[str, Any]:
    """Frozen-rule CPCV: 12 blocks, k=2, 66 splits, 11 reconstructed OOS paths.

    Because S1/S5 have no fitted parameters, the return at time t does not depend
    on which other block is held out with it. The 11 paths still differ in which
    *pair* of blocks they stitch; they are not a model-selection distribution.
    """
    if trades.empty:
        return {"n_splits": 0, "n_paths": 0, "paths_positive": None}
    d0, d1, _ = d_bounds()
    span = d1 - d0
    edges = [d0 + int(round(span * i / n_blocks)) for i in range(n_blocks + 1)]
    dec = trades["decision_time_ms"].to_numpy(dtype=np.int64)
    block_id = np.searchsorted(edges, dec, side="right") - 1
    block_id = np.clip(block_id, 0, n_blocks - 1)

    split_stats = []
    from itertools import combinations

    for a, b in combinations(range(n_blocks), 2):
        mask = (block_id == a) | (block_id == b)
        te = trades.loc[mask]
        mu = float(te["ret_net"].mean()) if len(te) else None
        split_stats.append(
            {"blocks": [a, b], "n": int(len(te)), "mean_net": mu, "positive": mu is not None and mu > 0}
        )

    # 11 paths: for offset o=1..11, stitch pairs (i, i+o mod 12) for even i covering
    paths = []
    for o in range(1, n_blocks):
        used = np.zeros(len(trades), dtype=bool)
        rets = []
        for i in range(n_blocks):
            j = (i + o) % n_blocks
            if i >= j:
                continue
            mask = ((block_id == i) | (block_id == j)) & ~used
            used |= mask
            if mask.any():
                rets.append(trades.loc[mask, "ret_net"].to_numpy(dtype=float))
        if not rets:
            continue
        arr = np.concatenate(rets)
        paths.append(
            {
                "offset": o,
                "n": int(arr.size),
                "mean_net": float(np.nanmean(arr)),
                "positive": bool(np.nanmean(arr) > 0),
            }
        )

    n_pos = int(sum(1 for p in paths if p["positive"]))
    return {
        "n_blocks": n_blocks,
        "n_splits": len(split_stats),
        "splits_positive": int(sum(1 for s in split_stats if s["positive"])),
        "n_paths": len(paths),
        "paths_positive": n_pos,
        "path_means": [p["mean_net"] for p in paths],
        "splits": split_stats,
        "note": "Frozen rule: CPCV does not refit. Paths differ only by which block pairs are scored together.",
    }


def walk_forward_report(trades: pd.DataFrame) -> Dict[str, Any]:
    folds = []
    leak_ok = True
    stitched = []
    for spec in rolling_folds():
        sp = fold_split(trades, spec)
        leak_ok = leak_ok and sp["leak_ok"]
        te = sp["test"]
        mu = float(te["ret_net"].mean()) if len(te) else None
        pnl = float(te["ret_net"].sum()) if len(te) else 0.0
        folds.append(
            {
                **{k: spec[k] for k in spec},
                "n_train": int(len(sp["train"])),
                "n_test": int(len(te)),
                "n_purged": int(len(sp["purged"])),
                "leak_ok": sp["leak_ok"],
                "test_mean_net": mu,
                "test_pnl_sum": pnl,
                "positive": mu is not None and mu > 0,
            }
        )
        if len(te):
            stitched.append(te)
    stitched_df = pd.concat(stitched, ignore_index=True) if stitched else trades.iloc[0:0]
    pos = int(sum(1 for f in folds if f["positive"]))
    pnls = np.array([f["test_pnl_sum"] for f in folds], dtype=float)
    tot = pnls.sum()
    max_share = float(np.max(np.abs(pnls)) / abs(tot)) if tot != 0 else None
    anchored = []
    for spec in anchored_folds():
        sp = fold_split(trades, spec)
        te = sp["test"]
        mu = float(te["ret_net"].mean()) if len(te) else None
        anchored.append({"fold": spec["fold"], "test_mean_net": mu, "positive": mu is not None and mu > 0})
    roll_sign = np.sign([f["test_mean_net"] or 0 for f in folds])
    anc_sign = np.sign([f["test_mean_net"] or 0 for f in anchored])
    return {
        "folds": folds,
        "n_positive": pos,
        "stitched_n": int(len(stitched_df)),
        "stitched_mean_net": float(stitched_df["ret_net"].mean()) if len(stitched_df) else None,
        "max_fold_pnl_share": max_share,
        "leak_ok_all": leak_ok,
        "anchored": anchored,
        "rolling_anchored_sign_agree": bool(np.all(roll_sign == anc_sign))
        if len(folds) == len(anchored)
        else False,
        "stitched": stitched_df,
    }


def symbol_month_concentration(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {"max_symbol_share": None, "max_month_share": None}
    pnl = trades["ret_net"]
    tot = float(pnl.sum())
    if tot == 0:
        return {"max_symbol_share": None, "max_month_share": None}
    by_s = trades.groupby("symbol")["ret_net"].sum() / tot
    month = pd.to_datetime(trades["decision_time_ms"], unit="ms", utc=True).dt.to_period("M")
    by_m = trades.groupby(month)["ret_net"].sum() / tot
    return {
        "max_symbol_share": float(by_s.abs().max()),
        "max_month_share": float(by_m.abs().max()),
        "top_symbol": str(by_s.abs().idxmax()),
        "top_month": str(by_m.abs().idxmax()),
    }


def kmax_sensitivity(trades: pd.DataFrame, ks=(2, 5, 10, 20)) -> Dict[str, Any]:
    out = {}
    signs = []
    for k in ks:
        bun = portfolio_bundle(trades, max_concurrent=k)
        mu = float(bun["daily"].mean()) if len(bun["daily"]) else None
        out[str(k)] = {
            "daily_mean": mu,
            "sharpe_ann": bun["sharpe_ann"],
            "rejection_rate": bun["rejection_rate"],
            "accepted": bun["accepted"],
        }
        signs.append(np.sign(mu) if mu is not None else 0)
    out["sign_stable"] = bool(len(set(int(s) for s in signs if s != 0)) <= 1)
    return out
