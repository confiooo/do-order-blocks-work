"""Family-A inference and the pre-registered decision gates for S1 and S5.

Turns simulated trades into the confidence intervals, gate outcomes and verdicts that the
pre-registration defines. Orchestration (loading, simulation, reporting) lives in ``run``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from hawk_research.stats.multiple_testing import deflated_sharpe_ratio
from hawk_research.stats.resampling import paired_block_bootstrap_diff_ci
from hawk_research.studies.family_a.rules import BPS, P_STAR, assert_tier_d
from hawk_research.studies.family_a.simulation import with_flat_cost
from hawk_research.studies.family_a.statistics import (
    cpcv_paths,
    hit_rate_bundle,
    kmax_sensitivity,
    lo_sharpe_se,
    measure_rho_bar,
    n_eff_from,
    portfolio_bundle,
    studentised_mean_ci,
    symbol_month_concentration,
    uniqueness,
    walk_forward_report,
    widest_ci,
)

SEED = 7


B_CI = 10_000


B_FAST = 2_000  # K_max / extra axes only


def _ci_inside(ci: Dict, lo: float, hi: float) -> bool:
    a, b = ci.get("ci_low"), ci.get("ci_high")
    if a is None or b is None:
        return False
    return float(a) >= lo and float(b) <= hi


def _ci_excludes_zero(ci: Dict) -> Optional[bool]:
    a, b = ci.get("ci_low"), ci.get("ci_high")
    if a is None or b is None:
        return None
    return bool(a > 0 or b < 0)


def cell_expectancy_ci(sim: pd.DataFrame, cell: str) -> Dict[str, Any]:
    sub = sim[(sim["s1_cell"] == cell) & (sim["fill_status"] == "filled")]
    # FAR is scored, not traded; still use filled FAR rows
    if sub.empty:
        return {"n": 0, "stat": None, "ci_low": None, "ci_high": None}
    col = "ret_resid_net" if sub["ret_resid_net"].notna().any() else "ret_net"
    return studentised_mean_ci(
        sub[col].to_numpy(dtype=float) * BPS,
        sub["decision_time_ms"].to_numpy(dtype=np.int64),
        block_days=10,
        n_samples=B_CI,
        seed=SEED,
    )


def paired_delta_bps(a: pd.DataFrame, b: pd.DataFrame, col: str) -> Dict[str, Any]:
    aa = a.dropna(subset=[col])
    bb = b.dropna(subset=[col])
    res = paired_block_bootstrap_diff_ci(
        aa[col].to_numpy(dtype=float) * BPS,
        aa["decision_time_ms"].to_numpy(dtype=np.int64),
        bb[col].to_numpy(dtype=float) * BPS,
        bb["decision_time_ms"].to_numpy(dtype=np.int64),
        block_hours=10 * 24,
        n_samples=B_CI,
        seed=SEED,
    )
    return res


def evaluate_trades(label: str, filled: pd.DataFrame, unconstrained: bool = False) -> Dict[str, Any]:
    assert_msg = filled["decision_time_ms"].to_numpy()
    assert_tier_d(assert_msg, label)
    u = uniqueness(filled)
    rho = measure_rho_bar(filled)
    neff = n_eff_from(u, rho)
    neff_pess = n_eff_from(u, 0.70)
    hits = hit_rate_bundle(filled, neff)
    port = portfolio_bundle(filled, max_concurrent=5, unconstrained=unconstrained)
    daily = port["daily"]
    d_ci = (
        widest_ci(daily.to_numpy(), daily.index.to_numpy(dtype=np.int64), n_samples=B_CI, seed=SEED)
        if len(daily)
        else {"primary_b10": {}, "widest": {}}
    )
    trade_ci = {
        "primary_b10": studentised_mean_ci(
            filled["ret_net"].to_numpy(dtype=float) * BPS,
            filled["decision_time_ms"].to_numpy(dtype=np.int64),
            10,
            B_CI,
            SEED + 1,
        ),
        "widest": widest_ci(
            filled["ret_net"].to_numpy(dtype=float) * BPS,
            filled["decision_time_ms"].to_numpy(dtype=np.int64),
            n_samples=B_FAST,
            seed=SEED + 1,
        )["widest"],
    }
    gross_ci = studentised_mean_ci(
        filled["ret_gross"].to_numpy(dtype=float) * BPS,
        filled["decision_time_ms"].to_numpy(dtype=np.int64),
        10,
        B_CI,
        SEED + 2,
    )
    wf = walk_forward_report(filled)
    conc = symbol_month_concentration(wf["stitched"] if len(wf["stitched"]) else filled)
    ksens = kmax_sensitivity(filled)
    cpcv = cpcv_paths(filled)
    net22 = with_flat_cost(filled, 22.0)
    net32 = with_flat_cost(filled, 32.0)
    net52 = with_flat_cost(filled, 52.0)
    sr = port["sharpe_ann"]
    dsr = deflated_sharpe_ratio(
        float(sr or 0.0),
        n_trials=7,
        sr_variance=0.25,
        n_obs=max(2, len(daily)),
        skew=float(pd.Series(daily).skew()) if len(daily) else 0.0,
        kurtosis=float(pd.Series(daily).kurtosis() + 3.0) if len(daily) else 3.0,
    )
    lo_se = lo_sharpe_se(daily.to_numpy()) if len(daily) else None
    amb = hits.get("ambiguous_frac")
    # Descriptive only, not used by any gate. Despite the key name ("trim5"), this is a
    # 5%/95% *winsorised* mean: returns are clipped at the full-sample quantiles, not dropped.
    # The pre-registration's sign check against a 5%-trimmed mean was not implemented
    # (docs/deviations.md).
    ret_net = filled["ret_net"]
    trim = (
        float(ret_net.clip(lower=ret_net.quantile(0.05), upper=ret_net.quantile(0.95)).mean())
        if len(filled)
        else None
    )
    mean_net = float(filled["ret_net"].mean())
    median_net = float(filled["ret_net"].median())
    return {
        "label": label,
        "n_filled": int(len(filled)),
        "uniqueness": u,
        "rho_bar": rho,
        "n_eff": neff,
        "n_eff_rho070": neff_pess,
        "hits": hits,
        "mean_gross_bps": float(filled["ret_gross"].mean() * BPS),
        "mean_net_bps": mean_net * BPS,
        "median_net_bps": median_net * BPS,
        "trim5_net_bps": None if trim is None else trim * BPS,
        "gross_ci_bps": gross_ci,
        "trade_net_ci_bps": trade_ci,
        "flat_cost_net_bps": {
            "22": float(net22["ret_net"].mean() * BPS),
            "32": float(net32["ret_net"].mean() * BPS),
            "52": float(net52["ret_net"].mean() * BPS),
        },
        "portfolio": {
            "rejection_rate": port["rejection_rate"],
            "capacity_utilization": port["capacity_utilization"],
            "accepted": port["accepted"],
            "rejected": port["rejected"],
            "sharpe_ann": sr,
            "max_drawdown": port["equity_metrics"].get("max_drawdown"),
            "daily_mean": float(daily.mean()) if len(daily) else None,
            "n_daily": int(len(daily)),
            "daily_ci": d_ci["primary_b10"],
            "daily_ci_widest": d_ci["widest"],
            "lo_sharpe_se": lo_se,
        },
        "walk_forward": {k: v for k, v in wf.items() if k != "stitched"},
        "concentration": conc,
        "kmax": ksens,
        "cpcv": {k: v for k, v in cpcv.items() if k != "splits"},
        "cpcv_splits_positive": cpcv.get("splits_positive"),
        "dsr": dsr,
        "ambiguous_frac": amb,
        "n_losing": int((filled["ret_net"] < 0).sum()),
        "fill_status_source_n": None,
    }


def gates_s1(
    ev: Dict[str, Any], identity_ok: bool, delta_b: Dict, delta_n: Dict, fill_break: Dict
) -> Dict[str, Any]:
    hits = ev["hits"]
    wilson = hits.get("wilson_n_eff") or {}
    amb = ev.get("ambiguous_frac") or 0.0
    wf = ev["walk_forward"]
    n_pos = wf.get("n_positive") or 0
    net32 = ev["flat_cost_net_bps"]["32"]
    net52 = ev["flat_cost_net_bps"]["52"]
    net22 = ev["flat_cost_net_bps"]["22"]
    lb32 = ev["trade_net_ci_bps"]["primary_b10"].get("ci_low")
    ub_hit = wilson.get("high") if wilson else None

    g0 = True  # pre-registration file exists and is the contract for this run
    g1_ok = bool(identity_ok) and (amb <= 0.10)
    g1_void = (not identity_ok) or (amb > 0.10)

    g2_pass = (
        net32 > 0
        and (lb32 is not None and lb32 > 0)
        and net52 > 0
        and (wilson.get("low") is not None and wilson["low"] > P_STAR)
    )
    g2_falsified = (ub_hit is not None and ub_hit <= P_STAR) or (net22 <= -10)

    g3_pass = (
        n_pos >= 5
        and (ev["portfolio"]["sharpe_ann"] or 0) > 0
        and (wf.get("max_fold_pnl_share") is not None and wf["max_fold_pnl_share"] <= 0.50)
        and (ev["concentration"].get("max_symbol_share") or 1) <= 0.25
        and (ev["concentration"].get("max_month_share") or 1) <= 0.35
        and wf.get("rolling_anchored_sign_agree")
        and ev["kmax"].get("sign_stable")
    )
    g3_falsified = n_pos <= 2

    g4_pass = False  # SPA/StepM across five systems not runnable on two trials
    g5_pass = (ev["dsr"].get("dsr") or 0) >= 0.95 and (ev["cpcv"].get("paths_positive") or 0) >= 8
    g5_falsified = ev["cpcv"].get("path_means") and np.nanmedian(ev["cpcv"]["path_means"]) <= 0

    g7_pass = ev["n_eff"] >= 300 and ev["n_losing"] >= 0

    s1_delta_ok = (
        _ci_excludes_zero(delta_b)
        and (delta_b.get("diff") or 0) > 0
        and (delta_b.get("diff") or 0) > (delta_n.get("diff") or 0) > 0
    )
    both_inside = _ci_inside(delta_b, -5, 5) and _ci_inside(delta_n, -5, 5)
    break_unreach = (fill_break.get("fill_rate") or 1) < 0.15 and (fill_break.get("shadow_ci_contains_0"))

    if g1_void:
        verdict = "VOID"
    elif g2_falsified or g3_falsified or g5_falsified or both_inside or break_unreach:
        verdict = "FALSIFIED"
    elif g0 and g1_ok and g2_pass and g3_pass and g4_pass and g5_pass and g7_pass and s1_delta_ok:
        verdict = "EVIDENCE OF AN EDGE"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "gate0_preregistration": {"pass": g0, "note": "docs/research/preregistration.md is the contract"},
        "gate1_leakage": {"pass": g1_ok, "void": g1_void, "identity_ok": identity_ok, "ambiguous_frac": amb},
        "gate2_economic": {
            "pass": g2_pass,
            "falsified": g2_falsified,
            "net32_bps": net32,
            "net32_lb": lb32,
            "net52_bps": net52,
            "hit_wilson": wilson,
        },
        "gate3_walk_forward": {"pass": g3_pass, "falsified": g3_falsified, "folds_positive": n_pos, "of": 6},
        "gate4_multiplicity": {
            "pass": g4_pass,
            "note": "Hansen SPA across five Family-A curves is not defined with only S1 and S5 run. Gate 4 cannot pass.",
        },
        "gate5_overfitting": {
            "pass": g5_pass,
            "falsified": bool(g5_falsified),
            "dsr": ev["dsr"].get("dsr"),
            "cpcv_paths_positive": ev["cpcv"].get("paths_positive"),
        },
        "gate6_holdout": {"pass": False, "not_run": True, "note": "H is sealed. Not touched."},
        "gate7_sample": {"pass": g7_pass, "n_eff": ev["n_eff"]},
        "s1_delta_requirement": {
            "pass": s1_delta_ok,
            "both_ci_inside_5bps": both_inside,
            "break_unreachable": break_unreach,
        },
        "verdict": verdict,
    }


def gates_s5(
    ev: Dict,
    delta_zone: Dict,
    balance_ok: bool,
    near_miss: bool,
    n_fold_dz_pos: int = 0,
    delta_retest_ci: Optional[Dict] = None,
) -> Dict[str, Any]:
    hits = ev["hits"]
    wilson = hits.get("wilson_n_eff") or {}
    amb = ev.get("ambiguous_frac") or 0.0
    wf = ev["walk_forward"]
    n_pos = wf.get("n_positive") or 0
    net32 = ev["flat_cost_net_bps"]["32"]
    net52 = ev["flat_cost_net_bps"]["52"]
    lb32 = ev["trade_net_ci_bps"]["primary_b10"].get("ci_low")
    ub_hit = wilson.get("high") if wilson else None
    dz = delta_zone

    g1_ok = amb <= 0.10
    g1_void = amb > 0.10
    g2_pass = (
        net32 > 0
        and (lb32 is not None and lb32 > 0)
        and net52 > 0
        and (wilson.get("low") is not None and wilson["low"] > P_STAR)
    )
    g2_falsified = ub_hit is not None and ub_hit <= P_STAR
    g3_pass = n_pos >= 5
    g3_falsified = n_pos <= 2 or n_fold_dz_pos <= 2
    g4_pass = False
    g5_pass = (ev["dsr"].get("dsr") or 0) >= 0.95
    g7_pass = ev["n_eff"] >= 300

    inside = _ci_inside(dz, -5, 5)
    harmful = dz.get("diff") is not None and dz["diff"] < 0 and _ci_excludes_zero(dz)
    evidence_delta = (dz.get("diff") or 0) > 5 and _ci_excludes_zero(dz)
    retest_geometry = bool(delta_retest_ci) and _ci_inside(delta_retest_ci, -3, 3)
    note = None

    if g1_void:
        verdict = "VOID"
        note = "Gate 1 VOID (ambiguity > 10%)"
    elif not balance_ok:
        verdict = "INCONCLUSIVE"
        note = "covariate imbalance > 0.10; matching rule not changed (would be a new trial)"
    elif near_miss:
        verdict = "INCONCLUSIVE"
        note = "control arm concentrated near detector gates"
    elif inside or harmful or g3_falsified or g2_falsified or retest_geometry:
        verdict = "FALSIFIED"
        if inside:
            note = "event marker"
        elif harmful:
            note = "harmful gates"
        elif retest_geometry:
            note = "Δ_retest CI inside ±3 pp (geometry)"
        else:
            note = "gate falsification"
    elif g2_pass and g3_pass and g4_pass and g5_pass and g7_pass and evidence_delta and balance_ok:
        verdict = "EVIDENCE OF AN EDGE"
        note = None
    else:
        verdict = "INCONCLUSIVE"
        note = None

    return {
        "gate0_preregistration": {"pass": True},
        "gate1_leakage": {"pass": g1_ok, "void": g1_void, "ambiguous_frac": amb},
        "gate2_economic": {
            "pass": g2_pass,
            "falsified": g2_falsified,
            "net32_bps": net32,
            "net52_bps": net52,
        },
        "gate3_walk_forward": {
            "pass": g3_pass,
            "falsified": g3_falsified,
            "folds_positive": n_pos,
            "delta_zone_folds_positive": n_fold_dz_pos,
        },
        "gate4_multiplicity": {"pass": False, "note": "SPA across five systems not runnable on two trials"},
        "gate5_overfitting": {"pass": g5_pass},
        "gate6_holdout": {"pass": False, "not_run": True},
        "gate7_sample": {"pass": g7_pass, "n_eff": ev["n_eff"]},
        "delta_zone_inside_5bps": inside,
        "delta_zone_harmful": harmful,
        "delta_retest_geometry": retest_geometry,
        "n_folds_delta_zone_positive": n_fold_dz_pos,
        "covariate_balance_ok": balance_ok,
        "near_miss_inconclusive": near_miss,
        "verdict": verdict,
        "note": note,
    }


def earn_next(s1: Dict, s5: Dict) -> Dict[str, Any]:
    dz = s5.get("delta_zone_bps") or {}
    db = s1.get("delta_B_bps") or {}
    zone_ci_defined = dz.get("ci_low") is not None and dz.get("ci_high") is not None
    b_ci_defined = db.get("ci_low") is not None and db.get("ci_high") is not None
    zone_inside = _ci_inside(dz, -5, 5) if zone_ci_defined else False
    b_excludes = bool(_ci_excludes_zero(db)) if b_ci_defined else False
    event_marker_stop = zone_inside  # §13.2 first row; requires an actual CI
    authorize_s3 = (zone_ci_defined and not zone_inside) or b_excludes
    if zone_inside and not b_excludes:
        decision = "STOP"
        authorize = []
        why = "S5 Δ_zone 95% CI ⊂ [−5, +5] bps and S1 Δ_B CI includes 0. §9.4: do not authorize S3/S4/S2."
    elif authorize_s3:
        decision = "AUTHORIZE_S3"
        authorize = ["S3"]
        why = (
            (
                "S5 Δ_zone CI is not contained in ±5 bps"
                if (zone_ci_defined and not zone_inside)
                else "S1 Δ_B CI excludes zero"
            )
            + ". §9.4 item 2 authorizes S3 only. S4/S2 remain unauthorized until S3 is at least INCONCLUSIVE with a positive point estimate."
        )
    else:
        decision = "STOP"
        authorize = []
        why = "Δ_zone and/or Δ_B CIs are undefined, so §9.4 cannot authorize S3. S4/S2 remain unauthorized."
    return {
        "decision": decision,
        "authorize": authorize,
        "s3_authorized": authorize_s3 and decision == "AUTHORIZE_S3",
        "s4_authorized": False,
        "s2_authorized": False,
        "event_marker_stop_fired": event_marker_stop,
        "why": why,
        "delta_zone_inside_pm5": zone_inside,
        "delta_B_excludes_zero": b_excludes,
    }
