"""Execute pre-registered Family-A trials S1 and S5 on tier D only.

Does not read holdout (tier H). Does not run S2/S3/S4. Does not touch the
operational database. Logs two trials to the research ledger (``hawk_research.ledger``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hawk_research.data import loader, paths
from hawk_research.features import point_in_time as feat
from hawk_research.labels import outcomes as lab
from hawk_research.ledger import TrialLedger
from hawk_research.provenance import git_commit
from hawk_research.stats.resampling import paired_block_bootstrap_diff_ci
from hawk_research.studies.family_a.controls import (
    MATCH_CONT,
    MATCH_KEYS,
    btc_p99_map,
    load_ohlcv,
    match_controls_for_symbol,
    prepare_bar_panel,
    standardised_smd,
)
from hawk_research.studies.family_a.gates import (
    B_CI,
    B_FAST,
    SEED,
    cell_expectancy_ci,
    earn_next,
    evaluate_trades,
    gates_s1,
    gates_s5,
    paired_delta_bps,
)
from hawk_research.studies.family_a.rules import (
    BPS,
    apply_common_gates,
    d_bounds,
    filter_d,
    s1_side_and_cell,
    s3_side,
)
from hawk_research.studies.family_a.simulation import filled_trades, simulate_maker_outcomes
from hawk_research.studies.family_a.statistics import rolling_folds, studentised_mean_ci

log = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.bool_)):
        v = float(obj) if not isinstance(obj, np.bool_) else bool(obj)
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame {obj.shape}>"
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def break_fill_diagnostics(sim: pd.DataFrame) -> Dict[str, Any]:
    br = sim[sim["s1_cell"] == "BREAK"]
    n = len(br)
    if n == 0:
        return {"n": 0, "fill_rate": None, "shadow_ci_contains_0": True}
    filled = br[br["fill_status"] == "filled"]
    expired = br[br["fill_status"] == "expired_no_fill"]
    fill_rate = float(len(filled) / n)
    shadow = br.dropna(subset=["shadow_ret_gross"]) if "shadow_ret_gross" in br.columns else expired
    if len(shadow) >= 5 and "shadow_ret_gross" in shadow.columns:
        ci = studentised_mean_ci(
            shadow["shadow_ret_gross"].to_numpy(dtype=float) * BPS,
            shadow["decision_time_ms"].to_numpy(dtype=np.int64),
            10,
            B_FAST,
            SEED,
        )
        contains0 = ci["ci_low"] is not None and ci["ci_low"] <= 0 <= (ci["ci_high"] or 0)
    else:
        ci, contains0 = {"stat": None}, True
    return {
        "n": int(n),
        "n_filled": int(len(filled)),
        "fill_rate": fill_rate,
        "shadow_ci": ci,
        "shadow_ci_contains_0": contains0,
    }


def attach_match_keys(z: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    cols = list(MATCH_KEYS) + list(MATCH_CONT) + ["open_time"]
    cols = [c for c in cols if c in panel.columns]
    m = panel[cols].drop_duplicates("open_time")
    out = z.merge(m, left_on="source_open_time", right_on="open_time", how="left", suffixes=("", "_p"))
    return out


def run_s1(ds_r: pd.DataFrame, ohlcv, btc, btc_p99, identity_ok: bool) -> Dict[str, Any]:
    d = filter_d(ds_r)
    d = apply_common_gates(d, btc_p99)
    skipped = {
        "n_d": int(len(d)),
        "n_pass_gates": int(d["passes_common_gates"].sum()),
        "skip_vol": int((~d["gate_vol"]).sum()),
        "skip_dislocation": int((~d["gate_dislocation"]).sum()),
        "skip_integrity": int((~d["gate_integrity"]).sum()),
        "skip_universe": int((~d["gate_universe"]).sum()),
    }
    d = d[d["passes_common_gates"]].copy()
    d = s1_side_and_cell(d)
    cells = d["s1_cell"].value_counts().to_dict()
    sim = simulate_maker_outcomes(d, ohlcv, btc)
    filled_tradeable = filled_trades(sim, tradeable_only=True)
    filled_all = filled_trades(sim, tradeable_only=False)
    ev = evaluate_trades("S1", filled_tradeable)
    ev["fill_status_counts"] = sim["fill_status"].value_counts().to_dict()
    ev["n_candidates"] = int(len(d))
    ev["n_tradeable"] = int(d["tradeable"].sum())
    ev["n_placed"] = int((sim["fill_status"] != "not_placed").sum())
    ev["fill_rate_tradeable"] = (
        float((sim.loc[sim["tradeable"] == True, "fill_status"] == "filled").mean())  # noqa: E712
        if len(sim)
        else None
    )

    far = filled_all[filled_all["s1_cell"] == "FAR"]
    brk = filled_all[filled_all["s1_cell"] == "BREAK"]
    ner = filled_all[filled_all["s1_cell"] == "NEAR"]
    col = "ret_resid_net"
    delta_b = (
        paired_delta_bps(brk, far, col)
        if len(brk) and len(far)
        else {"diff": None, "ci_low": None, "ci_high": None}
    )
    delta_n = (
        paired_delta_bps(ner, far, col)
        if len(ner) and len(far)
        else {"diff": None, "ci_low": None, "ci_high": None}
    )
    far_ci = cell_expectancy_ci(sim, "FAR")
    fill_break = break_fill_diagnostics(sim)
    per_cell = {}
    for cell, sub in (("BREAK", brk), ("NEAR", ner), ("FAR", far)):
        per_cell[cell] = {
            "n_filled": int(len(sub)),
            "fill_rate": float((sim.loc[sim["s1_cell"] == cell, "fill_status"] == "filled").mean())
            if (sim["s1_cell"] == cell).any()
            else None,
            "mean_gross_bps": float(sub["ret_gross"].mean() * BPS) if len(sub) else None,
            "mean_net_bps": float(sub["ret_net"].mean() * BPS) if len(sub) else None,
            "mean_resid_net_bps": float(sub[col].mean() * BPS)
            if len(sub) and sub[col].notna().any()
            else None,
        }

    gates = gates_s1(ev, identity_ok, delta_b, delta_n, fill_break)
    return {
        "system": "S1",
        "skipped": skipped,
        "cell_counts_gated_d": {str(k): int(v) for k, v in cells.items()},
        "evaluation": ev,
        "delta_B_bps": delta_b,
        "delta_N_bps": delta_n,
        "delta_FAR_bps": far_ci,
        "per_cell": per_cell,
        "break_fill": fill_break,
        "gates": gates,
        "verdict": gates["verdict"],
    }


def _void_s5(n_signal: int, dropped: int, note: str, gate_note: str) -> Dict[str, Any]:
    return {
        "system": "S5",
        "verdict": "VOID",
        "note": note,
        "n_signal": n_signal,
        "n_dropped_unmatched": dropped,
        "gates": {"verdict": "VOID", "note": gate_note},
        "delta_zone_bps": {"diff": None, "ci_low": None, "ci_high": None},
        "delta_rv": {"diff": None, "ci_low": None, "ci_high": None},
    }


def _match_s5_controls(
    zones: pd.DataFrame, pass_r: pd.DataFrame, ohlcv, btc, btc_p99, pit: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    """Three matched control bars per zone, symbol by symbol.

    Returns (zones with at least one control, controls, zones dropped, controls near the
    detector's gates, controls in total).
    """
    from hawk_research.data.backfill import to_candles
    from hawk_research.replay.round1 import btc_range_series

    d0, d1, _ = d_bounds()
    pass_r_by_sym = {
        s: set(g["source_open_time"].astype("int64").tolist()) for s, g in pass_r.groupby("symbol")
    }
    btc_range = btc_range_series(to_candles(btc))
    pit = pit.copy()

    ctrl_frames: List[pd.DataFrame] = []
    zone_keyed: List[pd.DataFrame] = []
    dropped = 0
    n_near_gate = 0
    n_ctrl_total = 0

    symbols = sorted(zones["symbol"].unique())
    for i, symbol in enumerate(symbols, 1):
        z = zones[zones["symbol"] == symbol].copy()
        if z.empty:
            continue
        bars = ohlcv[symbol]
        lab_path = paths.LABELS_DIR / "15m" / f"{symbol}.parquet"
        if not lab_path.exists():
            dropped += len(z)
            continue
        labels = pd.read_parquet(lab_path)
        fmat = feat.build_symbol_features(bars, btc)
        pit_s = pit[pit["symbol"] == symbol][["date", "rank_qv_1d", "listed_days"]]
        panel = prepare_bar_panel(
            bars, labels, fmat, pit_s, btc_range, pass_r_by_sym.get(symbol, set()), btc_p99, d0, d1
        )
        panel["symbol"] = symbol
        z = attach_match_keys(z, panel)
        ctrl, drop = match_controls_for_symbol(z, panel, n_controls=3, match_bars=192)
        dropped += drop
        if not ctrl.empty:
            ctrl_frames.append(ctrl)
            n_ctrl_total += len(ctrl)
            vr = ctrl["det_volume_ratio"].to_numpy(dtype=float)
            wr = ctrl["det_wick_ratio"].to_numpy(dtype=float)
            n_near_gate += int(((np.abs(vr - 1.35) / 1.35 <= 0.10) | (np.abs(wr - 1.2) / 1.2 <= 0.10)).sum())
        # keep only zones that received ≥1 control
        if not ctrl.empty:
            keep = set(ctrl["matched_zone_uid"].astype(str))
            zone_keyed.append(z[z["zone_uid"].astype(str).isin(keep)])
        log.info(
            f"{i}/{len(symbols)} {str(symbol).encode('ascii', 'replace').decode()} "
            f"zones={len(z)} ctrl={0 if ctrl.empty else len(ctrl)} drop={drop}"
        )

    z_paired = pd.concat(zone_keyed, ignore_index=True) if zone_keyed else zones.iloc[0:0]
    controls = pd.concat(ctrl_frames, ignore_index=True) if ctrl_frames else pd.DataFrame()
    return z_paired, controls, dropped, n_near_gate, n_ctrl_total


def _write_match_table(controls: pd.DataFrame) -> Path:
    match_path = paths.FEATURES_DIR / "s5_control_matches.parquet"
    keep_match = [
        c
        for c in (
            [
                "symbol",
                "open_time",
                "source_open_time",
                "zone_uid",
                "matched_zone_uid",
                "match_distance",
                "zone_high",
                "zone_low",
            ]
            + list(MATCH_KEYS)
            + list(MATCH_CONT)
            + ["det_volume_ratio", "det_wick_ratio", "sigma_8h", "rv_24h", "rv_4h"]
        )
        if c in controls.columns
    ]
    controls[keep_match].to_parquet(match_path, index=False)
    log.info(f"wrote matching table {match_path} n={len(controls)}")
    return match_path


def _s3_control_arm(controls: pd.DataFrame, z_paired: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the S3 side rule to the control arm and keep only zones that still have a control."""
    controls = s3_side(controls)
    controls["tradeable"] = controls["side"] != 0
    controls = controls[controls["tradeable"]].copy()
    if not controls.empty:
        keep = set(controls["matched_zone_uid"].astype(str))
        z_paired = z_paired[z_paired["zone_uid"].astype(str).isin(keep)].copy()
    if "zone_uid" not in controls.columns or (len(controls) and controls["zone_uid"].duplicated().any()):
        controls["zone_uid"] = (
            controls["symbol"].astype(str)
            + "@"
            + controls["open_time"].astype(str)
            + "#c"
            + np.arange(len(controls)).astype(str)
        )
    return controls, z_paired


def _retest_delta(
    z_paired: pd.DataFrame, controls: pd.DataFrame, ohlcv
) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
    """Zone vs control retest rate: (zone rate, control rate, difference in pp, its interval)."""
    z_retest_flag = (
        z_paired["retested"].fillna(False).astype(float)
        if "retested" in z_paired.columns
        else pd.Series(np.nan, index=z_paired.index)
    )
    z_re = float(z_retest_flag.mean()) if z_retest_flag.notna().any() else None
    # control retest via lifecycle
    retest_ctrl = []
    for symbol, g in controls.groupby("symbol"):
        bars = ohlcv[symbol]
        pz = pd.DataFrame(
            {
                "symbol": symbol,
                "source_open_time": g["source_open_time"].to_numpy(dtype=np.int64),
                "zone_high": g["zone_high"].to_numpy(dtype=float),
                "zone_low": g["zone_low"].to_numpy(dtype=float),
                "direction": np.where(g["side"].to_numpy() > 0, "bullish", "bearish"),
            }
        )
        retest_ctrl.append(lab.zone_lifecycle(bars, pz))
    life_c = pd.concat(retest_ctrl, ignore_index=True) if retest_ctrl else pd.DataFrame()
    c_re = float(life_c["retested"].fillna(False).mean()) if len(life_c) else None
    delta_retest_pp = None if z_re is None or c_re is None else (z_re - c_re) * 100.0
    if z_re is not None and c_re is not None and len(z_paired) and len(life_c):
        delta_retest_ci = paired_block_bootstrap_diff_ci(
            z_retest_flag.to_numpy(dtype=float) * 100.0,
            z_paired["source_open_time"].to_numpy(dtype=np.int64) + lab.BAR_MS,
            life_c["retested"].fillna(False).to_numpy(dtype=float) * 100.0,
            life_c["source_open_time"].to_numpy(dtype=np.int64) + lab.BAR_MS,
            block_hours=10 * 24,
            n_samples=B_CI,
            seed=SEED + 11,
        )
    else:
        delta_retest_ci = {"diff": delta_retest_pp, "ci_low": None, "ci_high": None}
    return z_re, c_re, delta_retest_pp, delta_retest_ci


def _delta_rv(z_paired: pd.DataFrame, controls: pd.DataFrame) -> Dict[str, Any]:
    """Zone minus control forward 4h realised volatility, with a paired block-bootstrap interval."""
    if "fwd_rv_4h" in z_paired.columns and "fwd_rv_4h" in controls.columns:
        return paired_block_bootstrap_diff_ci(
            z_paired["fwd_rv_4h"].to_numpy(dtype=float),
            z_paired["source_open_time"].to_numpy(dtype=np.int64) + lab.BAR_MS,
            controls["fwd_rv_4h"].to_numpy(dtype=float),
            controls["source_open_time"].to_numpy(dtype=np.int64) + lab.BAR_MS,
            block_hours=10 * 24,
            n_samples=B_CI,
            seed=SEED,
        )
    return {"diff": None, "ci_low": None, "ci_high": None}


def _covariate_balance(z_paired: pd.DataFrame, controls: pd.DataFrame) -> Tuple[Dict[str, Any], bool]:
    """Standardised mean difference per matching covariate; balanced if every |SMD| <= 0.10."""
    balance = {}
    ok_bal = True
    for k in MATCH_CONT:
        if k not in z_paired.columns or k not in controls.columns:
            continue
        smd = standardised_smd(z_paired[k].to_numpy(), controls[k].to_numpy())
        balance[k] = smd
        if smd is not None and abs(smd) > 0.10:
            ok_bal = False
    return balance, ok_bal


def _fold_delta_zone(
    fill_z: pd.DataFrame, fill_c: pd.DataFrame, col: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Zone minus control mean per rolling test fold, and how many folds are positive."""
    fold_dz = []
    for spec in rolling_folds():
        zf = fill_z[
            (fill_z["decision_time_ms"] >= spec["test_start_ms"])
            & (fill_z["decision_time_ms"] < spec["test_end_ms"])
        ]
        cf = fill_c[
            (fill_c["decision_time_ms"] >= spec["test_start_ms"])
            & (fill_c["decision_time_ms"] < spec["test_end_ms"])
        ]
        if len(zf) and len(cf):
            dlt = float(zf[col].mean() - cf[col].mean()) * BPS
        else:
            dlt = None
        fold_dz.append({"fold": spec["fold"], "delta_zone_bps": dlt, "positive": dlt is not None and dlt > 0})
    return fold_dz, int(sum(1 for f in fold_dz if f["positive"]))


def run_s5(
    ds_p: pd.DataFrame, pass_r: pd.DataFrame, ohlcv, btc, btc_p99, pit: pd.DataFrame
) -> Dict[str, Any]:
    zones = filter_d(ds_p)
    zones = apply_common_gates(zones, btc_p99)
    zones = zones[zones["passes_common_gates"]].copy()
    zones = s3_side(zones)
    zones = zones[zones["tradeable"]].copy()
    n_signal = int(len(zones))

    z_paired, controls, dropped, n_near_gate, n_ctrl_total = _match_s5_controls(
        zones, pass_r, ohlcv, btc, btc_p99, pit
    )
    if controls.empty or z_paired.empty:
        return _void_s5(n_signal, dropped, "no matched controls", "no matched controls")
    match_path = _write_match_table(controls)

    controls, z_paired = _s3_control_arm(controls, z_paired)
    if controls.empty or z_paired.empty:
        return _void_s5(
            n_signal, dropped, "S3 magnitude skip left no paired controls", "S3 skip emptied the control arm"
        )

    sim_z = simulate_maker_outcomes(z_paired, ohlcv, btc)
    sim_c = simulate_maker_outcomes(controls, ohlcv, btc)
    fill_z = filled_trades(sim_z, tradeable_only=True)
    fill_c = filled_trades(sim_c, tradeable_only=True)

    ev_z = evaluate_trades("S5_zone", fill_z)
    # control arm unconstrained
    ev_c = evaluate_trades("S5_control", fill_c, unconstrained=True)

    # paired Δ_zone on zones that have both a filled zone and mean filled control
    col = "ret_net"
    delta_zone = (
        paired_delta_bps(fill_z, fill_c, col)
        if len(fill_z) and len(fill_c)
        else {"diff": None, "ci_low": None, "ci_high": None}
    )
    z_re, c_re, delta_retest_pp, delta_retest_ci = _retest_delta(z_paired, controls, ohlcv)
    delta_rv = _delta_rv(z_paired, controls)
    balance, ok_bal = _covariate_balance(z_paired, controls)

    near_frac = (n_near_gate / n_ctrl_total) if n_ctrl_total else 0.0
    near_miss = near_frac > 0.40

    fill_z_rate = float((sim_z["fill_status"] == "filled").mean()) if len(sim_z) else None
    fill_c_rate = float((sim_c["fill_status"] == "filled").mean()) if len(sim_c) else None

    fold_dz, n_fold_dz_pos = _fold_delta_zone(fill_z, fill_c, col)

    gates = gates_s5(ev_z, delta_zone, ok_bal, near_miss, n_fold_dz_pos, delta_retest_ci)
    return {
        "system": "S5",
        "n_signal_s3_on_pass_p_d": n_signal,
        "n_zone_paired": int(len(z_paired)),
        "n_controls": int(len(controls)),
        "n_dropped_unmatched": int(dropped),
        "match_table": paths.rel(match_path),
        "evaluation_zone": ev_z,
        "evaluation_control": ev_c,
        "delta_zone_bps": delta_zone,
        "delta_zone_folds": fold_dz,
        "n_folds_delta_zone_positive": n_fold_dz_pos,
        "delta_retest_pp": delta_retest_pp,
        "delta_retest_ci": delta_retest_ci,
        "zone_retest_rate": z_re,
        "control_retest_rate": c_re,
        "delta_rv": delta_rv,
        "delta_fill_pp": None
        if fill_z_rate is None or fill_c_rate is None
        else (fill_z_rate - fill_c_rate) * 100,
        "fill_rate_zone": fill_z_rate,
        "fill_rate_control": fill_c_rate,
        "covariate_smd": balance,
        "covariate_balance_ok": ok_bal,
        "near_gate_frac": near_frac,
        "near_miss": near_miss,
        "gates": gates,
        "verdict": gates["verdict"],
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    s1, s5, nxt = report["S1"], report["S5"], report["earn_the_next_trial"]
    db = s1.get("delta_B_bps") or {}
    dn = s1.get("delta_N_bps") or {}
    dz = s5.get("delta_zone_bps") or {}
    drv = s5.get("delta_rv") or {}
    ev = s1.get("evaluation") or {}
    lines = [
        "# Family A — S1 and S5 results (tier D only)",
        "",
        f"**Run:** {report.get('created_at')}",
        f"**Pre-registration SHA:** `{report.get('preregistration_sha')}`",
        f"**Code SHA:** `{report.get('git_commit')}`",
        "**Holdout (tier H):** not read.",
        "**Trials logged:** 2 (S1, S5). S2/S3/S4 not run.",
        "",
        "## Verdicts",
        "",
        f"- **S1:** `{s1.get('verdict')}`",
        f"- **S5:** `{s5.get('verdict')}`",
        f"- **Event-marker stop (§13.2):** {'FIRED' if nxt.get('event_marker_stop_fired') else 'not fired'}",
        f"- **S3 authorized:** {'yes' if nxt.get('s3_authorized') else 'no'}",
        f"- **Decision:** {nxt.get('decision')} — {nxt.get('why')}",
        "",
        "## S1 — Level-mechanism split (Osler)",
        "",
        f"- Δ_B (BREAK − FAR, residual net, bps): **{db.get('diff')}** 95% CI [{db.get('ci_low')}, {db.get('ci_high')}]",
        f"- Δ_N (NEAR − FAR, residual net, bps): **{dn.get('diff')}** 95% CI [{dn.get('ci_low')}, {dn.get('ci_high')}]",
        f"- FAR cell mean residual net, bps: **{(s1.get('delta_FAR_bps') or {}).get('stat')}** "
        f"CI [{(s1.get('delta_FAR_bps') or {}).get('ci_low')}, {(s1.get('delta_FAR_bps') or {}).get('ci_high')}]",
        f"- Net expectancy (ladder, bps/trade): **{ev.get('mean_net_bps')}** "
        f"CI [{(ev.get('trade_net_ci_bps') or {}).get('primary_b10', {}).get('ci_low')}, "
        f"{(ev.get('trade_net_ci_bps') or {}).get('primary_b10', {}).get('ci_high')}]",
        f"- Gross expectancy, bps/trade: **{ev.get('mean_gross_bps')}**",
        f"- Folds positive (rolling, of 6): **{(ev.get('walk_forward') or {}).get('n_positive')}**",
        f"- n_filled: {ev.get('n_filled')}; n_eff: {ev.get('n_eff')}; Σū: {(ev.get('uniqueness') or {}).get('sum_u')}",
        f"- Hit rate: {(ev.get('hits') or {}).get('hit_rate')} (target/stop/timeout = "
        f"{(ev.get('hits') or {}).get('n_target')}/{(ev.get('hits') or {}).get('n_stop')}/{(ev.get('hits') or {}).get('n_timeout')})",
        f"- Ambiguous fraction: {ev.get('ambiguous_frac')}",
        f"- BREAK fill rate: {(s1.get('break_fill') or {}).get('fill_rate')}",
        f"- Pass-R identity: {report.get('pass_r_identity_ok')}",
        "- Gate 4: not passable (SPA on five systems not run).",
        f"- Cells on gated D: {s1.get('cell_counts_gated_d')}",
        "",
        "## S5 — Zone vs matched control (S3 rule on both arms)",
        "",
        f"- Δ_zone (net bps): **{dz.get('diff')}** 95% CI [{dz.get('ci_low')}, {dz.get('ci_high')}]",
        f"- Δ_rv (fwd_rv_4h): **{drv.get('diff')}** 95% CI [{drv.get('ci_low')}, {drv.get('ci_high')}]",
        f"- Δ_retest (pp): **{s5.get('delta_retest_pp')}** CI [{(s5.get('delta_retest_ci') or {}).get('ci_low')}, {(s5.get('delta_retest_ci') or {}).get('ci_high')}] (zone {s5.get('zone_retest_rate')} vs control {s5.get('control_retest_rate')})",
        f"- Fill rate zone/control: {s5.get('fill_rate_zone')} / {s5.get('fill_rate_control')}",
        f"- n_zone_paired: {s5.get('n_zone_paired')}; n_controls: {s5.get('n_controls')}; unmatched drops: {s5.get('n_dropped_unmatched')}",
        f"- Covariate balance ok (SMD<0.10): {s5.get('covariate_balance_ok')}; SMDs: {s5.get('covariate_smd')}",
        f"- Near-miss controls: {s5.get('near_miss')} (frac {s5.get('near_gate_frac')})",
        f"- Zone-arm folds positive: {(s5.get('evaluation_zone') or {}).get('walk_forward', {}).get('n_positive')}",
        f"- Zone-arm n_eff: {(s5.get('evaluation_zone') or {}).get('n_eff')}",
        "",
        "## Earn-the-next-trial (§9.4 / §13.2)",
        "",
        f"{nxt.get('why')}",
        "",
        "**S4 and S2 remain unauthorized.** Holdout remains sealed.",
        "",
        "## What this is not",
        "",
        "Round 1's 84-config screen was not rerun. Round 1's unsigned zone-vs-matched "
        "close-to-close test is not S5. S5 uses S3's signed maker-entry 1.0σ/0.5σ rule "
        "on both arms. A clean INCONCLUSIVE or FALSIFIED is the expected modal outcome.",
        "",
        "## Artifacts",
        "",
    ]
    for p in report.get("paths", {}).items():
        lines.append(f"- `{p[0]}`: `{p[1]}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    t0 = time.time()
    paths.ensure_dirs()
    d0, d1, h0 = d_bounds()
    log.info(f"D = [{d0}, {d1})  H starts {h0} (not read)")

    pass_r_path = paths.ZONES_DIR / "replay_zones_pass_r.parquet"
    pass_r_ds_path = paths.FEATURES_DIR / "zone_dataset_pass_r.parquet"
    ident_path = paths.ZONES_DIR / "replay_pass_r_summary.json"
    if not pass_r_ds_path.exists() or not pass_r_path.exists():
        raise SystemExit("Pass R dataset missing. Run `hawk-research run s3r` first.")

    ident = json.loads(ident_path.read_text(encoding="utf-8")) if ident_path.exists() else {}
    identity_ok = not bool((ident.get("identity") or {}).get("void_s1_if_not_identical", True))
    log.info(f"Pass R identity ok={identity_ok}")

    log.info("loading datasets (D filter applied immediately)")
    ds_r = pd.read_parquet(pass_r_ds_path)
    ds_p = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")
    pass_r_zones = pd.read_parquet(pass_r_path, columns=["symbol", "source_open_time"])
    # drop H/F before any statistic
    ds_r = filter_d(ds_r)
    ds_p = filter_d(ds_p)
    pass_r_zones = filter_d(pass_r_zones)

    log.info("loading OHLCV + BTC + PIT")
    ohlcv = load_ohlcv()
    btc = loader.load_ohlcv("BTCUSDT")
    btc_p99 = btc_p99_map(btc)
    pit = pd.read_parquet(paths.UNIVERSE_DIR / "pit_universe.parquet")
    listing = pd.read_parquet(paths.UNIVERSE_DIR / "listing_history.parquet")
    n_delisted = (
        int((~listing["still_listed_at_snapshot"]).sum())
        if "still_listed_at_snapshot" in listing.columns
        else 0
    )
    log.info(f"listing_history delisted_before_snapshot={n_delisted} (protocol §9.1 item 2)")

    log.info("S1")
    s1 = run_s1(ds_r, ohlcv, btc, btc_p99, identity_ok)
    s1_ckpt = paths.REPORTS_DIR / "s1_checkpoint.json"
    s1_ckpt.write_text(json.dumps(_jsonable(s1), indent=2, default=str), encoding="utf-8")
    log.info(f"S1 verdict={s1['verdict']} Δ_B={s1['delta_B_bps'].get('diff')} (checkpoint {s1_ckpt})")

    log.info("S5")
    s5 = run_s5(ds_p, pass_r_zones, ohlcv, btc, btc_p99, pit)
    log.info(f"S5 verdict={s5['verdict']} Δ_zone={s5.get('delta_zone_bps', {}).get('diff')}")

    nxt = earn_next(s1, s5)
    log.info(f"{nxt['decision']} event_marker_stop={nxt['event_marker_stop_fired']}")

    ledger = TrialLedger(
        "familyA",
        description="Pre-registered S1 and S5 on tier D",
        dataset_start="2024-09-01",
        dataset_end="2026-02-28",
    )
    id1 = ledger.log_trial(
        {"system": "S1", "horizon": "8h", "barriers": "1.0s/0.5s", "entry": "maker_far_edge"},
        {
            "verdict": s1["verdict"],
            "delta_B": s1["delta_B_bps"],
            "mean_net_bps": (s1.get("evaluation") or {}).get("mean_net_bps"),
        },
        label="S1",
    )
    id5 = ledger.log_trial(
        {"system": "S5", "horizon": "8h", "barriers": "1.0s/0.5s", "entry": "maker_far_edge", "rule": "S3"},
        {"verdict": s5["verdict"], "delta_zone": s5.get("delta_zone_bps")},
        label="S5",
    )

    # Content hash of the pre-registration as shipped (docs/archive). The original
    # run recorded the git commit of that file instead; see docs/PROVENANCE.md.
    prereg = Path("docs") / "archive" / "preregistration.md"
    pre_sha = hashlib.sha256(prereg.read_bytes()).hexdigest() if prereg.exists() else "unknown"

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration_sha256": pre_sha,
        "git_commit": git_commit(),
        "tier": "D",
        "holdout_touched": False,
        "n_delisted_in_listing_history": n_delisted,
        "pass_r_identity_ok": identity_ok,
        "pass_r_summary": ident,
        "trials_logged": 2,
        "ledger_experiment_ids": {"S1": id1, "S5": id5},
        "ledger_db": paths.rel(paths.RESEARCH_DB),
        "S1": s1,
        "S5": s5,
        "earn_the_next_trial": nxt,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "pass_r_zones": paths.rel(pass_r_path),
            "pass_r_dataset": paths.rel(pass_r_ds_path),
            "json_report": paths.rel(paths.REPORTS_DIR / "s1_s5_results.json"),
            "markdown": paths.rel(paths.REPORTS_DIR / "s1_s5_results.md"),
            "ledger_sqlite": paths.rel(paths.RESEARCH_DB),
            "ledger_jsonl": paths.rel(paths.REPORTS_DIR / "experiments_ledger.jsonl"),
        },
    }
    jpath = paths.REPORTS_DIR / "s1_s5_results.json"
    jpath.write_text(json.dumps(_jsonable(report), indent=2, default=str), encoding="utf-8")
    md = paths.REPORTS_DIR / "s1_s5_results.md"
    write_markdown(report, md)
    log.info(f"wrote {jpath} and {md} in {report['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
