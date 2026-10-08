"""Compare regenerated results with the committed ``results/headline_metrics.json``.

Each headline number quoted in the docs is extracted from the stage reports by a
JSON path and compared with the committed value within a stated tolerance:

* counts, verdicts and booleans must match exactly;
* mean returns: absolute 1e-6 (0.01 bps);
* bootstrap interval bounds, Sharpe, PBO, DSR and rates: absolute 1e-4.

Bootstrap intervals use fixed seeds, so on the same library versions they match
exactly; the looser tolerance allows for floating-point differences between
platforms and numpy versions. Every comparison is printed, including passes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXACT = 0.0
RET = 1e-6
STAT = 1e-4

# (label, report path relative to the data dir, JSON path, tolerance)
Metric = Tuple[str, str, Sequence[Any], float]

NB = "reports/null_baselines.json"
NBS = "reports/null_baselines_smc.json"
METRICS: List[Metric] = [
    # Round 1
    ("round1 zones replayed", "zones/replay_summary.json", ["n_detections_pre_gate"], EXACT),
    ("round1 zones live-equivalent", "zones/replay_summary.json", ["n_live_equivalent"], EXACT),
    ("round1 leakage audit passed", "reports/leakage_audit.json", ["passed"], EXACT),
    (
        "round1 harness positive control",
        "reports/harness_controls.json",
        ["positive_control", "verdict"],
        EXACT,
    ),
    ("round1 4h zone - matched control (signed)", NB, ["roles", "directional_signal_4h", "diff"], RET),
    ("round1 4h zone - matched control CI low", NB, ["roles", "directional_signal_4h", "ci_low"], STAT),
    ("round1 4h zone - matched control CI high", NB, ["roles", "directional_signal_4h", "ci_high"], STAT),
    ("round1 4h forward vol - control", NB, ["roles", "volatility_filter_4h", "diff"], RET),
    (
        "round1 direction 4h gross",
        NB,
        ["directional", "direction", "4h|mexc_futures_taker", "gross_mean"],
        RET,
    ),
    (
        "round1 direction 4h net (MEXC)",
        NB,
        ["directional", "direction", "4h|mexc_futures_taker", "net_mean"],
        RET,
    ),
    (
        "round1 direction 4h win rate",
        NB,
        ["directional", "direction", "4h|mexc_futures_taker", "win_rate"],
        STAT,
    ),
    ("round1 walk-forward folds", NB, ["walk_forward", "summary", "n_folds"], EXACT),
    ("round1 folds gross-positive", NB, ["walk_forward", "summary", "folds_gross_positive"], EXACT),
    ("round1 folds net-positive", NB, ["walk_forward", "summary", "folds_net_positive"], EXACT),
    ("round1 train-test rank corr", NB, ["walk_forward", "summary", "train_test_rank_corr"], STAT),
    ("round1 PBO", NB, ["pbo", "pbo"], STAT),
    ("round1 deflated Sharpe", NB, ["deflated_sharpe", "dsr"], STAT),
    ("round1 DSR trials", NB, ["deflated_sharpe", "n_trials"], EXACT),
    ("round1 best observed Sharpe", NB, ["deflated_sharpe", "observed_sr"], STAT),
    ("round1 zone retest rate", "reports/location_control.json", ["zone_retest_rate"], STAT),
    ("round1 control retest rate", "reports/location_control.json", ["control_retest_rate"], STAT),
    (
        "bearish retest 2:1 gross",
        "reports/bearish_only.json",
        ["strategies", "bearish|retest|barrier|0.02/0.01|mexc_futures_taker", "mean_ret_gross"],
        RET,
    ),
    (
        "bearish retest 2:1 net (MEXC)",
        "reports/bearish_only.json",
        ["strategies", "bearish|retest|barrier|0.02/0.01|mexc_futures_taker", "mean_ret_net"],
        RET,
    ),
    (
        "capacity cap3 retest 24h net bps",
        "reports/capacity_check.json",
        ["results", 0, "mean_ret_net_bps"],
        1e-2,
    ),
    ("capacity cap3 rejection rate", "reports/capacity_check.json", ["results", 0, "rejection_rate"], STAT),
    # Family A
    ("S1 verdict", "reports/s1_s5_results.json", ["S1", "verdict"], EXACT),
    ("S5 verdict", "reports/s1_s5_results.json", ["S5", "verdict"], EXACT),
    ("S1 mean net bps", "reports/s1_s5_results.json", ["S1", "evaluation", "mean_net_bps"], 1e-2),
    ("S5 delta_zone bps", "reports/s1_s5_results.json", ["S5", "delta_zone_bps", "stat"], 1e-2),
    ("Pass R events", "reports/s1_s5_results.json", ["pass_r_summary", "n_pass_r"], EXACT),
    # Round 2 (SMC)
    ("smc zones", "zones/replay_smc_summary.json", ["n_zones"], EXACT),
    (
        "smc OB candles also round-1 zones %",
        "zones/replay_smc_summary.json",
        ["overlap_with_round1", "smc_ob_candle_also_round1_zone_pct"],
        STAT,
    ),
    ("smc leakage audit passed", "reports/leakage_audit_smc.json", ["passed"], EXACT),
    ("smc 4h zone - matched control (signed)", NBS, ["roles", "directional_signal_4h", "diff"], RET),
    ("smc 4h forward vol - control", NBS, ["roles", "volatility_filter_4h", "diff"], RET),
    ("smc folds net-positive", NBS, ["walk_forward", "summary", "folds_net_positive"], EXACT),
    ("smc PBO", NBS, ["pbo", "pbo"], STAT),
    ("smc deflated Sharpe", NBS, ["deflated_sharpe", "dsr"], STAT),
    (
        "smc fvg retest 24h net (MEXC)",
        "reports/smc_eval.json",
        ["strategies", "fvg|full|retest|time|-/-|mexc_futures_taker", "mean_ret_net"],
        RET,
    ),
    (
        "smc all retest 24h net (MEXC)",
        "reports/smc_eval.json",
        ["strategies", "all|full|retest|time|-/-|mexc_futures_taker", "mean_ret_net"],
        RET,
    ),
]


def _get(doc: Any, path: Sequence[Any]) -> Any:
    for key in path:
        doc = doc[key]
    return doc


def extract(data_dir: Optional[Path] = None, metrics: Optional[Sequence[Metric]] = None) -> Dict[str, Any]:
    """``{label: value}`` for every metric whose report exists (missing -> None)."""
    # Imported here: ``paths`` fixes the data directory at import, and the CLI sets it first.
    from hawk_research.data import paths

    root = Path(data_dir) if data_dir is not None else paths.RESEARCH_DIR
    cache: Dict[str, Any] = {}
    out: Dict[str, Any] = {}
    for label, rel, path, _tol in METRICS if metrics is None else metrics:
        f = root / rel
        if rel not in cache:
            cache[rel] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        try:
            out[label] = _get(cache[rel], path) if cache[rel] is not None else None
        except (KeyError, IndexError, TypeError):
            out[label] = None
    return out


def _same(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is b
    if tol == EXACT:
        return a == b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


def run(
    results_dir: Path, only: Optional[Sequence[str]] = None, metrics: Optional[Sequence[Metric]] = None
) -> Dict[str, Any]:
    metrics = METRICS if metrics is None else metrics
    committed = json.loads((results_dir / "headline_metrics.json").read_text(encoding="utf-8"))["metrics"]
    regenerated = extract(metrics=metrics)
    rows = []
    for label, rel, _path, tol in metrics:
        if only and not any(o in rel or o in label for o in only):
            continue
        new, old = regenerated.get(label), committed.get(label)
        if new is None:
            status = "MISSING"
        else:
            status = "OK" if _same(new, old, tol) else "DIFF"
        rows.append(
            {
                "label": label,
                "report": rel,
                "committed": old,
                "regenerated": new,
                "tolerance": tol,
                "status": status,
            }
        )
    ok = all(r["status"] == "OK" for r in rows if r["status"] != "MISSING")
    return {"ok": ok and any(r["status"] == "OK" for r in rows), "rows": rows}


def format_table(report: Dict[str, Any]) -> str:
    lines = [f"{'status':<8}{'metric':<46}{'committed':>18}{'regenerated':>18}  tol"]
    for r in report["rows"]:

        def fmt(v: Any) -> str:
            return f"{v:.6g}" if isinstance(v, float) else str(v)

        lines.append(
            f"{r['status']:<8}{r['label']:<46}{fmt(r['committed']):>18}{fmt(r['regenerated']):>18}  {r['tolerance']:g}"
        )
    n = {s: sum(1 for r in report["rows"] if r["status"] == s) for s in ("OK", "DIFF", "MISSING")}
    lines.append(
        f"\n{n['OK']} ok, {n['DIFF']} differ, {n['MISSING']} not regenerated (run those stages first)"
    )
    return "\n".join(lines)


def write_committed(results_dir: Path, metrics: Optional[Sequence[Metric]] = None) -> Path:
    """Update ``results/headline_metrics.json`` with ``metrics`` from the current data directory.

    Labels of the other table (main pipeline or side studies) already in the file are kept.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "headline_metrics.json"
    kept = json.loads(out.read_text(encoding="utf-8"))["metrics"] if out.exists() else {}
    doc = {
        "_note": "Headline numbers extracted from a full run; `hawk-research verify` compares against these.",
        "tolerances": {"exact": EXACT, "mean_return": RET, "statistic": STAT},
        "metrics": {**kept, **extract(metrics=metrics)},
    }
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return out
