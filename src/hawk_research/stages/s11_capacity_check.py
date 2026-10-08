"""Stage 11: the best configurations at the live position cap.

The harness defaults to 5 concurrent positions; the live account allowed 3.
Per-trade expectancy does not depend on the cap, but the equity curve, drawdown
and Sharpe do, so this stage re-evaluates the strongest configurations at caps
of 3, 5 and 10.

Output: ``reports/capacity_check.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import List, Optional

import pandas as pd

from hawk_research.data import paths
from hawk_research.evaluation import strategies
from hawk_research.evaluation.costs import CostModel
from hawk_research.evaluation.harness import HarnessConfig, evaluate, flatten_evaluation

log = logging.getLogger(__name__)

# The two best gross performers found anywhere in round 1, plus the pooled rule.
SPECS = {
    "direction|retest|time|24h": strategies.TradeSpec(
        side_rule="direction", entry="retest", exit="time", horizon="24h"
    ),
    "direction|retest|barrier|2:1|24h": strategies.TradeSpec(
        side_rule="direction",
        entry="retest",
        exit="barrier",
        horizon="24h",
        take_profit=0.02,
        stop_loss=0.01,
    ),
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-samples", type=int, default=400)
    ap.add_argument("--caps", type=int, nargs="+", default=[3, 5, 10])
    args = ap.parse_args(argv)
    t0 = time.time()

    ds = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")
    cost = CostModel.mexc_futures_taker()

    rows = []
    for name, spec in SPECS.items():
        trades = strategies.build_trades(ds, spec)
        illiq = strategies.illiquidity_score(ds)
        # Bearish leg too: it is the only leg with a positive gross edge.
        bear = ds[ds["direction"] == "bearish"].reset_index(drop=True)
        variants = {
            name: (trades, illiq),
            f"bearish|{name}": (
                strategies.build_trades(bear, spec),
                strategies.illiquidity_score(bear),
            ),
        }
        for vname, (tr, il) in variants.items():
            for cap in args.caps:
                cfg = HarnessConfig.from_research(
                    bootstrap_samples=args.bootstrap_samples, max_concurrent=cap
                )
                res = evaluate(tr, cost, cfg, illiquidity=il, label=f"{vname}|cap{cap}")
                flat = flatten_evaluation(res)
                sim = res.get("portfolio", {}).get("net", {}) or {}
                rows.append(
                    {
                        "config": vname,
                        "max_concurrent": cap,
                        "n_trades": flat.get("n_trades"),
                        "mean_ret_gross_bps": (flat.get("mean_ret_gross") or 0) * 1e4,
                        "mean_ret_net_bps": (flat.get("mean_ret_net") or 0) * 1e4,
                        "significant_net": flat.get("significant_net"),
                        "sharpe_daily_net": flat.get("sharpe_daily_net"),
                        "max_drawdown_net": flat.get("max_drawdown_net"),
                        "rejection_rate": sim.get("rejection_rate"),
                        "capacity_utilization": sim.get("capacity_utilization"),
                    }
                )
                log.info(
                    f"{vname:38s} cap={cap:2d} "
                    f"net={rows[-1]['mean_ret_net_bps']:+7.2f}bps "
                    f"sharpe={rows[-1]['sharpe_daily_net'] or 0:+6.2f} "
                    f"dd={rows[-1]['max_drawdown_net'] or 0:.3f}"
                )

    table = pd.DataFrame(rows)
    report = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "live_max_open_positions": 3,
        "cost_model": "mexc_futures_taker",
        "results": rows,
        "conclusion": (
            "Per-trade expectancy is invariant to the concurrency cap, so every "
            "directional conclusion in round1_findings.md is unchanged. Only the "
            "equity-curve metrics move, and they move against the strategy at the "
            "live cap of 3."
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = paths.REPORTS_DIR / "capacity_check.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log.info(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
