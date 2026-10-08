"""Stage 10: the bearish leg on its own.

The pooled ``direction`` rule averages a bearish leg that is right 51.4% of the
time with a bullish leg that is right 47.0% of the time. This stage checks
whether the bearish leg alone clears costs, through the same harness (holdout,
concurrency cap, cross-sectional bootstrap) and logging its trials to the same
ledger so they count towards the deflated Sharpe.

Output: ``reports/bearish_only.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from hawk_research import config as research_config
from hawk_research.data import paths
from hawk_research.evaluation import strategies
from hawk_research.evaluation.costs import CostModel
from hawk_research.evaluation.harness import HarnessConfig, evaluate, flatten_evaluation
from hawk_research.ledger import TrialLedger
from hawk_research.stats.resampling import cross_sectional_bootstrap_ci

log = logging.getLogger(__name__)

COSTS = {
    "zero": CostModel.zero(),
    "mexc_futures_taker": CostModel.mexc_futures_taker(),
    "binance_spot_taker": CostModel.binance_spot_taker(),
}


def leg_stats(ds: pd.DataFrame, mask: np.ndarray, horizon: str, n_samples: int) -> dict:
    sign = np.where(ds["direction"].to_numpy() == "bullish", 1.0, -1.0)
    r = sign * ds[f"fwd_ret_{horizon}"].to_numpy(dtype=float)
    t = ds["decision_time_ms"].to_numpy(dtype=np.int64)
    ok = mask & np.isfinite(r)
    ci = cross_sectional_bootstrap_ci(
        r[ok], t[ok], n_samples=n_samples, seed=research_config.get().seeds.strategy_bootstrap
    )
    return {
        "n": int(ok.sum()),
        "gross_mean": float(np.mean(r[ok])),
        "ci": [ci.get("ci_low"), ci.get("ci_high")],
        "win_rate": float(np.mean(r[ok] > 0)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    _CFG = research_config.get()
    ap.add_argument("--bootstrap-samples", type=int, default=_CFG.evaluation.bootstrap_samples)
    ap.add_argument("--holdout-fraction", type=float, default=_CFG.evaluation.holdout_fraction)
    args = ap.parse_args(argv)
    t0 = time.time()

    ds = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")
    ds = ds.sort_values("decision_time_ms").reset_index(drop=True)
    cut = ds["decision_time_ms"].quantile(1.0 - args.holdout_fraction)
    dev = ds[ds["decision_time_ms"] <= cut]
    hold = ds[ds["decision_time_ms"] > cut]

    bear = (ds["direction"] == "bearish").to_numpy()
    bull = ~bear
    report = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "holdout_start": str(pd.to_datetime(cut, unit="ms", utc=True)),
        "legs": {},
        "leg_out_of_sample": {},
        "strategies": {},
    }
    for h in ("1h", "4h", "24h", "48h"):
        report["legs"][f"bearish_{h}"] = leg_stats(ds, bear, h, args.bootstrap_samples)
        report["legs"][f"bullish_{h}"] = leg_stats(ds, bull, h, args.bootstrap_samples)

    # Development vs holdout: an edge that only shows up in development is a fit.
    for name, frame in (("development", dev), ("holdout", hold)):
        m = (frame["direction"] == "bearish").to_numpy()
        report["leg_out_of_sample"][f"bearish_4h_{name}"] = leg_stats(
            frame.reset_index(drop=True), m, "4h", args.bootstrap_samples
        )
        report["leg_out_of_sample"][f"bearish_24h_{name}"] = leg_stats(
            frame.reset_index(drop=True), m, "24h", args.bootstrap_samples
        )

    cfg = HarnessConfig.from_research(bootstrap_samples=args.bootstrap_samples)
    ledger = TrialLedger(
        name_prefix="s10-bearish-only",
        description="Bearish-leg-only evaluation of the replayed zone population",
    )

    bear_only = ds[ds["direction"] == "bearish"].reset_index(drop=True)
    illiq_bear = strategies.illiquidity_score(bear_only)
    for entry in ("close", "retest"):
        for exitmode, tp, sl in (("time", None, None), ("barrier", 0.02, 0.01)):
            spec = strategies.TradeSpec(
                side_rule="direction",
                entry=entry,
                exit=exitmode,
                horizon="24h",
                take_profit=tp,
                stop_loss=sl,
            )
            trades = strategies.build_trades(bear_only, spec)
            for cname, cost in COSTS.items():
                key = f"bearish|{entry}|{exitmode}|{tp or '-'}/{sl or '-'}|{cname}"
                res = evaluate(trades, cost, cfg, illiquidity=illiq_bear, label=key)
                flat = flatten_evaluation(res)
                ledger.log_trial({**spec.describe(), "leg": "bearish", "cost_model": cname}, flat, label=key)
                report["strategies"][key] = {
                    k: flat.get(k)
                    for k in (
                        "n_trades",
                        "mean_ret_gross",
                        "mean_ret_net",
                        "ci_low_cross",
                        "ci_high_cross",
                        "significant_net",
                        "win_rate_net",
                        "profit_factor_net",
                        "sharpe_daily_net",
                        "max_drawdown_net",
                    )
                }
                report["strategies"][key].update(
                    strategies.ambiguity_sensitivity(trades, spec, flat.get("mean_ret_net"))
                )
    report["n_trials_logged_this_stage"] = len(ledger.trials())
    report["elapsed_s"] = round(time.time() - t0, 1)

    out = paths.REPORTS_DIR / "bearish_only.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(report, indent=2, default=str))
    log.info(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
