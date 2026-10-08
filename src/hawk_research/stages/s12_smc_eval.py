"""Stage 12: textbook order blocks after costs.

Runs the SMC zones (``zone_dataset_smc.parquet``) through the same harness, cost
models, concurrency cap, bootstrap and ledger as s10. Two variants, fixed in
advance:

- ``all``: every BOS-confirmed order block;
- ``fvg``: only those whose impulse leg left a fair value gap.

Each is traded in the BOS direction, entering at the BOS bar's close or on the
first retest (a limit at the near edge), with a 24h time exit or a 2:1 barrier.
Directions are reported separately, and every headline number is split into
development and the last-20% holdout.

Output: ``reports/smc_eval.json``.
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
VARIANTS = {
    "all": lambda ds: np.ones(len(ds), dtype=bool),
    "fvg": lambda ds: ds["has_fvg"].fillna(False).to_numpy(dtype=bool),
}
KEEP = (
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
    ap.add_argument("--tag", default="smc")
    ap.add_argument("--bootstrap-samples", type=int, default=_CFG.evaluation.bootstrap_samples)
    ap.add_argument("--holdout-fraction", type=float, default=_CFG.evaluation.holdout_fraction)
    args = ap.parse_args(argv)
    t0 = time.time()

    ds = pd.read_parquet(paths.FEATURES_DIR / f"zone_dataset_{args.tag}.parquet")
    ds = ds.sort_values("decision_time_ms").reset_index(drop=True)
    cut = ds["decision_time_ms"].quantile(1.0 - args.holdout_fraction)
    periods = {
        "development": (ds["decision_time_ms"] <= cut).to_numpy(),
        "holdout": (ds["decision_time_ms"] > cut).to_numpy(),
    }
    bull = (ds["direction"] == "bullish").to_numpy()
    report = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "holdout_start": str(pd.to_datetime(cut, unit="ms", utc=True)),
        "legs": {},
        "strategies": {},
    }
    for vname, vmask in VARIANTS.items():
        base = vmask(ds)
        for h in ("1h", "4h", "24h", "48h"):
            for leg, lmask in (("pooled", base), ("bullish", base & bull), ("bearish", base & ~bull)):
                report["legs"][f"{vname}|{leg}|{h}"] = leg_stats(ds, lmask, h, args.bootstrap_samples)
            for pname, pmask in periods.items():
                report["legs"][f"{vname}|pooled|{h}|{pname}"] = leg_stats(
                    ds, base & pmask, h, args.bootstrap_samples
                )

    cfg = HarnessConfig.from_research(bootstrap_samples=args.bootstrap_samples)
    ledger = TrialLedger(
        name_prefix=f"s12-{args.tag}-eval",
        description="Textbook SMC order blocks traded in the BOS direction",
    )
    for vname, vmask in VARIANTS.items():
        for pname, pmask in (("full", np.ones(len(ds), dtype=bool)), *periods.items()):
            sub = ds[vmask(ds) & pmask].reset_index(drop=True)
            illiq = strategies.illiquidity_score(sub)
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
                    trades = strategies.build_trades(sub, spec)
                    for cname, cost in COSTS.items():
                        key = f"{vname}|{pname}|{entry}|{exitmode}|{tp or '-'}/{sl or '-'}|{cname}"
                        res = evaluate(trades, cost, cfg, illiquidity=illiq, label=key)
                        flat = flatten_evaluation(res)
                        # Only full-sample runs are trials; the period splits
                        # re-read the same trial and must not inflate N.
                        if pname == "full":
                            ledger.log_trial(
                                {**spec.describe(), "variant": vname, "cost_model": cname},
                                flat,
                                label=key,
                            )
                        report["strategies"][key] = {k: flat.get(k) for k in KEEP}
                        report["strategies"][key].update(
                            strategies.ambiguity_sensitivity(trades, spec, flat.get("mean_ret_net"))
                        )
    report["n_trials_logged_this_stage"] = len(ledger.trials())
    report["elapsed_s"] = round(time.time() - t0, 1)

    out = paths.REPORTS_DIR / f"{args.tag}_eval.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(report, indent=2, default=str))
    log.info(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
