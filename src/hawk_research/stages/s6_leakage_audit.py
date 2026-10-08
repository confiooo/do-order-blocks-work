"""Stage 6: run the leakage and integrity audit and write the report.

Exits non-zero if any hard check fails, so it gates the downstream stages.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Dict, List, Optional

import pandas as pd

from hawk_research import config as research_config
from hawk_research.audit import leakage as audit
from hawk_research.data import loader, paths

log = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _CFG = research_config.get()
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=_CFG.seeds.leakage_audit)
    parser.add_argument("--tag", default="", help="zone population suffix, e.g. smc")
    args = parser.parse_args(argv)
    sfx = f"_{args.tag}" if args.tag else ""

    ds = pd.read_parquet(paths.FEATURES_DIR / f"zone_dataset{sfx}.parquet")
    symbols = sorted(ds["symbol"].unique())
    bars = {s: loader.load_ohlcv(s) for s in symbols}
    btc = loader.load_ohlcv("BTCUSDT")

    log.info("structural checks")
    structural = audit.structural_checks(ds, bars)
    log.info("gap masking")
    gaps = audit.gap_masking_check(ds)
    log.info("forward-fill contamination")
    ffill = audit.forward_fill_check(bars)
    log.info(f"feature truncation test ({args.samples} zones)")
    trunc = audit.truncation_test(ds, bars, btc, n_samples=args.samples, seed=args.seed)
    log.info(f"label truncation test ({args.samples} zones)")
    ltrunc = audit.label_truncation_test(ds, bars, n_samples=args.samples, seed=args.seed + 1)

    report: Dict = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_zones": int(len(ds)),
        "n_columns": int(ds.shape[1]),
        "structural": structural,
        "gap_masking": gaps,
        "forward_fill": ffill,
        "feature_truncation_test": trunc,
        "label_truncation_test": ltrunc,
    }
    hard_pass = (
        structural["passed"]
        and gaps.get("passed", True)
        and trunc.get("passed", False)
        and ltrunc.get("passed", False)
    )
    report["passed"] = bool(hard_pass)

    out = paths.REPORTS_DIR / f"leakage_audit{sfx}.json"
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps({k: v for k, v in report.items() if k != "forward_fill"}, indent=2, default=str))
    log.info(f"-> {out}  PASSED={hard_pass}")
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
