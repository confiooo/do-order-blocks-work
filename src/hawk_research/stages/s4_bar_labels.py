"""Stage 4: forward labels and matching covariates for every 15m bar.

Labels are computed for all bars, not only zone bars: the null tests compare
zones with the other bars, and matching needs the detector's quantities on bars
that did not qualify.

Writes ``labels/15m/<symbol>.parquet``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import as_completed
from datetime import UTC, datetime
from typing import Dict, List, Optional

import numpy as np

from hawk_research.data import loader, paths, universe
from hawk_research.features import point_in_time as feat
from hawk_research.labels import outcomes as lab
from hawk_research.parallel import process_pool

log = logging.getLogger(__name__)

# Covariates persisted for every bar so the matched-control baseline can be
# built without recomputing the full feature matrix over 5.7M rows.
MATCH_COLUMNS = (
    "det_volume_ratio",
    "det_wick_ratio",
    "det_distance_to_swing",
    "det_distance_to_price",
    "det_zone_size_pct",
    "det_direction",
    "rv_24h",
    "atr_14_pct",
    "close_position_in_range",
    "wick_asymmetry",
    "body_to_range",
)

OUT_DIR = paths.LABELS_DIR / "15m"


def process(symbol: str) -> Dict:
    bars = loader.load_ohlcv(symbol)
    if bars.empty:
        return {"symbol": symbol, "rows": 0}
    out = lab.per_bar_labels(bars)

    cov = {}
    for block in (
        feat.detector_features(bars),
        feat.volatility_features(bars),
        feat.bar_shape_features(bars),
        feat.range_position_features(bars),
    ):
        cov.update(block)
    for col in MATCH_COLUMNS:
        if col in cov:
            out[col] = np.asarray(cov[col], dtype=np.float32)
    out["close"] = bars["close"].to_numpy(dtype=np.float32)
    out["symbol"] = symbol

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{symbol}.parquet", index=False, compression="zstd")
    return {
        "symbol": symbol,
        "rows": int(len(out)),
        "with_fwd_24h": int(out["fwd_ret_24h"].notna().sum()),
        "with_fwd_48h": int(out["fwd_ret_48h"].notna().sum()),
        "gap_masked_bars": int((out["fwd_bars_available"] < lab.MAX_HORIZON_BARS).sum()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    symbols = universe.available("round1")
    t0 = time.time()
    records: List[Dict] = []
    with process_pool(args.workers) as pool:
        futures = {pool.submit(process, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            records.append(rec)
            log.info(
                f"{i}/{len(symbols)} {rec['symbol']} rows={rec['rows']} fwd48h={rec.get('with_fwd_48h')}"
            )

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "symbols": len(records),
        "total_bars": int(sum(r["rows"] for r in records)),
        "bars_with_48h_forward": int(sum(r.get("with_fwd_48h", 0) for r in records)),
        "bars_masked_by_gaps_or_edge": int(sum(r.get("gap_masked_bars", 0) for r in records)),
        "return_horizons": lab.RETURN_HORIZONS,
        "barrier_levels": list(lab.BARRIER_LEVELS),
        "barrier_pairs": [list(p) for p in lab.BARRIER_PAIRS],
        "barrier_horizons": lab.BARRIER_HORIZONS,
        "ambiguity_convention": (
            "When the up and down barriers are both inside one 15m bar's range, "
            "the ordering is unknowable from 15m data. Those cases are flagged "
            "and resolved as the DOWN barrier first (the pessimistic choice for a long)."
        ),
        "elapsed_s": round(time.time() - t0, 1),
        "per_symbol": sorted(records, key=lambda r: r["symbol"]),
    }
    (paths.LABELS_DIR / "bar_labels_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info(json.dumps({k: v for k, v in summary.items() if k != "per_symbol"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
