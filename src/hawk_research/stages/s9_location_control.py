"""Location control: are zones retested more often than ordinary bars?

About 78% of zones are revisited within 24h, but on 15m bars an arbitrary bar's
high-low band is also revisited often. This stage runs the same lifecycle
(``labels.zone_lifecycle``) on the matched control bars from s8, using each
control bar's high-low band as the zone and ``close > open`` as its direction,
and compares the two retest rates.

Output: ``reports/location_control.json``.
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
from hawk_research.labels import outcomes as lab
from hawk_research.stats.resampling import cross_sectional_bootstrap_ci

log = logging.getLogger(__name__)


def pseudo_zones(bars: pd.DataFrame, open_times: np.ndarray, symbol: str) -> pd.DataFrame:
    """A control bar dressed as a zone: its own high-low band is the band."""
    sub = bars[bars["open_time"].isin(open_times)]
    if sub.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "symbol": symbol,
            "source_open_time": sub["open_time"].to_numpy(dtype=np.int64),
            "zone_high": sub["high"].to_numpy(dtype=float),
            "zone_low": sub["low"].to_numpy(dtype=float),
            "direction": np.where(
                sub["close"].to_numpy(dtype=float) > sub["open"].to_numpy(dtype=float),
                "bullish",
                "bearish",
            ),
        }
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    _CFG = research_config.get()
    ap.add_argument("--bootstrap-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=_CFG.seeds.location_control)
    args = ap.parse_args(argv)
    t0 = time.time()

    # Reuse the controls s8 already drew, so the location claim and the return
    # claims are made against the identical control population.
    controls = pd.read_parquet(paths.LABELS_DIR / "matched_controls.parquet")
    rows = []
    symbols = sorted(controls["symbol"].unique().tolist())
    for n, symbol in enumerate(symbols, 1):
        bp = paths.ohlcv_path(symbol, "15m")
        if not bp.exists():
            continue
        bars = pd.read_parquet(bp)
        ot = controls.loc[controls["symbol"] == symbol, "open_time"].to_numpy(dtype=np.int64)
        pz = pseudo_zones(bars, ot, symbol)
        if pz.empty:
            continue
        rows.append(lab.zone_lifecycle(bars, pz))
        if n % 10 == 0:
            log.info(f"{n}/{len(symbols)} symbols")

    control_life = pd.concat(rows, ignore_index=True)
    ds = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")

    z_rate = float(ds["retested"].fillna(False).mean())
    c_rate = float(control_life["retested"].fillna(False).mean())

    # Cross-sectional bootstrap so the CI respects the fact that all 80 symbols
    # move together; a per-bar binomial interval would be far too narrow.
    z_ci = cross_sectional_bootstrap_ci(
        ds["retested"].fillna(False).to_numpy(dtype=float),
        ds["decision_time_ms"].to_numpy(dtype=np.int64),
        n_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    c_ci = cross_sectional_bootstrap_ci(
        control_life["retested"].fillna(False).to_numpy(dtype=float),
        control_life["source_open_time"].to_numpy(dtype=np.int64) + lab.BAR_MS,
        n_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    # Retest *speed* matters too: a zone that is only revisited after 20h is a
    # worse location than one revisited in 2h, even at the same hit rate.
    z_lag = ds.loc[ds["retested"] == True, "retest_lag_bars"].dropna()  # noqa: E712
    c_lag = control_life.loc[control_life["retested"] == True, "retest_lag_bars"].dropna()  # noqa: E712

    report = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "n_zones": int(len(ds)),
        "n_matched_controls": int(len(control_life)),
        "zone_retest_rate": z_rate,
        "zone_retest_ci": [z_ci.get("ci_low"), z_ci.get("ci_high")],
        "control_retest_rate": c_rate,
        "control_retest_ci": [c_ci.get("ci_low"), c_ci.get("ci_high")],
        "difference_pp": (z_rate - c_rate) * 100.0,
        "zone_median_retest_lag_bars": float(z_lag.median()) if len(z_lag) else None,
        "control_median_retest_lag_bars": float(c_lag.median()) if len(c_lag) else None,
        "zone_mean_retest_lag_bars": float(z_lag.mean()) if len(z_lag) else None,
        "control_mean_retest_lag_bars": float(c_lag.mean()) if len(c_lag) else None,
        "interpretation": (
            "If the two rates overlap, the zone is not a special location: price "
            "returns to an arbitrary matched bar's band just as readily, and the "
            "high retest rate is a property of 15m price oscillation, not of the "
            "detector."
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = paths.REPORTS_DIR / "location_control.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(report, indent=2, default=str))
    log.info(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
