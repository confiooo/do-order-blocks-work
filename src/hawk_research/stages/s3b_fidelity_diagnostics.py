"""Stage 3b: explain each live zone the replay did not reproduce.

The expected cause is short history. ``detect_order_blocks`` takes its lookback
from ``len(candles)``, and for a newly listed symbol the live collector's first
call got fewer than 400 bars, so it ran with a lookback of
``int(len(candles) * 0.15)`` instead of 60. The replay does not emit those, so
they show up here as misses inside the 400-bar warm-up.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import pandas as pd

from hawk_research.data import live_snapshot, loader, paths
from hawk_research.replay import round1 as replay

log = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    zones = pd.read_parquet(paths.ZONES_DIR / "replay_zones.parquet")
    live = live_snapshot.load_zones()

    rep_keys = set(zip(zones["symbol"], zones["source_open_time"].astype("int64")))
    live["_key"] = list(zip(live["symbol"], live["source_open_time"].astype("int64")))
    missed = live[~live["_key"].isin(rep_keys)].copy()

    attribution: List[Dict] = []
    for symbol, grp in missed.groupby("symbol"):
        bars = loader.load_ohlcv(symbol, columns=["open_time"])
        pos = {int(t): i for i, t in enumerate(bars["open_time"].tolist())}
        n_bars = len(bars)
        for row in grp.itertuples(index=False):
            idx = pos.get(int(row.source_open_time))
            if idx is None:
                cause = "bar absent from backfilled history"
            elif n_bars < replay.LIVE_WINDOW:
                cause = "symbol history shorter than the 400-bar live window"
            elif idx < replay.WARMUP_BARS:
                cause = "inside the 400-bar warm-up (live ran a shorter lookback here)"
            else:
                cause = "UNEXPLAINED"
            attribution.append(
                {
                    "symbol": symbol,
                    "source_open_time": int(row.source_open_time),
                    "bar_index": idx,
                    "symbol_bar_count": n_bars,
                    "live_detection_lag_bars": row.detection_lag_bars,
                    "implied_live_lookback": (int(min(n_bars, replay.LIVE_WINDOW) * replay.LOOKBACK_RATIO)),
                    "cause": cause,
                }
            )

    att = pd.DataFrame(attribution)
    summary = {
        "n_live_zones": int(len(live)),
        "n_missed": int(len(missed)),
        "recall": float(1 - len(missed) / len(live)),
        "by_cause": att["cause"].value_counts().to_dict() if len(att) else {},
        "by_symbol": att.groupby("symbol")
        .agg(
            n=("cause", "size"),
            bars=("symbol_bar_count", "max"),
            implied_lookback=("implied_live_lookback", "max"),
        )
        .reset_index()
        .to_dict("records")
        if len(att)
        else [],
        "unexplained": att[att["cause"] == "UNEXPLAINED"].to_dict("records") if len(att) else [],
    }
    out = paths.ZONES_DIR / "fidelity_misses.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
