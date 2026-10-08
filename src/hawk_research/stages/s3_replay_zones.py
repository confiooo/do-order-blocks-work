"""Stage 3: replay the unmodified live detector over the full backfilled history.

Produces ``data/research/zones/replay_zones.parquet``: every detector hit on
every symbol, with the live gating flags attached (``btc_stable`` /
``live_equivalent``, ``passes_signal_gate``, ``would_alert``) so the pre-gate
population stays available alongside the live-equivalent one.

Also runs the fidelity check: re-detect over the live overlap window and compare
field-by-field against the production ``zones`` table. That comparison is the
only evidence that the replayed population is the same object as the live book.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from concurrent.futures import as_completed
from datetime import UTC, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.data import backfill, live_snapshot, loader, paths, universe
from hawk_research.parallel import process_pool
from hawk_research.replay import round1 as replay

log = logging.getLogger(__name__)

COMPARE_FIELDS = (
    "direction",
    "zone_high",
    "zone_low",
    "zone_midpoint",
    "zone_size_pct",
    "volume_ratio",
    "wick_ratio",
    "distance_to_swing",
    "distance_to_price",
)

_BTC_RANGE: Dict[int, float] = {}
_PARAMS: Optional[replay.DetectorParams] = None


def _init_worker(btc_range: Dict[int, float], params: replay.DetectorParams) -> None:
    global _BTC_RANGE, _PARAMS
    _BTC_RANGE = btc_range
    _PARAMS = params


def _replay_one(symbol: str) -> bytes:
    df = loader.load_ohlcv(symbol)
    candles = backfill.to_candles(df)
    out = replay.replay_symbol(symbol, candles, _PARAMS, _BTC_RANGE)
    return out.to_parquet(index=False) if not out.empty else b""


def _symbols() -> List[str]:
    return universe.available("round1")


def _live_zones() -> pd.DataFrame:
    return live_snapshot.load_zones()


def fidelity_report(replayed: pd.DataFrame, live: pd.DataFrame) -> Dict:
    """How much of the live zones table does the replay reproduce, and why not?"""
    lo, hi = int(live["source_open_time"].min()), int(live["source_open_time"].max())
    rep = replayed[(replayed["source_open_time"] >= lo) & (replayed["source_open_time"] <= hi)].copy()
    rep_live_eq = rep[rep["live_equivalent"]].copy()

    key = ["symbol", "source_open_time"]
    merged = live.merge(rep_live_eq, on=key, how="left", suffixes=("_live", "_rep"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    unmatched = merged[merged["_merge"] == "left_only"].copy()

    field_stats: Dict[str, Dict] = {}
    exact = np.ones(len(matched), dtype=bool)
    for f in COMPARE_FIELDS:
        a, b = matched[f"{f}_live"], matched[f"{f}_rep"]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            ok = (a.astype(str) == b.astype(str)).to_numpy()
            maxdiff = None
        else:
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            ok = np.isclose(av, bv, rtol=1e-9, atol=1e-12)
            maxdiff = float(np.nanmax(np.abs(av - bv))) if len(av) else 0.0
        field_stats[f] = {
            "equal": int(ok.sum()),
            "unequal": int((~ok).sum()),
            "max_abs_diff": maxdiff,
        }
        exact &= ok

    # Misses that were detected but removed by the BTC gate.
    rep_all_keys = set(map(tuple, rep[key].to_numpy().tolist()))
    unmatched_keys = list(map(tuple, unmatched[key].to_numpy().tolist()))
    detected_but_gated = sum(1 for k in unmatched_keys if k in rep_all_keys)
    have_bars = {s for s in rep["symbol"].unique()}
    no_history = sum(1 for k in unmatched_keys if k[0] not in have_bars)

    # Extra replayed zones the live book does not contain (universe membership).
    live_keys = set(map(tuple, live[key].to_numpy().tolist()))
    extra = rep_live_eq[
        ~pd.Series(list(map(tuple, rep_live_eq[key].to_numpy().tolist()))).isin(live_keys).to_numpy()
    ]
    live_symbols = set(live["symbol"].unique())
    extra_on_unscanned_symbols = int((~extra["symbol"].isin(live_symbols)).sum())

    return {
        "live_window_start_iso": backfill._iso(lo),
        "live_window_end_iso": backfill._iso(hi),
        "n_live_zones": int(len(live)),
        "n_replayed_in_window_pre_gate": int(len(rep)),
        "n_replayed_in_window_live_equivalent": int(len(rep_live_eq)),
        "matched_on_key": int(len(matched)),
        "matched_all_fields_exact": int(exact.sum()),
        "recall_key": float(len(matched) / len(live)) if len(live) else None,
        "recall_exact": float(exact.sum() / len(live)) if len(live) else None,
        "field_stats": field_stats,
        "unmatched_live_zones": int(len(unmatched)),
        "unmatched_detected_but_btc_gated_in_replay": int(detected_but_gated),
        "unmatched_symbol_absent_from_backfill": int(no_history),
        "unmatched_by_symbol": unmatched["symbol"].value_counts().head(20).to_dict(),
        "extra_replayed_not_in_live": int(len(extra)),
        "extra_on_symbols_live_never_scanned": extra_on_unscanned_symbols,
        "note": (
            "Recall is the meaningful number. Extra replayed zones are expected: "
            "live only scans the top-40 symbols by 24h quote volume in each cycle, "
            "while the replay evaluates every backfilled symbol on every bar."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    params = replay.DetectorParams.from_research_config()
    log.info(
        f"effective lookback = {replay.EFFECTIVE_LOOKBACK} bars "
        f"(window {replay.LIVE_WINDOW} x ratio {replay.LOOKBACK_RATIO})"
    )
    log.info(f"params: {params}")

    btc_range = replay.checked_btc_range_series(backfill.to_candles(loader.load_ohlcv("BTCUSDT")))

    symbols = _symbols()
    frames: List[pd.DataFrame] = []
    t0 = time.time()
    with process_pool(args.workers, initializer=_init_worker, initargs=(btc_range, params)) as pool:
        futures = {pool.submit(_replay_one, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), start=1):
            sym = futures[fut]
            blob = fut.result()
            if blob:
                frames.append(pd.read_parquet(io.BytesIO(blob)))
            n = len(frames[-1]) if blob else 0
            log.info(f"{i}/{len(symbols)} {sym} zones={n}")

    zones = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    zones = zones.sort_values(["source_open_time", "symbol"], kind="mergesort").reset_index(drop=True)
    zones["zone_uid"] = zones["symbol"] + "@" + zones["source_open_time"].astype(str)
    out = paths.ZONES_DIR / "replay_zones.parquet"
    zones.to_parquet(out, index=False, compression="zstd")
    log.info(f"{len(zones)} detections -> {out}  ({time.time() - t0:.0f}s)")

    # Compare with the live bot's zones table.
    if paths.has_live_snapshot():
        fid = {**fidelity_report(zones, _live_zones()), "live_zones_source": live_snapshot.source()}
    else:
        fid = {
            "skipped": "no live zones (snapshot/live_zones.parquet missing); "
            "see docs/PROVENANCE.md for the published fidelity result"
        }
        log.info("fidelity check skipped: no live snapshot")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "effective_lookback_bars": replay.EFFECTIVE_LOOKBACK,
        "live_window_bars": replay.LIVE_WINDOW,
        "warmup_bars": replay.WARMUP_BARS,
        "detector_params": params.__dict__,
        "n_symbols": int(zones["symbol"].nunique()),
        "n_detections_pre_gate": int(len(zones)),
        "n_live_equivalent": int(zones["live_equivalent"].sum()),
        "n_passes_signal_gate": int((zones["live_equivalent"] & zones["passes_signal_gate"]).sum()),
        "n_would_alert": int(zones["would_alert"].sum()),
        "period_start_iso": backfill._iso(int(zones["source_open_time"].min())),
        "period_end_iso": backfill._iso(int(zones["source_open_time"].max())),
        "by_direction": zones["direction"].value_counts().to_dict(),
        "fidelity": fid,
    }
    (paths.ZONES_DIR / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    log.info(json.dumps({k: v for k, v in summary.items() if k != "fidelity"}, indent=2, default=str))
    log.info(json.dumps(fid, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
