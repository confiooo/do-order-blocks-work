"""Stage 5: assemble the analysis-ready zone dataset.

Joins, per replayed zone:

* the detector's own frozen snapshot (from stage 3),
* the full point-in-time feature matrix at the decision bar,
* the causal forward labels at the decision bar (stage 4),
* the live-equivalent lifecycle: retest, hypothetical entry, MFE/MAE,
* forward labels anchored on the retest entry,
* cross-sectional ranks against the rest of the universe at the same bar,
* the survivorship-aware point-in-time universe rank (stage 2b), if present.

Also cross-checks the recomputed lifecycle against the live ``zones`` table on
the overlap window. If the recomputed retest/MFE/MAE disagree with production,
the forward labels are measuring something other than what the live system
measures, and every downstream number would be about a different strategy.

Writes ``data/research/features/zone_dataset.parquet``.
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

from hawk_research.constants import BAR_MS
from hawk_research.data import live_snapshot, loader, paths
from hawk_research.features import point_in_time as feat
from hawk_research.labels import outcomes as lab
from hawk_research.parallel import process_pool

log = logging.getLogger(__name__)

XS_COLUMNS = ("trend_ret_4h", "trend_ret_24h", "quote_volume_24h", "quote_volume_30d", "rv_24h")

PIT_COLUMNS = {
    "qv_1d": "pit_qv_prev_1d",
    "qv_30d": "pit_qv_prev_30d",
    "rank_qv_1d": "pit_rank_prev_1d",
    "rank_qv_30d": "pit_rank_prev_30d",
    "listed_days": "pit_listed_days",
    "n_listed": "pit_n_listed",
}

_BTC: Optional[pd.DataFrame] = None


def _init(btc_bytes: bytes) -> None:
    global _BTC
    _BTC = pd.read_parquet(io.BytesIO(btc_bytes))


def process_symbol(symbol: str, zone_blob: bytes) -> bytes:
    """One symbol's zone rows: detector snapshot, features, labels and lifecycle."""
    return _symbol_rows(symbol, zone_blob)[0].to_parquet(index=False)


def _symbol_rows(symbol: str, zone_blob: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The symbol's zone rows, and its per-bar values that get ranked across the universe."""
    zones = pd.read_parquet(io.BytesIO(zone_blob))
    bars = loader.load_ohlcv(symbol)
    labels = pd.read_parquet(paths.LABELS_DIR / "15m" / f"{symbol}.parquet")
    fmat = feat.build_symbol_features(bars, _BTC)

    merged = (
        zones.merge(fmat, left_on="source_open_time", right_on="open_time", how="left")
        .drop(columns=["open_time"])
        .merge(
            labels.drop(columns=[c for c in ("symbol", "close") if c in labels.columns]),
            left_on="source_open_time",
            right_on="open_time",
            how="left",
        )
        .drop(columns=["open_time"])
    )

    life = lab.zone_lifecycle(bars, zones)
    merged = merged.merge(life, on=["symbol", "source_open_time"], how="left")

    entry = lab.entry_anchored_labels(bars, life)
    if not entry.empty:
        merged = merged.merge(entry, on=["symbol", "source_open_time"], how="left")

    merged = lab.add_barrier_pair_labels(merged)
    merged = lab.add_barrier_pair_labels(merged, prefix="entry_")

    xs = fmat[["open_time"] + [c for c in XS_COLUMNS if c in fmat.columns]].copy()
    xs["symbol"] = symbol
    return merged, xs


def _process_symbol_with_xs(symbol: str, zone_blob: bytes) -> tuple[bytes, bytes]:
    rows, xs = _symbol_rows(symbol, zone_blob)
    return rows.to_parquet(index=False), xs.to_parquet(index=False)


def attach_pit_ranks(zones: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Join the point-in-time universe ranks of the PREVIOUS UTC day.

    A day's own quote volume is not known until that day closes, so ranking on it would peek.
    """
    prev = pit[["date", "symbol", *PIT_COLUMNS]].rename(columns=PIT_COLUMNS)
    prev["join_date"] = prev.pop("date") + pd.Timedelta(days=1)
    day = pd.to_datetime(zones["source_open_time"], unit="ms", utc=True).dt.normalize()
    return (
        zones.assign(_date=day)
        .merge(prev, left_on=["_date", "symbol"], right_on=["join_date", "symbol"], how="left")
        .drop(columns=["_date", "join_date"])
    )


def live_outcome_crosscheck(zones: pd.DataFrame) -> Dict:
    """Recomputed lifecycle vs the live zones table, on the overlap."""
    live = live_snapshot.load_zones()
    m = live.merge(zones, on=["symbol", "source_open_time"], how="inner", suffixes=("_live", "_rep"))
    if m.empty:
        return {"n": 0}

    live_retested = m["status"].isin(["entered", "complete"])
    live_resolved = m["status"].isin(["entered", "complete", "no_retest"])
    rep_retested = m["retested"].fillna(False).astype(bool)

    res = m[live_resolved]
    agree = res["status"].isin(["entered", "complete"]).to_numpy() == res["retested"].fillna(False).to_numpy()
    out: Dict[str, object] = {
        "n_overlap_zones": int(len(m)),
        "n_live_resolved": int(live_resolved.sum()),
        "retest_flag_agreement": float(agree.mean()) if len(res) else None,
        "live_retest_rate": float(live_retested[live_resolved].mean()) if live_resolved.any() else None,
        "replay_retest_rate": float(rep_retested[live_resolved].mean()) if live_resolved.any() else None,
    }

    both = m[live_retested & rep_retested]
    if len(both):
        live_retest_ms = (
            pd.to_datetime(both["retest_at"], utc=True, errors="coerce")
            .astype("datetime64[ms, UTC]")
            .astype("int64")
        )
        same_bar = live_retest_ms.to_numpy() == both["retest_open_time"].to_numpy()
        out["retest_bar_identical"] = float(same_bar.mean())
        out["entry_price_identical"] = float(
            np.isclose(
                both["hypothetical_entry_price_live"].to_numpy(dtype=float),
                both["hypothetical_entry_price_rep"].to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-12,
            ).mean()
        )
    excursions = {}
    for h in lab.MFE_MAE_HORIZONS:
        for kind in ("mfe", "mae"):
            lcol, rcol = f"{kind}_{h}h_live", f"{kind}_{h}h_rep"
            if lcol not in m.columns or rcol not in m.columns:
                continue
            sub = m[[lcol, rcol]].dropna()
            if sub.empty:
                continue
            a = sub[lcol].to_numpy(dtype=float)
            b = sub[rcol].to_numpy(dtype=float)
            excursions[f"{kind}_{h}h"] = {
                "n_both_present": int(len(sub)),
                "identical": int(np.isclose(a, b, rtol=1e-9, atol=1e-9).sum()),
                "within_1e-4": int(np.isclose(a, b, rtol=0, atol=1e-4).sum()),
                "max_abs_diff": float(np.max(np.abs(a - b))),
                "corr": float(np.corrcoef(a, b)[0, 1]) if len(sub) > 2 else None,
            }
    out["excursions"] = excursions
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="", help="zone population suffix, e.g. smc")
    args = parser.parse_args(argv)
    sfx = f"_{args.tag}" if args.tag else ""

    paths.ensure_dirs()
    t0 = time.time()
    zones = pd.read_parquet(paths.ZONES_DIR / f"replay_zones{sfx}.parquet")
    btc = loader.load_ohlcv("BTCUSDT")
    btc_bytes = btc.to_parquet(index=False)
    symbols = sorted(zones["symbol"].unique())

    frames: List[pd.DataFrame] = []
    xs_parts: Dict[str, pd.DataFrame] = {}
    with process_pool(args.workers, initializer=_init, initargs=(btc_bytes,)) as pool:
        futures = {
            pool.submit(_process_symbol_with_xs, s, zones[zones["symbol"] == s].to_parquet(index=False)): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            rows_blob, xs_blob = fut.result()
            frames.append(pd.read_parquet(io.BytesIO(rows_blob)))
            xs_parts[futures[fut]] = pd.read_parquet(io.BytesIO(xs_blob))
            log.info(f"{i}/{len(symbols)} {futures[fut]} rows={len(frames[-1])}")

    ds = pd.concat(frames, ignore_index=True)

    log.info("cross-sectional ranks")
    panel = pd.concat([xs_parts[s] for s in symbols], ignore_index=True)
    xs = feat.cross_sectional_ranks(panel, [c for c in XS_COLUMNS if c in panel.columns])
    ds = ds.merge(
        xs, left_on=["source_open_time", "symbol"], right_on=["open_time", "symbol"], how="left"
    ).drop(columns=["open_time"])
    # Consolidate once after the wide merge; otherwise each insert below warns about fragmentation.
    ds = ds.copy()

    ds["bars_since_prev_zone"] = feat.time_since_previous_zone(ds)

    pit_path = paths.UNIVERSE_DIR / "pit_universe.parquet"
    if pit_path.exists():
        log.info("point-in-time universe join")
        ds = attach_pit_ranks(ds, pd.read_parquet(pit_path))

    ds["decision_time_ms"] = ds["source_open_time"].astype("int64") + BAR_MS
    ds["mode"] = "replay"
    ds["detection_lag_bars"] = 0
    ds = ds.sort_values(["source_open_time", "symbol"], kind="mergesort").reset_index(drop=True)

    out = paths.FEATURES_DIR / f"zone_dataset{sfx}.parquet"
    ds.to_parquet(out, index=False, compression="zstd")

    # The live zones table exists only for the round-1 detector.
    crosscheck = live_outcome_crosscheck(ds) if not args.tag and paths.has_live_snapshot() else None
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "path": paths.rel(out),
        "n_zones": int(len(ds)),
        "n_symbols": int(ds["symbol"].nunique()),
        "n_columns": int(ds.shape[1]),
        "period_start_iso": str(pd.to_datetime(ds["source_open_time"].min(), unit="ms", utc=True)),
        "period_end_iso": str(pd.to_datetime(ds["source_open_time"].max(), unit="ms", utc=True)),
        "n_live_equivalent": int(ds["live_equivalent"].sum()),
        "n_retested": int(ds["retested"].fillna(False).sum()),
        "n_complete_24h": int((ds["lifecycle_status"] == "complete").sum()),
        "n_with_fwd_48h": int(ds["fwd_ret_48h"].notna().sum()),
        "lifecycle_status_counts": ds["lifecycle_status"].value_counts().to_dict(),
        "live_outcome_crosscheck": crosscheck,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (paths.FEATURES_DIR / f"zone_dataset{sfx}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
