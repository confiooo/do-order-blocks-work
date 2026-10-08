"""Pass R: replay the unmodified detector with the swing gate turned off.

Pre-registration §2.1: call ``detect_order_blocks`` with
``min_distance_from_high_low = 0.0`` over the same 15m history and the same
400-bar rolling window (60-bar lookback). Every other detector parameter stays
identical to live. Tag ``passes_swing_gate = (distance_to_swing >= 0.5)``.

The subset with ``passes_swing_gate == 1`` must be bit-identical to Pass P
(``replay_zones.parquet``) on the detector-produced fields. A mismatch voids S1.

Then attach the same causal features/labels used for ``zone_dataset.parquet``
by reusing ``s5_zone_dataset.process_symbol``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from concurrent.futures import as_completed
from dataclasses import replace
from datetime import UTC, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.data import backfill, loader, paths, universe
from hawk_research.labels import outcomes as lab
from hawk_research.parallel import process_pool
from hawk_research.replay import round1 as replay
from hawk_research.stages import s3_replay_zones as s3mod
from hawk_research.stages import s5_zone_dataset as s5mod

log = logging.getLogger(__name__)

IDENTITY_FIELDS = (
    "zone_high",
    "zone_low",
    "volume_ratio",
    "wick_ratio",
    "distance_to_swing",
)


def bit_identity(pass_r: pd.DataFrame, pass_p: pd.DataFrame) -> Dict:
    """Pass R ∩ swing-gate vs Pass P on overlapping (symbol, source_open_time)."""
    key = ["symbol", "source_open_time"]
    gated = pass_r.loc[pass_r["passes_swing_gate"].to_numpy().astype(bool)].copy()
    left = pass_p.merge(gated, on=key, how="outer", suffixes=("_p", "_r"), indicator=True)
    both = left[left["_merge"] == "both"].copy()
    p_only = left[left["_merge"] == "left_only"]
    r_only = left[left["_merge"] == "right_only"]

    field_stats: Dict[str, Dict] = {}
    exact = np.ones(len(both), dtype=bool)
    for f in IDENTITY_FIELDS:
        a = both[f"{f}_p"] if f"{f}_p" in both.columns else both[f]
        b = both[f"{f}_r"] if f"{f}_r" in both.columns else both[f]
        av = a.to_numpy(dtype=float)
        bv = b.to_numpy(dtype=float)
        ok = np.isclose(av, bv, rtol=1e-9, atol=1e-12)
        field_stats[f] = {
            "equal": int(ok.sum()),
            "unequal": int((~ok).sum()),
            "max_abs_diff": float(np.nanmax(np.abs(av - bv))) if len(av) else 0.0,
        }
        exact &= ok

    n_p, n_r_gate = int(len(pass_p)), int(len(gated))
    identical = bool(len(p_only) == 0 and len(r_only) == 0 and exact.all())
    return {
        "n_pass_p": n_p,
        "n_pass_r": int(len(pass_r)),
        "n_pass_r_swing_gate": n_r_gate,
        "n_overlap": int(len(both)),
        "n_pass_p_only": int(len(p_only)),
        "n_pass_r_gate_only": int(len(r_only)),
        "n_bit_identical": int(exact.sum()),
        "all_overlap_bit_identical": bool(exact.all()) if len(both) else False,
        "populations_identical": identical,
        "field_stats": field_stats,
        "void_s1_if_not_identical": not identical,
    }


def classify_cells(ds: pd.DataFrame) -> pd.DataFrame:
    """BREAK / NEAR / FAR from swing geometry. Used only for Pass-R counts."""
    broke_h = ds["broke_swing_high"].to_numpy(dtype=float)
    broke_l = ds["broke_swing_low"].to_numpy(dtype=float)
    dist = ds["distance_to_swing"].to_numpy(dtype=float)
    cell = np.full(len(ds), "FAR", dtype=object)
    outside = (broke_h == 1.0) & (broke_l == 1.0)
    brk = ((broke_h == 1.0) | (broke_l == 1.0)) & ~outside
    near = (~brk) & (~outside) & (dist < 0.5)
    cell[outside] = "SKIP"
    cell[brk] = "BREAK"
    cell[near] = "NEAR"
    out = ds.copy()
    out["s1_cell"] = cell
    return out


def attach_dataset(zones: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Attach the same causal features/labels as ``zone_dataset.parquet``.

    Overlap with Pass P is copied from that file (already validated). New Pass-R
    rows go through ``s5_zone_dataset.process_symbol`` unchanged.
    """
    key = ["symbol", "source_open_time"]
    pass_p_ds = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")
    pass_p_ds = pass_p_ds.copy()
    pass_p_ds["passes_swing_gate"] = True
    pass_p_ds["mode"] = "replay_pass_r"

    merged = zones.merge(pass_p_ds[key].drop_duplicates(), on=key, how="left", indicator=True)
    new_zones = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    log.info(f"overlap={len(zones) - len(new_zones)} new={len(new_zones)}")

    frames: List[pd.DataFrame] = [pass_p_ds]
    if len(new_zones):
        btc = loader.load_ohlcv("BTCUSDT")
        btc_bytes = btc.to_parquet(index=False)
        symbols = sorted(new_zones["symbol"].unique())
        with process_pool(workers, initializer=s5mod._init, initargs=(btc_bytes,)) as pool:
            futures = {
                pool.submit(
                    s5mod.process_symbol, s, new_zones[new_zones["symbol"] == s].to_parquet(index=False)
                ): s
                for s in symbols
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                part = pd.read_parquet(io.BytesIO(fut.result()))
                frames.append(part)
                log.info(
                    f"{i}/{len(symbols)} "
                    f"{str(futures[fut]).encode('ascii', 'replace').decode()} rows={len(part)}"
                )

    ds = pd.concat(frames, ignore_index=True, sort=False)
    miss = ds["passes_swing_gate"].isna()
    if miss.any():
        ds.loc[miss, "passes_swing_gate"] = ds.loc[miss, "distance_to_swing"].to_numpy(dtype=float) >= 0.5

    pit_path = paths.UNIVERSE_DIR / "pit_universe.parquet"
    if pit_path.exists() and "pit_rank_prev_1d" in ds.columns:
        need = ds["pit_rank_prev_1d"].isna()
        if need.any():
            stale = [c for c in s5mod.PIT_COLUMNS.values() if c in ds.columns]
            sub = s5mod.attach_pit_ranks(ds.loc[need].drop(columns=stale), pd.read_parquet(pit_path))
            ds = pd.concat([ds.loc[~need], sub], ignore_index=True)

    if "decision_time_ms" not in ds.columns:
        ds["decision_time_ms"] = ds["source_open_time"].astype("int64") + lab.BAR_MS
    ds["mode"] = "replay_pass_r"
    if "broke_swing_high" in ds.columns:
        ds = classify_cells(ds)
    return ds.sort_values(["source_open_time", "symbol"], kind="mergesort").reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    t0 = time.time()

    live_params = replay.DetectorParams.from_research_config()
    if float(live_params.min_distance_from_high_low) != 0.5:
        raise SystemExit(
            f"live min_distance_from_high_low is {live_params.min_distance_from_high_low}, "
            "Pass P was specified at 0.5"
        )
    params = replace(live_params, min_distance_from_high_low=0.0)
    log.info(f"live params: {live_params}")
    log.info(f"pass-R params (only swing floor changed): {params}")

    out_zones = paths.ZONES_DIR / "replay_zones_pass_r.parquet"
    btc_range = replay.checked_btc_range_series(backfill.to_candles(loader.load_ohlcv("BTCUSDT")))
    symbols = universe.available("round1")
    frames: List[pd.DataFrame] = []
    with process_pool(args.workers, initializer=s3mod._init_worker, initargs=(btc_range, params)) as pool:
        futures = {pool.submit(s3mod._replay_one, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), start=1):
            blob = fut.result()
            n = 0
            if blob:
                frames.append(pd.read_parquet(io.BytesIO(blob)))
                n = len(frames[-1])
            log.info(f"{i}/{len(symbols)} {futures[fut]} zones={n}")

    zones = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    zones = zones.sort_values(["source_open_time", "symbol"], kind="mergesort").reset_index(drop=True)
    zones["zone_uid"] = zones["symbol"] + "@" + zones["source_open_time"].astype(str)
    zones["passes_swing_gate"] = zones["distance_to_swing"].to_numpy(dtype=float) >= 0.5
    zones.to_parquet(out_zones, index=False, compression="zstd")
    log.info(f"{len(zones)} detections -> {out_zones}")

    pass_p = pd.read_parquet(paths.ZONES_DIR / "replay_zones.parquet")
    identity = bit_identity(zones, pass_p)
    log.info(json.dumps({k: v for k, v in identity.items() if k != "field_stats"}, indent=2))
    if identity["void_s1_if_not_identical"]:
        log.info("IDENTITY FAILED — S1 is VOID")
    else:
        log.info("identity assertion PASSED")

    ds_path = paths.FEATURES_DIR / "zone_dataset_pass_r.parquet"
    ds = attach_dataset(zones, workers=min(args.workers, 8))
    ds.to_parquet(ds_path, index=False, compression="zstd")
    cell_counts = ds["s1_cell"].value_counts().to_dict() if "s1_cell" in ds.columns else {}
    log.info(f"dataset {len(ds)} x {ds.shape[1]} -> {ds_path}")

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "pass": "R",
        "detector_params": params.__dict__,
        "live_params": live_params.__dict__,
        "n_pass_r": int(len(zones)),
        "n_passes_swing_gate": int(zones["passes_swing_gate"].sum()),
        "n_fails_swing_gate": int((~zones["passes_swing_gate"]).sum()),
        "identity": identity,
        "cell_counts": {str(k): int(v) for k, v in cell_counts.items()},
        "n_BREAK": int(cell_counts.get("BREAK", 0)),
        "n_NEAR": int(cell_counts.get("NEAR", 0)),
        "n_FAR": int(cell_counts.get("FAR", 0)),
        "n_SKIP_outside_bar": int(cell_counts.get("SKIP", 0)),
        "zones_path": paths.rel(out_zones),
        "dataset_path": paths.rel(ds_path),
        "elapsed_s": round(time.time() - t0, 1),
        "note": (
            "n_BREAK / n_NEAR / n_FAR are the Pass-R population after attaching "
            "swing features; they are not yet gated to tier D."
        ),
    }
    summary_path = paths.ZONES_DIR / "replay_pass_r_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps({k: v for k, v in summary.items() if k != "identity"}, indent=2, default=str))
    log.info(f"summary -> {summary_path} ({summary['elapsed_s']}s)")
    return 0 if identity["populations_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
