"""Stage 3s: replay the textbook (SMC) order-block detector over full history.

Writes ``zones/replay_zones_smc.parquet`` with the same key columns as
``replay_zones.parquet``, so s5, s8 and s12 take it with ``--tag smc``.
``source_open_time`` is the BOS bar; the OB candle is ``ob_open_time``.

The detector is causal (see its tests), so it runs over each symbol's whole
series at once. Zones whose BOS bar falls in the first ``replay.WARMUP_BARS``
bars are dropped, as in the round-1 replay. The summary also reports how many
SMC order blocks the round-1 detector flagged too.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from concurrent.futures import as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.data import backfill, loader, paths, universe
from hawk_research.detectors.smc_order_blocks import detect_smc_order_blocks
from hawk_research.parallel import process_pool
from hawk_research.replay import round1 as replay

log = logging.getLogger(__name__)

_BTC_RANGE: Dict[int, float] = {}
_BTC_THRESHOLD = 0.0


def _init_worker(btc_range: Dict[int, float], btc_threshold: float) -> None:
    global _BTC_RANGE, _BTC_THRESHOLD
    _BTC_RANGE = btc_range
    _BTC_THRESHOLD = btc_threshold


def _replay_one(symbol: str) -> bytes:
    df = loader.load_ohlcv(symbol)
    candles = backfill.to_candles(df)
    zones = [z for z in detect_smc_order_blocks(candles) if z.bar_index >= replay.WARMUP_BARS]
    if not zones:
        return b""
    out = pd.DataFrame([asdict(z) for z in zones])
    out.insert(0, "symbol", symbol)
    brange = out["source_open_time"].map(_BTC_RANGE)
    out["btc_range_4h_pct"] = brange.astype(float)
    out["btc_stable"] = (brange.notna() & (brange < _BTC_THRESHOLD)).astype(bool)
    out["live_equivalent"] = out["btc_stable"]
    return out.to_parquet(index=False)


def _symbols() -> List[str]:
    """The round-1 zone universe (76 symbols), so the two populations are comparable."""
    return universe.available("round1_zone_symbols")


def overlap_with_round1(smc: pd.DataFrame) -> Dict:
    """Share of SMC order-block candles that the round-1 detector also flagged."""
    old_path = paths.ZONES_DIR / "replay_zones.parquet"
    if not old_path.exists():
        return {"available": False}
    old = pd.read_parquet(old_path, columns=["symbol", "source_open_time"])
    old_keys = set(zip(old["symbol"], old["source_open_time"].astype("int64")))
    ob_hit = np.array([(s, int(t)) in old_keys for s, t in zip(smc["symbol"], smc["ob_open_time"])])
    bos_hit = np.array([(s, int(t)) in old_keys for s, t in zip(smc["symbol"], smc["source_open_time"])])
    return {
        "available": True,
        "n_round1_zones": int(len(old)),
        "n_smc_zones": int(len(smc)),
        "smc_ob_candle_also_round1_zone": int(ob_hit.sum()),
        "smc_ob_candle_also_round1_zone_pct": float(ob_hit.mean() * 100) if len(smc) else None,
        "smc_bos_bar_also_round1_zone_pct": float(bos_hit.mean() * 100) if len(smc) else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    t0 = time.time()
    paths.ensure_dirs()
    params = replay.DetectorParams.from_research_config()
    btc = loader.load_ohlcv("BTCUSDT")
    btc_range = replay.btc_range_series(backfill.to_candles(btc))

    symbols = _symbols()
    frames: List[pd.DataFrame] = []
    with process_pool(
        args.workers,
        initializer=_init_worker,
        initargs=(btc_range, params.btc_volatility_threshold),
    ) as pool:
        futures = {pool.submit(_replay_one, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), start=1):
            blob = fut.result()
            n = 0
            if blob:
                frames.append(pd.read_parquet(io.BytesIO(blob)))
                n = len(frames[-1])
            log.info(f"{i}/{len(symbols)} {futures[fut]} zones={n}")

    zones = pd.concat(frames, ignore_index=True)
    zones = zones.sort_values(["source_open_time", "symbol"], kind="mergesort").reset_index(drop=True)
    zones["zone_uid"] = zones["symbol"] + "@" + zones["source_open_time"].astype(str)
    if not zones["zone_uid"].is_unique:
        raise SystemExit("more than one SMC zone on the same symbol and BOS bar")
    out = paths.ZONES_DIR / "replay_zones_smc.parquet"
    zones.to_parquet(out, index=False, compression="zstd")

    days = (zones["source_open_time"].max() - zones["source_open_time"].min()) / 86_400_000
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "path": paths.rel(out),
        "n_zones": int(len(zones)),
        "n_symbols": int(zones["symbol"].nunique()),
        "zones_per_symbol_per_day": float(len(zones) / zones["symbol"].nunique() / days),
        "direction_counts": zones["direction"].value_counts().to_dict(),
        "has_fvg_pct": float(zones["has_fvg"].mean() * 100),
        "n_live_equivalent": int(zones["live_equivalent"].sum()),
        "median_ob_age_bars": float((zones["bar_index"] - zones["ob_index"]).median()),
        "median_zone_size_pct": float(zones["zone_size_pct"].median()),
        "overlap_with_round1": overlap_with_round1(zones),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (paths.ZONES_DIR / "replay_smc_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
