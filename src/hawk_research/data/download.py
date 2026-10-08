"""Download OHLCV from the public Binance archive and check it against the manifest.

Resumable: a file that already covers the pinned window is skipped unless
``force``. Every file is validated (hard corruption raises) and compared with the
published content checksum; mismatches are reported, never repaired. A local
manifest of what was downloaded, and how it compares, is written to
``data/ohlcv/coverage_manifest.json``.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from hawk_research.data import backfill, manifest, paths, universe, validation

log = logging.getLogger(__name__)

DEFAULT_UNIVERSES = ("round1", "market_neutral_tier2_extra")


def plan(
    universes: Sequence[str] = DEFAULT_UNIVERSES,
    symbols: Optional[Sequence[str]] = None,
    intervals: Sequence[str] = ("15m",),
    include_btc_1h: bool = False,
) -> List[Tuple[str, str]]:
    """Files to fetch. BTCUSDT 1h is opt-in: the original project downloaded it for the live
    system's stability check, but no stage here reads it, and its published checksum includes
    a last bar that was still forming when it was downloaded, so it can never match again."""
    names = sorted(set(symbols)) if symbols else sorted({s for u in universes for s in universe.symbols(u)})
    jobs = [(s, iv) for iv in intervals for s in names]
    if include_btc_1h and ("BTCUSDT", "1h") not in jobs:
        jobs.append(("BTCUSDT", "1h"))
    return jobs


def _complete(symbol: str, interval: str, start_ms: int, end_ms: int, pub: Optional[Dict[str, Any]]) -> bool:
    """True if the local file already covers the window (so a short ``--since`` download is not
    mistaken for a complete one later)."""
    path = paths.ohlcv_path(symbol, interval)
    if not path.exists():
        return False
    try:
        times = pd.read_parquet(path, columns=["open_time"])["open_time"]
    except Exception:  # noqa: BLE001 - an unreadable file is simply re-downloaded
        return False
    if times.empty or int(times.max()) < end_ms:
        return False
    # A symbol listed after start_ms legitimately starts later; the manifest says when.
    first_allowed = (
        max(start_ms, int(pub["first_open_time"])) if pub and pub.get("first_open_time") else start_ms
    )
    return int(times.min()) <= first_allowed or pub is None


def fetch_one(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    force: bool,
    published: Dict[str, Dict[str, Any]],
    offline: bool = False,
) -> Dict[str, Any]:
    """Fetch one file (or, offline, only check the local copy) and compare it with the manifest."""
    path = paths.ohlcv_path(symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"symbol": symbol, "interval": interval}
    pub = published.get(f"{symbol}|{interval}")
    if offline or (not force and _complete(symbol, interval, start_ms, end_ms, pub)):
        df = pd.read_parquet(path)
        record["cached"] = True
    else:
        # The original 1m backfill bounded the daily-archive probe to 45 days; 15m and 1h did not.
        lookback = 45 if interval == "1m" else None
        df, cov = backfill.download_symbol(symbol, interval, start_ms, end_ms, daily_lookback_days=lookback)
        record["cached"] = False
        record["coverage"] = cov.as_dict()
        if df.empty:
            record["error"] = "no data returned"
            return record
    record["checks"] = validation.validate_ohlcv(df, interval, symbol).as_dict()
    if not record["cached"]:
        df.to_parquet(path, index=False, compression="zstd")
    local = manifest.describe(symbol, interval, df)
    record.update(local)
    full_window = start_ms == paths.HISTORY_START_MS and end_ms == paths.SNAPSHOT_END_MS
    if not full_window:
        record["manifest"] = "not-compared (custom window)"
        return record
    record["manifest"] = (
        "not-in-manifest" if pub is None else ("match" if not manifest.compare(local, pub) else "mismatch")
    )
    if pub is not None and record["manifest"] == "mismatch":
        record["manifest_diffs"] = manifest.compare(local, pub)
    return record


def run(
    jobs: Sequence[Tuple[str, str]],
    workers: int = 8,
    force: bool = False,
    offline: bool = False,
    start_ms: int = paths.HISTORY_START_MS,
    end_ms: int = paths.SNAPSHOT_END_MS,
) -> Dict[str, Any]:
    """Download (or, offline, only check) ``jobs`` and write the local coverage manifest."""
    paths.ensure_dirs()
    published = manifest.entries_by_key(manifest.load_published())
    records: List[Dict[str, Any]] = []
    t0 = time.time()

    if offline:
        for symbol, interval in jobs:
            if not paths.ohlcv_path(symbol, interval).exists():
                records.append({"symbol": symbol, "interval": interval, "error": "missing (offline)"})
                continue
            records.append(fetch_one(symbol, interval, start_ms, end_ms, False, published, offline=True))
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(fetch_one, s, iv, start_ms, end_ms, force, published): (s, iv) for s, iv in jobs
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                s, iv = futures[fut]
                try:
                    rec = fut.result()
                except validation.CorruptOHLCV:
                    raise
                except Exception as exc:  # noqa: BLE001 - network failures are recorded per file
                    rec = {"symbol": s, "interval": iv, "error": repr(exc)}
                records.append(rec)
                log.info(
                    "%d/%d %s %s rows=%s %s manifest=%s",
                    i,
                    len(jobs),
                    s,
                    iv,
                    rec.get("rows"),
                    "cached" if rec.get("cached") else "",
                    rec.get("manifest", rec.get("error")),
                )

    records.sort(key=lambda r: (r["interval"], r["symbol"]))
    totals = {
        "jobs": len(jobs),
        "ok": sum(1 for r in records if "error" not in r),
        "errors": [f"{r['symbol']}|{r['interval']}: {r['error']}" for r in records if "error" in r],
        "manifest_match": sum(1 for r in records if r.get("manifest") == "match"),
        "manifest_mismatch": [
            f"{r['symbol']}|{r['interval']}" for r in records if r.get("manifest") == "mismatch"
        ],
        "not_in_manifest": [
            f"{r['symbol']}|{r['interval']}" for r in records if r.get("manifest") == "not-in-manifest"
        ],
    }
    out = {
        "created_at": datetime.now(UTC).isoformat(),
        "offline": offline,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source": "https://data.binance.vision monthly+daily klines, REST tail",
        "elapsed_s": round(time.time() - t0, 1),
        "totals": totals,
        "files": records,
    }
    paths.DATA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    paths.DATA_MANIFEST.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out
