"""Stage 2b: point-in-time universe.

The live symbol list is today's top volume names. Using it for two years of
history keeps coins that became liquid later and drops coins that were liquid and
later delisted; both bias a backtest upwards.

This stage lists every USDT spot pair with a kline archive (the bucket listing,
not ``exchangeInfo``, which no longer knows fully delisted pairs), downloads daily
klines for all of them and ranks them by trailing 30d and 24h quote volume as of
each date. A symbol is in the universe on a date only if it was listed and liquid
on that date.

Writes ``universe/pit_universe.parquet`` (one row per date and symbol: trailing
volumes and ranks) and ``universe_membership.parquet`` (top-N flags per date).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import List, Optional

import pandas as pd
import requests

from hawk_research.data import backfill, paths, universe

log = logging.getLogger(__name__)

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TOP_N = (10, 20, 40, 80)


def list_archived_symbols(interval: str = "1d") -> List[str]:
    """Every symbol with a monthly kline archive, including delisted ones."""
    prefix = "data/spot/monthly/klines/"
    symbols: List[str] = []
    token: Optional[str] = None
    session = requests.Session()
    while True:
        params = {"delimiter": "/", "prefix": prefix, "list-type": "2"}
        if token:
            params["continuation-token"] = token
        resp = session.get(BUCKET, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for cp in root.findall("s3:CommonPrefixes/s3:Prefix", NS):
            name = (cp.text or "").rstrip("/").rsplit("/", 1)[-1]
            if name:
                symbols.append(name)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=NS) or "").lower()
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if truncated != "true" or not token:
            break
    return sorted(set(symbols))


def _fetch_daily(symbol: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
    try:
        df, _cov = backfill.download_symbol(symbol, "1d", start_ms, end_ms, daily_lookback_days=45)
    except Exception:  # noqa: BLE001
        return None
    if df.empty:
        return None
    df = df[["open_time", "close", "quote_volume"]].copy()
    df["symbol"] = symbol
    return df


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-ms", type=int, default=paths.HISTORY_START_MS)
    parser.add_argument("--end-ms", type=int, default=paths.SNAPSHOT_END_MS)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    t0 = time.time()
    all_symbols = list_archived_symbols()
    usdt = [s for s in all_symbols if s.endswith("USDT")]
    log.info(f"{len(all_symbols)} archived spot symbols, {len(usdt)} USDT pairs")

    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_daily, s, args.start_ms, args.end_ms): s for s in usdt}
        for i, fut in enumerate(as_completed(futures), start=1):
            df = fut.result()
            if df is not None:
                frames.append(df)
            if i % 50 == 0:
                log.info(f"{i}/{len(usdt)} symbols, {len(frames)} with data ({time.time() - t0:.0f}s)")

    if not frames:
        log.warning("no data")
        return 1

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["open_time"], unit="ms", utc=True).dt.normalize()
    panel = panel.sort_values(["symbol", "open_time"], kind="mergesort")

    # Trailing quote volume as of each date, inclusive of that date's bar (the
    # daily bar has closed before the next day's decisions are taken).
    g = panel.groupby("symbol", sort=False)["quote_volume"]
    panel["qv_1d"] = panel["quote_volume"]
    panel["qv_30d"] = g.rolling(30, min_periods=10).sum().reset_index(level=0, drop=True)
    panel["qv_7d"] = g.rolling(7, min_periods=3).sum().reset_index(level=0, drop=True)
    panel["listed_days"] = panel.groupby("symbol", sort=False).cumcount() + 1

    # Ranks are computed per date across everything listed on that date.
    for col in ("qv_1d", "qv_30d"):
        panel[f"rank_{col}"] = panel.groupby("date", sort=False)[col].rank(ascending=False, method="first")
    panel["n_listed"] = panel.groupby("date", sort=False)["symbol"].transform("size")

    for n in TOP_N:
        panel[f"in_top{n}_30d"] = panel["rank_qv_30d"] <= n
        panel[f"in_top{n}_1d"] = panel["rank_qv_1d"] <= n

    out = paths.UNIVERSE_DIR / "pit_universe.parquet"
    keep = [
        "date",
        "open_time",
        "symbol",
        "close",
        "quote_volume",
        "qv_1d",
        "qv_7d",
        "qv_30d",
        "rank_qv_1d",
        "rank_qv_30d",
        "listed_days",
        "n_listed",
    ] + [f"in_top{n}_{w}" for n in TOP_N for w in ("30d", "1d")]
    panel[keep].to_parquet(out, index=False, compression="zstd")

    # Listing start of each symbol, so dates before it are known to be pre-listing.
    listing = panel.groupby("symbol", as_index=False).agg(
        first_open_time=("open_time", "min"), last_open_time=("open_time", "max"), days=("open_time", "size")
    )
    listing["first_iso"] = listing["first_open_time"].map(backfill._iso)
    listing["last_iso"] = listing["last_open_time"].map(backfill._iso)
    listing["still_listed_at_snapshot"] = listing["last_open_time"] >= args.end_ms - 3 * 86_400_000
    listing.to_parquet(paths.UNIVERSE_DIR / "listing_history.parquet", index=False)

    live_syms = universe.available("round1")
    last_date = panel["date"].max()
    today_top40 = set(panel[(panel["date"] == last_date) & (panel["rank_qv_1d"] <= 40)]["symbol"])
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "archived_spot_symbols": len(all_symbols),
        "usdt_pairs_considered": len(usdt),
        "usdt_pairs_with_data": int(panel["symbol"].nunique()),
        "dates": int(panel["date"].nunique()),
        "rows": int(len(panel)),
        "delisted_before_snapshot": int((~listing["still_listed_at_snapshot"]).sum()),
        "median_n_listed_per_day": float(panel.groupby("date")["symbol"].size().median()),
        "survivorship_note": (
            "The live 80-symbol list is selected by TODAY's volume. Ranking here is "
            "recomputed from trailing volume as of each date across every USDT pair "
            "Binance ever archived, including "
            f"{int((~listing['still_listed_at_snapshot']).sum())} that were delisted "
            "before the snapshot. Strategy research should filter on "
            "in_topN_* rather than on the live tokens table. Residual bias: pairs "
            "delisted early enough that Binance removed their archives entirely "
            "cannot be recovered from this source."
        ),
        "backfilled_15m_symbols_in_today_top40": sorted(set(live_syms) & today_top40),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (paths.UNIVERSE_DIR / "universe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    log.info(
        json.dumps(
            {k: v for k, v in summary.items() if k != "backfilled_15m_symbols_in_today_top40"},
            indent=2,
            default=str,
        )
    )
    log.info(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
