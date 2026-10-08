"""Binance USD-M funding rates and 1h perpetual contract prices.

Archives come from data.binance.vision. The unfinished tail comes from the
public USD-M REST API. Open interest is not backfilled. Rows are written in
full, including the sealed holdout; readers clip that window themselves.
"""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from experiments.market_neutral.calendar import development_bounds
from experiments.market_neutral.funding_basis.loaders import funding_path, perpetual_path
from experiments.market_neutral.funding_basis.universe import MIN_DEVELOPMENT_DAYS, PHASE1_NAMES
from hawk_research.data.backfill import (
    KEEP_COLUMNS,
    _download_zip,
    _months,
    _normalise_epoch,
    _parse_kline_csv,
    _session,
    interval_ms,
)

log = logging.getLogger(__name__)

VISION = "https://data.binance.vision/data/futures/um"
FAPI = "https://fapi.binance.com"
# BTCUSDT perpetual listing is September 2019. Earlier months 404 and are skipped.
HISTORY_START_MS = 1567296000000  # 2019-09-01T00:00:00Z
# Two years is roughly 25-30k funding rows and ~200k hourly bars; full listing
# history is longer. A count off by an order of magnitude stops the write.
FUNDING_ROWS_MIN = 2_500
FUNDING_ROWS_MAX = 300_000
PERP_ROWS_MIN = 20_000
PERP_ROWS_MAX = 2_000_000
HOUR_MS = interval_ms("1h")
MS_DAY = 24 * HOUR_MS


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    """One monthly or daily funding archive. Timestamps stay as published."""
    first = raw.split(b"\n", 1)[0].lower()
    has_header = b"calc_time" in first or b"funding" in first
    frame = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if not has_header:
        cols = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        if frame.shape[1] < 3:
            raise ValueError(f"funding archive has {frame.shape[1]} columns")
        frame = frame.iloc[:, :3]
        frame.columns = cols
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    time_col = next((c for c in ("calc_time", "fundingtime", "funding_time") if c in frame.columns), None)
    rate_col = next(
        (c for c in ("last_funding_rate", "funding_rate", "fundingrate") if c in frame.columns),
        None,
    )
    if time_col is None or rate_col is None:
        raise ValueError(f"unexpected funding columns: {list(frame.columns)}")
    out = pd.DataFrame(
        {
            "funding_time_ms": _normalise_epoch(frame[time_col]),
            "funding_rate": pd.to_numeric(frame[rate_col], errors="coerce"),
        }
    )
    if "funding_interval_hours" in frame.columns:
        out["funding_interval_hours"] = pd.to_numeric(frame["funding_interval_hours"], errors="coerce")
    else:
        out["funding_interval_hours"] = np.nan
    return out.dropna(subset=["funding_time_ms", "funding_rate"])


def _funding_from_rest(payload: Sequence[dict]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["funding_time_ms", "funding_rate", "funding_interval_hours"])
    frame = pd.DataFrame(payload)
    out = pd.DataFrame(
        {
            "funding_time_ms": _normalise_epoch(frame["fundingTime"]),
            "funding_rate": pd.to_numeric(frame["fundingRate"], errors="coerce"),
            "funding_interval_hours": np.nan,
        }
    )
    return out.dropna(subset=["funding_time_ms", "funding_rate"])


def _fill_interval_hours(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("funding_time_ms", kind="mergesort").reset_index(drop=True)
    if len(out) < 2:
        return out
    delta_h = np.diff(out["funding_time_ms"].to_numpy(dtype=np.int64)) / HOUR_MS
    filled = out["funding_interval_hours"].to_numpy(dtype=float).copy()
    missing = ~np.isfinite(filled)
    # The interval on row i is the gap from the previous settlement, when the archive omitted it.
    missing_body = missing[1:] & np.isfinite(delta_h) & (delta_h > 0)
    filled[1:][missing_body] = delta_h[missing_body]
    if missing[0] and np.isfinite(filled[1:]).any():
        filled[0] = next(x for x in filled[1:] if np.isfinite(x))
    out["funding_interval_hours"] = filled
    return out


def _finalize_funding(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["funding_time_ms", "funding_rate", "funding_interval_hours"])
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates("funding_time_ms", keep="first")
    frame = _fill_interval_hours(frame)
    frame["funding_time_ms"] = frame["funding_time_ms"].astype("int64")
    return frame.reset_index(drop=True)


def _zip_csv(url: str) -> Optional[pd.DataFrame]:
    raw = _download_zip(_session(), url)
    if raw is None:
        return None
    return parse_funding_csv(raw)


def _kline_zip(url: str) -> Optional[pd.DataFrame]:
    raw = _download_zip(_session(), url)
    if raw is None:
        return None
    return _parse_kline_csv(raw)


def _gather(urls: Sequence[str], kind: str, workers: int = 8) -> Tuple[List[pd.DataFrame], List[str]]:
    frames: List[pd.DataFrame] = []
    errors: List[str] = []
    parser = _zip_csv if kind == "funding" else _kline_zip
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(parser, url): url for url in urls}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                frame = future.result()
            except Exception as exc:  # noqa: BLE001 — recorded per archive
                errors.append(f"{url}: {exc}")
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
    return frames, errors


def _rest_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    session = _session()
    frames: List[pd.DataFrame] = []
    cursor = int(start_ms)
    for _ in range(50):
        if cursor > end_ms:
            break
        resp = session.get(
            f"{FAPI}/fapi/v1/fundingRate",
            params={"symbol": symbol, "startTime": cursor, "endTime": int(end_ms), "limit": 1000},
            timeout=30,
        )
        if resp.status_code in (400, 404):
            break
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            break
        frame = _funding_from_rest(payload)
        frame = frame[frame["funding_time_ms"] <= end_ms]
        if frame.empty:
            break
        frames.append(frame)
        nxt = int(frame["funding_time_ms"].max()) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(payload) < 1000:
            break
    return _finalize_funding(frames)


def _rest_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    session = _session()
    frames: List[pd.DataFrame] = []
    cursor = int(start_ms)
    for _ in range(200):
        if cursor > end_ms:
            break
        resp = session.get(
            f"{FAPI}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": 1500,
            },
            timeout=30,
        )
        if resp.status_code in (400, 404):
            break
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            break
        raw = pd.DataFrame(
            payload,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        raw["open_time"] = _normalise_epoch(raw["open_time"])
        for col in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw[(raw["open_time"] >= start_ms) & (raw["open_time"] <= end_ms)]
        if raw.empty:
            break
        frames.append(raw[list(KEEP_COLUMNS)])
        nxt = int(raw["open_time"].max()) + HOUR_MS
        if nxt <= cursor:
            break
        cursor = nxt
        if len(payload) < 1500:
            break
    if not frames:
        return pd.DataFrame(columns=list(KEEP_COLUMNS))
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates("open_time", keep="first").sort_values("open_time", kind="mergesort")


def _month_urls(symbol: str, kind: str, start_ms: int, end_ms: int) -> List[str]:
    urls = []
    for year, month in _months(start_ms, end_ms):
        if kind == "funding":
            urls.append(
                f"{VISION}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
            )
        else:
            urls.append(f"{VISION}/monthly/klines/{symbol}/1h/{symbol}-1h-{year:04d}-{month:02d}.zip")
    return urls


def download_funding(symbol: str, start_ms: int, end_ms: int) -> Tuple[pd.DataFrame, List[str]]:
    frames, errors = _gather(_month_urls(symbol, "funding", start_ms, end_ms), "funding")
    partial = _finalize_funding(frames)
    covered = int(partial["funding_time_ms"].max()) + 1 if len(partial) else start_ms
    if covered <= end_ms:
        try:
            tail = _rest_funding(symbol, covered, end_ms)
        except requests.RequestException as exc:
            errors.append(f"rest funding: {exc}")
            tail = pd.DataFrame()
        if len(tail):
            partial = _finalize_funding([partial, tail] if len(partial) else [tail])
    if len(partial):
        partial = partial[
            (partial["funding_time_ms"] >= start_ms) & (partial["funding_time_ms"] <= end_ms)
        ].reset_index(drop=True)
    return partial, errors


def download_perpetual(symbol: str, start_ms: int, end_ms: int) -> Tuple[pd.DataFrame, List[str]]:
    frames, errors = _gather(_month_urls(symbol, "klines", start_ms, end_ms), "klines")
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates("open_time", keep="first")
        covered = int(frame["open_time"].max()) + HOUR_MS
    else:
        frame = pd.DataFrame(columns=list(KEEP_COLUMNS))
        covered = start_ms
    if covered <= end_ms:
        try:
            tail = _rest_klines(symbol, covered, end_ms)
        except requests.RequestException as exc:
            errors.append(f"rest klines: {exc}")
            tail = pd.DataFrame()
        if len(tail):
            frame = pd.concat([frame, tail], ignore_index=True) if len(frame) else tail
    if frame.empty:
        return frame, errors
    frame = frame[(frame["open_time"] >= start_ms) & (frame["open_time"] <= end_ms)]
    frame = frame.drop_duplicates("open_time", keep="first")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    frame["open_time"] = frame["open_time"].astype("int64")
    frame, errors = _fill_kline_gaps(symbol, frame, errors)
    return frame[list(KEEP_COLUMNS)], errors


def _fill_kline_gaps(symbol: str, frame: pd.DataFrame, errors: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Monthly archives sometimes omit the last days of a month. Daily zips cover those hours."""
    if len(frame) < 2:
        return frame, errors
    times = frame["open_time"].to_numpy(dtype=np.int64)
    diffs = times[1:] - times[:-1]
    seen = set()
    urls: List[str] = []
    for i, gap in enumerate(diffs.tolist()):
        if gap <= HOUR_MS:
            continue
        cursor = int(times[i]) + HOUR_MS
        stop = int(times[i + 1])
        while cursor < stop:
            day = datetime.fromtimestamp(cursor / 1000.0, tz=UTC).strftime("%Y-%m-%d")
            if day not in seen:
                seen.add(day)
                urls.append(f"{VISION}/daily/klines/{symbol}/1h/{symbol}-1h-{day}.zip")
            cursor += MS_DAY
    if not urls:
        return frame, errors
    extra, more = _gather(urls, "klines")
    errors.extend(more)
    if not extra:
        return frame, errors
    merged = pd.concat([frame, *extra], ignore_index=True)
    merged = merged.drop_duplicates("open_time", keep="first")
    merged = merged.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    merged["open_time"] = merged["open_time"].astype("int64")
    return merged[list(KEEP_COLUMNS)], errors


def missing_contract_reason(symbol: str) -> Optional[str]:
    """Set when the USD-M API does not list this exact symbol."""
    try:
        resp = _session().get(
            f"{FAPI}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 1},
            timeout=30,
        )
    except requests.RequestException as exc:
        return f"could not confirm a USD-M contract for {symbol}: {exc}"
    if resp.status_code == 400 and "Invalid symbol" in resp.text:
        return (
            f"Binance USD-M has no contract named {symbol}. "
            "A differently named multiplier contract, if one exists, was not ingested: "
            "it is outside the approved symbol list."
        )
    return None


def overlap_days(first_ms: Optional[int], last_ms: Optional[int]) -> float:
    """Days of [first, last] that sit inside tier D."""
    if first_ms is None or last_ms is None:
        return 0.0
    d0, d1 = development_bounds()
    start = max(int(first_ms), d0)
    end = min(int(last_ms), d1)
    if end <= start:
        return 0.0
    return (end - start) / MS_DAY


def inclusion(funding_days: float, perp_days: float) -> Tuple[bool, Optional[str]]:
    if funding_days < MIN_DEVELOPMENT_DAYS and perp_days < MIN_DEVELOPMENT_DAYS:
        return False, (
            f"funding overlap {funding_days:.1f}d and perpetual overlap {perp_days:.1f}d "
            f"are both under {MIN_DEVELOPMENT_DAYS}d inside tier D"
        )
    if funding_days < MIN_DEVELOPMENT_DAYS:
        return False, f"funding overlap {funding_days:.1f}d inside tier D is under {MIN_DEVELOPMENT_DAYS}d"
    if perp_days < MIN_DEVELOPMENT_DAYS:
        return False, f"perpetual overlap {perp_days:.1f}d inside tier D is under {MIN_DEVELOPMENT_DAYS}d"
    return True, None


def _span(frame: pd.DataFrame, col: str) -> Tuple[Optional[int], Optional[int]]:
    if frame.empty:
        return None, None
    return int(frame[col].iloc[0]), int(frame[col].iloc[-1])


def _iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()


def counts_are_sane(funding_rows: int, perp_rows: int) -> Optional[str]:
    if not FUNDING_ROWS_MIN <= funding_rows <= FUNDING_ROWS_MAX:
        return (
            f"funding rows {funding_rows} are outside {FUNDING_ROWS_MIN}..{FUNDING_ROWS_MAX}, "
            "an order of magnitude off the expected ~25–30k for two years "
            "(full history may run higher, but not this far)"
        )
    if not PERP_ROWS_MIN <= perp_rows <= PERP_ROWS_MAX:
        return (
            f"perpetual rows {perp_rows} are outside {PERP_ROWS_MIN}..{PERP_ROWS_MAX}, "
            "an order of magnitude off the expected ~200k hourly bars"
        )
    return None


def ingest(symbols: Sequence[str] = PHASE1_NAMES, end_ms: Optional[int] = None) -> Dict[str, object]:
    """Download, sanity-check, and write funding plus 1h perpetual klines."""
    now = int(datetime.now(UTC).timestamp() * 1000) if end_ms is None else int(end_ms)
    # Last completed 1h bar. Its close is knowable; the current hour is not.
    last_open = (now // HOUR_MS) * HOUR_MS - HOUR_MS
    pulled: List[dict] = []
    for symbol in symbols:
        log.info(f"{symbol}")
        funding, funding_errors = download_funding(symbol, HISTORY_START_MS, now)
        perp, perp_errors = download_perpetual(symbol, HISTORY_START_MS, last_open)
        f0, f1 = _span(funding, "funding_time_ms")
        p0, p1 = _span(perp, "open_time")
        funding_days = overlap_days(f0, f1)
        perp_days = overlap_days(p0, p1 + HOUR_MS if p1 is not None else None)
        included, reason = inclusion(funding_days, perp_days)
        if funding.empty and perp.empty:
            contract_reason = missing_contract_reason(symbol)
            if contract_reason:
                included, reason = False, contract_reason
        interval = None
        if len(funding) and funding["funding_interval_hours"].notna().any():
            interval = float(funding["funding_interval_hours"].median())
        missing_bars = None
        if p0 is not None and p1 is not None:
            missing_bars = int((p1 - p0) // HOUR_MS + 1 - len(perp))
        pulled.append(
            {
                "symbol": symbol,
                "funding": funding,
                "perpetual": perp,
                "record": {
                    "symbol": symbol,
                    "funding_rows": int(len(funding)),
                    "funding_first_ms": f0,
                    "funding_last_ms": f1,
                    "funding_first_iso": _iso(f0),
                    "funding_last_iso": _iso(f1),
                    "funding_interval_hours_median": interval,
                    "funding_errors": funding_errors,
                    "perp_rows": int(len(perp)),
                    "perp_first_ms": p0,
                    "perp_last_ms": p1,
                    "perp_first_iso": _iso(p0),
                    "perp_last_iso": _iso(p1),
                    "perp_missing_bars": missing_bars,
                    "perp_errors": perp_errors,
                    "development_funding_days": round(funding_days, 2),
                    "development_perp_days": round(perp_days, 2),
                    "included_in_family_b": included,
                    "exclusion_reason": reason,
                },
            }
        )
        log.info(f"{symbol} funding={len(funding)} from {_iso(f0)} perp={len(perp)} included={included}")

    funding_rows = sum(len(item["funding"]) for item in pulled)
    perp_rows = sum(len(item["perpetual"]) for item in pulled)
    problem = counts_are_sane(funding_rows, perp_rows)
    from experiments.market_neutral.funding_basis.loaders import _root

    log_path = _root() / "ingestion_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "venue": "binance_usd_m",
        "price_series": (
            "USD-M perpetual contract klines (last price) at 1h. "
            "Mark-price klines were not requested and were not downloaded."
        ),
        "funding_source": "data.binance.vision futures/um fundingRate, REST tail /fapi/v1/fundingRate",
        "perp_source": "data.binance.vision futures/um klines 1h, REST tail /fapi/v1/klines",
        "open_interest": "not ingested; historical OI stays missing",
        "history_start_probed": _iso(HISTORY_START_MS),
        "end_ms": now,
        "last_perp_open_ms": last_open,
        "funding_rows_total": funding_rows,
        "perp_rows_total": perp_rows,
        "sanity_problem": problem,
        "written": problem is None,
        "min_development_days": MIN_DEVELOPMENT_DAYS,
        "symbols": [item["record"] for item in pulled],
    }
    if problem:
        log_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        raise RuntimeError(problem + f" Log: {log_path}. Parquets were not written.")

    for item in pulled:
        symbol = item["symbol"]
        funding_file = funding_path(symbol)
        perp_file = perpetual_path(symbol, "1h")
        if item["funding"].empty and item["perpetual"].empty:
            funding_file.unlink(missing_ok=True)
            perp_file.unlink(missing_ok=True)
            continue
        funding_file.parent.mkdir(parents=True, exist_ok=True)
        perp_file.parent.mkdir(parents=True, exist_ok=True)
        item["funding"].to_parquet(funding_file, index=False)
        item["perpetual"].to_parquet(perp_file, index=False)
    log_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info(f"wrote {funding_rows} funding rows and {perp_rows} perpetual bars -> {log_path}")
    return summary


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()
