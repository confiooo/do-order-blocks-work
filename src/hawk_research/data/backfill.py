"""Historical 15m OHLCV from Binance's bulk archives.

``data.binance.vision`` has monthly and daily kline ZIPs; two years of 15m bars
for ~80 symbols is about 2,000 monthly files instead of ~5,600 paged REST
requests. REST is only used for the tail that is not archived yet.

Archive quirks handled here:

- files from 2025 on have a CSV header row, older ones do not;
- from 2025 the timestamps are in microseconds, not milliseconds. The unit is
  inferred per block from the magnitude and converted to milliseconds.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

BULK_BASE = "https://data.binance.vision/data/spot"
REST_BASE = "https://api.binance.com"

KLINE_COLUMNS = [
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
]

KEEP_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_quote",
]

_NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]


@dataclass
class SymbolCoverage:
    symbol: str
    interval: str
    rows: int = 0
    first_open_ms: Optional[int] = None
    last_open_ms: Optional[int] = None
    monthly_files: int = 0
    daily_files: int = 0
    rest_rows: int = 0
    missing_bars: int = 0
    gap_count: int = 0
    duplicates_dropped: int = 0
    nan_rows_dropped: int = 0
    out_of_order_fixed: bool = False
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["first_open_iso"] = _iso(self.first_open_ms)
        d["last_open_iso"] = _iso(self.last_open_ms)
        d["errors"] = list(self.errors)
        return d


def _iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=UTC).isoformat()


def interval_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]


def _normalise_epoch(values: pd.Series) -> pd.Series:
    """Coerce Binance epoch columns to integer milliseconds.

    2024 archives are in ms (~1.7e12); 2025+ archives are in us (~1.7e15).
    Anything at or above 1e14 is therefore microseconds.
    """
    v = pd.to_numeric(values, errors="coerce")
    v = v.where(v.notna(), other=np.nan)
    us_mask = v >= 1e14
    v = v.mask(us_mask, v // 1000)
    return v.astype("int64")


def _parse_kline_csv(raw: bytes) -> pd.DataFrame:
    first_line = raw.split(b"\n", 1)[0]
    has_header = b"open_time" in first_line.lower()
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else KLINE_COLUMNS,
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("open_time", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected archive columns: {list(df.columns)}")
    for col in ("quote_volume", "trades", "taker_buy_quote"):
        if col not in df.columns:
            df[col] = np.nan
    df["open_time"] = _normalise_epoch(df["open_time"])
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[KEEP_COLUMNS]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "hawk-research-backfill/1.0"})
    return s


def _download_zip(session: requests.Session, url: str, retries: int = 3) -> Optional[bytes]:
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                return zf.read(name)
        except (requests.RequestException, zipfile.BadZipFile, IndexError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None


def _months(start_ms: int, end_ms: int) -> List[Tuple[int, int]]:
    start = datetime.fromtimestamp(start_ms / 1000.0, tz=UTC)
    end = datetime.fromtimestamp(end_ms / 1000.0, tz=UTC)
    out: List[Tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _days(start_ms: int, end_ms: int) -> List[datetime]:
    start = datetime.fromtimestamp(start_ms / 1000.0, tz=UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = datetime.fromtimestamp(end_ms / 1000.0, tz=UTC)
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def fetch_rest_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """REST fallback for bars newer than the published archives."""
    step = interval_ms(interval)
    frames: List[pd.DataFrame] = []
    cursor = start_ms
    for _ in range(200):
        if cursor > end_ms:
            break
        try:
            resp = session.get(
                f"{REST_BASE}/api/v3/klines",
                params={  # type: ignore[arg-type]  # requests' stub is narrower than its runtime
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": int(cursor),
                    "endTime": int(end_ms + step - 1),
                    "limit": 1000,
                },
                timeout=30,
            )
            if resp.status_code in (400, 404):
                break
            if resp.status_code == 429:
                time.sleep(5)
                continue
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException:
            break
        if not isinstance(raw, list) or not raw:
            break
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        df["open_time"] = _normalise_epoch(df["open_time"])
        for col in _NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[KEEP_COLUMNS]
        df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
        if df.empty:
            break
        frames.append(df)
        cursor = int(df["open_time"].max()) + step
    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def download_symbol(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    session: Optional[requests.Session] = None,
    daily_lookback_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, SymbolCoverage]:
    """Assemble one symbol's history: monthly archives, daily archives, REST tail.

    ``end_ms`` is the inclusive open time of the last bar to keep. Returned rows
    are deduplicated, sorted, and clamped to ``[start_ms, end_ms]``.

    ``daily_lookback_days`` bounds how far back the daily archives are probed.
    Left ``None`` the daily scan runs from wherever the monthly archives stop,
    which is correct but pathological for a delisted pair: there is nothing after
    its last month, so it walks hundreds of 404s. Since Binance only leaves a
    window uncovered by monthly archives at the very end of the series, bounding
    the probe to the last ~45 days loses nothing for a wide symbol sweep.
    """
    session = session or _session()
    cov = SymbolCoverage(symbol=symbol, interval=interval)
    frames: List[pd.DataFrame] = []

    for year, month in _months(start_ms, end_ms):
        url = f"{BULK_BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
        try:
            raw = _download_zip(session, url)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            cov.errors.append(f"monthly {year}-{month:02d}: {exc}")
            continue
        if raw is None:
            continue
        try:
            frames.append(_parse_kline_csv(raw))
            cov.monthly_files += 1
        except Exception as exc:  # noqa: BLE001
            cov.errors.append(f"parse monthly {year}-{month:02d}: {exc}")

    covered_to = max((int(f["open_time"].max()) for f in frames), default=start_ms - 1)
    step = interval_ms(interval)

    daily_floor = start_ms
    if daily_lookback_days is not None:
        daily_floor = max(start_ms, end_ms - daily_lookback_days * 86_400_000)
    if covered_to < end_ms:
        for day in _days(max(covered_to + step, daily_floor), end_ms):
            url = f"{BULK_BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
            try:
                raw = _download_zip(session, url)
            except Exception as exc:  # noqa: BLE001
                cov.errors.append(f"daily {day:%Y-%m-%d}: {exc}")
                continue
            if raw is None:
                continue
            try:
                frames.append(_parse_kline_csv(raw))
                cov.daily_files += 1
            except Exception as exc:  # noqa: BLE001
                cov.errors.append(f"parse daily {day:%Y-%m-%d}: {exc}")

    covered_to = max((int(f["open_time"].max()) for f in frames), default=start_ms - 1)
    if covered_to < end_ms:
        rest = fetch_rest_klines(session, symbol, interval, max(covered_to + step, start_ms), end_ms)
        if not rest.empty:
            cov.rest_rows = len(rest)
            frames.append(rest)

    if not frames:
        cov.errors.append("no data available from bulk archives or REST")
        return pd.DataFrame(columns=KEEP_COLUMNS), cov

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
    before = len(df)
    df = df.drop_duplicates(subset="open_time", keep="first")
    cov.duplicates_dropped = before - len(df)
    cov.out_of_order_fixed = not df["open_time"].is_monotonic_increasing
    df = df.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    cov.nan_rows_dropped = before - len(df)
    df["open_time"] = df["open_time"].astype("int64")

    cov.rows = len(df)
    if cov.rows:
        cov.first_open_ms = int(df["open_time"].iloc[0])
        cov.last_open_ms = int(df["open_time"].iloc[-1])
        expected = (cov.last_open_ms - cov.first_open_ms) // step + 1
        cov.missing_bars = int(expected - cov.rows)
        deltas = np.diff(df["open_time"].to_numpy())
        cov.gap_count = int((deltas > step).sum())
    return df, cov


def to_candles(df: pd.DataFrame) -> List[Tuple[int, float, float, float, float, float]]:
    """The 6-tuple shape the live detector and outcome tracker consume."""
    return list(
        zip(
            df["open_time"].astype("int64").tolist(),
            df["open"].astype(float).tolist(),
            df["high"].astype(float).tolist(),
            df["low"].astype(float).tolist(),
            df["close"].astype(float).tolist(),
            df["volume"].astype(float).tolist(),
        )
    )
