"""Assembling one symbol's history from monthly archives, daily archives and the REST tail,
against a fake HTTP session (no network)."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pandas as pd

from hawk_research.data import backfill

H = 3_600_000


def _ms(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp() * 1000)


def _rows(first: str, n: int, close: float, unit: int = 1) -> list:
    t0 = _ms(first)
    return [
        [
            (t0 + i * H) * unit,
            close,
            close + 1,
            close - 1,
            close,
            5.0,
            (t0 + (i + 1) * H - 1) * unit,
            50.0,
            7,
            2.0,
            20.0,
            0,
        ]
        for i in range(n)
    ]


def _zip(rows: list, header: bool) -> bytes:
    csv = pd.DataFrame(rows, columns=backfill.KLINE_COLUMNS).to_csv(index=False, header=header).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("klines.csv", csv)
    return buf.getvalue()


class _Resp:
    def __init__(self, status: int, content: bytes = b"", payload=None):
        self.status_code, self.content, self._payload = status, content, payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise backfill.requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class FakeSession:
    """Monthly Jan 2025 (ms, header, one bar before the window, one bar overlapping the daily
    file); no Feb monthly; daily 2025-02-01 in microseconds without a header, rate-limited once;
    REST tail for 2025-02-02 including one bar past the window end."""

    def __init__(self):
        self.urls: list = []
        self.rate_limited = False

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        if url.endswith("XUSDT-1h-2025-01.zip"):
            return _Resp(200, _zip(_rows("2025-01-31 20:00", 5, close=10.0), header=True))
        if url.endswith("XUSDT-1h-2025-02-01.zip"):
            if not self.rate_limited:
                self.rate_limited = True
                return _Resp(429)
            return _Resp(200, _zip(_rows("2025-02-01 00:00", 24, close=20.0, unit=1000), header=False))
        if url.endswith("/api/v3/klines"):
            rows = _rows("2025-02-02 00:00", 5, close=30.0)
            return _Resp(200, payload=[r for r in rows if r[0] >= params["startTime"]])
        return _Resp(404)


def test_download_symbol_stitches_archives_and_rest_tail(monkeypatch):
    monkeypatch.setattr(backfill.time, "sleep", lambda s: None)
    session = FakeSession()
    start, end = _ms("2025-01-31 22:00"), _ms("2025-02-02 03:00")
    df, cov = backfill.download_symbol("XUSDT", "1h", start, end, session=session)

    # 2 bars on Jan 31, 24 on Feb 1, 4 on Feb 2: clipped to the window, one per hour, in ms
    assert len(df) == 30 and cov.rows == 30
    assert df["open_time"].tolist() == [start + i * H for i in range(30)]
    assert (cov.first_open_ms, cov.last_open_ms, cov.missing_bars, cov.gap_count) == (start, end, 0, 0)
    assert (cov.monthly_files, cov.daily_files, cov.rest_rows) == (1, 1, 4)
    # the monthly and daily files both hold 2025-02-01 00:00; the first (monthly) copy is kept
    assert cov.duplicates_dropped == 1
    assert df.set_index("open_time").loc[_ms("2025-02-01 00:00"), "close"] == 10.0
    assert df.set_index("open_time").loc[_ms("2025-02-01 01:00"), "close"] == 20.0  # from µs epochs
    assert df.set_index("open_time").loc[end, "close"] == 30.0
    assert list(df.columns) == backfill.KEEP_COLUMNS and not cov.errors
    assert session.rate_limited  # the 429 was retried, not treated as missing


def test_download_symbol_with_nothing_published_reports_an_error(monkeypatch):
    class Empty:
        def get(self, url, params=None, timeout=None):
            return _Resp(404) if url.endswith(".zip") else _Resp(200, payload=[])

    df, cov = backfill.download_symbol(
        "NOPEUSDT", "1h", _ms("2025-01-01 00:00"), _ms("2025-01-02 00:00"), session=Empty()
    )
    assert df.empty and cov.errors == ["no data available from bulk archives or REST"]
