"""The one way to read downloaded OHLCV.

Every stage and study reads bars through :func:`load_ohlcv`. Differences that
used to be separate copy-pasted loaders (a column subset, sorting, a missing
file being an error or an empty frame) are parameters here.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from hawk_research.data import paths


class MissingOHLCV(FileNotFoundError):
    """Raised when a symbol's bars have not been downloaded."""


def load_ohlcv(
    symbol: str,
    interval: str = "15m",
    columns: Optional[Sequence[str]] = None,
    sort: bool = False,
    missing_ok: bool = False,
) -> pd.DataFrame:
    """Bars for one symbol and interval, as stored by the downloader.

    ``sort`` re-sorts by ``open_time`` (stable); stored files are already sorted,
    so it only matters for callers that historically sorted defensively.
    ``missing_ok`` returns an empty frame instead of raising.
    """
    path = paths.ohlcv_path(symbol, interval)
    if not path.exists():
        if missing_ok:
            return pd.DataFrame(columns=list(columns) if columns else None)
        raise MissingOHLCV(f"no {interval} OHLCV for {symbol} at {path}; run `hawk-research download`")
    frame = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if sort:
        frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    return frame


def load_many(
    symbols: Iterable[str],
    interval: str = "15m",
    columns: Optional[Sequence[str]] = None,
    sort: bool = False,
) -> Dict[str, pd.DataFrame]:
    """``{symbol: bars}`` for every symbol whose file exists."""
    out: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if paths.ohlcv_path(symbol, interval).exists():
            out[symbol] = load_ohlcv(symbol, interval, columns=columns, sort=sort)
    return out
