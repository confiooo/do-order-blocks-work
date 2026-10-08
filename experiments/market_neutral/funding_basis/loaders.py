"""Funding and perpetual-bar access.

Historical funding and USD-M perpetual klines are read from the ingestion
store. The default window is tier D: firewall, holdout, and the post-holdout
September slice are dropped before the caller sees them. Holdout rows are
returned only through ``load_holdout_funding`` / ``load_holdout_perpetual``,
which consume the one-touch audit log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from experiments.market_neutral.calendar import development_bounds, holdout_bounds
from experiments.market_neutral.holdout import unseal_holdout
from hawk_research.data import paths


class MissingMarketData(FileNotFoundError):
    """A Family B input is not on disk. Nothing was invented in its place."""


def _root() -> Path:
    return paths.MARKET_NEUTRAL_DIR


def funding_path(symbol: str) -> Path:
    return _root() / "funding" / f"{symbol}.parquet"


def perpetual_path(symbol: str, interval: str = "1h") -> Path:
    return _root() / "perpetual" / interval / f"{symbol}.parquet"


def _missing(kind: str, symbol: str, path: Path) -> MissingMarketData:
    return MissingMarketData(
        f"{kind} for {symbol} is not on disk ({path}). "
        "Family B will not fabricate funding or perpetual prices."
    )


def _read(path: Path, kind: str, symbol: str) -> pd.DataFrame:
    if not path.exists():
        raise _missing(kind, symbol, path)
    return pd.read_parquet(path)


def _development(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    _, end_ms = development_bounds()
    times = frame[time_col].to_numpy(dtype=np.int64)
    return frame.loc[times < end_ms].reset_index(drop=True)


def _holdout(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    start_ms, end_ms = holdout_bounds()
    times = frame[time_col].to_numpy(dtype=np.int64)
    return frame.loc[(times >= start_ms) & (times < end_ms)].reset_index(drop=True)


def load_funding(symbol: str) -> pd.DataFrame:
    """Funding settlements with time before the firewall, including history from before tier D."""
    frame = _read(funding_path(symbol), "funding history", symbol)
    return _development(frame, "funding_time_ms")


def load_perpetual(symbol: str, interval: str = "1h") -> pd.DataFrame:
    """USD-M perpetual contract bars whose open time is inside tier D."""
    frame = _read(perpetual_path(symbol, interval), "perpetual prices", symbol)
    return _development(frame, "open_time")


def load_holdout_funding(symbol: str, reason: str, audit_path: Optional[Path] = None) -> pd.DataFrame:
    """One-touch read of funding inside 2026-04-01..2026-08-31."""

    def _load() -> pd.DataFrame:
        frame = _read(funding_path(symbol), "funding history", symbol)
        return _holdout(frame, "funding_time_ms")

    return unseal_holdout("funding_basis", reason, _load, audit_path)


def load_holdout_perpetual(
    symbol: str,
    reason: str,
    interval: str = "1h",
    audit_path: Optional[Path] = None,
) -> pd.DataFrame:
    """One-touch read of perpetual bars inside 2026-04-01..2026-08-31."""

    def _load() -> pd.DataFrame:
        frame = _read(perpetual_path(symbol, interval), "perpetual prices", symbol)
        return _holdout(frame, "open_time")

    return unseal_holdout("funding_basis", reason, _load, audit_path)
