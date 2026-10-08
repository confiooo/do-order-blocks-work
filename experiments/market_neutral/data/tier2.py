"""Tier-2 relative-value book.

The eligible band is point-in-time: ``in_top80_30d`` and not ``in_top20_30d``.
Phase-1 names and pegged pairs are removed. A name must sit in that band on at
least half of tier-D days — the same persistence fraction ``select_names``
already uses — and the book is then capped at 15 by median trailing 30-day
quote volume. The cap and the floor are not relaxed to admit more names.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from experiments.market_neutral.calendar import DEV_END, DEV_START, development_bounds
from experiments.market_neutral.data.universe import PEGGED
from experiments.market_neutral.funding_basis.universe import PHASE1_NAMES
from hawk_research.data import paths
from hawk_research.data.backfill import download_symbol

BAND_DAY_FRACTION = 0.50
NAME_CAP = 15
BAR_15_MS = 900_000

TIER2_NOTE = (
    "Tier-2 book: in_top80_30d and not in_top20_30d on at least 50% of tier-D days, "
    "excluding the phase-1 twelve and pegged pairs, capped at 15 by median trailing "
    "30-day quote volume. The rank band and the persistence floor were not relaxed. "
    "Each name is paired with BTC only."
)


def select_tier2_names(
    pit: pd.DataFrame,
    *,
    n_days: int,
    cap: int = NAME_CAP,
    min_fraction: float = BAND_DAY_FRACTION,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the capped book, highest trailing volume first."""
    if n_days < 1:
        raise ValueError("n_days must be positive")
    if "in_top80_30d" not in pit.columns or "in_top20_30d" not in pit.columns:
        raise ValueError("pit frame is missing the top-80 / top-20 flags")
    blocked = set(PHASE1_NAMES) | set(PEGGED) | set(exclude)
    band = pit[pit["in_top80_30d"].fillna(False) & ~pit["in_top20_30d"].fillna(False)]
    grouped = band.groupby("symbol", sort=False).agg(
        band_days=("date", "nunique"),
        median_qv_30d=("qv_30d", "median"),
        median_rank_qv_30d=("rank_qv_30d", "median"),
    )
    grouped = grouped[~grouped.index.isin(blocked)]
    keep = [symbol for symbol in grouped.index if str(symbol).isascii() and str(symbol).endswith("USDT")]
    grouped = grouped.loc[keep].copy()
    grouped["band_fraction"] = grouped["band_days"] / float(n_days)
    grouped = grouped[grouped["band_fraction"] >= min_fraction]
    grouped = grouped.sort_values(["median_qv_30d", "symbol"], ascending=[False, True], kind="mergesort")
    return grouped.head(cap)


def load_tier2_frame() -> Tuple[pd.DataFrame, int]:
    columns = ["date", "symbol", "qv_30d", "rank_qv_30d", "in_top80_30d", "in_top20_30d"]
    frame = pd.read_parquet(paths.UNIVERSE_DIR / "pit_universe.parquet", columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    start = pd.Timestamp(DEV_START, tz="UTC")
    end = pd.Timestamp(DEV_END, tz="UTC")
    frame = frame[(frame["date"] >= start) & (frame["date"] < end)]
    return frame, int(frame["date"].nunique())


def tier2_names() -> List[str]:
    frame, n_days = load_tier2_frame()
    selected = select_tier2_names(frame, n_days=n_days)
    return [str(symbol) for symbol in selected.index]


def tier2_pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    return [(name, "BTCUSDT") for name in names]


def ensure_spot_15m(symbols: Sequence[str]) -> Dict[str, object]:
    """Download tier-D 15m spot bars for names that are not already on disk.

    The window stops at the development boundary. Holdout bars are not fetched
    for these names.
    """
    start_ms, end_ms = development_bounds()
    last_open = end_ms - BAR_15_MS
    report: Dict[str, object] = {}
    for symbol in symbols:
        path = paths.ohlcv_path(symbol, "15m")
        if path.exists():
            report[symbol] = {"downloaded": False, "path": paths.rel(path)}
            continue
        frame, coverage = download_symbol(symbol, "15m", start_ms, last_open, daily_lookback_days=45)
        path.parent.mkdir(parents=True, exist_ok=True)
        if frame.empty:
            report[symbol] = {
                "downloaded": False,
                "rows": 0,
                "errors": coverage.errors,
                "excluded": "no spot 15m archive inside tier D",
            }
            continue
        frame.to_parquet(path, index=False)
        report[symbol] = {
            "downloaded": True,
            "rows": int(len(frame)),
            "first_open_ms": coverage.first_open_ms,
            "last_open_ms": coverage.last_open_ms,
            "errors": coverage.errors,
        }
    return report
