"""Point-in-time name selection inside the survivor OHLCV pool.

Ranks come from ``pit_universe.parquet``. Prices exist only for today's
80-symbol archive, so the selected book is survivor-biased; every experiment
records ``UNIVERSE_BIAS_NOTE``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.market_neutral.calendar import (
    DEV_END,
    DEV_START,
    UNIVERSE_BIAS_NOTE,
    development_bounds,
)
from hawk_research.data import paths

# Spot pairs that are pegged, tokenized cash, or metal. They are not relative-value legs.
PEGGED = frozenset(
    {
        "USDCUSDT",
        "FDUSDUSDT",
        "USD1USDT",
        "RLUSDUSDT",
        "EURUSDT",
        "PAXGUSDT",
        "XAUTUSDT",
        "TUSDUSDT",
        "BUSDUSDT",
        "USDPUSDT",
        "DAIUSDT",
        "USDEUSDT",
        "USTCUSDT",
    }
)

Pair = Tuple[str, str]


def load_pit(symbols: Sequence[str] | None = None) -> pd.DataFrame:
    path = paths.UNIVERSE_DIR / "pit_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"point-in-time universe is missing: {path}")
    columns = ["date", "symbol", "rank_qv_30d", "in_top40_30d", "in_top20_30d"]
    frame = pd.read_parquet(path, columns=columns)
    if symbols is not None:
        wanted = set(symbols)
        frame = frame[frame["symbol"].isin(wanted)]
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    start = pd.Timestamp(DEV_START, tz="UTC")
    end = pd.Timestamp(DEV_END, tz="UTC")
    return frame[(frame["date"] >= start) & (frame["date"] < end)].reset_index(drop=True)


def symbols_covering_development() -> List[str]:
    """Round-1 15m symbols whose bar span covers tier D.

    Coverage is read from ``config/data_manifest.json``, so this check does not
    pull firewall or holdout rows into a frame. Candidates are the round-1
    universe: the published run read a coverage manifest that listed exactly
    those symbols, before the tier-2 names were downloaded.
    """
    from hawk_research.data import manifest
    from hawk_research.data import universe as named_universes

    start_ms, end_ms = development_bounds()
    by_key = manifest.entries_by_key(manifest.load_published())
    found: List[str] = []
    for symbol in named_universes.symbols("round1"):
        row = by_key.get(f"{symbol}|15m")
        if row is None or not symbol.isascii() or symbol in PEGGED:
            continue
        first = int(row["first_open_time"])
        last = int(row["last_open_time"])
        if first <= start_ms and last >= end_ms - 900_000 and paths.ohlcv_path(symbol, "15m").exists():
            found.append(symbol)
    return sorted(set(found))


def select_names(n_names: int = 12, min_top40_fraction: float = 0.50) -> List[str]:
    """BTC, ETH, then the next most liquid full-history survivors, up to ``n_names``."""
    if n_names < 2:
        raise ValueError("n_names must be at least 2")
    eligible = symbols_covering_development()
    if "BTCUSDT" not in eligible or "ETHUSDT" not in eligible:
        raise RuntimeError("BTCUSDT and ETHUSDT must have full tier-D 15m history")
    pit = load_pit(eligible)
    rows = []
    for symbol, group in pit.groupby("symbol", sort=False):
        rank = group["rank_qv_30d"].to_numpy(dtype=float)
        finite = rank[np.isfinite(rank)]
        if finite.size == 0:
            continue
        top40 = group["in_top40_30d"].fillna(False).to_numpy(dtype=bool)
        rows.append(
            {
                "symbol": str(symbol),
                "median_rank": float(np.median(finite)),
                "top40_fraction": float(top40.mean()),
            }
        )
    scored = pd.DataFrame(rows)
    if scored.empty:
        raise RuntimeError("no point-in-time ranks for the survivor OHLCV pool")
    liquid = scored[
        (scored["symbol"].isin(["BTCUSDT", "ETHUSDT"])) | (scored["top40_fraction"] >= min_top40_fraction)
    ].copy()
    liquid = liquid.sort_values(["median_rank", "symbol"], kind="mergesort")
    ordered = ["BTCUSDT", "ETHUSDT"]
    for symbol in liquid["symbol"]:
        if symbol not in ordered:
            ordered.append(str(symbol))
        if len(ordered) >= n_names:
            break
    if len(ordered) < n_names:
        raise RuntimeError(
            f"only {len(ordered)} full-history liquid names cleared the PIT screen; need {n_names}"
        )
    return ordered


def select_pairs(names: Sequence[str]) -> List[Pair]:
    """Each non-BTC name versus BTC, plus the first other name versus ETH."""
    if "BTCUSDT" not in names or "ETHUSDT" not in names:
        raise ValueError("names must include BTCUSDT and ETHUSDT")
    pairs: List[Pair] = [(name, "BTCUSDT") for name in names if name != "BTCUSDT"]
    cross = next(name for name in names if name not in ("BTCUSDT", "ETHUSDT"))
    pairs.append((cross, "ETHUSDT"))
    return pairs


def prior_day_ranks(symbols: Sequence[str]) -> pd.DataFrame:
    """30d quote-volume rank keyed by the decision date that may use it.

    A pit row dated D includes D's volume and is known after D closes, so a
    decision on day T uses the row dated T-1.
    """
    pit = load_pit(symbols)
    pit = pit.dropna(subset=["rank_qv_30d"]).copy()
    pit["known_on"] = pit["date"] + pd.Timedelta(days=1)
    return pit[["known_on", "symbol", "rank_qv_30d"]].reset_index(drop=True)


def rank_lookup(symbols: Sequence[str]) -> Dict[tuple, float]:
    frame = prior_day_ranks(symbols)
    out: Dict[tuple, float] = {}
    known = frame["known_on"].to_numpy()
    syms = frame["symbol"].to_numpy()
    ranks = frame["rank_qv_30d"].to_numpy(dtype=float)
    for day, symbol, rank in zip(known, syms, ranks):
        out[(pd.Timestamp(day).tz_convert("UTC").normalize(), str(symbol))] = float(rank)
    return out


def describe_universe(names: Sequence[str], pairs: Sequence[Pair]) -> dict:
    return {
        "names": list(names),
        "pairs": [list(pair) for pair in pairs],
        "n_names": len(names),
        "n_pairs": len(pairs),
        "priced_from": "data/research/ohlcv/15m survivor archive",
        "ranks_from": "data/research/universe/pit_universe.parquet",
        "universe_bias_note": UNIVERSE_BIAS_NOTE,
    }
