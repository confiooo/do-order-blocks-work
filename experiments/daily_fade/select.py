"""Pick the completed UTC day's biggest mover and fade it.

The decision uses only bars whose open time falls on that UTC day. The fill
is the next day's first 15m open, which is the first price after the daily
candle has closed. The biggest gainer is a short. The biggest loser is a long.
A day with no positive gainer, or no negative loser, skips that side.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.daily_fade.sessions import BARS_PER_DAY, DAY_MS

STABLES = frozenset(
    {
        "USDCUSDT",
        "FDUSDUSDT",
        "USD1USDT",
        "RLUSDUSDT",
        "EURUSDT",
        "TUSDUSDT",
        "BUSDUSDT",
        "USDPUSDT",
        "DAIUSDT",
        "USDEUSDT",
    }
)


def daily_candles(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per complete UTC day. Incomplete days are dropped."""
    if bars.empty:
        return pd.DataFrame(columns=["day_ms", "open", "high", "low", "close", "bars"])
    frame = bars.sort_values("open_time", kind="mergesort")
    day = (frame["open_time"].to_numpy(dtype=np.int64) // DAY_MS) * DAY_MS
    grouped = frame.assign(day_ms=day).groupby("day_ms", sort=True)
    out = pd.DataFrame(
        {
            "day_ms": grouped["open_time"].min().index.to_numpy(dtype=np.int64),
            "open": grouped["open"].first().to_numpy(dtype=float),
            "high": grouped["high"].max().to_numpy(dtype=float),
            "low": grouped["low"].min().to_numpy(dtype=float),
            "close": grouped["close"].last().to_numpy(dtype=float),
            "bars": grouped["open_time"].size().to_numpy(dtype=int),
        }
    )
    # first() follows sort order, so the first open is the 00:00 bar when the day is complete.
    return out.loc[out["bars"] == BARS_PER_DAY].reset_index(drop=True)


def day_returns(candles: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close return. The first day has no prior close and is dropped."""
    out = candles.copy()
    prior = out["close"].shift(1)
    out["prior_close"] = prior
    out["return_24h"] = out["close"] / prior - 1.0
    return out.dropna(subset=["return_24h"]).reset_index(drop=True)


def _return_panel(by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol, candles in by_symbol.items():
        if symbol in STABLES or not str(symbol).isascii():
            continue
        returned = day_returns(candles)
        if returned.empty:
            continue
        part = returned[["day_ms", "open", "high", "low", "close", "prior_close", "return_24h"]].copy()
        part["symbol"] = symbol
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rank_side(by_symbol: Dict[str, pd.DataFrame], side: int) -> pd.DataFrame:
    """One row per day. Side −1 is the largest gainer. Side +1 is the largest loser."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 (short the gainer) or +1 (long the loser)")
    panel = _return_panel(by_symbol)
    if panel.empty:
        return panel
    if side < 0:
        panel = panel[panel["return_24h"] > 0.0]
        ascending = False
    else:
        panel = panel[panel["return_24h"] < 0.0]
        ascending = True
    if panel.empty:
        return pd.DataFrame()
    panel = panel.sort_values(
        ["day_ms", "return_24h", "symbol"],
        ascending=[True, ascending, True],
        kind="mergesort",
    )
    picked = panel.groupby("day_ms", sort=True).head(1).reset_index(drop=True)
    picked["side"] = side
    return picked


def rank_gainers(by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per day: the symbol with the largest positive 24h return, faded short."""
    return rank_side(by_symbol, -1)


def rank_losers(by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per day: the symbol with the most negative 24h return, faded long."""
    return rank_side(by_symbol, 1)


def forward_day(bars: pd.DataFrame, day_ms: int) -> pd.DataFrame:
    """The 96 fifteen-minute bars of the UTC day after ``day_ms``."""
    start = int(day_ms) + DAY_MS
    end = start + DAY_MS
    times = bars["open_time"].to_numpy(dtype=np.int64)
    keep = (times >= start) & (times < end)
    out = bars.loc[keep].sort_values("open_time", kind="mergesort")
    if len(out) != BARS_PER_DAY:
        return out.iloc[0:0].copy()
    return out.reset_index(drop=True)
