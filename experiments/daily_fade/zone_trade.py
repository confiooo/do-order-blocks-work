"""A daily-fade bot whose entry and target are fixed when the candle closes.

The rollback zone is known before the next day starts. Its near edge is the
signal close. Its far edge is a retrace of that candle, taken from the same
kind of day in the earlier sample only. The trade starts at that near edge
and tries to reach the far edge. Nothing here is chosen by looking at the
later sample.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.daily_fade.exits import ROUND_TRIP_COST
from experiments.daily_fade.select import forward_day
from experiments.daily_fade.study import load_universe
from hawk_research.data import paths

log = logging.getLogger(__name__)

# Decisions on days before this instant are the fitting sample.
# The bot is scored only on days from this instant forward.
TEST_START = "2026-01-01"
MIN_TRAIN_CELL = 30


def _last_pivot(highs: np.ndarray, lows: np.ndarray, close: float, side: int) -> float:
    """Last 15m swing on the stop side of the close. Two bars each side."""
    if side < 0:
        found = None
        for i in range(2, highs.size - 2):
            if (
                highs[i] > close
                and highs[i] >= highs[i - 2 : i].max()
                and highs[i] > highs[i + 1 : i + 3].max()
            ):
                found = float(highs[i])
        return float(highs.max()) if found is None else found
    found = None
    for i in range(2, lows.size - 2):
        if lows[i] < close and lows[i] <= lows[i - 2 : i].min() and lows[i] < lows[i + 1 : i + 3].min():
            found = float(lows[i])
    return float(lows.min()) if found is None else found


def structural_level(day: pd.DataFrame, side: int, kind: str) -> float:
    """A resistance or support taken only from the closed signal day."""
    highs = day["high"].to_numpy(dtype=float)
    lows = day["low"].to_numpy(dtype=float)
    close = float(day["close"].iloc[-1])
    if kind == "day_extreme":
        return float(highs.max()) if side < 0 else float(lows.min())
    if kind == "last_4h":
        return float(highs[-16:].max()) if side < 0 else float(lows[-16:].min())
    if kind == "last_1h":
        return float(highs[-4:].max()) if side < 0 else float(lows[-4:].min())
    if kind == "swing":
        return _last_pivot(highs, lows, close, side)
    raise ValueError(kind)


def behind_level(level: float, side: int, buffer: float) -> float:
    """Place the stop just past the level, on the side that invalidates the fade."""
    if side < 0:
        return float(level) * (1.0 + buffer)
    return float(level) * (1.0 - buffer)


def at_extreme(close: float, low: float, high: float, side: int) -> bool:
    span = float(high - low)
    if span <= 0.0:
        return False
    location = (float(close) - float(low)) / span
    if side < 0:
        return location >= 0.85
    return location <= 0.15


def zone_prices(
    close: float, prior: float, side: int, retrace: float, day_high: float, day_low: float
) -> Tuple[float, float]:
    """Far edge of the rollback zone, and the signal-day extreme as invalidation."""
    impulse = abs(float(close) - float(prior))
    if side < 0:
        return float(close) - retrace * impulse, float(day_high)
    return float(close) + retrace * impulse, float(day_low)


def _walk(
    forward: pd.DataFrame, start: int, entry: float, side: int, target: float, stop: float, use_stop: bool
) -> Tuple[float, str, int]:
    opens = forward["open"].to_numpy(dtype=float)
    highs = forward["high"].to_numpy(dtype=float)
    lows = forward["low"].to_numpy(dtype=float)
    closes = forward["close"].to_numpy(dtype=float)
    for i in range(start, opens.size):
        price_open = float(opens[i])
        if side < 0:
            if use_stop and price_open >= stop:
                return price_open, "stop", i
            if price_open <= target:
                return price_open, "target", i
            if use_stop and float(highs[i]) >= stop:
                return float(stop), "stop", i
            if float(lows[i]) <= target:
                return float(target), "target", i
        else:
            if use_stop and price_open <= stop:
                return price_open, "stop", i
            if price_open >= target:
                return price_open, "target", i
            if use_stop and float(lows[i]) <= stop:
                return float(stop), "stop", i
            if float(highs[i]) >= target:
                return float(target), "target", i
    return float(closes[-1]), "time", int(opens.size - 1)


def _limit_fill(
    forward: pd.DataFrame, close: float, side: int, target: float
) -> Optional[Tuple[int, float, bool]]:
    """Fill a limit at the close. None means the zone was missed."""
    opens = forward["open"].to_numpy(dtype=float)
    highs = forward["high"].to_numpy(dtype=float)
    lows = forward["low"].to_numpy(dtype=float)
    for i in range(opens.size):
        price_open = float(opens[i])
        high = float(highs[i])
        low = float(lows[i])
        if side < 0:
            if price_open >= close and price_open > target:
                return i, price_open, True
            if price_open >= close and price_open <= target:
                return None
            if low <= target:
                return None
            if high >= close:
                return i, float(close), False
        else:
            if price_open <= close and price_open < target:
                return i, price_open, True
            if price_open <= close and price_open >= target:
                return None
            if high >= target:
                return None
            if low <= close:
                return i, float(close), False
    return None


def simulate(
    forward: pd.DataFrame,
    close: float,
    prior: float,
    side: int,
    day_high: float,
    day_low: float,
    retrace: float,
    max_rollback: float,
    entry_mode: str,
    stop_mode: str,
) -> Optional[dict]:
    target, extreme = zone_prices(close, prior, side, retrace, day_high, day_low)
    if not np.isfinite(target) or target <= 0.0 or forward.empty:
        return None
    use_stop = stop_mode == "day_extreme"
    if entry_mode == "market_open":
        price_open = float(forward["open"].iloc[0])
        if side < 0 and price_open <= target:
            return None
        if side > 0 and price_open >= target:
            return None
        filled = (0, price_open, True)
    elif entry_mode == "limit_close":
        filled = _limit_fill(forward, close, side, target)
        if filled is None:
            return None
    else:
        raise ValueError(entry_mode)
    index, entry, entry_is_open = filled
    start = index if entry_is_open else index + 1
    if start >= len(forward):
        exit_px, reason, exit_i = entry, "time", index
    else:
        exit_px, reason, exit_i = _walk(forward, start, entry, side, target, extreme, use_stop)
    if side < 0:
        ret_gross = (entry - exit_px) / entry
        gained = (entry - exit_px) / close
    else:
        ret_gross = (exit_px - entry) / entry
        gained = (exit_px - entry) / close
    capture = 0.0 if max_rollback <= 0.0 else max(0.0, gained) / max_rollback
    return {
        "entry": float(entry),
        "exit": float(exit_px),
        "reason": reason,
        "exit_bar": int(exit_i),
        "ret_gross": float(ret_gross),
        "ret_net": float(ret_gross) - ROUND_TRIP_COST,
        "capture": float(capture),
        "target_hit": reason == "target",
    }


def _prepare(events: pd.DataFrame, bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for rec in events.itertuples(index=False):
        forward = forward_day(bars[rec.symbol], int(rec.day_ms))
        if forward.empty or float(rec.day_close) <= 0.0:
            continue
        rows.append(
            {
                "day_ms": int(rec.day_ms),
                "symbol": rec.symbol,
                "side": int(rec.side),
                "close": float(rec.day_close),
                "prior": float(rec.prior_close),
                "day_high": float(rec.day_high),
                "day_low": float(rec.day_low),
                "abs_ret": abs(float(rec.return_24h)),
                "retrace": float(rec.retrace_of_candle),
                "max_rollback": float(rec.max_rollback),
                "at_extreme": at_extreme(
                    float(rec.day_close), float(rec.day_low), float(rec.day_high), int(rec.side)
                ),
                "forward": forward,
            }
        )
    return pd.DataFrame(rows)


def fit_cells(train: pd.DataFrame) -> Tuple[np.ndarray, Dict[Tuple[int, bool], Dict[str, float]]]:
    edges = np.quantile(train["abs_ret"].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(train["abs_ret"].to_numpy(dtype=float), edges)
    table: Dict[Tuple[int, bool], Dict[str, float]] = {}
    frame = train.copy()
    frame["bin"] = bins
    for (bin_id, extreme), group in frame.groupby(["bin", "at_extreme"], sort=True):
        if len(group) < MIN_TRAIN_CELL:
            continue
        table[(int(bin_id), bool(extreme))] = {
            "p30": float(group["retrace"].quantile(0.30)),
            "p50": float(group["retrace"].quantile(0.50)),
            "n": float(len(group)),
        }
    return edges, table


def _cell_retrace(
    table, edges, abs_ret: float, extreme: bool, percentile: str, only_half: bool
) -> Optional[float]:
    bin_id = int(np.digitize([abs_ret], edges)[0])
    cell = table.get((bin_id, bool(extreme)))
    if cell is None:
        return None
    if only_half and float(cell["p50"]) < 0.5:
        return None
    return float(cell["p30"] if percentile == "p30" else cell["p50"])


def _score(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    net = trades["ret_net"].to_numpy(dtype=float)
    daily = trades.groupby("day_ms")["ret_net"].mean().sort_index()
    equity = np.cumprod(1.0 + daily.to_numpy(dtype=float))
    peak = np.maximum.accumulate(equity)
    drawdown = float(np.max(1.0 - equity / peak)) if equity.size else 0.0
    return {
        "n": int(len(trades)),
        "hit_rate": float(trades["target_hit"].mean()),
        "stop_rate": float((trades["reason"] == "stop").mean()),
        "time_rate": float((trades["reason"] == "time").mean()),
        "win_rate": float(np.mean(net > 0.0)),
        "mean_net": float(np.mean(net)),
        "median_net": float(np.median(net)),
        "sum_net": float(np.sum(net)),
        "median_capture": float(trades["capture"].median()),
        "mean_capture": float(trades["capture"].mean()),
        "daily_compound": float(equity[-1] - 1.0),
        "max_drawdown": drawdown,
        "n_days": int(daily.size),
    }


def _run_rule(
    frame: pd.DataFrame, edges, table, entry_mode: str, percentile: str, stop_mode: str, only_half: bool
) -> pd.DataFrame:
    taken = []
    for rec in frame.itertuples(index=False):
        retrace = _cell_retrace(table, edges, float(rec.abs_ret), bool(rec.at_extreme), percentile, only_half)
        if retrace is None or retrace <= 0.0:
            continue
        result = simulate(
            rec.forward,
            float(rec.close),
            float(rec.prior),
            int(rec.side),
            float(rec.day_high),
            float(rec.day_low),
            retrace,
            float(rec.max_rollback),
            entry_mode,
            stop_mode,
        )
        if result is None:
            continue
        taken.append({"day_ms": int(rec.day_ms), "symbol": rec.symbol, "side": int(rec.side), **result})
    return pd.DataFrame(taken)


def _select(scored: List[dict]) -> dict:
    """Prefer a rule that keeps most of the rollback and still makes money in the fitting sample."""
    full = [row for row in scored if row.get("n", 0) > 0 and "median_capture" in row]
    qualified = [
        row
        for row in full
        if row["n"] >= 80
        and row["median_capture"] >= 0.60
        and row["hit_rate"] >= 0.45
        and row["mean_net"] > 0.0
    ]
    pool = qualified or full
    return max(pool, key=lambda row: (row["median_capture"], row["mean_net"]))


def run_bot() -> dict:
    events = pd.read_parquet(paths.DAILY_FADE_DIR / "events.parquet")
    prepared = _prepare(events, load_universe())
    cutoff = int(pd.Timestamp(TEST_START, tz="UTC").timestamp() * 1000)
    train = prepared[prepared["day_ms"] < cutoff].reset_index(drop=True)
    test = prepared[prepared["day_ms"] >= cutoff].reset_index(drop=True)
    edges, table = fit_cells(train)
    rules = []
    for entry_mode in ("market_open", "limit_close"):
        for percentile in ("p30", "p50"):
            for stop_mode in ("time_24h", "day_extreme"):
                for only_half in (False, True):
                    rules.append((entry_mode, percentile, stop_mode, only_half))
    train_scores = []
    for entry_mode, percentile, stop_mode, only_half in rules:
        name = f"{entry_mode}|{percentile}|{stop_mode}|{'half_candle' if only_half else 'all_days'}"
        score = _score(_run_rule(train, edges, table, entry_mode, percentile, stop_mode, only_half))
        score["rule"] = name
        train_scores.append(score)
    chosen = _select(train_scores)
    entry_mode, percentile, stop_mode, half_flag = chosen["rule"].split("|")
    only_half = half_flag == "half_candle"
    test_trades = _run_rule(test, edges, table, entry_mode, percentile, stop_mode, only_half)
    test_score = _score(test_trades)
    test_score["rule"] = chosen["rule"]
    others = []
    for entry_mode, percentile, stop_mode, only_half in rules:
        name = f"{entry_mode}|{percentile}|{stop_mode}|{'half_candle' if only_half else 'all_days'}"
        if name == chosen["rule"]:
            continue
        score = _score(_run_rule(test, edges, table, entry_mode, percentile, stop_mode, only_half))
        score["rule"] = name
        others.append(score)
    cells = [{"bin": key[0], "at_extreme": key[1], **value} for key, value in sorted(table.items())]
    summary = {
        "orders_sent": False,
        "test_start": TEST_START,
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "round_trip_cost": ROUND_TRIP_COST,
        "selection": "fitting sample only: median capture of the rollback at least 60%, target hit at least 45%, positive mean net, then the highest capture",
        "chosen_train": chosen,
        "chosen_test": test_score,
        "cells": cells,
        "train_rules": train_scores,
        "other_test_rules": others,
    }
    folder = paths.DAILY_FADE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "zone_bot.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not test_trades.empty:
        test_trades.drop(columns=[], errors="ignore").to_parquet(
            folder / "zone_bot_test.parquet", index=False
        )
    return summary


def run_tp_stops() -> dict:
    """Fixed p50 target, market entry, with percent stops and stops behind the signal day's structure.

    The target table is fit on days before 2026-01-01. Every stop below is scored
    on that sample and on the later sample. The later sample does not pick the stop.
    """
    from experiments.daily_fade.sessions import DAY_MS

    events = pd.read_parquet(paths.DAILY_FADE_DIR / "events.parquet")
    bars = load_universe()
    prepared = _prepare(events, bars)
    cutoff = int(pd.Timestamp(TEST_START, tz="UTC").timestamp() * 1000)
    train = prepared[prepared["day_ms"] < cutoff].reset_index(drop=True)
    edges, table = fit_cells(train)
    signal_days: Dict[Tuple[str, int], pd.DataFrame] = {}
    for rec in prepared.itertuples(index=False):
        frame = bars[rec.symbol]
        times = frame["open_time"].to_numpy(dtype=np.int64)
        keep = (times >= int(rec.day_ms)) & (times < int(rec.day_ms) + DAY_MS)
        day = frame.loc[keep].sort_values("open_time", kind="mergesort")
        if len(day) == 96:
            signal_days[(rec.symbol, int(rec.day_ms))] = day.reset_index(drop=True)

    specs: List[Tuple[str, Optional[float], Optional[str], float]] = [("no stop", None, None, 0.0)]
    for pct in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        specs.append((f"{pct:.0%} of price", pct, None, 0.0))
    for label, kind, buffer in (
        ("at the day high/low", "day_extreme", 0.0),
        ("0.5% past the day high/low", "day_extreme", 0.005),
        ("at the last 4h high/low", "last_4h", 0.0),
        ("0.5% past the last 4h high/low", "last_4h", 0.005),
        ("0.5% past the last 1h high/low", "last_1h", 0.005),
        ("0.5% past the last swing", "swing", 0.005),
    ):
        specs.append((label, None, kind, buffer))

    scores = []
    for name, pct, kind, buffer in specs:
        taken = []
        for rec in prepared.itertuples(index=False):
            retrace = _cell_retrace(table, edges, float(rec.abs_ret), bool(rec.at_extreme), "p50", False)
            if retrace is None or retrace <= 0.0:
                continue
            target, _extreme = zone_prices(
                float(rec.close),
                float(rec.prior),
                int(rec.side),
                retrace,
                float(rec.day_high),
                float(rec.day_low),
            )
            entry = float(rec.forward["open"].iloc[0])
            if int(rec.side) < 0 and entry <= target:
                continue
            if int(rec.side) > 0 and entry >= target:
                continue
            if pct is None and kind is None:
                exit_px, reason, exit_i = _walk(rec.forward, 0, entry, int(rec.side), target, entry, False)
                dist = np.nan
            else:
                if pct is not None:
                    stop = entry * (1.0 + pct) if int(rec.side) < 0 else entry * (1.0 - pct)
                else:
                    day = signal_days.get((rec.symbol, int(rec.day_ms)))
                    if day is None:
                        continue
                    stop = behind_level(structural_level(day, int(rec.side), kind), int(rec.side), buffer)
                dist = (stop - entry) / entry if int(rec.side) < 0 else (entry - stop) / entry
                exit_px, reason, exit_i = _walk(rec.forward, 0, entry, int(rec.side), target, stop, True)
            if int(rec.side) < 0:
                gross = (entry - exit_px) / entry
                gained = (entry - exit_px) / float(rec.close)
            else:
                gross = (exit_px - entry) / entry
                gained = (exit_px - entry) / float(rec.close)
            max_rb = float(rec.max_rollback)
            taken.append(
                {
                    "day_ms": int(rec.day_ms),
                    "reason": reason,
                    "ret_net": float(gross) - ROUND_TRIP_COST,
                    "capture": 0.0 if max_rb <= 0.0 else max(0.0, gained) / max_rb,
                    "target_hit": reason == "target",
                    "stop_dist": dist,
                    "sample": "test" if int(rec.day_ms) >= cutoff else "train",
                }
            )
        frame = pd.DataFrame(taken)
        for sample in ("train", "test"):
            part = frame[frame["sample"] == sample] if not frame.empty else frame
            score = (
                _score(part.drop(columns=["sample", "stop_dist"], errors="ignore"))
                if not part.empty
                else {"n": 0}
            )
            score["rule"] = name
            score["sample"] = sample
            if not part.empty and part["stop_dist"].notna().any():
                score["median_stop_dist"] = float(part["stop_dist"].median())
            scores.append(score)
    summary = {
        "orders_sent": False,
        "target": "p50 retrace from days before 2026-01-01",
        "entry": "next 15m open",
        "scores": scores,
    }
    folder = paths.DAILY_FADE_DIR
    (folder / "tp_stops.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_bot()
    test = summary["chosen_test"]
    log.info(
        json.dumps(
            {
                "rule": test.get("rule"),
                "n": test.get("n"),
                "mean_net": test.get("mean_net"),
                "orders_sent": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
