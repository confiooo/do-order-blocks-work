"""Historical daily-fade study.

For each qualifying day it records the short that would be placed at the next
UTC open and compares a fixed list of exits. Nothing here places orders.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.daily_fade.exits import EXIT_RULES, ROUND_TRIP_COST, evaluate_exit, session_path
from experiments.daily_fade.rollback import maximum_rollback
from experiments.daily_fade.select import daily_candles, forward_day, rank_gainers, rank_losers
from hawk_research.data import loader, paths, universe

log = logging.getLogger(__name__)


def load_universe() -> Dict[str, pd.DataFrame]:
    """15m bars for the daily-fade universe (config/universe.json ``daily_fade``)."""
    names = [s for s in universe.symbols("daily_fade") if s.isascii() and s.endswith("USDT")]
    return loader.load_many(names, "15m", columns=["open_time", "open", "high", "low", "close"], sort=True)


def build_events(bars_by_symbol: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candles = {symbol: daily_candles(bars) for symbol, bars in bars_by_symbol.items()}
    picked = pd.concat(
        [rank_gainers(candles), rank_losers(candles)],
        ignore_index=True,
    )
    if picked.empty:
        return picked, pd.DataFrame()
    events = []
    exits = []
    for row in picked.itertuples(index=False):
        forward = forward_day(bars_by_symbol[row.symbol], int(row.day_ms))
        if forward.empty:
            continue
        entry = float(forward["open"].iloc[0])
        close = float(row.close)
        if not np.isfinite(entry) or entry <= 0.0 or not np.isfinite(close) or close <= 0.0:
            continue
        side = int(row.side)
        day_range = float(row.high - row.low)
        rollback = maximum_rollback(forward, close, side, float(row.prior_close))
        event = {
            "day_ms": int(row.day_ms),
            "symbol": row.symbol,
            "side": side,
            "return_24h": float(row.return_24h),
            "day_open": float(row.open),
            "day_high": float(row.high),
            "day_low": float(row.low),
            "day_close": close,
            "prior_close": float(row.prior_close),
            "entry_price": entry,
            "day_range": day_range,
            **rollback,
        }
        if side < 0:
            event.update(session_path(forward, entry))
            for rule in EXIT_RULES:
                result = evaluate_exit(forward, entry, float(row.prior_close), day_range, rule)
                exits.append({"day_ms": int(row.day_ms), "symbol": row.symbol, "side": side, **result})
        events.append(event)
    return pd.DataFrame(events), pd.DataFrame(exits)


def _rule_summary(exits: pd.DataFrame) -> List[dict]:
    rows = []
    for rule, group in exits.groupby("rule", sort=False):
        net = group["ret_net"].to_numpy(dtype=float)
        gross = group["ret_gross"].to_numpy(dtype=float)
        rows.append(
            {
                "rule": rule,
                "n": int(len(group)),
                "mean_gross": float(np.mean(gross)),
                "mean_net": float(np.mean(net)),
                "median_net": float(np.median(net)),
                "win_rate_net": float(np.mean(net > 0.0)),
                "mean_hold_hours": float(group["hold_hours"].mean()),
                "stop_rate": float(np.mean(group["exit_reason"] == "stop")),
                "target_rate": float(np.mean(group["exit_reason"] == "target")),
            }
        )
    return rows


def _rollback_summary(events: pd.DataFrame) -> Dict[str, object]:
    """Average of each day's deepest rollback, from the signal close."""
    if events.empty:
        return {}
    rollback = events["max_rollback"].to_numpy(dtype=float)
    retrace = events["retrace_of_candle"].to_numpy(dtype=float)

    def block(frame: pd.DataFrame) -> Dict[str, object]:
        values = frame["max_rollback"].to_numpy(dtype=float)
        given = frame["retrace_of_candle"].to_numpy(dtype=float)
        return {
            "n": int(len(frame)),
            "share_with_any_rollback": float(np.mean(values > 0.0)),
            "mean_max_rollback": float(np.mean(values)),
            "median_max_rollback": float(np.median(values)),
            "p10_max_rollback": float(np.quantile(values, 0.10)),
            "p90_max_rollback": float(np.quantile(values, 0.90)),
            "mean_retrace_of_candle": float(np.mean(given)),
            "median_retrace_of_candle": float(np.median(given)),
            "mean_by_london_open": float(frame["by_london_open"].mean()),
            "mean_by_new_york_open": float(frame["by_new_york_open"].mean()),
            "mean_by_new_york_close": float(frame["by_new_york_close"].mean()),
            "mean_by_next_close": float(frame["by_next_close"].mean()),
            "mean_max_continuation": float(frame["max_continuation"].mean()),
            "median_rollback_hours": float(np.nanmedian(frame["rollback_hours"].to_numpy(dtype=float))),
        }

    shorts = events[events["side"] < 0]
    longs = events[events["side"] > 0]
    return {
        "definition": "deepest print back through the signal close over the next 24h, as a fraction of that close",
        "all": block(events),
        "short_biggest_gainer": block(shorts) if len(shorts) else {},
        "long_biggest_loser": block(longs) if len(longs) else {},
        "mean_max_rollback_check": float(np.mean(rollback)),
        "mean_retrace_check": float(np.mean(retrace)),
    }


def _path_summary(events: pd.DataFrame) -> Dict[str, float]:
    columns = [
        "london_open",
        "new_york_open",
        "new_york_close",
        "next_daily_close",
        "during_asia",
        "during_london",
        "during_new_york",
        "during_late",
    ]
    return {column: float(events[column].mean()) for column in columns if column in events.columns}


def run_study() -> Dict[str, object]:
    bars = load_universe()
    events, exits = build_events(bars)
    folder = paths.DAILY_FADE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    events.to_parquet(folder / "events.parquet", index=False)
    exits.to_parquet(folder / "exits.parquet", index=False)
    summary = {
        "n_symbols": len(bars),
        "n_events": int(len(events)),
        "first_day_ms": int(events["day_ms"].min()) if len(events) else None,
        "last_day_ms": int(events["day_ms"].max()) if len(events) else None,
        "round_trip_cost": ROUND_TRIP_COST,
        "n_shorts": int((events["side"] < 0).sum()) if len(events) else 0,
        "n_longs": int((events["side"] > 0).sum()) if len(events) else 0,
        "side": "short the largest gainer and long the largest loser",
        "entry": "first 15m open of the next UTC day",
        "rollback": _rollback_summary(events) if len(events) else {},
        "session_path_mean_short_return": _path_summary(events[events["side"] < 0]) if len(events) else {},
        "exits": _rule_summary(exits) if len(exits) else [],
        "orders_sent": False,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_study()
    log.info(json.dumps({k: summary[k] for k in ("n_symbols", "n_events", "orders_sent")}, indent=2))


if __name__ == "__main__":
    main()
