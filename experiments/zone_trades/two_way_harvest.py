"""Two-way harvest backtest: dual TP at zone edges, SL just beyond the other TP.

Entry is the signal bar close (alert time = next 15m open). No 1m archive:
intra-bar path is open->low->high->close on up bars, open->high->low->close on
down bars. A 15m print that spans a TP and the opposite SL is ambiguous and
resolved pessimistically (one-way: near TP then far SL).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.data import loader
from hawk_research.data.paths import FEATURES_DIR, REPORTS_DIR

log = logging.getLogger(__name__)

HORIZON_BARS = 16  # 4h
BUFFER_FRAC = 0.05
MIN_ZONE_PCT = 0.50  # skip boxes where fees dominate
MID_CLOSE_MAX_DIST = 0.45  # same as Telegram gate, percent of price
TAKER_FEE = 0.0006  # MEXC API taker
N_FILLS = 4
FEE_DRAG = TAKER_FEE * N_FILLS
SPREAD_DRAG = 0.0016 * 2  # 16 bps quoted spread × two positions, round-trip


def _fee_drag(n_fills: int) -> float:
    return TAKER_FEE * float(n_fills)


def simulate_path(
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    entry: float,
    tp_l: float,
    tp_s: float,
    sl_l: float,
    sl_s: float,
) -> tuple[str, float, float, int, int]:
    """Walk forward bars. Returns outcome, long_pnl, short_pnl, bars_used, n_exits."""
    long_open = True
    short_open = True
    long_pnl = 0.0
    short_pnl = 0.0
    n_exits = 0

    def close_long(px: float) -> None:
        nonlocal long_open, long_pnl, n_exits
        if long_open:
            long_pnl = (px - entry) / entry
            long_open = False
            n_exits += 1

    def close_short(px: float) -> None:
        nonlocal short_open, short_pnl, n_exits
        if short_open:
            short_pnl = (entry - px) / entry
            short_open = False
            n_exits += 1

    n = len(h)
    for i in range(n):
        hi = float(h[i])
        lo = float(low[i])
        op = float(o[i])
        cl = float(c[i])

        if long_open and short_open:
            both_tp = (hi >= tp_l) and (lo <= tp_s)
            hit_sl_s = hi >= sl_s
            hit_sl_l = lo <= sl_l
            if both_tp and not hit_sl_s and not hit_sl_l:
                close_long(tp_l)
                close_short(tp_s)
                return "win", long_pnl, short_pnl, i + 1, n_exits
            if (hit_sl_s and lo <= tp_s) or (hit_sl_l and hi >= tp_l) or (hit_sl_s and hit_sl_l):
                # Ambiguous 15m span: pessimistic one-way by larger excursion.
                if (hi - entry) >= (entry - lo):
                    close_long(tp_l)
                    close_short(sl_s)
                else:
                    close_short(tp_s)
                    close_long(sl_l)
                return "ambiguous", long_pnl, short_pnl, i + 1, n_exits

        if cl >= op:
            path = (op, lo, hi, cl)
        else:
            path = (op, hi, lo, cl)
        for px in path:
            if long_open and px >= tp_l:
                close_long(tp_l)
            if long_open and px <= sl_l:
                close_long(sl_l)
            if short_open and px <= tp_s:
                close_short(tp_s)
            if short_open and px >= sl_s:
                close_short(sl_s)
            if not long_open and not short_open:
                break
        if not long_open and not short_open:
            if long_pnl > 0 and short_pnl > 0:
                return "win", long_pnl, short_pnl, i + 1, n_exits
            return "loss", long_pnl, short_pnl, i + 1, n_exits

    last = float(c[-1]) if n else entry
    if long_open:
        close_long(last)
    if short_open:
        close_short(last)
    return "timeout", long_pnl, short_pnl, n, n_exits


def run_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sym, grp in events.groupby("symbol", sort=False):
        bars = loader.load_ohlcv(sym, columns=["open_time", "open", "high", "low", "close"], missing_ok=True)
        if bars.empty:
            continue
        bars = bars.sort_values("open_time")
        opens = bars["open_time"].to_numpy(dtype=np.int64)
        o = bars["open"].to_numpy(dtype=np.float64)
        h = bars["high"].to_numpy(dtype=np.float64)
        low = bars["low"].to_numpy(dtype=np.float64)
        c = bars["close"].to_numpy(dtype=np.float64)

        for z in grp.itertuples(index=False):
            start = int(z.source_open_time) + BAR_MS
            i0 = int(np.searchsorted(opens, start, side="left"))
            i1 = min(i0 + HORIZON_BARS, len(opens))
            if i0 >= len(opens) or i1 <= i0:
                continue
            # Contiguous 15m window only.
            if int(opens[i1 - 1]) != start + (i1 - i0 - 1) * BAR_MS:
                span = int(opens[i1 - 1] - start) // BAR_MS + 1
                if span != (i1 - i0):
                    continue
            entry = float(z.close)
            zh, zl = float(z.zone_high), float(z.zone_low)
            buf = (zh - zl) * BUFFER_FRAC
            outcome, lp, sp, held, n_exits = simulate_path(
                o[i0:i1],
                h[i0:i1],
                low[i0:i1],
                c[i0:i1],
                entry,
                zh,
                zl,
                zl - buf,
                zh + buf,
            )
            fills = 2 + n_exits
            gross = lp + sp
            rows.append(
                {
                    "symbol": sym,
                    "source_open_time": int(z.source_open_time),
                    "zone_pct": float(z.zone_pct),
                    "outcome": outcome,
                    "long_pnl": lp,
                    "short_pnl": sp,
                    "gross": gross,
                    "net_fee": gross - _fee_drag(fills),
                    "net_fee_spread": gross - _fee_drag(fills) - SPREAD_DRAG,
                    "fills": fills,
                    "bars_held": held,
                    "arm": z.arm,
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, arm: str) -> dict[str, Any]:
    sub = df.loc[df["arm"] == arm]
    n = int(len(sub))
    if n == 0:
        return {"arm": arm, "n": 0}
    ordered = sub.sort_values("source_open_time")
    equity = ordered["net_fee"].to_numpy(dtype=np.float64).cumsum()
    peak = np.maximum.accumulate(equity)
    dd = float((equity - peak).min()) if n else 0.0
    gp = float(sub.loc[sub["net_fee"] > 0, "net_fee"].sum())
    gl = float(-sub.loc[sub["net_fee"] < 0, "net_fee"].sum())
    two_way = float((sub["outcome"] == "win").mean())
    zone = sub["zone_pct"].mean() / 100.0
    # Break-even two-way under stylized WIN=zone, LOSS=-buffer-fees
    win_net = zone - FEE_DRAG
    loss_net = -(zone * BUFFER_FRAC) - FEE_DRAG
    be = float(loss_net / (loss_net - win_net)) if win_net != loss_net else np.nan
    return {
        "arm": arm,
        "n": n,
        "two_way_hit_rate": two_way,
        "loss_rate": float((sub["outcome"] == "loss").mean()),
        "timeout_rate": float((sub["outcome"] == "timeout").mean()),
        "ambiguous_rate": float((sub["outcome"] == "ambiguous").mean()),
        "expectancy_gross_bps": float(sub["gross"].mean() * 1e4),
        "expectancy_net_fee_bps": float(sub["net_fee"].mean() * 1e4),
        "expectancy_net_spread_bps": float(sub["net_fee_spread"].mean() * 1e4),
        "total_net_fee": float(sub["net_fee"].sum()),
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "max_dd_fee": dd,
        "mean_zone_pct": float(sub["zone_pct"].mean()),
        "stylized_breakeven_two_way": be,
        "win_mean_net_bps": float(sub.loc[sub["outcome"] == "win", "net_fee"].mean() * 1e4)
        if (sub["outcome"] == "win").any()
        else None,
        "loss_mean_net_bps": float(sub.loc[sub["outcome"] == "loss", "net_fee"].mean() * 1e4)
        if (sub["outcome"] == "loss").any()
        else None,
        "timeout_mean_net_bps": float(sub.loc[sub["outcome"] == "timeout", "net_fee"].mean() * 1e4)
        if (sub["outcome"] == "timeout").any()
        else None,
        "ambiguous_mean_net_bps": float(sub.loc[sub["outcome"] == "ambiguous", "net_fee"].mean() * 1e4)
        if (sub["outcome"] == "ambiguous").any()
        else None,
    }


def size_buckets(df: pd.DataFrame) -> list[dict[str, Any]]:
    bins = [0.5, 1.0, 2.0, 4.0, 100.0]
    labels = ["0.5–1%", "1–2%", "2–4%", ">4%"]
    sub = df.loc[df["arm"] == "hawk"].copy()
    sub["bucket"] = pd.cut(sub["zone_pct"], bins=bins, labels=labels, right=False)
    out = []
    for lab in labels:
        g = sub.loc[sub["bucket"] == lab]
        if g.empty:
            continue
        out.append(
            {
                "bucket": lab,
                "n": int(len(g)),
                "two_way": float((g["outcome"] == "win").mean()),
                "net_bps": float(g["net_fee"].mean() * 1e4),
                "gross_bps": float(g["gross"].mean() * 1e4),
            }
        )
    return out


def load_hawk_events(zone_path: Path) -> pd.DataFrame:
    cols = [
        "symbol",
        "source_open_time",
        "zone_high",
        "zone_low",
        "zone_size_pct",
        "distance_to_price",
        "would_alert",
        "passes_signal_gate",
    ]
    z = pd.read_parquet(zone_path, columns=cols)
    z = z.rename(columns={"zone_size_pct": "zone_pct"})
    z = z.loc[z["would_alert"].astype(bool)]
    z = z.loc[np.isfinite(z["zone_high"]) & np.isfinite(z["zone_low"])]
    z = z.loc[(z["zone_high"] > z["zone_low"]) & (z["zone_pct"] >= MIN_ZONE_PCT)]
    z = z.loc[z["distance_to_price"] <= MID_CLOSE_MAX_DIST]
    closes = []
    for sym, g in z.groupby("symbol", sort=False):
        bars = loader.load_ohlcv(sym, columns=["open_time", "close"], missing_ok=True)
        if bars.empty:
            continue
        m = g.merge(bars, left_on="source_open_time", right_on="open_time", how="inner")
        closes.append(m.drop(columns=["open_time"]))
    if not closes:
        raise RuntimeError("no hawk events joined to OHLCV")
    z = pd.concat(closes, ignore_index=True)
    z = z.loc[np.isfinite(z["close"])]
    z["arm"] = "hawk"
    return z.reset_index(drop=True)


def build_controls(hawk: pd.DataFrame) -> pd.DataFrame:
    hawk_keys = set(zip(hawk["symbol"], hawk["source_open_time"].astype("int64")))
    parts: list[pd.DataFrame] = []
    for sym, g in hawk.groupby("symbol", sort=False):
        bars = loader.load_ohlcv(sym, columns=["open_time", "open", "high", "low", "close"], missing_ok=True)
        if bars.empty:
            continue
        rng = bars["high"] - bars["low"]
        zone_pct = rng / bars["close"].replace(0, np.nan) * 100.0
        mid = (bars["high"] + bars["low"]) / 2.0
        dist = (mid - bars["close"]).abs() / bars["close"].replace(0, np.nan) * 100.0
        close_pos = (bars["close"] - bars["low"]) / rng.replace(0, np.nan)
        cand = bars.assign(zone_pct=zone_pct, distance_to_price=dist, close_pos=close_pos)
        cand = cand.loc[
            (cand["zone_pct"] >= MIN_ZONE_PCT)
            & (cand["distance_to_price"] <= MID_CLOSE_MAX_DIST)
            & cand["close_pos"].between(0.35, 0.65)
        ]
        mask = [(sym, int(t)) not in hawk_keys for t in cand["open_time"].to_numpy(dtype=np.int64)]
        cand = cand.loc[mask]
        if cand.empty:
            continue
        h = g[["zone_pct"]].copy()
        h = h.sort_values("zone_pct")
        cand = cand.sort_values("zone_pct")
        hp = h["zone_pct"].to_numpy()
        cp = cand["zone_pct"].to_numpy()
        j = 0
        used = np.zeros(len(cand), dtype=bool)
        picks = []
        for i in range(len(hp)):
            while j < len(cp) and cp[j] < hp[i] * 0.75:
                j += 1
            k = j
            found = -1
            best = 1e9
            while k < len(cp) and cp[k] <= hp[i] * 1.25:
                if not used[k]:
                    d = abs(cp[k] - hp[i])
                    if d < best:
                        best = d
                        found = k
                k += 1
            if found < 0:
                continue
            used[found] = True
            r = cand.iloc[found]
            picks.append(
                {
                    "symbol": sym,
                    "source_open_time": int(r.open_time),
                    "zone_high": float(r.high),
                    "zone_low": float(r.low),
                    "close": float(r.close),
                    "zone_pct": float(r.zone_pct),
                    "arm": "control",
                }
            )
            j = found + 1
        if picks:
            parts.append(pd.DataFrame(picks))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def bootstrap_diff(hawk_net: np.ndarray, ctrl_net: np.ndarray, n: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(7)
    diffs = np.empty(n)
    for i in range(n):
        a = hawk_net[rng.integers(0, len(hawk_net), len(hawk_net))].mean()
        b = ctrl_net[rng.integers(0, len(ctrl_net), len(ctrl_net))].mean()
        diffs[i] = a - b
    return {
        "diff_bps": float((hawk_net.mean() - ctrl_net.mean()) * 1e4),
        "ci_low_bps": float(np.percentile(diffs, 2.5) * 1e4),
        "ci_high_bps": float(np.percentile(diffs, 97.5) * 1e4),
    }


def main() -> None:
    zone_path = FEATURES_DIR / "zone_dataset.parquet"
    hawk = load_hawk_events(zone_path)
    log.info(f"hawk events after gate+min_zone: {len(hawk)}")
    controls = build_controls(hawk)
    log.info(f"matched controls: {len(controls)}")
    events = pd.concat([hawk, controls], ignore_index=True)
    trades = run_events(events)
    log.info(f"simulated: {len(trades)}")

    hawk_s = summarize(trades, "hawk")
    ctrl_s = summarize(trades, "control")
    diff = None
    hnet = trades.loc[trades["arm"] == "hawk", "net_fee"].to_numpy()
    cnet = trades.loc[trades["arm"] == "control", "net_fee"].to_numpy()
    if len(hnet) and len(cnet):
        diff = bootstrap_diff(hnet, cnet)

    # Chronological split: first 70% / last 30% by time on hawk only
    ht = trades.loc[trades["arm"] == "hawk"].sort_values("source_open_time")
    cut = int(len(ht) * 0.7)
    is_s = summarize(ht.iloc[:cut].assign(arm="hawk"), "hawk") if cut else None
    oos_s = summarize(ht.iloc[cut:].assign(arm="hawk"), "hawk") if cut < len(ht) else None

    report = {
        "spec": {
            "horizon_hours": 4,
            "buffer_frac_of_zone": BUFFER_FRAC,
            "min_zone_pct": MIN_ZONE_PCT,
            "mid_close_max_dist_pct": MID_CLOSE_MAX_DIST,
            "taker_fee_bps": TAKER_FEE * 1e4,
            "fills_if_both_close": N_FILLS,
            "fee_drag_bps": FEE_DRAG * 1e4,
            "spread_drag_bps": SPREAD_DRAG * 1e4,
            "path": "15m OHLC; up-bar O-L-H-C; down-bar O-H-L-C; TP+opposite-SL same bar = pessimistic one-way",
            "population": "would_alert (Telegram gate) and zone_pct >= 0.50 and close within 0.45% of midpoint",
            "period": "2024-09-05 → 2026-09-17",
            "source": "data/research replay + Binance 15m archive",
        },
        "hawk": hawk_s,
        "control": ctrl_s,
        "hawk_minus_control_net_fee": diff,
        "hawk_is": is_s,
        "hawk_oos": oos_s,
        "by_zone_size": size_buckets(trades),
        "n_hawk_events_input": int(len(hawk)),
        "n_control_events_input": int(len(controls)),
        "n_hawk_simulated": int((trades["arm"] == "hawk").sum()),
        "n_control_simulated": int((trades["arm"] == "control").sum()),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "two_way_harvest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades_path = REPORTS_DIR / "two_way_harvest_trades.parquet"
    trades.to_parquet(trades_path, index=False)
    log.info(json.dumps(report, indent=2))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
