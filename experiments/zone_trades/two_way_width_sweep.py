"""Sweep synthetic straddle width around the HAWK box center (alert price).

Ignores the real zone high/low. Entry = midpoint. TP at center ± half-width.
SL 5% of the synthetic box beyond each TP. Same 15m path rules as two_way_harvest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.zone_trades.common import forward_window, map_symbols, symbol_bars
from hawk_research.data import loader
from hawk_research.data.paths import FEATURES_DIR, REPORTS_DIR

log = logging.getLogger(__name__)

BUFFER_FRAC = 0.05
TAKER_FEE = 0.0006
WIDTHS_PCT = np.array(
    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0],
    dtype=np.float64,
)
# Snapshot bars: 4h, 8h, 24h
HORIZON_BARS = (16, 32, 96)
HORIZON_NAMES = ("4h", "8h", "24h")
MAX_BARS = 96


def load_alerts(zone_path: Path) -> pd.DataFrame:
    z = pd.read_parquet(
        zone_path,
        columns=["symbol", "source_open_time", "zone_high", "zone_low", "would_alert"],
    )
    z = z.loc[z["would_alert"].astype(bool)].copy()
    z["mid"] = (z["zone_high"].to_numpy() + z["zone_low"].to_numpy()) / 2.0
    z = z.loc[np.isfinite(z["mid"]) & (z["mid"] > 0)]
    parts = []
    for sym, g in z.groupby("symbol", sort=False):
        bars = loader.load_ohlcv(sym, columns=["open_time"], missing_ok=True)
        if bars.empty:
            continue
        times = bars["open_time"].astype("int64")
        keep = g["source_open_time"].astype("int64").isin(set(times))
        gg = g.loc[keep]
        if not gg.empty:
            parts.append(gg)
    return pd.concat(parts, ignore_index=True) if parts else z.iloc[0:0]


def walk_all_widths(
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    entry: float,
    half: np.ndarray,
    buf: np.ndarray,
    snapshots: tuple[int, ...] = HORIZON_BARS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One path, every width. Snapshot (net, outcome) after 16/32/96 bars."""
    n_w = len(half)
    tp_l = entry * (1.0 + half)
    tp_s = entry * (1.0 - half)
    sl_l = entry * (1.0 - half - buf)
    sl_s = entry * (1.0 + half + buf)
    long_open = np.ones(n_w, dtype=bool)
    short_open = np.ones(n_w, dtype=bool)
    long_pnl = np.zeros(n_w, dtype=np.float64)
    short_pnl = np.zeros(n_w, dtype=np.float64)
    n_exits = np.zeros(n_w, dtype=np.int8)
    outcome = np.full(n_w, 2, dtype=np.int8)
    done = np.zeros(n_w, dtype=bool)
    snap_at = {int(s): i for i, s in enumerate(snapshots)}
    snaps: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(snapshots)

    def close_long(idx: np.ndarray, px: np.ndarray) -> None:
        long_pnl[idx] = (px - entry) / entry
        long_open[idx] = False
        n_exits[idx] += 1

    def close_short(idx: np.ndarray, px: np.ndarray) -> None:
        short_pnl[idx] = (entry - px) / entry
        short_open[idx] = False
        n_exits[idx] += 1

    def snapshot(px: float) -> tuple[np.ndarray, np.ndarray]:
        lp = long_pnl.copy()
        sp = short_pnl.copy()
        nex = n_exits.copy()
        oc = outcome.copy()
        lo = long_open.copy()
        so = short_open.copy()
        if lo.any():
            k = np.flatnonzero(lo)
            lp[k] = (px - entry) / entry
            nex[k] += 1
            oc[k] = 2
        if so.any():
            k = np.flatnonzero(so)
            sp[k] = (entry - px) / entry
            nex[k] += 1
            oc[k] = 2
        net = lp + sp - TAKER_FEE * (2 + nex).astype(np.float64)
        return net, oc

    n = len(h)
    for i in range(n):
        hi = float(h[i])
        lo_px = float(low[i])
        op = float(o[i])
        cl = float(c[i])
        if not done.all():
            both = (~done) & long_open & short_open
            if both.any():
                bidx = np.flatnonzero(both)
                both_tp = (hi >= tp_l[bidx]) & (lo_px <= tp_s[bidx])
                hit_sl_s = hi >= sl_s[bidx]
                hit_sl_l = lo_px <= sl_l[bidx]
                clean = both_tp & ~hit_sl_s & ~hit_sl_l
                if clean.any():
                    k = bidx[clean]
                    close_long(k, tp_l[k])
                    close_short(k, tp_s[k])
                    outcome[k] = 0
                    done[k] = True
                messy = (
                    (hit_sl_s & (lo_px <= tp_s[bidx]))
                    | (hit_sl_l & (hi >= tp_l[bidx]))
                    | (hit_sl_s & hit_sl_l)
                ) & ~clean
                if messy.any():
                    k = bidx[messy]
                    # Same bar hits both sides; use the farther wick as the fill path.
                    if (hi - entry) >= (entry - lo_px):
                        close_long(k, tp_l[k])
                        close_short(k, sl_s[k])
                    else:
                        close_short(k, tp_s[k])
                        close_long(k, sl_l[k])
                    outcome[k] = 3
                    done[k] = True

            path = (op, lo_px, hi, cl) if cl >= op else (op, hi, lo_px, cl)
            for px in path:
                if done.all():
                    break
                still = ~done
                hit_tl = still & long_open & (px >= tp_l)
                hit_slong = still & long_open & (px <= sl_l)
                hit_ts = still & short_open & (px <= tp_s)
                hit_ss = still & short_open & (px >= sl_s)
                if hit_tl.any():
                    close_long(np.flatnonzero(hit_tl), tp_l[hit_tl])
                if hit_slong.any():
                    close_long(np.flatnonzero(hit_slong), sl_l[hit_slong])
                if hit_ts.any():
                    close_short(np.flatnonzero(hit_ts), tp_s[hit_ts])
                if hit_ss.any():
                    close_short(np.flatnonzero(hit_ss), sl_s[hit_ss])
                just = still & ~long_open & ~short_open
                if just.any():
                    j = np.flatnonzero(just)
                    win = (long_pnl[j] > 0) & (short_pnl[j] > 0)
                    outcome[j] = np.where(win, 0, 1)
                    done[j] = True

        bar_no = i + 1
        if bar_no in snap_at:
            snaps[snap_at[bar_no]] = snapshot(cl)

    return snaps


def summarize_bucket(
    nets: np.ndarray, outcomes: np.ndarray, width_pct: float, horizon: str, buffer_frac: float
) -> dict[str, Any]:
    n = int(len(nets))
    two_way = float((outcomes == 0).mean())
    w = width_pct / 100.0
    win_styl = w - (TAKER_FEE * 4)
    loss_styl = -(w * buffer_frac) - (TAKER_FEE * 4)
    be = float(loss_styl / (loss_styl - win_styl)) * 100.0
    gp = float(nets[nets > 0].sum())
    gl = float(-nets[nets < 0].sum())
    return {
        "horizon": horizon,
        "width_pct": float(width_pct),
        "n": n,
        "two_way_pct": two_way * 100.0,
        "loss_pct": float((outcomes == 1).mean() * 100.0),
        "timeout_pct": float((outcomes == 2).mean() * 100.0),
        "ambiguous_pct": float((outcomes == 3).mean() * 100.0),
        "net_avg_pct": float(nets.mean() * 100.0),
        "profit_factor": (gp / gl) if gl > 0 else None,
        "stylized_be_two_way_pct": be,
        "profitable_net": bool(nets.mean() > 0),
    }


def _walk_symbol(symbol: str, alerts: pd.DataFrame, half: np.ndarray, buf: np.ndarray) -> list:
    bars = symbol_bars(symbol)
    if bars is None:
        return []
    out = []
    for z in alerts.itertuples(index=False):
        w = forward_window(bars["open_time"], z.source_open_time, MAX_BARS)
        if w is not None:
            out.append(
                walk_all_widths(
                    bars["open"][w],
                    bars["high"][w],
                    bars["low"][w],
                    bars["close"][w],
                    float(z.mid),
                    half,
                    buf,
                )
            )
    return out


def run_sweep(
    tag: str, buffer_frac: float, buf: np.ndarray, buffer_note: str, report_name: str
) -> dict[str, Any]:
    """Every width and horizon over every alert, with stop buffer ``buf`` beyond each TP."""
    alerts = load_alerts(FEATURES_DIR / "zone_dataset.parquet")
    log.info(f"[{tag}] alerts={len(alerts)}")
    half = (WIDTHS_PCT / 100.0) / 2.0
    n_w = len(WIDTHS_PCT)
    n_h = len(HORIZON_BARS)

    nets: list[list[list[float]]] = [[[] for _ in range(n_w)] for _ in range(n_h)]
    outs: list[list[list[int]]] = [[[] for _ in range(n_w)] for _ in range(n_h)]
    for paths in map_symbols(_walk_symbol, alerts, half=half, buf=buf):
        for snaps in paths:
            for hi, pack in enumerate(snaps):
                if pack is None:
                    continue
                net, outcome = pack
                for wi in range(n_w):
                    nets[hi][wi].append(net[wi])
                    outs[hi][wi].append(outcome[wi])

    rows: list[dict[str, Any]] = []
    for hi, name in enumerate(HORIZON_NAMES):
        for wi, width_pct in enumerate(WIDTHS_PCT):
            if not nets[hi][wi]:
                continue
            row = summarize_bucket(
                np.asarray(nets[hi][wi], dtype=np.float64),
                np.asarray(outs[hi][wi], dtype=np.int8),
                float(width_pct),
                name,
                buffer_frac,
            )
            rows.append(row)
            log.info(
                f"  {name} w={width_pct:5.2f}%  two-way={row['two_way_pct']:5.2f}%  "
                f"net={row['net_avg_pct']:+.4f}%  need={row['stylized_be_two_way_pct']:.1f}%  "
                f"{'PROFIT' if row['profitable_net'] else 'loss'}"
            )

    profitable = [r for r in rows if r["profitable_net"]]
    best = max(rows, key=lambda r: r["net_avg_pct"]) if rows else None
    report = {
        "entry": "box midpoint at alert (real high/low ignored)",
        "buffer": buffer_note,
        "fees": "0.06% taker × number of fills (usually 0.24%)",
        "n_alerts": int(len(alerts)),
        "rows": rows,
        "any_profitable": bool(profitable),
        "profitable_rows": profitable,
        "best": best,
    }
    out = REPORTS_DIR / report_name
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"[{tag}] wrote {out}  any_profit={bool(profitable)}")
    return report


def sweep() -> dict[str, Any]:
    return run_sweep(
        "sweep",
        BUFFER_FRAC,
        (WIDTHS_PCT / 100.0) * BUFFER_FRAC,
        "5% of synthetic full width beyond each TP",
        "two_way_width_sweep.json",
    )


if __name__ == "__main__":
    sweep()
