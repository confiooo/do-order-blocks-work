"""Breakout + trailing stop from the HAWK alert box.

Wait for the first touch of zone_high or zone_low after the alert bar.
Market-enter in that direction. Trail a stop from the favourable extreme.
Ignores the two-way center entry. Sweep trail size and hold time.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from experiments.zone_trades.common import forward_window, map_symbols, symbol_bars
from experiments.zone_trades.two_way_harvest import build_controls, load_hawk_events
from hawk_research.data.paths import FEATURES_DIR, REPORTS_DIR

log = logging.getLogger(__name__)

WAIT_BARS = 96  # 24h to catch a break
HOLD_BARS = 96  # 24h after entry
HOLD_SNAPS = (16, 32, 96)  # 4h, 8h, 24h after entry
HOLD_NAMES = ("4h", "8h", "24h")
TAKER_FEE = 0.0006
N_FILLS = 2
FEE_DRAG = TAKER_FEE * N_FILLS
SPREAD_DRAG = 0.0016  # mid-cap quoted spread, one position round-trip
TRAIL_PCT = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=np.float64)
BOX_MULT = np.array([0.25, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
N_PCT = len(TRAIL_PCT)
N_BOX = len(BOX_MULT)
N_TRAIL = N_PCT + N_BOX
MAX_BARS = WAIT_BARS + HOLD_BARS


def _path(op: float, hi: float, lo_px: float, cl: float) -> tuple[float, float, float, float]:
    if cl >= op:
        return (op, lo_px, hi, cl)
    return (op, hi, lo_px, cl)


def simulate(
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    zh: float,
    zl: float,
    trails: np.ndarray,
) -> dict[str, Any] | None:
    """Return None if no break inside WAIT_BARS."""
    n = len(h)
    n_t = len(trails)
    wait_n = min(WAIT_BARS, n)
    if wait_n <= 0 or zh <= zl:
        return None

    hit = (h[:wait_n] >= zh) | (low[:wait_n] <= zl)
    if not bool(hit.any()):
        return None
    i0 = int(np.argmax(hit))

    side = 0
    entry = 0.0
    resume_k = 0
    for i in range(i0, wait_n):
        op = float(o[i])
        hi = float(h[i])
        lo_px = float(low[i])
        cl = float(c[i])
        pts = _path(op, hi, lo_px, cl)
        for k, px in enumerate(pts):
            if px >= zh:
                side = 1
                entry = op if (k == 0 and op > zh) else zh
                resume_k = k
                i0 = i
                break
            if px <= zl:
                side = -1
                entry = op if (k == 0 and op < zl) else zl
                resume_k = k
                i0 = i
                break
        if side:
            break
    if side == 0 or not np.isfinite(entry) or entry <= 0:
        return None

    extreme = np.full(n_t, entry, dtype=np.float64)
    best = np.full(n_t, entry, dtype=np.float64)
    worst = np.full(n_t, entry, dtype=np.float64)
    alive = np.ones(n_t, dtype=bool)
    exit_px = np.full(n_t, np.nan)
    last_close = float(c[i0])
    n_h = len(HOLD_SNAPS)
    snaps_px = np.full((n_h, n_t), np.nan)
    snaps_stop = np.zeros((n_h, n_t), dtype=bool)
    snaps_mfe = np.zeros((n_h, n_t), dtype=np.float64)
    snaps_mae = np.zeros((n_h, n_t), dtype=np.float64)
    snap_at = {int(s): i for i, s in enumerate(HOLD_SNAPS)}

    def manage(px: float, gapped: bool) -> None:
        if not alive.any():
            return
        a = alive
        if side == 1:
            extreme[a] = np.maximum(extreme[a], px)
            best[a] = np.maximum(best[a], px)
            worst[a] = np.minimum(worst[a], px)
            stop = extreme * (1.0 - trails)
            hit_sl = a & (px <= stop)
            if hit_sl.any():
                exit_px[hit_sl] = px if gapped else stop[hit_sl]
                alive[hit_sl] = False
        else:
            extreme[a] = np.minimum(extreme[a], px)
            best[a] = np.minimum(best[a], px)
            worst[a] = np.maximum(worst[a], px)
            stop = extreme * (1.0 + trails)
            hit_sl = a & (px >= stop)
            if hit_sl.any():
                exit_px[hit_sl] = px if gapped else stop[hit_sl]
                alive[hit_sl] = False

    def mark_favour() -> tuple[np.ndarray, np.ndarray]:
        if side == 1:
            return (best - entry) / entry, (entry - worst) / entry
        return (entry - best) / entry, (worst - entry) / entry

    def snapshot(si: int) -> None:
        snaps_px[si] = np.where(alive, last_close, exit_px)
        snaps_stop[si] = ~alive
        mfe_now, mae_now = mark_favour()
        snaps_mfe[si] = mfe_now
        snaps_mae[si] = mae_now

    pts0 = _path(float(o[i0]), float(h[i0]), float(low[i0]), float(c[i0]))
    for k in range(resume_k, 4):
        manage(pts0[k], gapped=(k == 0))
    last_close = pts0[3]

    bars_used = 1
    if bars_used in snap_at:
        snapshot(snap_at[bars_used])

    j_end = min(i0 + HOLD_BARS, n)
    for j in range(i0 + 1, j_end):
        if not alive.any():
            for need, si in snap_at.items():
                if need >= bars_used and not np.isfinite(snaps_px[si, 0]):
                    snapshot(si)
            break
        op = float(o[j])
        hi = float(h[j])
        lo_px = float(low[j])
        cl = float(c[j])
        pts = _path(op, hi, lo_px, cl)
        manage(pts[0], gapped=True)
        for px in pts[1:]:
            manage(px, gapped=False)
        last_close = cl
        bars_used += 1
        if bars_used in snap_at:
            snapshot(snap_at[bars_used])
        if bars_used >= HOLD_BARS:
            break

    avail = n - i0
    for si in range(len(HOLD_SNAPS)):
        if not np.isfinite(snaps_px[si]).all():
            if alive.any():
                pass
            else:
                snapshot(si)
    valid = np.zeros((n_h, n_t), dtype=bool)
    for si, need in enumerate(HOLD_SNAPS):
        reached = np.isfinite(snaps_px[si])
        valid[si] = reached & (snaps_stop[si] | (avail >= int(need)))

    if side == 1:
        gross = (snaps_px - entry) / entry
    else:
        gross = (entry - snaps_px) / entry

    return {
        "side": side,
        "wait_bars": i0 + 1,
        "gross": gross,
        "mfe": snaps_mfe,
        "mae": snaps_mae,
        "stopped": snaps_stop,
        "valid": valid,
    }


def trail_labels() -> list[tuple[str, str, float]]:
    rows = [("pct", f"{p:g}% of price", float(p)) for p in TRAIL_PCT]
    rows += [("box", f"{m:g}× box", float(m)) for m in BOX_MULT]
    return rows


def summarize(
    gross: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    stopped: np.ndarray,
    sides: np.ndarray,
    n_alerts: int,
    label: str,
    kind: str,
    param: float,
    horizon: str,
) -> dict[str, Any]:
    n_traded = int(len(gross))
    net = gross - FEE_DRAG
    net_sp = gross - FEE_DRAG - SPREAD_DRAG
    wins = net > 0
    gp = float(net[net > 0].sum()) if n_traded else 0.0
    gl = float(-net[net < 0].sum()) if n_traded else 0.0
    n_long = int((sides == 1).sum())
    n_short = int((sides == -1).sum())
    per_alert = float(net.sum() / n_alerts) * 100.0 if n_alerts else 0.0
    return {
        "horizon": horizon,
        "kind": kind,
        "label": label,
        "param": param,
        "n_alerts": n_alerts,
        "n_traded": n_traded,
        "break_pct": (n_traded / n_alerts * 100.0) if n_alerts else 0.0,
        "long_pct": (n_long / n_traded * 100.0) if n_traded else 0.0,
        "short_pct": (n_short / n_traded * 100.0) if n_traded else 0.0,
        "stop_pct": float(stopped.mean() * 100.0) if n_traded else 0.0,
        "timeout_pct": float((~stopped).mean() * 100.0) if n_traded else 0.0,
        "win_pct": float(wins.mean() * 100.0) if n_traded else 0.0,
        "gross_avg_pct": float(gross.mean() * 100.0) if n_traded else 0.0,
        "net_avg_pct": float(net.mean() * 100.0) if n_traded else 0.0,
        "net_spread_avg_pct": float(net_sp.mean() * 100.0) if n_traded else 0.0,
        "net_per_alert_pct": per_alert,
        "win_mean_pct": float(net[wins].mean() * 100.0) if wins.any() else None,
        "loss_mean_pct": float(net[~wins].mean() * 100.0) if (~wins).any() else None,
        "mfe_avg_pct": float(mfe.mean() * 100.0) if n_traded else 0.0,
        "mae_avg_pct": float(mae.mean() * 100.0) if n_traded else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else None,
        "profitable_traded": bool(n_traded and net.mean() > 0),
        "profitable_per_alert": bool(n_alerts and net.sum() > 0),
    }


def _bucket_rows(
    grosses: list[list[list[float]]],
    mfes: list[list[list[float]]],
    maes: list[list[list[float]]],
    stops: list[list[list[bool]]],
    sides: list[list[list[int]]],
    n_alerts: int,
    arm: str,
) -> list[dict[str, Any]]:
    labels = trail_labels()
    rows = []
    for hi, name in enumerate(HOLD_NAMES):
        for ti, (kind, lab, param) in enumerate(labels):
            g_arr = np.asarray(grosses[hi][ti], dtype=np.float64)
            row = summarize(
                g_arr,
                np.asarray(mfes[hi][ti], dtype=np.float64),
                np.asarray(maes[hi][ti], dtype=np.float64),
                np.asarray(stops[hi][ti], dtype=bool),
                np.asarray(sides[hi][ti], dtype=np.int8),
                n_alerts,
                lab,
                kind,
                param,
                name,
            )
            rows.append(row)
            log.info(
                f"  {arm} {name} {lab:14s}  break={row['break_pct']:5.1f}%  "
                f"win={row['win_pct']:5.1f}%  net={row['net_avg_pct']:+.3f}%  "
                f"per_alert={row['net_per_alert_pct']:+.3f}%  "
                f"{'PROFIT' if row['profitable_traded'] else 'loss'}"
            )
    return rows


def _simulate_symbol(symbol: str, events: pd.DataFrame) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    """Alerts with a usable window, and the simulation of each one that broke out."""
    bars = symbol_bars(symbol)
    if bars is None:
        return 0, []
    n_alerts = 0
    breaks = []
    for zh, zl, src in zip(
        events["zone_high"].to_numpy(dtype=np.float64),
        events["zone_low"].to_numpy(dtype=np.float64),
        events["source_open_time"].to_numpy(dtype=np.int64),
    ):
        w = forward_window(bars["open_time"], src, MAX_BARS)
        mid = 0.5 * (zh + zl)
        if w is None or mid <= 0:
            continue
        n_alerts += 1
        box_frac = (zh - zl) / mid
        trails = np.empty(N_TRAIL, dtype=np.float64)
        trails[:N_PCT] = TRAIL_PCT / 100.0
        trails[N_PCT:] = BOX_MULT * box_frac
        trails = np.maximum(trails, 1e-6)
        out = simulate(
            bars["open"][w], bars["high"][w], bars["low"][w], bars["close"][w], float(zh), float(zl), trails
        )
        if out is not None:
            breaks.append((int(src), out))
    return n_alerts, breaks


def run_arm(events: pd.DataFrame, arm: str) -> dict[str, Any]:
    n_h = len(HOLD_SNAPS)
    grosses: list[list[list[float]]] = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
    mfes: list[list[list[float]]] = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
    maes: list[list[list[float]]] = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
    stops: list[list[list[bool]]] = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
    sides: list[list[list[int]]] = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
    times: list[list[list[int]]] = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
    waits: list[int] = []
    n_alerts = 0
    n_break = 0

    for sym_alerts, sym_breaks in map_symbols(_simulate_symbol, events):
        n_alerts += sym_alerts
        for src, out in sym_breaks:
            n_break += 1
            waits.append(int(out["wait_bars"]))
            side = int(out["side"])
            for hi in range(n_h):
                for ti in range(N_TRAIL):
                    if not bool(out["valid"][hi, ti]):
                        continue
                    grosses[hi][ti].append(float(out["gross"][hi, ti]))
                    mfes[hi][ti].append(float(out["mfe"][hi, ti]))
                    maes[hi][ti].append(float(out["mae"][hi, ti]))
                    stops[hi][ti].append(bool(out["stopped"][hi, ti]))
                    sides[hi][ti].append(side)
                    times[hi][ti].append(src)

    rows = _bucket_rows(grosses, mfes, maes, stops, sides, n_alerts, arm)
    split = None
    if arm == "hawk" and times[0][0]:
        cut_t = float(np.quantile(np.asarray(times[0][0], dtype=np.int64), 0.7))
        split = {}
        for tag, pred in (("is", lambda t: t <= cut_t), ("oos", lambda t: t > cut_t)):
            g2 = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
            m2 = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
            a2 = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
            s2 = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
            d2 = [[[] for _ in range(N_TRAIL)] for _ in range(n_h)]
            n_al = 0
            # n_alerts in a time split isn't stored per event for no-break;
            # use traded-only expectancy and skip per-alert. Count alerts
            # approximately via overall 70/30 of n_alerts.
            for hi in range(n_h):
                for ti in range(N_TRAIL):
                    for k, t in enumerate(times[hi][ti]):
                        if pred(t):
                            g2[hi][ti].append(grosses[hi][ti][k])
                            m2[hi][ti].append(mfes[hi][ti][k])
                            a2[hi][ti].append(maes[hi][ti][k])
                            s2[hi][ti].append(stops[hi][ti][k])
                            d2[hi][ti].append(sides[hi][ti][k])
            n_al = int(round(n_alerts * (0.7 if tag == "is" else 0.3)))
            split[tag] = {
                "arm": f"hawk_{tag}",
                "n_alerts": n_al,
                "rows": _bucket_rows(g2, m2, a2, s2, d2, n_al, f"hawk_{tag}"),
            }

    return {
        "arm": arm,
        "n_alerts": n_alerts,
        "n_break": n_break,
        "median_wait_bars": float(np.median(waits)) if waits else None,
        "rows": rows,
        "time_split": split,
    }


def main() -> None:
    hawk = load_hawk_events(FEATURES_DIR / "zone_dataset.parquet")
    log.info(f"hawk alerts={len(hawk)}")
    hawk_res = run_arm(hawk, "hawk")
    controls = build_controls(hawk)
    log.info(f"controls={len(controls)}")
    ctrl_res = (
        run_arm(controls, "control")
        if len(controls)
        else {"rows": [], "n_alerts": 0, "n_break": 0, "time_split": None}
    )

    def best_of(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        traded = [r for r in rows if r["n_traded"] > 0]
        return max(traded, key=lambda r: r["net_avg_pct"]) if traded else None

    report = {
        "spec": {
            "entry": "market on first touch of zone_high (long) or zone_low (short)",
            "wait": "24h after alert for a break; no trade if neither edge prints",
            "stop": "trailing from favourable extreme; initial stop = entry ± trail",
            "path": "15m OHLC; up-bar O-L-H-C; down-bar O-H-L-C; gap through stop fills at open",
            "fees": "0.06% taker × 2 fills = 0.12%",
            "spread": "0.16% mid-cap round-trip, shown separately",
            "population": "would_alert, zone ≥ 0.50%, close within 0.45% of midpoint",
            "period": "2024-09-05 → 2026-09-17",
        },
        "hawk": {k: v for k, v in hawk_res.items() if k != "time_split"},
        "control": {k: v for k, v in ctrl_res.items() if k != "time_split"},
        "time_split": hawk_res.get("time_split"),
        "best_hawk": best_of(hawk_res["rows"]),
        "any_profitable_hawk": any(r["profitable_traded"] for r in hawk_res["rows"]),
        "any_profitable_control": any(r["profitable_traded"] for r in ctrl_res.get("rows", [])),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "breakout_trail.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"wrote {out}  any_profit_hawk={report['any_profitable_hawk']}")


if __name__ == "__main__":
    main()
