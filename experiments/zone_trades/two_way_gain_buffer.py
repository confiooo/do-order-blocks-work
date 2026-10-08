"""Gain-buffer trailing stop on the two-way center straddle.

Open long+short at box center. TPs at ± half-width. When the first TP fills
(+X), the remaining stop is set to a loss of alpha*X from the original entry.
For alpha < 1 that stop is already through the market (price is at +X, stop at
+alpha*X), so the remaining leg is flattened at the TP fill. Alpha therefore never
changes a fill: each (horizon, width) result is reported once per alpha. Same-bar
prints of both TPs still count as two-way. 4 taker fills × 0.06% on every alert.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from experiments.zone_trades.common import forward_window, symbol_bars
from experiments.zone_trades.two_way_width_sweep import (
    HORIZON_BARS,
    HORIZON_NAMES,
    MAX_BARS,
    load_alerts,
)
from hawk_research.data.paths import FEATURES_DIR, REPORTS_DIR

log = logging.getLogger(__name__)

TAKER_FEE = 0.0006
N_FILLS = 4
FEE_DRAG = TAKER_FEE * N_FILLS
WIDTHS_PCT = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=np.float64)
ALPHAS = np.array([0.25, 0.50, 0.75], dtype=np.float64)
UNVIABLE_MSG = (
    "Gain-buffer trailing confirmed unviable. The tightened stop gets triggered "
    "prematurely by market noise before the price can complete the full-box reversal."
)


def first_touch(
    h: np.ndarray,
    low: np.ndarray,
    entry: float,
    half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First bar that tags a TP, and whether that bar also tagged the other TP."""
    n_w = len(half)
    n = len(h)
    first_i = np.full(n_w, -1, dtype=np.int32)
    both = np.zeros(n_w, dtype=bool)
    tp_l = entry * (1.0 + half)
    tp_s = entry * (1.0 - half)
    pending = np.ones(n_w, dtype=bool)
    for i in range(n):
        if not pending.any():
            break
        hi = float(h[i])
        lo_px = float(low[i])
        hit_l = hi >= tp_l
        hit_s = lo_px <= tp_s
        hit = pending & (hit_l | hit_s)
        if hit.any():
            first_i[hit] = i
            both[hit] = hit_l[hit] & hit_s[hit]
            pending[hit] = False
    return first_i, both


def summarize(
    two_way: np.ndarray,
    one_way: np.ndarray,
    timeout: np.ndarray,
    nets: np.ndarray,
    width_pct: float,
    alpha: float,
    horizon: str,
) -> dict[str, Any]:
    n = int(len(nets))
    w = width_pct / 100.0
    be = (FEE_DRAG / w) * 100.0 if w > 0 else None
    gp = float(nets[nets > 0].sum()) if n else 0.0
    gl = float(-nets[nets < 0].sum()) if n else 0.0
    return {
        "horizon": horizon,
        "width_pct": float(width_pct),
        "alpha_pct": float(alpha * 100.0),
        "n": n,
        "two_way_pct": float(two_way.mean() * 100.0) if n else 0.0,
        "one_way_pct": float(one_way.mean() * 100.0) if n else 0.0,
        "timeout_pct": float(timeout.mean() * 100.0) if n else 0.0,
        "net_avg_pct": float(nets.mean() * 100.0) if n else 0.0,
        "stylized_be_two_way_pct": be,
        "profit_factor": (gp / gl) if gl > 0 else None,
        "profitable_net": bool(n and nets.mean() > 0),
    }


def sweep() -> dict[str, Any]:
    alerts = load_alerts(FEATURES_DIR / "zone_dataset.parquet")
    log.info(f"alerts={len(alerts)}")
    half = (WIDTHS_PCT / 100.0) / 2.0
    n_w = len(WIDTHS_PCT)
    n_h = len(HORIZON_BARS)

    # Alpha cannot change a fill (see the module docstring), so each (horizon, width)
    # cell is computed once and reported under every alpha.
    flags_tw: list[list[list[bool]]] = [[[] for _ in range(n_w)] for _ in range(n_h)]
    flags_ow: list[list[list[bool]]] = [[[] for _ in range(n_w)] for _ in range(n_h)]
    flags_to: list[list[list[bool]]] = [[[] for _ in range(n_w)] for _ in range(n_h)]
    nets: list[list[list[float]]] = [[[] for _ in range(n_w)] for _ in range(n_h)]

    for sym, g in alerts.groupby("symbol", sort=False):
        bars = symbol_bars(sym)
        if bars is None:
            continue
        log.info(f"{sym} n={len(g)}")
        for z in g.itertuples(index=False):
            w = forward_window(bars["open_time"], z.source_open_time, MAX_BARS)
            if w is None:
                continue
            first_i, both = first_touch(bars["high"][w], bars["low"][w], float(z.mid), half)
            for hi, need in enumerate(HORIZON_BARS):
                for wi in range(n_w):
                    fi = int(first_i[wi])
                    tw = bool(fi >= 0 and fi < need and both[wi])
                    ow = bool(fi >= 0 and fi < need and not both[wi])
                    # One-way flattens at the TP (gross ~ 0); two-way earns the full width.
                    gross = float(WIDTHS_PCT[wi] / 100.0) if tw else 0.0
                    flags_tw[hi][wi].append(tw)
                    flags_ow[hi][wi].append(ow)
                    flags_to[hi][wi].append(not tw and not ow)
                    nets[hi][wi].append(gross - FEE_DRAG)

    rows: list[dict[str, Any]] = []
    for hi, name in enumerate(HORIZON_NAMES):
        for wi, width_pct in enumerate(WIDTHS_PCT):
            for alpha in ALPHAS:
                row = summarize(
                    np.asarray(flags_tw[hi][wi], dtype=bool),
                    np.asarray(flags_ow[hi][wi], dtype=bool),
                    np.asarray(flags_to[hi][wi], dtype=bool),
                    np.asarray(nets[hi][wi], dtype=np.float64),
                    float(width_pct),
                    float(alpha),
                    name,
                )
                rows.append(row)
                log.info(
                    f"  {name} w={width_pct:4.2f}% a={alpha * 100:2.0f}%  "
                    f"two-way={row['two_way_pct']:5.2f}%  need={row['stylized_be_two_way_pct']:.1f}%  "
                    f"one-way={row['one_way_pct']:5.1f}%  net={row['net_avg_pct']:+.4f}%  "
                    f"{'PROFIT' if row['profitable_net'] else 'loss'}"
                )

    profitable = [r for r in rows if r["profitable_net"]]
    best = max(rows, key=lambda r: r["net_avg_pct"]) if rows else None
    unviable = not bool(profitable)
    report = {
        "entry": "box midpoint; dual long+short",
        "rule": (
            "first TP +X locks a buffer; remaining SL = -alpha*X from original "
            "entry. For alpha<1 that SL is already through; remaining fills at the TP"
        ),
        "fees": "0.06% taker × 4 fills = 0.24% every alert",
        "n_alerts": int(len(alerts)),
        "rows": rows,
        "any_profitable": bool(profitable),
        "profitable_rows": profitable,
        "best": best,
        "unviable": unviable,
        "unviable_message": UNVIABLE_MSG if unviable else None,
    }
    out = REPORTS_DIR / "two_way_gain_buffer.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"wrote {out}  any_profit={bool(profitable)}")
    if unviable:
        log.info(UNVIABLE_MSG)
    return report


if __name__ == "__main__":
    sweep()
