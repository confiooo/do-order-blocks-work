"""Path simulator for one dollar-neutral pair. One position at a time."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd

from experiments.market_neutral.data.bars import HOUR_MS


def simulate_pair(
    times_ms: np.ndarray,
    close_a: np.ndarray,
    close_b: np.ndarray,
    z: np.ndarray,
    beta: np.ndarray,
    *,
    entry: float,
    exit_band: float,
    max_hold_bars: int,
    symbol: str,
    beta_mode: str,
    window: int,
) -> pd.DataFrame:
    """Enter on a completed-bar close when ``|z|`` exceeds ``entry``.

    ``z > entry`` shorts A and longs B. ``z < -entry`` does the opposite.
    Exit when the spread has converged (through 0, or inside ``exit_band``)
    or when ``max_hold_bars`` is reached. The next entry is the following bar,
    so a trade cannot be reopened on its own exit close.

    ``ret_gross`` is the equal-weight two-leg return on posted capital
    (each leg is half of capital).
    """
    times = np.asarray(times_ms, dtype=np.int64)
    pa = np.asarray(close_a, dtype=float)
    pb = np.asarray(close_b, dtype=float)
    zz = np.asarray(z, dtype=float)
    bb = np.asarray(beta, dtype=float)
    n = times.size
    if min(pa.size, pb.size, zz.size, bb.size) != n:
        raise ValueError("pair arrays must be the same length")
    if entry <= 0.0 or max_hold_bars < 1:
        raise ValueError("entry threshold and max hold must be positive")
    if exit_band < 0.0:
        raise ValueError("exit band must be >= 0")

    # Plain floats: indexing numpy arrays one element at a time is several times slower.
    pa_l, pb_l, zz_l, bb_l, times_l = pa.tolist(), pb.tolist(), zz.tolist(), bb.tolist(), times.tolist()
    rows: List[dict] = []
    i = 0
    while i < n - 1:
        zi = zz_l[i]
        # Strict: |z| == entry does not enter, and a non-finite z does not enter.
        if not math.isfinite(zi) or abs(zi) <= entry:
            i += 1
            continue
        if zi > entry:
            side_a = -1
        elif zi < -entry:
            side_a = 1
        else:
            i += 1
            continue
        side_b = -side_a
        limit = min(n - 1, i + int(max_hold_bars))
        exit_j = limit
        reason = "max_hold"
        aborted = False
        for j in range(i + 1, limit + 1):
            zj = zz_l[j]
            if not math.isfinite(zj) or not math.isfinite(pa_l[j]) or not math.isfinite(pb_l[j]):
                if j == i + 1:
                    aborted = True
                else:
                    exit_j = j - 1
                    reason = "gap"
                break
            if exit_band == 0.0:
                converged = zj <= 0.0 if side_a < 0 else zj >= 0.0
            else:
                converged = abs(zj) <= exit_band
            if converged:
                exit_j = j
                reason = "z_exit"
                break
        if aborted or exit_j <= i:
            i += 1
            continue
        if not (pa_l[i] > 0.0 and pb_l[i] > 0.0 and pa_l[exit_j] > 0.0 and pb_l[exit_j] > 0.0):
            i += 1
            continue
        ret_a = side_a * (pa_l[exit_j] / pa_l[i] - 1.0)
        ret_b = side_b * (pb_l[exit_j] / pb_l[i] - 1.0)
        entry_ms = int(times_l[i] + HOUR_MS)
        exit_ms = int(times_l[exit_j] + HOUR_MS)
        rows.append(
            {
                "zone_uid": f"{symbol}|{entry_ms}|e{entry}|x{exit_band}|h{max_hold_bars}|{beta_mode}|w{window}",
                "symbol": symbol,
                "symbol_a": symbol.split("|", 1)[0],
                "symbol_b": symbol.split("|", 1)[1],
                "decision_time_ms": entry_ms,
                "entry_time_ms": entry_ms,
                "exit_time_ms": exit_ms,
                "side": int(side_a),
                "side_b": int(side_b),
                "ret_gross": float(0.5 * (ret_a + ret_b)),
                "ret_leg_a": float(ret_a),
                "ret_leg_b": float(ret_b),
                "weight": 1.0,
                "entry_z": float(zi),
                "exit_z": float(zz_l[exit_j]),
                "beta_entry": float(bb_l[i]) if math.isfinite(bb_l[i]) else np.nan,
                "hold_bars": int(exit_j - i),
                "exit_reason": reason,
                "gross_exposure": 1.0,
                "net_exposure": 0.0,
            }
        )
        i = exit_j + 1

    if not rows:
        return _empty()
    return pd.DataFrame.from_records(rows)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "zone_uid",
            "symbol",
            "symbol_a",
            "symbol_b",
            "decision_time_ms",
            "entry_time_ms",
            "exit_time_ms",
            "side",
            "side_b",
            "ret_gross",
            "ret_leg_a",
            "ret_leg_b",
            "weight",
            "entry_z",
            "exit_z",
            "beta_entry",
            "hold_bars",
            "exit_reason",
            "gross_exposure",
            "net_exposure",
        ]
    )
