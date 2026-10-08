"""Maker-fill simulation: queue-aware limit fills, barriers, funding and round-trip costs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.studies.family_a.rules import (
    BPS,
    ENTRY_BARS,
    HOLD_BARS,
    PASSIVE_MIN,
    THROUGH,
    spread_bps,
    tod_mult,
)


def _funding_settlements(entry_ms: int, exit_ms: int) -> int:
    """MEXC 8h settlements at 00:00/08:00/16:00 UTC strictly inside (entry, exit]."""
    if exit_ms <= entry_ms:
        return 0
    t0 = pd.Timestamp(entry_ms, unit="ms", tz="UTC")
    # next 8h boundary strictly after entry
    hour = t0.hour
    nxt_hour = ((hour // 8) + 1) * 8
    first = t0.normalize() + pd.Timedelta(hours=int(nxt_hour))
    n = 0
    t = first
    end = pd.Timestamp(exit_ms, unit="ms", tz="UTC")
    while t <= end:
        n += 1
        t = t + pd.Timedelta(hours=8)
        if n > 8:
            break
    return n


def round_trip_cost_bps(
    rank: float,
    decision_ms: int,
    entry_ms: int,
    exit_ms: int,
    close_px: float,
) -> float:
    """Maker-in / taker-out ladder from ``config/costs.json``, per symbol rank."""
    s = float(spread_bps(np.array([rank]))[0])
    if not np.isfinite(s):
        s = 16.0
    fees = 4.0 + 6.0
    tod = float(tod_mult(np.array([decision_ms], dtype=np.int64))[0])
    slip = 0.25 * s * tod * 1.0
    fund = float(_funding_settlements(entry_ms, exit_ms)) * 1.0
    tick = 1.0 if (np.isfinite(close_px) and close_px < 1.0) else 0.0
    return fees + s + slip + fund + tick


def _scan_fill(
    side: int,
    limit: float,
    high: np.ndarray,
    low: np.ndarray,
    times: np.ndarray,
    i: int,
) -> Optional[int]:
    n = len(times)
    last = min(n - 1, i + ENTRY_BARS)
    for j in range(i + 1, last + 1):
        if times[j] - times[j - 1] != BAR_MS:
            return None
        if side > 0:
            if low[j] <= limit * (1.0 - THROUGH):
                return j
        else:
            if high[j] >= limit * (1.0 + THROUGH):
                return j
    return None


def _scan_barriers(
    side: int,
    fill: float,
    t_frac: float,
    l_frac: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    times: np.ndarray,
    start: int,
    btc_close: np.ndarray,
    beta: float,
    btc_fill: float,
) -> Dict[str, Any]:
    n = len(times)
    last = min(n - 1, start + HOLD_BARS)
    up_px = fill * (1.0 + t_frac) if side > 0 else fill * (1.0 + l_frac)
    dn_px = fill * (1.0 - l_frac) if side > 0 else fill * (1.0 - t_frac)
    hit_up = 0
    hit_dn = 0
    amb = False
    exit_j = last
    reason = "timeout"
    for j in range(start + 1, last + 1):
        if times[j] - times[j - 1] != BAR_MS:
            exit_j = j - 1
            reason = "timeout"
            break
        u = high[j] >= up_px
        d = low[j] <= dn_px
        if u and d:
            amb = True
            hit_dn = j
            exit_j = j
            reason = "stop"
            break
        if u:
            hit_up = j
            exit_j = j
            reason = "target" if side > 0 else "stop"
            break
        if d:
            hit_dn = j
            exit_j = j
            reason = "stop" if side > 0 else "target"
            break
    if exit_j <= start:
        if start + 1 < n and times[start + 1] - times[start] == BAR_MS:
            exit_j = start + 1
            reason = "timeout"
        else:
            return {
                "ret_gross": np.nan,
                "exit_reason": "incomplete",
                "ambiguous": False,
                "exit_index": int(start),
                "exit_time_ms": int(times[start] + BAR_MS + BAR_MS),
                "ret_resid": np.nan,
                "resid_exit_reason": "incomplete",
                "resid_ambiguous": False,
                "hit_up": 0,
                "hit_dn": 0,
            }
    if reason == "target":
        ret = t_frac
    elif reason == "stop":
        ret = -l_frac
    else:
        ret = side * (close[exit_j] / fill - 1.0)

    # Residual close-path barriers (scientific). Closes only, same T/L.
    resid_reason = "timeout"
    resid_ret = np.nan
    resid_amb = False
    if np.isfinite(beta) and np.isfinite(btc_fill) and btc_fill != 0:
        signed = None
        for j in range(start + 1, last + 1):
            if times[j] - times[j - 1] != BAR_MS:
                break
            b = btc_close[j]
            if not np.isfinite(b) or b == 0:
                continue
            p_res = close[j] / fill - beta * (b / btc_fill - 1.0)
            sret = side * (p_res - 1.0)
            if sret >= t_frac:
                resid_reason = "target"
                signed = t_frac
                break
            if sret <= -l_frac:
                resid_reason = "stop"
                signed = -l_frac
                break
            signed = sret
        resid_ret = float(signed) if signed is not None else float("nan")
        if resid_reason == "timeout" and np.isfinite(resid_ret):
            pass
    return {
        "ret_gross": float(ret),
        "exit_reason": reason,
        "ambiguous": bool(amb),
        "exit_index": int(exit_j),
        "exit_time_ms": int(times[exit_j] + BAR_MS),
        "ret_resid": resid_ret,
        "resid_exit_reason": resid_reason,
        "resid_ambiguous": bool(resid_amb),
        "hit_up": int(hit_up),
        "hit_dn": int(hit_dn),
    }


def simulate_maker_outcomes(
    events: pd.DataFrame,
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    btc: pd.DataFrame,
) -> pd.DataFrame:
    """§3.2–3.3 maker fill + 1.0σ/0.5σ first-touch, plus the residual close path."""
    if events.empty:
        return events.iloc[0:0].copy()
    btc = btc.sort_values("open_time")
    btc_t = btc["open_time"].to_numpy(dtype=np.int64)
    btc_c = btc["close"].to_numpy(dtype=float)

    rows: List[Dict[str, Any]] = []
    for symbol, grp in events.groupby("symbol", sort=False):
        bars = ohlcv_by_symbol.get(str(symbol))
        if bars is None or bars.empty:
            continue
        times = bars["open_time"].to_numpy(dtype=np.int64)
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        pos = {int(t): i for i, t in enumerate(times.tolist())}
        btc_ix = np.searchsorted(btc_t, times, side="right") - 1
        btc_al = np.where(btc_ix >= 0, btc_c[np.clip(btc_ix, 0, len(btc_c) - 1)], np.nan)
        btc_al[btc_ix < 0] = np.nan

        for rec in grp.itertuples(index=False):
            side = int(rec.side)
            src = int(rec.source_open_time)
            i = pos.get(src)
            uid = str(rec.zone_uid)
            base = {
                "zone_uid": uid,
                "symbol": str(symbol),
                "source_open_time": src,
                "decision_time_ms": src + BAR_MS,
                "side": side,
                "s1_cell": getattr(rec, "s1_cell", None),
                "tradeable": bool(getattr(rec, "tradeable", True)),
                "pit_rank_prev_1d": float(getattr(rec, "pit_rank_prev_1d", np.nan)),
                "sigma_8h": float(getattr(rec, "sigma_8h", np.nan)),
                "rv_24h": float(getattr(rec, "rv_24h", np.nan)),
                "fwd_rv_4h": float(getattr(rec, "fwd_rv_4h", np.nan))
                if hasattr(rec, "fwd_rv_4h")
                else np.nan,
                "retested": bool(getattr(rec, "retested", False)) if hasattr(rec, "retested") else None,
            }
            if i is None or side == 0 or not np.isfinite(base["sigma_8h"]):
                base.update(
                    {
                        "fill_status": "skipped",
                        "ret_gross": np.nan,
                        "ret_resid": np.nan,
                        "entry_time_ms": src + BAR_MS,
                        "exit_time_ms": src + BAR_MS + HOLD_BARS * BAR_MS,
                    }
                )
                rows.append(base)
                continue

            z_high = float(rec.zone_high)
            z_low = float(rec.zone_low)
            limit = z_low if side > 0 else z_high
            px = float(close[i])
            t_frac = float(base["sigma_8h"])
            l_frac = 0.5 * t_frac
            base["limit_price"] = limit
            base["T"] = t_frac
            base["L"] = l_frac
            base["close"] = px

            if not np.isfinite(limit) or px == 0 or abs(limit - px) / px < PASSIVE_MIN:
                base["fill_status"] = "not_placed"
                base["ret_gross"] = np.nan
                base["ret_resid"] = np.nan
                base["entry_time_ms"] = src + BAR_MS
                base["exit_time_ms"] = src + BAR_MS + HOLD_BARS * BAR_MS
                rows.append(base)
                continue

            fill_j = _scan_fill(side, limit, high, low, times, i)
            beta = float(getattr(rec, "beta_btc_24h", np.nan))

            # Called only within this iteration, so the loop variables it reads are current.
            def _shadow(start_j: int) -> Dict[str, Any]:
                bfill = btc_al[start_j] if start_j < len(btc_al) else np.nan
                return _scan_barriers(
                    side,
                    limit,
                    t_frac,
                    l_frac,
                    high,
                    low,
                    close,
                    times,
                    start_j,
                    btc_al,
                    beta,
                    bfill,
                )

            if fill_j is None:
                # expiry bar = decision + 32; shadow from there
                exp_j = min(i + ENTRY_BARS, len(times) - 1)
                sh = _shadow(exp_j)
                base.update(
                    {
                        "fill_status": "expired_no_fill",
                        "fill_lag_bars": np.nan,
                        "traded_through": False,
                        "entry_time_ms": int(times[exp_j] + BAR_MS),
                        "exit_time_ms": sh["exit_time_ms"],
                        "ret_gross": np.nan,
                        "ret_resid": np.nan,
                        "exit_reason": None,
                        "ambiguous": False,
                        "shadow_ret_gross": sh["ret_gross"],
                        "shadow_ret_resid": sh["ret_resid"],
                        "shadow_exit_reason": sh["exit_reason"],
                        "shadow_ambiguous": sh["ambiguous"],
                    }
                )
                rows.append(base)
                continue

            bfill = btc_al[fill_j]
            outc = _scan_barriers(
                side, limit, t_frac, l_frac, high, low, close, times, fill_j, btc_al, beta, bfill
            )
            entry_ms = int(times[fill_j] + BAR_MS)
            exit_ms = int(outc["exit_time_ms"])
            cost = round_trip_cost_bps(base["pit_rank_prev_1d"], src + BAR_MS, entry_ms, exit_ms, px)
            base.update(
                {
                    "fill_status": "filled",
                    "fill_lag_bars": int(fill_j - i),
                    "traded_through": True,
                    "fill_open_time": int(times[fill_j]),
                    "entry_time_ms": entry_ms,
                    "exit_time_ms": exit_ms,
                    "ret_gross": outc["ret_gross"],
                    "ret_resid": outc["ret_resid"],
                    "exit_reason": outc["exit_reason"],
                    "ambiguous": outc["ambiguous"],
                    "resid_exit_reason": outc["resid_exit_reason"],
                    "cost_bps": cost,
                    "ret_net": outc["ret_gross"] - cost / BPS,
                    "ret_resid_net": (
                        outc["ret_resid"] - cost / BPS if np.isfinite(outc["ret_resid"]) else np.nan
                    ),
                    "n_funding": _funding_settlements(entry_ms, exit_ms),
                }
            )
            rows.append(base)

    return pd.DataFrame.from_records(rows)


def filled_trades(sim: pd.DataFrame, tradeable_only: bool = True) -> pd.DataFrame:
    out = sim[sim["fill_status"] == "filled"].copy()
    if "exit_reason" in out.columns:
        out = out[out["exit_reason"] != "incomplete"]
    if tradeable_only and "tradeable" in out.columns:
        out = out[out["tradeable"].astype(bool)]
    out = out.dropna(subset=["ret_gross", "entry_time_ms", "exit_time_ms"])
    if "ret_net" not in out.columns:
        out["ret_net"] = out["ret_gross"]
    out["weight"] = 1.0
    if "zone_uid" in out.columns:
        out = out.drop_duplicates("zone_uid")
    return out.reset_index(drop=True)


def with_flat_cost(trades: pd.DataFrame, c_bps: float) -> pd.DataFrame:
    out = trades.copy()
    out["ret_net"] = out["ret_gross"].to_numpy(dtype=float) - c_bps / BPS
    if "ret_resid" in out.columns:
        out["ret_resid_net"] = out["ret_resid"].to_numpy(dtype=float) - c_bps / BPS
    return out
