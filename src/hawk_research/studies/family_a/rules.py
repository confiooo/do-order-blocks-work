"""Pre-registered constants, tier-D calendar, signal gates and side rules for S1 and S5."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from hawk_research.constants import MS_PER_DAY, MS_PER_HOUR

MS_DAY = MS_PER_DAY
ENTRY_BARS = 32
HOLD_BARS = 32
MAX_LABEL_BARS = ENTRY_BARS + HOLD_BARS  # 16 h
THROUGH = 1e-4
PASSIVE_MIN = 0.0005
VOL_LO = 0.0060  # 0.60%
VOL_HI = 0.0400  # 4.00%
P_STAR = 0.420
P0 = 1.0 / 3.0
BPS = 10_000.0

QUEUE_BY = ["entry_time_ms", "pit_rank_prev_1d", "zone_uid"]

SPREAD_LADDER = (
    (10, 4.0),
    (25, 8.0),
    (45, 16.0),
    (65, 28.0),
    (80, 45.0),
)


def d_bounds() -> Tuple[int, int, int]:
    d0 = int(pd.Timestamp("2024-09-01", tz="UTC").timestamp() * 1000)
    d1 = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp() * 1000)
    h0 = int(pd.Timestamp("2026-04-01", tz="UTC").timestamp() * 1000)
    return d0, d1, h0


def assert_tier_d(times: np.ndarray, what: str = "events") -> None:
    """Refuse to proceed if any timestamp belongs to F or H."""
    _d0, d1, h0 = d_bounds()
    t = np.asarray(times, dtype=np.int64)
    if t.size == 0:
        return
    if int(t.max()) >= d1:
        n_f = int(((t >= d1) & (t < h0)).sum())
        n_h = int((t >= h0).sum())
        raise RuntimeError(
            f"{what} include {n_f} firewall and {n_h} holdout timestamps; Family A may evaluate tier D only"
        )


def filter_d(df: pd.DataFrame, time_col: str = "source_open_time") -> pd.DataFrame:
    d0, d1, _ = d_bounds()
    t = df[time_col].to_numpy(dtype=np.int64)
    out = df.loc[(t >= d0) & (t < d1)].copy()
    assert_tier_d(out[time_col].to_numpy(), time_col)
    return out


def alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the s5 merge suffixes onto the names the rules use."""
    out = df.copy()
    aliases = {
        "rv_24h": ["rv_24h", "rv_24h_x", "rv_24h_y"],
        "det_volume_ratio": ["det_volume_ratio", "det_volume_ratio_x", "volume_ratio"],
        "det_wick_ratio": ["det_wick_ratio", "det_wick_ratio_x", "wick_ratio"],
        "det_distance_to_swing": ["det_distance_to_swing", "det_distance_to_swing_x", "distance_to_swing"],
    }
    for dest, cands in aliases.items():
        if dest in out.columns and out[dest].notna().any():
            continue
        for c in cands:
            if c in out.columns:
                out[dest] = out[c]
                break
    return out


def sigma_8h(rv_24h: np.ndarray) -> np.ndarray:
    return np.asarray(rv_24h, dtype=float) * math.sqrt(32.0)


def resid_ret_4h(df: pd.DataFrame) -> np.ndarray:
    return df["trend_ret_4h"].to_numpy(dtype=float) - df["beta_btc_24h"].to_numpy(dtype=float) * df[
        "btc_ret_4h"
    ].to_numpy(dtype=float)


def expanding_p99(values: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.expanding(min_periods=96).quantile(0.99).to_numpy()


def tod_mult(ts_ms: np.ndarray) -> np.ndarray:
    """Slippage multiplier by UTC time of day: 1.0 in the 13-17h overlap, 1.5 at weekends, else 1.3."""
    ms = np.asarray(ts_ms, dtype=np.int64)
    hour = (ms // MS_PER_HOUR) % 24
    weekend = (ms // MS_DAY + 3) % 7 >= 5  # 1970-01-01 was a Thursday (weekday 3)
    overlap = (hour >= 13) & (hour < 17)
    out = np.full(len(ts_ms), 1.3, dtype=float)
    out[overlap] = 1.0
    out[weekend] = 1.5
    return out


def spread_bps(rank: np.ndarray) -> np.ndarray:
    r = np.asarray(rank, dtype=float)
    out = np.full(r.shape, np.nan, dtype=float)
    prev = 0
    for hi, s in SPREAD_LADDER:
        out[(r > prev) & (r <= hi)] = s
        prev = hi
    return out


def apply_common_gates(
    df: pd.DataFrame,
    btc_p99_by_time: Dict[int, float],
) -> pd.DataFrame:
    """§2.2–2.3 gates. ``btc_stable`` / ``signal_gate`` are NOT applied."""
    out = alias_columns(df)
    sig = sigma_8h(out["rv_24h"].to_numpy(dtype=float))
    vol_ok = np.isfinite(sig) & (sig >= VOL_LO) & (sig <= VOL_HI)

    br = out["btc_range_4h_pct"].to_numpy(dtype=float)
    p99 = np.array(
        [btc_p99_by_time.get(int(t), np.nan) for t in out["source_open_time"].to_numpy()],
        dtype=float,
    )
    disloc_ok = np.isfinite(br) & np.isfinite(p99) & (br <= p99)

    fwd = out["fwd_bars_available"].to_numpy(dtype=float)
    integ_ok = np.isfinite(fwd) & (fwd >= MAX_LABEL_BARS)

    rank = out["pit_rank_prev_1d"].to_numpy(dtype=float)
    uni_ok = np.isfinite(rank) & (rank >= 1) & (rank <= 80)
    listed = (
        out["pit_listed_days"].to_numpy(dtype=float)
        if "pit_listed_days" in out.columns
        else np.full(len(out), 30.0)
    )
    listed_ok = np.isfinite(listed) & (listed >= 30)

    out["sigma_8h"] = sig
    out["gate_vol"] = vol_ok
    out["gate_dislocation"] = disloc_ok
    out["gate_integrity"] = integ_ok
    out["gate_universe"] = uni_ok & listed_ok
    out["passes_common_gates"] = vol_ok & disloc_ok & integ_ok & uni_ok & listed_ok
    return out


def s1_side_and_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-registration §4 item 3. FAR is scored with the NEAR sign, not traded."""
    out = alias_columns(df)
    broke_h = out["broke_swing_high"].to_numpy(dtype=float)
    broke_l = out["broke_swing_low"].to_numpy(dtype=float)
    dist = out["det_distance_to_swing"].to_numpy(dtype=float)
    d_hi = out["dist_to_swing_high_pct"].to_numpy(dtype=float)
    d_lo = out["dist_to_swing_low_pct"].to_numpy(dtype=float)

    outside = (broke_h == 1.0) & (broke_l == 1.0)
    brk = ((broke_h == 1.0) | (broke_l == 1.0)) & ~outside
    near = (~brk) & (~outside) & (dist < 0.5)
    far = (~brk) & (~outside) & (dist >= 0.5)

    cell = np.full(len(out), "FAR", dtype=object)
    cell[outside] = "SKIP"
    cell[brk] = "BREAK"
    cell[near] = "NEAR"
    cell[far] = "FAR"

    side = np.zeros(len(out), dtype=int)
    side[brk & (broke_h == 1.0)] = 1
    side[brk & (broke_l == 1.0)] = -1
    near_or_far = near | far
    short_nf = near_or_far & (d_hi < d_lo)
    long_nf = near_or_far & ~short_nf
    side[short_nf] = -1
    side[long_nf] = 1
    side[outside] = 0

    out["s1_cell"] = cell
    out["side"] = side
    out["tradeable"] = (cell == "BREAK") | (cell == "NEAR")
    return out


def s3_side(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-registration §5 / §8 item 3. Identical on both S5 arms."""
    out = alias_columns(df)
    resid = resid_ret_4h(out)
    move = out["rv_4h"].to_numpy(dtype=float) * math.sqrt(16.0)
    rank = out["pit_rank_prev_1d"].to_numpy(dtype=float)
    tier = np.full(len(out), "T3", dtype=object)
    tier[rank <= 15] = "T1"
    tier[(rank > 15) & (rank <= 45)] = "T2"
    skip = ~np.isfinite(resid) | ~np.isfinite(move) | (np.abs(resid) < 1.0 * move)
    sgn = np.sign(resid)
    sgn[~np.isfinite(sgn) | (sgn == 0)] = 0
    side = np.where(tier == "T1", sgn, -sgn).astype(int)
    side[skip] = 0
    out["resid_ret_4h"] = resid
    out["resid_abs"] = np.abs(resid)
    out["s3_tier"] = tier
    out["s3_skip_magnitude"] = skip
    out["side"] = side
    out["tradeable"] = side != 0
    return out
