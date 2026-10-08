"""Matched controls for S5: bar panel, matching keys and per-symbol control selection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.data import loader, universe
from hawk_research.studies.family_a.rules import (
    MAX_LABEL_BARS,
    VOL_HI,
    VOL_LO,
    expanding_p99,
    resid_ret_4h,
    sigma_8h,
)


def btc_p99_map(btc: pd.DataFrame) -> Dict[int, float]:
    from hawk_research.data.backfill import to_candles
    from hawk_research.replay.round1 import BTC_RANGE_BARS, btc_range_series

    series = btc_range_series(to_candles(btc), bars=BTC_RANGE_BARS)
    times = np.array(sorted(series), dtype=np.int64)
    vals = np.array([series[int(t)] for t in times], dtype=float)
    p99 = expanding_p99(vals)
    return {int(t): float(p) for t, p in zip(times.tolist(), p99.tolist())}


def load_ohlcv() -> Dict[str, pd.DataFrame]:
    """15m bars for the round-1 universe (the files present when Family A ran)."""
    return loader.load_many(universe.symbols("round1"), "15m")


def standardised_smd(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return None
    var = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    if var <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / math.sqrt(var))


MATCH_KEYS = (
    "btc_tercile",
    "atr_decile",
    "pit_tier",
    "hour6",
    "resid_sign",
    "resid_decile",
)
MATCH_CONT = (
    "btc_range_4h_pct",
    "atr_48_pct",
    "pit_rank_prev_1d",
    "hour_of_day_utc",
    "resid_ret_4h",
    "resid_abs",
)


def _bin_series(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    try:
        cat = pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop")
        return cat.to_numpy()
    except (ValueError, IndexError):
        return np.full(len(x), 0)


def prepare_bar_panel(
    bars: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    pit_symbol: pd.DataFrame,
    btc_range: Dict[int, float],
    pass_r_times: set,
    btc_p99: Dict[int, float],
    d0: int,
    d1: int,
) -> pd.DataFrame:
    """One-symbol bar panel with S5 matching keys. Outcomes attached later."""
    df = bars[["open_time", "high", "low", "close"]].copy()
    df = df.merge(labels, on="open_time", how="left", suffixes=("", "_lab"))
    keep_f = [
        c
        for c in (
            "open_time",
            "atr_48_pct",
            "rv_4h",
            "rv_24h",
            "trend_ret_4h",
            "btc_ret_4h",
            "beta_btc_24h",
            "hour_of_day_utc",
            "det_volume_ratio",
            "det_wick_ratio",
        )
        if c in features.columns
    ]
    df = df.merge(features[keep_f], on="open_time", how="left", suffixes=("", "_f"))
    df["btc_range_4h_pct"] = [btc_range.get(int(t), np.nan) for t in df["open_time"].to_numpy()]
    df["_date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("D")
    if pit_symbol is not None and len(pit_symbol):
        p = pit_symbol.copy()
        p["join_date"] = pd.to_datetime(p["date"], utc=True).dt.floor("D") + pd.Timedelta(days=1)
        df = df.merge(
            p[["join_date", "rank_qv_1d", "listed_days"]].rename(
                columns={"rank_qv_1d": "pit_rank_prev_1d", "listed_days": "pit_listed_days"}
            ),
            left_on="_date",
            right_on="join_date",
            how="left",
        )
        df = df.drop(columns=["join_date", "_date"], errors="ignore")
    else:
        df["pit_rank_prev_1d"] = np.nan
        df["pit_listed_days"] = np.nan
        df = df.drop(columns=["_date"], errors="ignore")

    t = df["open_time"].to_numpy(dtype=np.int64)
    in_d = (t >= d0) & (t < d1)
    df = df.loc[in_d].copy()
    if df.empty:
        return df

    # collapse possible suffix collisions
    if "rv_24h" not in df.columns:
        for c in ("rv_24h_x", "rv_24h_y", "rv_24h_f"):
            if c in df.columns:
                df["rv_24h"] = df[c]
                break
    if "det_volume_ratio" not in df.columns and "det_volume_ratio_f" in df.columns:
        df["det_volume_ratio"] = df["det_volume_ratio_f"]
    if "det_wick_ratio" not in df.columns and "det_wick_ratio_f" in df.columns:
        df["det_wick_ratio"] = df["det_wick_ratio_f"]
    for c in (
        "trend_ret_4h",
        "btc_ret_4h",
        "beta_btc_24h",
        "rv_24h",
        "rv_4h",
        "atr_48_pct",
        "hour_of_day_utc",
        "det_volume_ratio",
        "det_wick_ratio",
        "fwd_bars_available",
    ):
        if c not in df.columns:
            df[c] = np.nan

    df["resid_ret_4h"] = resid_ret_4h(df)
    df["resid_abs"] = np.abs(df["resid_ret_4h"].to_numpy(dtype=float))
    df["sigma_8h"] = sigma_8h(df["rv_24h"].to_numpy(dtype=float))
    pr_arr = (
        np.fromiter((int(x) for x in pass_r_times), dtype=np.int64)
        if pass_r_times
        else np.array([], dtype=np.int64)
    )
    df["is_pass_r"] = np.isin(df["open_time"].to_numpy(dtype=np.int64), pr_arr)
    fail_gate = (df["det_volume_ratio"].to_numpy(dtype=float) < 1.35) | (
        df["det_wick_ratio"].to_numpy(dtype=float) < 1.2
    )
    vol_ok = (df["sigma_8h"] >= VOL_LO) & (df["sigma_8h"] <= VOL_HI)
    br = df["btc_range_4h_pct"].to_numpy(dtype=float)
    p99 = np.array([btc_p99.get(int(t), np.nan) for t in df["open_time"].to_numpy()], dtype=float)
    disloc_ok = np.isfinite(br) & np.isfinite(p99) & (br <= p99)
    integ_ok = df["fwd_bars_available"].to_numpy(dtype=float) >= MAX_LABEL_BARS
    rank = df["pit_rank_prev_1d"].to_numpy(dtype=float)
    uni_ok = (
        np.isfinite(rank) & (rank >= 1) & (rank <= 80) & (df["pit_listed_days"].to_numpy(dtype=float) >= 30)
    )
    df["control_eligible"] = (
        (~df["is_pass_r"].astype(bool)) & fail_gate & vol_ok & disloc_ok & integ_ok & uni_ok
    )

    df["btc_tercile"] = _bin_series(df["btc_range_4h_pct"].to_numpy(dtype=float), 3)
    df["atr_decile"] = _bin_series(df["atr_48_pct"].to_numpy(dtype=float), 10)
    df["resid_decile"] = _bin_series(df["resid_abs"].to_numpy(dtype=float), 10)
    hour = df["hour_of_day_utc"].to_numpy(dtype=float)
    hour = np.where(np.isfinite(hour), hour, 0.0)
    df["hour6"] = (hour // 6).astype("int32")
    df["resid_sign"] = np.sign(df["resid_ret_4h"].to_numpy(dtype=float))
    df["pit_tier"] = np.where(rank <= 15, 1, np.where(rank <= 45, 2, 3))
    df["zone_high"] = df["high"]
    df["zone_low"] = df["low"]
    df["source_open_time"] = df["open_time"]
    return df


def match_controls_for_symbol(
    zones: pd.DataFrame,
    panel: pd.DataFrame,
    n_controls: int = 3,
    match_bars: int = 192,
) -> Tuple[pd.DataFrame, int]:
    """Nearest-neighbour exact-bin matching, without replacement. Returns control rows + drop count."""
    leak_cols = list(MATCH_KEYS) + list(MATCH_CONT)
    for c in leak_cols:
        if c.startswith(("fwd_", "entry_", "first_up_", "first_dn_", "mfe_", "mae_")) or c in {
            "retested",
            "lifecycle_status",
        }:
            raise RuntimeError(f"S5 matching key {c} is a post-decision / outcome column")
    if zones.empty or panel.empty:
        return pd.DataFrame(), int(len(zones))
    need = [c for c in list(MATCH_CONT) + list(MATCH_KEYS) if c in panel.columns]
    elig = panel.loc[panel["control_eligible"]].copy()
    elig = elig.dropna(subset=need)
    elig = elig.sort_values("open_time").reset_index(drop=True)
    if elig.empty:
        return pd.DataFrame(), int(len(zones))

    mu = elig[list(MATCH_CONT)].mean()
    sd = elig[list(MATCH_CONT)].std().replace(0, 1.0)
    z_e = ((elig[list(MATCH_CONT)] - mu) / sd).to_numpy(dtype=float)

    ot = elig["open_time"].to_numpy(dtype=np.int64)
    keys = elig[list(MATCH_KEYS)].to_numpy()
    taken = np.zeros(len(elig), dtype=bool)
    picked: List[int] = []
    matched_uid: List[str] = []
    match_dist: List[float] = []
    dropped = 0

    z_sorted = zones.sort_values("source_open_time")
    for rec in z_sorted.itertuples(index=False):
        src = int(rec.source_open_time)
        lo, hi = src - match_bars * BAR_MS, src + match_bars * BAR_MS
        in_win = (ot >= lo) & (ot <= hi) & (ot != src) & ~taken
        if not in_win.any():
            dropped += 1
            continue
        k = np.array([getattr(rec, name, np.nan) for name in MATCH_KEYS], dtype=float)
        if not np.all(np.isfinite(k)):
            dropped += 1
            continue
        cand = np.flatnonzero(in_win)
        same = np.all(np.isclose(keys[cand].astype(float), k, equal_nan=False), axis=1)
        cand = cand[same]
        if cand.size == 0:
            dropped += 1
            continue
        z_row = (
            np.array([getattr(rec, c, np.nan) for c in MATCH_CONT], dtype=float) - mu.to_numpy()
        ) / sd.to_numpy()
        dist = np.sqrt(np.nansum((z_e[cand] - z_row) ** 2, axis=1))
        order = np.argsort(dist, kind="stable")
        take_n = min(n_controls, cand.size)
        take = cand[order[:take_n]]
        take_d = dist[order[:take_n]]
        # tie-break by ascending open_time: elig is time-sorted; stable argsort keeps earlier bars first
        taken[take] = True
        for ix, dval in zip(take.tolist(), take_d.tolist()):
            picked.append(ix)
            matched_uid.append(str(rec.zone_uid))
            match_dist.append(float(dval))

    if not picked:
        return pd.DataFrame(), dropped
    ctrl = elig.iloc[picked].copy()
    ctrl["matched_zone_uid"] = matched_uid
    ctrl["match_distance"] = match_dist
    if "symbol" in ctrl.columns:
        ctrl["zone_uid"] = ctrl["symbol"].astype(str) + "@" + ctrl["open_time"].astype(str) + "#ctrl"
    else:
        ctrl["zone_uid"] = "ctrl@" + ctrl["open_time"].astype(str)
    return ctrl, dropped
