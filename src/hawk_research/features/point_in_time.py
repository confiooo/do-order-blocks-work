"""Point-in-time features on 15m bars.

Every column is computable from bars that close at or before the decision bar's
close. The decision bar itself is included (its close is what triggers a
detection).

Two kinds of window:

- ``rolling(N)`` includes the decision bar: "what does this bar look like".
- ``shift(1).rolling(N)`` excludes it: for baselines the bar is compared against,
  as the detector does for its volume baseline and swing levels.

The detector's ``wick_ratio`` does not say which side the wick is on, so upper
and lower wicks are kept as separate features.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS

# Bars. 4h, 24h, 4d, 7d, 30d.
VOL_WINDOWS = {"4h": 16, "24h": 96, "4d": 384, "7d": 672}
ATR_WINDOWS = {"14": 14, "48": 48, "96": 96}
RANGE_WINDOWS = {"24h": 96, "4d": 384, "7d": 672}
TREND_WINDOWS = {"4h": 16, "24h": 96, "4d": 384, "7d": 672}
VOLZ_WINDOWS = {"20": 20, "96": 96, "384": 384}
DOLLAR_VOL_WINDOWS = {"24h": 96, "7d": 672, "30d": 2880}
BETA_WINDOWS = {"24h": 96, "7d": 672}

DETECTOR_LOOKBACK = 60  # must match replay.EFFECTIVE_LOOKBACK


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan)


def bar_shape_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Geometry of the decision bar itself, including wick side."""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    body = np.abs(c - o)
    rng = h - low
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - low

    close_pos = _safe_div(c - low, rng)
    return {
        "body_abs": body,
        "bar_range_pct": _safe_div(rng, c) * 100.0,
        "body_to_range": _safe_div(body, rng),
        "upper_wick_frac": _safe_div(upper, rng),
        "lower_wick_frac": _safe_div(lower, rng),
        "upper_wick_to_body": _safe_div(upper, body),
        "lower_wick_to_body": _safe_div(lower, body),
        "wick_asymmetry": _safe_div(upper - lower, rng),
        "wick_side": np.where(upper > lower, 1.0, np.where(lower > upper, -1.0, 0.0)),
        "close_position_in_range": close_pos,
        "closed_upper_third": (close_pos >= 2.0 / 3.0).astype(float),
        "closed_lower_third": (close_pos <= 1.0 / 3.0).astype(float),
        "bar_return": _safe_div(c - o, o),
    }


def _true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    return pd.concat([h - low, (h - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def volatility_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Trailing realized volatility and ATR, as-of the decision bar."""
    close = df["close"]
    logret = np.log(close / close.shift(1))
    out: Dict[str, np.ndarray] = {}
    for name, n in VOL_WINDOWS.items():
        out[f"rv_{name}"] = logret.rolling(n, min_periods=n).std().to_numpy()
    tr = _true_range(df)
    for name, n in ATR_WINDOWS.items():
        atr = tr.rolling(n, min_periods=n).mean()
        out[f"atr_{name}"] = atr.to_numpy()
        out[f"atr_{name}_pct"] = (atr / close).to_numpy()
    return out


def range_position_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Where the close sits inside the trailing high-low range (0 = low, 1 = high)."""
    close = df["close"]
    out: Dict[str, np.ndarray] = {}
    for name, n in RANGE_WINDOWS.items():
        hi = df["high"].rolling(n, min_periods=n).max()
        lo = df["low"].rolling(n, min_periods=n).min()
        out[f"range_pos_{name}"] = _safe_div((close - lo).to_numpy(), (hi - lo).to_numpy())
        out[f"dist_to_range_high_{name}_pct"] = _safe_div((hi - close).to_numpy(), close.to_numpy()) * 100.0
        out[f"dist_to_range_low_{name}_pct"] = _safe_div((close - lo).to_numpy(), close.to_numpy()) * 100.0
    return out


def swing_features(df: pd.DataFrame, lookback: int = DETECTOR_LOOKBACK) -> Dict[str, np.ndarray]:
    """Separate distances to the prior swing high and low.

    ``detect_order_blocks`` collapses these into ``min(|h-recent_high|,
    |l-recent_low|)``, which cannot distinguish a bar poking through the top of
    its range from one poking through the bottom. Both are kept here, over the
    same prior-``lookback`` window the detector uses.
    """
    prior_high = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = df["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    c = df["close"].to_numpy(dtype=float)
    return {
        "swing_high_prior": prior_high.to_numpy(),
        "swing_low_prior": prior_low.to_numpy(),
        "dist_to_swing_high_pct": _safe_div((prior_high - df["high"]).to_numpy(), c) * 100.0,
        "dist_to_swing_low_pct": _safe_div((df["low"] - prior_low).to_numpy(), c) * 100.0,
        "broke_swing_high": (df["high"] > prior_high).astype(float).to_numpy(),
        "broke_swing_low": (df["low"] < prior_low).astype(float).to_numpy(),
    }


def trend_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    close = df["close"]
    out: Dict[str, np.ndarray] = {}
    for name, n in TREND_WINDOWS.items():
        out[f"trend_ret_{name}"] = (close / close.shift(n) - 1.0).to_numpy()
        sma = close.rolling(n, min_periods=n).mean()
        out[f"close_vs_sma_{name}"] = _safe_div((close - sma).to_numpy(), sma.to_numpy())
    ema_fast = close.ewm(span=24, adjust=False, min_periods=24).mean()
    ema_slow = close.ewm(span=96, adjust=False, min_periods=96).mean()
    out["ema_fast_vs_slow"] = _safe_div((ema_fast - ema_slow).to_numpy(), ema_slow.to_numpy())
    return out


def volume_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Volume z-scores against *prior* bars, plus trailing traded value."""
    vol = df["volume"]
    out: Dict[str, np.ndarray] = {}
    for name, n in VOLZ_WINDOWS.items():
        prior = vol.shift(1)
        mu = prior.rolling(n, min_periods=n).mean()
        sd = prior.rolling(n, min_periods=n).std()
        out[f"volume_z_{name}"] = _safe_div((vol - mu).to_numpy(), sd.to_numpy())
        out[f"volume_ratio_{name}"] = _safe_div(vol.to_numpy(), mu.to_numpy())
    qv = df["quote_volume"] if "quote_volume" in df.columns else vol * df["close"]
    for name, n in DOLLAR_VOL_WINDOWS.items():
        s = qv.rolling(n, min_periods=max(1, n // 4)).sum()
        out[f"quote_volume_{name}"] = s.to_numpy()
        out[f"log_quote_volume_{name}"] = np.log1p(np.maximum(s.to_numpy(), 0.0))
    if "trades" in df.columns:
        out["trades_24h"] = df["trades"].rolling(96, min_periods=24).sum().to_numpy()
    return out


def btc_features(
    df: pd.DataFrame,
    btc: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """BTC context plus the symbol's rolling beta/correlation to BTC.

    BTC bars are aligned on ``open_time``; a missing BTC bar yields NaN rather
    than a forward-filled stale value, which would be a quiet leak of the last
    known price into a period where it was not known to be still current.
    """
    btc_idx = btc.set_index("open_time")
    aligned = btc_idx.reindex(df["open_time"].to_numpy())
    b_close = aligned["close"]
    b_logret = np.log(b_close / b_close.shift(1))
    s_logret = np.log(df["close"] / df["close"].shift(1))
    s_logret.index = b_logret.index

    out: Dict[str, np.ndarray] = {}
    for name, n in TREND_WINDOWS.items():
        out[f"btc_ret_{name}"] = (b_close / b_close.shift(n) - 1.0).to_numpy()
    for name, n in VOL_WINDOWS.items():
        out[f"btc_rv_{name}"] = b_logret.rolling(n, min_periods=n).std().to_numpy()
    for name, n in BETA_WINDOWS.items():
        cov = s_logret.rolling(n, min_periods=n).cov(b_logret)
        var = b_logret.rolling(n, min_periods=n).var()
        out[f"beta_btc_{name}"] = _safe_div(cov.to_numpy(), var.to_numpy())
        out[f"corr_btc_{name}"] = s_logret.rolling(n, min_periods=n).corr(b_logret).to_numpy()
    return out


def calendar_features(open_time: np.ndarray) -> Dict[str, np.ndarray]:
    """Decision-bar close in UTC. The bar opens at ``open_time`` and the decision
    is taken 15 minutes later, so the calendar stamp uses the close."""
    close_ms = open_time + BAR_MS
    ts = pd.to_datetime(close_ms, unit="ms", utc=True)
    hour = ts.hour.to_numpy().astype(float)
    dow = ts.dayofweek.to_numpy().astype(float)
    return {
        "hour_of_day_utc": hour,
        "day_of_week_utc": dow,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "is_weekend": ((dow >= 5).astype(float)),
    }


def detector_features(df: pd.DataFrame, lookback: int = DETECTOR_LOOKBACK) -> Dict[str, np.ndarray]:
    """The detector's four quantities, computed for every bar.

    Needed to match zones against bars that did not qualify. Same arithmetic as
    ``detect_order_blocks``, including its ``candles[i-lookback:i]`` baseline that
    leaves out the decision bar.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    prior_vol_mean = df["volume"].shift(1).rolling(lookback, min_periods=lookback).mean().to_numpy()
    prior_high = df["high"].shift(1).rolling(lookback, min_periods=lookback).max().to_numpy()
    prior_low = df["low"].shift(1).rolling(lookback, min_periods=lookback).min().to_numpy()

    body = np.abs(c - o)
    wick = np.maximum(np.abs(h - np.maximum(o, c)), np.abs(np.minimum(o, c) - low))
    wick_ratio = np.where(body > 0, _safe_div(wick, body), 0.0)
    midpoint = (h + low) / 2.0
    return {
        "det_volume_ratio": np.where(prior_vol_mean > 0, _safe_div(v, prior_vol_mean), 0.0),
        "det_wick_ratio": wick_ratio,
        "det_distance_to_swing": _safe_div(np.minimum(np.abs(h - prior_high), np.abs(low - prior_low)), c)
        * 100.0,
        "det_distance_to_price": _safe_div(np.abs(midpoint - c), c) * 100.0,
        "det_zone_size_pct": _safe_div(h - low, c) * 100.0,
        "det_direction": np.where(c > o, 1.0, -1.0),
    }


def build_symbol_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """All per-bar point-in-time features for one symbol."""
    out = pd.DataFrame({"open_time": df["open_time"].to_numpy(dtype=np.int64)})
    blocks: List[Dict[str, np.ndarray]] = [
        detector_features(df),
        bar_shape_features(df),
        volatility_features(df),
        range_position_features(df),
        swing_features(df),
        trend_features(df),
        volume_features(df),
        btc_features(df, btc),
        calendar_features(out["open_time"].to_numpy()),
    ]
    data = {}
    for block in blocks:
        for k, v in block.items():
            data[k] = np.asarray(v, dtype=np.float64)
    return pd.concat([out, pd.DataFrame(data)], axis=1)


def cross_sectional_ranks(
    panel: pd.DataFrame,
    value_cols: Sequence[str],
    time_col: str = "open_time",
) -> pd.DataFrame:
    """Rank each symbol against the others *at the same bar*.

    Purely contemporaneous: a rank at time t uses only values that were
    themselves knowable at t. It says nothing about the future cross-section.
    """
    out = panel[[time_col, "symbol"]].copy()
    grouped = panel.groupby(time_col, sort=False)
    for col in value_cols:
        out[f"xs_rank_{col}"] = grouped[col].rank(pct=True, method="average").to_numpy()
        out[f"xs_n_{col}"] = grouped[col].transform("count").to_numpy()
    return out


def time_since_previous_zone(zones: pd.DataFrame) -> np.ndarray:
    """Bars since the previous detection on the same symbol. NaN for the first."""
    z = zones.sort_values(["symbol", "source_open_time"], kind="mergesort")
    delta = z.groupby("symbol", sort=False)["source_open_time"].diff() / BAR_MS
    return delta.reindex(zones.index).to_numpy()
