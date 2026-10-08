"""Stage 8: round-1 null and baseline results.

1. Unconditional: zone bars vs all same-symbol bars, vs volume- and
   volatility-matched non-zone bars within ±24h, and vs the base rate.
2. Directional: the detector's ``direction`` vs a coin flip, gross and net, by
   horizon, regime and liquidity tier.
3. Reversed: fading the zone.
4. Feature screen: each point-in-time feature against forward outcomes,
   in-sample and out-of-sample, FDR-corrected.
5. Role: directional signal, location filter, volatility filter, entry trigger,
   trade-management reference or event marker, one number each.

The last 20% of the sample is split off as a holdout first and used only where
the code says so.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import time
from concurrent.futures import as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hawk_research import config as research_config
from hawk_research import provenance
from hawk_research.constants import BAR_MS
from hawk_research.data import loader, paths
from hawk_research.evaluation import regimes, strategies
from hawk_research.evaluation.costs import CostModel
from hawk_research.evaluation.harness import HarnessConfig, evaluate, flatten_evaluation, walk_forward_folds
from hawk_research.labels import outcomes as lab
from hawk_research.ledger import TrialLedger, historical_prior
from hawk_research.parallel import process_pool
from hawk_research.stats import nulls
from hawk_research.stats.multiple_testing import deflated_sharpe_ratio, pbo_cscv
from hawk_research.stats.resampling import cross_sectional_bootstrap_ci

log = logging.getLogger(__name__)

HORIZONS = ("1h", "4h", "24h", "48h")

COST_MODELS = {
    "zero": CostModel.zero(),
    "binance_spot_taker": CostModel.binance_spot_taker(),
    "binance_spot_taker_bnb": CostModel.binance_spot_taker_bnb(),
    "mexc_futures_taker": CostModel.mexc_futures_taker(),
}


def _controls_for_symbol(symbol: str, zone_times: List[int], seed: int) -> bytes:
    bars = pd.read_parquet(paths.LABELS_DIR / "15m" / f"{symbol}.parquet")
    ctrl = nulls.sample_matched_controls(bars, zone_times, seed=seed)
    return ctrl.to_parquet(index=False) if not ctrl.empty else b""


def controls_cache_key(ds: pd.DataFrame, seed: int) -> Dict[str, object]:
    """Everything the matched controls depend on. A cached file is reused only on an exact match."""
    zones = ds[["symbol", "source_open_time"]].sort_values(["symbol", "source_open_time"], kind="mergesort")
    h = hashlib.sha256()
    for sym, t in zones.itertuples(index=False):
        h.update(f"{sym}:{int(t)};".encode())
    params = {
        k: v.default
        for k, v in inspect.signature(nulls.sample_matched_controls).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return {
        "zones_sha256": h.hexdigest(),
        "seed": int(seed),
        "match_params": params,
        "data_manifest_sha256": provenance.data_manifest_hash(),
    }


def build_controls(ds: pd.DataFrame, workers: int, seed: int) -> pd.DataFrame:
    import io

    by_symbol = ds.groupby("symbol")["source_open_time"].apply(list).to_dict()
    frames = []
    with process_pool(workers) as pool:
        futures = {pool.submit(_controls_for_symbol, s, t, seed): s for s, t in by_symbol.items()}
        for fut in as_completed(futures):
            blob = fut.result()
            if blob:
                frames.append(pd.read_parquet(io.BytesIO(blob)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def all_bars_population(symbols: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    frames = []
    for s in symbols:
        df = pd.read_parquet(paths.LABELS_DIR / "15m" / f"{s}.parquet")
        frames.append(df[[c for c in columns if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def unconditional_tests(
    ds: pd.DataFrame, controls: pd.DataFrame, all_bars: pd.DataFrame, n_samples: int
) -> Dict:
    """Are zone bars distinguishable from non-zone bars at all?"""
    out: Dict[str, object] = {}
    zt = ds["decision_time_ms"].to_numpy(dtype=np.int64)
    ct = controls["open_time"].to_numpy(dtype=np.int64) + BAR_MS
    at = all_bars["open_time"].to_numpy(dtype=np.int64) + BAR_MS

    for horizon in HORIZONS:
        col = f"fwd_ret_{horizon}"
        block: Dict[str, object] = {}
        z = ds[col].to_numpy(dtype=float)

        block["vs_matched_controls"] = nulls.compare_means(
            z, zt, controls[col].to_numpy(dtype=float), ct, n_samples=n_samples
        )
        block["vs_all_bars"] = nulls.compare_means(
            z, zt, all_bars[col].to_numpy(dtype=float), at, n_samples=n_samples, seed=6
        )
        # Absolute return: magnitude rather than sign.
        block["abs_return_vs_matched"] = nulls.compare_means(
            np.abs(z),
            zt,
            np.abs(controls[col].to_numpy(dtype=float)),
            ct,
            n_samples=n_samples,
            seed=7,
        )
        out[horizon] = block

    # Forward realized volatility: the volatility-filter hypothesis.
    vol: Dict[str, object] = {}
    for name in ("4h", "24h", "48h"):
        col = f"fwd_rv_{name}"
        if col not in ds.columns or col not in controls.columns:
            continue
        vol[name] = nulls.compare_means(
            ds[col].to_numpy(dtype=float),
            zt,
            controls[col].to_numpy(dtype=float),
            ct,
            n_samples=n_samples,
            seed=8,
        )
    out["forward_realized_vol"] = vol

    # First-touch base rates, zone vs control vs everything.
    barriers: Dict[str, object] = {}
    for pair in ((0.01, 0.01), (0.02, 0.01), (0.01, 0.02)):
        tag = f"bar_{lab._level_tag(pair[0])}_{lab._level_tag(pair[1])}"
        for horizon in ("4h", "24h"):
            col = f"{tag}_{horizon}"
            if col not in ds.columns:
                continue
            entry: Dict[str, object] = {}
            for name, frame in (("zone", ds), ("matched_control", controls), ("all_bars", all_bars)):
                if name == "zone":
                    lbl = frame[col].to_numpy()
                else:
                    up = frame[f"first_up_{lab._level_tag(pair[0])}"].to_numpy()
                    dn = frame[f"first_dn_{lab._level_tag(pair[1])}"].to_numpy()
                    lbl, _ = lab.barrier_label(up, dn, lab.BARRIER_HORIZONS[horizon])
                resolved = lbl != 0
                entry[name] = nulls.proportion_test(int((lbl == 1).sum()), int(resolved.sum()))
            barriers[col] = entry
    out["barrier_base_rates"] = barriers
    return out


def directional_tests(ds: pd.DataFrame, n_samples: int) -> Dict:
    nominal = np.where(ds["direction"].to_numpy() == "bullish", 1, -1)
    out: Dict[str, object] = {}
    for rule, side in (("direction", nominal), ("fade", -nominal), ("always_long", np.ones(len(ds), int))):
        by_h = {}
        for horizon in HORIZONS:
            for cost_name, model in COST_MODELS.items():
                rt = model.round_trip_fee() + 2 * model.slippage_bps_base / 10_000.0
                key = f"{horizon}|{cost_name}"
                by_h[key] = nulls.directional_summary(ds, horizon, side, rt, n_samples=n_samples)
        out[rule] = by_h
    return out


def directional_barrier_tests(ds: pd.DataFrame) -> Dict:
    """Does the direction label predict which barrier is hit first?"""
    out: Dict[str, object] = {}
    bullish = (ds["direction"] == "bullish").to_numpy()
    for pair in ((0.01, 0.01), (0.02, 0.02), (0.02, 0.01)):
        tag = f"bar_{lab._level_tag(pair[0])}_{lab._level_tag(pair[1])}"
        for horizon in ("4h", "24h", "48h"):
            col = f"{tag}_{horizon}"
            if col not in ds.columns:
                continue
            lbl = ds[col].to_numpy()
            resolved = lbl != 0
            # "Correct" = up-first for a bullish zone, down-first for bearish.
            correct = np.where(bullish, lbl == 1, lbl == -1)
            out[col] = {
                "n_resolved": int(resolved.sum()),
                "direction_accuracy": nulls.proportion_test(
                    int((correct & resolved).sum()), int(resolved.sum())
                ),
                "ambiguous_fraction": float(ds[f"{col}_ambiguous"].mean()),
                "bullish_up_first": nulls.proportion_test(
                    int((bullish & resolved & (lbl == 1)).sum()),
                    int((bullish & resolved).sum()),
                ),
                "bearish_down_first": nulls.proportion_test(
                    int((~bullish & resolved & (lbl == -1)).sum()),
                    int((~bullish & resolved).sum()),
                ),
            }
    return out


def role_tests(ds: pd.DataFrame, controls: pd.DataFrame, n_samples: int) -> Dict:
    """Six concrete, falsifiable framings of "what is a zone for"."""
    zt = ds["decision_time_ms"].to_numpy(dtype=np.int64)
    ct = controls["open_time"].to_numpy(dtype=np.int64) + BAR_MS
    out: Dict[str, object] = {}

    # (i) Directional signal: same-direction rule applied at zones vs controls.
    zone_side = np.where(ds["direction"].to_numpy() == "bullish", 1, -1)
    ctrl_side = np.where(controls["det_direction"].to_numpy() > 0, 1, -1)
    for horizon in ("4h", "24h"):
        out[f"directional_signal_{horizon}"] = nulls.compare_means(
            zone_side * ds[f"fwd_ret_{horizon}"].to_numpy(dtype=float),
            zt,
            ctrl_side * controls[f"fwd_ret_{horizon}"].to_numpy(dtype=float),
            ct,
            n_samples=n_samples,
            seed=21,
        )

    # (ii) Volatility filter: forward realised vol at a zone vs a matched bar.
    out["volatility_filter_4h"] = nulls.compare_means(
        ds["fwd_rv_4h"].to_numpy(dtype=float),
        zt,
        controls["fwd_rv_4h"].to_numpy(dtype=float),
        ct,
        n_samples=n_samples,
        seed=22,
    )

    # (iii) Entry trigger: retest entry vs entry at the zone close, on zones that filled.
    filled = ds[ds["retested"].fillna(False).to_numpy().astype(bool)]
    side_f = np.where(filled["direction"].to_numpy() == "bullish", 1, -1)
    entry_trigger = {}
    for horizon in ("4h", "24h"):
        a = side_f * filled[f"entry_fwd_ret_{horizon}"].to_numpy(dtype=float)
        b = side_f * filled[f"fwd_ret_{horizon}"].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 50:
            continue
        diff = a[m] - b[m]
        ci = cross_sectional_bootstrap_ci(
            diff,
            filled["decision_time_ms"].to_numpy(dtype=np.int64)[m],
            n_samples=n_samples,
            seed=23,
        )
        entry_trigger[horizon] = {
            "n": int(m.sum()),
            "retest_entry_mean": float(np.mean(a[m])),
            "close_entry_mean": float(np.mean(b[m])),
            "improvement": float(np.mean(diff)),
            "improvement_ci": [ci.get("ci_low"), ci.get("ci_high")],
        }
    out["entry_trigger"] = entry_trigger

    # (iv) Location: zone retest rate (compared with control bars in s9).
    out["location_filter"] = {
        "zone_retest_rate": float(ds["retested"].fillna(False).mean()),
        "n": int(len(ds)),
        "note": (
            "A retest only requires price to re-enter the bar's own high-low band "
            "within 24h from the correct side. The comparable base rate for an "
            "arbitrary bar is in the matched-control retest column if present."
        ),
    }

    # (v) Trade-management reference: MFE/MAE asymmetry after a fill.
    mgmt = {}
    for h in (1, 4, 8, 24):
        mfe = ds[f"mfe_{h}h"].dropna().to_numpy(dtype=float)
        mae = ds[f"mae_{h}h"].dropna().to_numpy(dtype=float)
        if len(mfe) < 50:
            continue
        mgmt[f"{h}h"] = {
            "n": int(len(mfe)),
            "mean_mfe": float(mfe.mean()),
            "mean_mae": float(mae.mean()),
            "median_mfe": float(np.median(mfe)),
            "median_mae": float(np.median(mae)),
            "mfe_minus_mae": float(mfe.mean() - mae.mean()),
            "p_mae_negative": float((mae < 0).mean()),
        }
    out["trade_management"] = mgmt

    # (vi) Event marker: share of the panel that is a zone.
    out["event_density"] = {
        "zones": int(len(ds)),
        "zones_per_symbol_per_day": float(
            len(ds)
            / max(1, ds["symbol"].nunique())
            / max(1.0, (ds["source_open_time"].max() - ds["source_open_time"].min()) / 86_400_000)
        ),
    }
    return out


FEATURE_EXCLUDE_PREFIXES = (
    "fwd_",
    "entry_",
    "first_up_",
    "first_dn_",
    "bar_",
    "mfe_",
    "mae_",
)
FEATURE_EXCLUDE = {
    "source_open_time",
    "decision_time_ms",
    "bar_index",
    "zone_uid",
    "symbol",
    "direction",
    "mode",
    "detection_lag_bars",
    "live_equivalent",
    "would_alert",
    "passes_signal_gate",
    "retested",
    "retest_open_time",
    "retest_lag_bars",
    "hypothetical_entry_price",
    "lifecycle_status",
    "size",
    "zone_high",
    "zone_low",
    "zone_midpoint",
    "swing_high_prior",
    "swing_low_prior",
    "fwd_bars_available",
    "close",
    # SMC replay identifiers (s3s_replay_smc.py), not features.
    "ob_open_time",
    "ob_index",
    "swing_open_time",
    "swing_price",
}


def chronological_holdout(
    ds: pd.DataFrame, fraction: float, time_col: str = "decision_time_ms"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve the final ``fraction`` of the sample off by time, once, up front.

    The cut is on decision time, not row order, and the holdout is only used
    in the sections that say so.
    """
    ordered = ds.sort_values(time_col, kind="mergesort")
    cut = ordered[time_col].quantile(1.0 - fraction)
    development = ordered[ordered[time_col] < cut].copy()
    holdout = ordered[ordered[time_col] >= cut].copy()
    return development, holdout


def feature_columns(ds: pd.DataFrame) -> List[str]:
    cols = []
    for c in ds.columns:
        if c in FEATURE_EXCLUDE or c.startswith(FEATURE_EXCLUDE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(ds[c]):
            continue
        cols.append(c)
    return cols


def load_or_build_controls(ds: pd.DataFrame, sfx: str, workers: int, seed: int) -> pd.DataFrame:
    """Matched controls, reused from disk only when the cache key (zones, seed, parameters) matches."""
    ctrl_path = paths.LABELS_DIR / f"matched_controls{sfx}.parquet"
    key_path = ctrl_path.with_suffix(".key.json")
    key = controls_cache_key(ds, seed)
    cached = key_path.exists() and json.loads(key_path.read_text(encoding="utf-8")) == key
    if ctrl_path.exists() and cached:
        controls = pd.read_parquet(ctrl_path)
    else:
        log.info("sampling matched controls" + (" (cache key changed)" if ctrl_path.exists() else ""))
        controls = build_controls(ds, workers, seed)
        controls.to_parquet(ctrl_path, index=False, compression="zstd")
        key_path.write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
    log.info(f"{len(controls)} matched control bars")
    return controls


def strategy_grid() -> List[strategies.TradeSpec]:
    """The 12 trade specifications evaluated (each under two cost models: 24 trials)."""
    specs: List[strategies.TradeSpec] = []
    for rule in ("direction", "fade"):
        for entry in ("close", "retest"):
            for horizon in ("4h", "24h"):
                specs.append(strategies.TradeSpec(side_rule=rule, entry=entry, exit="time", horizon=horizon))
        for tp, sl in ((0.01, 0.01), (0.02, 0.01)):
            specs.append(
                strategies.TradeSpec(
                    side_rule=rule,
                    entry="retest",
                    exit="barrier",
                    horizon="24h",
                    take_profit=tp,
                    stop_loss=sl,
                )
            )
    return specs


def evaluate_strategies(
    ds: pd.DataFrame, cfg: HarnessConfig, regime_table: pd.DataFrame, ledger: TrialLedger
) -> Dict[str, Any]:
    """Run every grid configuration through the harness and log each to the trial ledger."""
    illiq_by_uid = dict(zip(ds["zone_uid"], strategies.illiquidity_score(ds)))
    strategy_results: Dict[str, Any] = {}
    for s in strategy_grid():
        trades = strategies.build_trades(ds, s)
        if trades.empty:
            continue
        illiq = pd.Series([illiq_by_uid.get(u, 0.0) for u in trades["zone_uid"]], index=trades.index)
        for cost_name in ("binance_spot_taker", "mexc_futures_taker"):
            key = (
                f"{s.side_rule}|{s.entry}|{s.exit}|{s.horizon}"
                f"{'' if s.exit == 'time' else f'|{s.take_profit}/{s.stop_loss}'}|{cost_name}"
            )
            res = evaluate(
                trades,
                COST_MODELS[cost_name],
                cfg,
                illiquidity=illiq,
                regime_table=regime_table,
                label=key,
            )
            flat = flatten_evaluation(res)
            ledger.log_trial({**s.describe(), "cost_model": cost_name}, flat, label=key)
            flat.update(strategies.ambiguity_sensitivity(trades, s, flat.get("mean_ret_net")))
            breakdown = res.get("regime_breakdown")
            flat["regime_breakdown"] = (
                breakdown.to_dict("records") if isinstance(breakdown, pd.DataFrame) else breakdown
            )
            strategy_results[key] = flat
    return strategy_results


def liquidity_tier_tests(ds: pd.DataFrame, rt: float, n_samples: int) -> Dict[str, Any]:
    liq = {}
    for tier, grp_idx in ds.groupby("liquidity_tier").groups.items():
        grp = ds.loc[grp_idx]
        s = np.where(grp["direction"].to_numpy() == "bullish", 1, -1)
        liq[str(tier)] = {
            h: nulls.directional_summary(grp, h, s, rt, n_samples=n_samples) for h in ("4h", "24h")
        }
    return liq


def feature_screens(
    ds: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    nominal: np.ndarray,
    holdout_fraction: float,
    sfx: str,
) -> Dict[str, Path]:
    """IS-vs-OOS feature screens (FDR corrected); returns the CSV written for each target."""
    fcols = feature_columns(ds)
    log.info(f"{len(fcols)} candidate features")
    files: Dict[str, Path] = {}
    for target, binary in (
        ("fwd_ret_4h", "bar_1_1_4h"),
        ("fwd_ret_24h", "bar_1_1_24h"),
    ):
        dev = development.assign(_signed=development[target])
        hold = holdout.assign(_signed=holdout[target])
        res = nulls.screen_is_oos(dev, hold, fcols, target, binary)
        files[target] = paths.REPORTS_DIR / f"feature_screen_{target}{sfx}.csv"
        res.to_csv(files[target], index=False)

    # Directional screen: features against the signed return of the direction rule.
    ds_signed = ds.assign(
        signed_ret_4h=nominal * ds["fwd_ret_4h"],
        signed_ret_24h=nominal * ds["fwd_ret_24h"],
    )
    dev_s, hold_s = chronological_holdout(ds_signed, holdout_fraction)
    screen_signed = nulls.screen_is_oos(dev_s, hold_s, fcols, "signed_ret_4h", None)
    files["signed_ret_4h"] = paths.REPORTS_DIR / f"feature_screen_signed_direction_4h{sfx}.csv"
    screen_signed.to_csv(files["signed_ret_4h"], index=False)
    return files


def walk_forward_summary(
    ds: pd.DataFrame, spec: strategies.TradeSpec, cfg: HarnessConfig
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Per-fold out-of-sample performance of one rule, and the fold summary."""
    trades_dir = strategies.build_trades(ds, spec)
    folds = walk_forward_folds(trades_dir, cfg)
    net_rt = float(COST_MODELS["mexc_futures_taker"].round_trip_fee())
    fold_perf = []
    for f in folds:
        train, test = f["train"], f["test"]
        tr = train["ret_gross"].to_numpy(dtype=float) if len(train) else np.array([])
        te = test["ret_gross"].to_numpy(dtype=float) if len(test) else np.array([])
        fold_perf.append(
            {
                "fold": f.get("fold"),
                "test_start": str(pd.to_datetime(f["test_start_ms"], unit="ms", utc=True).date()),
                "n_train": int(f.get("n_train", 0)),
                "n_test": int(f.get("n_test", 0)),
                "n_purged": int(f.get("n_purged", 0)),
                "n_embargoed": int(f.get("n_embargoed", 0)),
                "train_mean_gross": float(tr.mean()) if tr.size else None,
                "test_mean_gross": float(te.mean()) if te.size else None,
                "test_mean_net": float(te.mean() - net_rt) if te.size else None,
            }
        )
    tested = [f["test_mean_gross"] for f in fold_perf if f["test_mean_gross"] is not None]
    fold_summary = {
        "n_folds": len(fold_perf),
        "cost_model_for_net": "mexc_futures_taker",
        "folds_gross_positive": int(sum(1 for v in tested if v > 0)),
        "folds_net_positive": int(sum(1 for f in fold_perf if (f["test_mean_net"] or 0) > 0)),
        "mean_of_fold_test_gross": float(np.mean(tested)) if tested else None,
        "std_of_fold_test_gross": float(np.std(tested, ddof=1)) if len(tested) > 1 else None,
        "train_test_rank_corr": (
            float(
                pd.Series([f["train_mean_gross"] for f in fold_perf]).corr(
                    pd.Series([f["test_mean_gross"] for f in fold_perf]), method="spearman"
                )
            )
            if len(tested) > 2
            else None
        ),
        "note": (
            "Per-fold mean is the average signed per-trade return of the nominal "
            "direction rule on that fold's out-of-sample window. A rule with a real "
            "edge should be gross-positive in a clear majority of folds."
        ),
    }
    return fold_summary, fold_perf


def pbo_over_grid(ds: pd.DataFrame, rt: float) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """PBO (CSCV, 10 splits) over daily returns of the direction/fade x horizon grid."""
    perf_cols = []
    spec_names = []
    for rule in ("direction", "fade"):
        for horizon in ("1h", "4h", "24h", "48h"):
            s = np.where(ds["direction"].to_numpy() == "bullish", 1, -1)
            s = s if rule == "direction" else -s
            series = pd.Series(
                s * ds[f"fwd_ret_{horizon}"].to_numpy(dtype=float) - rt,
                index=pd.to_datetime(ds["decision_time_ms"], unit="ms", utc=True),
            )
            daily = series.resample("1D").mean()
            perf_cols.append(daily)
            spec_names.append(f"{rule}|{horizon}")
    matrix = pd.concat(perf_cols, axis=1).dropna()
    pbo = pbo_cscv(matrix.to_numpy(), n_splits=10)
    pbo["configurations"] = spec_names
    return matrix, pbo, spec_names


def deflated_best_sharpe(
    strategy_results: Dict[str, Any], matrix: pd.DataFrame, n_trials: int
) -> Tuple[Dict[str, Any], str]:
    """Deflated Sharpe of the best configuration's daily portfolio Sharpe.

    A per-trade Sharpe over ~150 simultaneous trades a day annualises to a number
    no account could earn, so the daily portfolio return is used.
    """
    sharpes = {
        k: v.get("sharpe_daily_net")
        for k, v in strategy_results.items()
        if v.get("sharpe_daily_net") is not None
    }
    best = max(sharpes, key=sharpes.get) if sharpes else matrix.mean().idxmax()
    sr = float(sharpes[best]) if sharpes else 0.0
    sr_var = float(np.var(list(sharpes.values()), ddof=1)) if len(sharpes) > 1 else 1e-9
    best_series = matrix[matrix.columns[int(np.argmax(matrix.mean().to_numpy()))]].to_numpy()
    dsr = deflated_sharpe_ratio(
        sr,
        n_trials=n_trials,
        sr_variance=max(sr_var, 1e-9),
        n_obs=len(best_series),
        skew=float(pd.Series(best_series).skew()),
        kurtosis=float(pd.Series(best_series).kurtosis() + 3.0),
    )
    return dsr, str(best)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _CFG = research_config.get()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=_CFG.evaluation.bootstrap_samples)
    parser.add_argument("--seed", type=int, default=_CFG.seeds.matched_controls)
    parser.add_argument("--holdout-fraction", type=float, default=_CFG.evaluation.holdout_fraction)
    parser.add_argument("--tag", default="", help="zone population suffix, e.g. smc")
    args = parser.parse_args(argv)
    sfx = f"_{args.tag}" if args.tag else ""

    t0 = time.time()
    paths.ensure_dirs()
    ds = pd.read_parquet(paths.FEATURES_DIR / f"zone_dataset{sfx}.parquet")
    ds["liquidity_tier"] = strategies.liquidity_tier(ds)
    log.info(f"dataset {len(ds)} zones, {ds['symbol'].nunique()} symbols")

    # Holdout is carved first and only used where explicitly stated.
    development, holdout = chronological_holdout(ds, args.holdout_fraction)
    log.info(
        f"development {len(development)} / holdout {len(holdout)} "
        f"(split at {pd.to_datetime(holdout['decision_time_ms'].min(), unit='ms', utc=True)})"
    )
    controls = load_or_build_controls(ds, sfx, args.workers, args.seed)

    label_cols = (
        ["open_time"]
        + [f"fwd_ret_{h}" for h in HORIZONS]
        + [f"fwd_rv_{h}" for h in ("4h", "24h", "48h")]
        + [f"first_{s}_{lab._level_tag(l)}" for l in lab.BARRIER_LEVELS for s in ("up", "dn")]
    )
    all_bars = all_bars_population(sorted(ds["symbol"].unique()), label_cols)
    log.info(f"{len(all_bars)} bars in the unconditional population")

    log.info("unconditional tests")
    uncond = unconditional_tests(ds, controls, all_bars, args.bootstrap_samples)

    log.info("directional tests")
    direction = directional_tests(ds, args.bootstrap_samples)
    barriers = directional_barrier_tests(ds)

    log.info("regime and liquidity breakdowns")
    btc = loader.load_ohlcv("BTCUSDT")
    regime_table = regimes.btc_regime_labels(btc)
    spec = strategies.TradeSpec(side_rule="direction", entry="close", exit="time", horizon="4h")
    cfg = HarnessConfig.from_research(bootstrap_samples=args.bootstrap_samples)
    prefix = f"s8-null-baselines{sfx}"
    # The published report deflated against every earlier session of this scan too.
    ledger = TrialLedger(
        name_prefix=prefix,
        description="First-round null and baseline evaluation of the replayed zone population",
        prior_trials=historical_prior(prefix),
    )
    strategy_results = evaluate_strategies(ds, cfg, regime_table, ledger)

    log.info("liquidity tiers")
    nominal = np.where(ds["direction"].to_numpy() == "bullish", 1, -1)
    rt = COST_MODELS["binance_spot_taker"].round_trip_fee() + 4e-4
    liq = liquidity_tier_tests(ds, rt, args.bootstrap_samples)

    log.info("feature screen (IS vs OOS, FDR corrected)")
    screen_files = feature_screens(ds, development, holdout, nominal, args.holdout_fraction, sfx)

    log.info("role tests")
    roles = role_tests(ds, controls, args.bootstrap_samples)

    log.info("walk-forward and overfitting diagnostics")
    fold_summary, fold_perf = walk_forward_summary(ds, spec, cfg)

    # PBO over the specification grid actually evaluated above.
    matrix, pbo, spec_names = pbo_over_grid(ds, rt)
    dsr, best = deflated_best_sharpe(strategy_results, matrix, ledger.n_trials or len(spec_names))

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "n_zones": int(len(ds)),
            "n_symbols": int(ds["symbol"].nunique()),
            "start": str(pd.to_datetime(ds["source_open_time"].min(), unit="ms", utc=True)),
            "end": str(pd.to_datetime(ds["source_open_time"].max(), unit="ms", utc=True)),
            "n_development": int(len(development)),
            "n_holdout": int(len(holdout)),
            "holdout_start": str(pd.to_datetime(holdout["decision_time_ms"].min(), unit="ms", utc=True)),
            "n_matched_controls": int(len(controls)),
            "n_all_bars": int(len(all_bars)),
        },
        "cost_models": {k: v.as_dict() for k, v in COST_MODELS.items()},
        "unconditional": uncond,
        "directional": direction,
        "directional_barriers": barriers,
        "liquidity_tiers": liq,
        "strategy_evaluations": strategy_results,
        "roles": roles,
        "walk_forward": {"summary": fold_summary, "folds": fold_perf},
        "pbo": pbo,
        "deflated_sharpe": {
            **dsr,
            "best_configuration": best,
            "sharpe_basis": "annualized daily portfolio return, net of costs",
            "n_trials_logged": ledger.n_trials,
            "n_trials_prior": ledger.prior_trials,
        },
        "feature_screen_files": {k: paths.rel(v) for k, v in screen_files.items()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = paths.REPORTS_DIR / f"null_baselines{sfx}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(f"-> {out}  ({report['elapsed_s']}s)")
    return 0
