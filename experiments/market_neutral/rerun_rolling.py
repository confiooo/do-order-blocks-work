"""Replace invalid rolling-beta executions with a corrected z-score run.

Fixed-beta rows are not rewritten. The zero-trade result stays on the row as
the invalid execution. No new experiment ids are inserted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.market_neutral.backtest.pair_costs import attach_costs
from experiments.market_neutral.backtest.simulate import simulate_pair
from experiments.market_neutral.calendar import test_window
from experiments.market_neutral.cointegration import filter_trades, fold_results, passing_windows
from experiments.market_neutral.data.bars import align_closes, load_closes
from experiments.market_neutral.data.universe import rank_lookup
from experiments.market_neutral.experiments.registry import output_dir
from experiments.market_neutral.features.spread import pair_zscore
from experiments.market_neutral.strategies.relative_value import BETA_MODES, WINDOWS_H
from experiments.market_neutral.validation.criteria import judge
from experiments.market_neutral.validation.score import (
    basket_returns,
    compound_daily,
    daily_calendar,
    daily_close_returns,
    oos_mask,
    score_configuration,
)
from hawk_research.data import paths
from hawk_research.evaluation.costs import CostModel, apply_costs
from hawk_research.stats.multiple_testing import deflated_sharpe_ratio, pbo_cscv

log = logging.getLogger(__name__)

INVALID_REASON = (
    "Not validly tested. The rolling-beta z-score was entirely non-finite: "
    "a NaN in the beta warmup poisoned every later rolling sum. This is not a no-signal result."
)

METRIC_COLUMNS = (
    "n_trades",
    "win_rate_net",
    "expectancy_net",
    "expectancy_gross",
    "avg_win_net",
    "avg_loss_net",
    "profit_factor_net",
    "total_ret_net_sum",
    "total_ret_gross_sum",
    "sharpe_gross_ann",
    "sharpe_daily_sr",
    "sortino_net_ann",
    "max_drawdown_net",
    "volatility_ann",
    "mean_hold_hours",
    "turnover_capital",
    "gross_exposure",
    "net_exposure",
    "beta_btc",
    "corr_btc",
    "corr_market",
    "n_days",
    "sharpe_net_ann",
    "sharpe_ci_low",
    "sharpe_ci_high",
    "sharpe_ci_width",
    "n_samples",
    "dsr",
    "expected_max_sr",
    "sr_variance",
    "n_trials",
    "status",
    "reasons",
)


def smoke_pairs() -> List[dict]:
    """Finite-rate and threshold crossings for the three diagnostic pairs."""
    from experiments.market_neutral.data.bars import development_1h

    rows = []
    specs = [("ETHUSDT", "BTCUSDT"), ("BNBUSDT", "BTCUSDT"), ("SOLUSDT", "BTCUSDT")]
    for left, right in specs:
        panel = align_closes(development_1h(left), development_1h(right))
        times = panel["open_time"].to_numpy(dtype=np.int64)
        ca = panel["close_a"].to_numpy(dtype=float)
        cb = panel["close_b"].to_numpy(dtype=float)
        for window in WINDOWS_H:
            for mode in ("fixed", "rolling_ols"):
                z, beta = pair_zscore(times, ca, cb, int(window), mode)
                finite_z = np.isfinite(z)
                finite_beta = np.isfinite(beta)
                spread_ok = finite_beta & np.isfinite(ca) & np.isfinite(cb) & (ca > 0) & (cb > 0)
                zz = z[finite_z]
                row = {
                    "pair": f"{left}|{right}",
                    "window_h": int(window),
                    "beta_mode": mode,
                    "beta_finite_pct": float(finite_beta.mean()),
                    "spread_finite_pct": float(spread_ok.mean()),
                    "z_finite_pct": float(finite_z.mean()),
                    "z_mean": float(zz.mean()) if zz.size else None,
                    "z_std": float(zz.std(ddof=1)) if zz.size > 1 else None,
                    "z_min": float(zz.min()) if zz.size else None,
                    "z_max": float(zz.max()) if zz.size else None,
                }
                for level in (1.5, 2.0, 2.5, 3.0):
                    row[f"cross_{level}"] = int((np.abs(zz) >= level).sum()) if zz.size else 0
                rows.append(row)
    return rows


def _names_and_pairs(frame: pd.DataFrame) -> tuple[list, list]:
    pairs = []
    seen = set()
    for symbol in frame["symbol"].astype(str):
        if symbol in seen:
            continue
        seen.add(symbol)
        left, right = symbol.split("|", 1)
        pairs.append((left, right))
    names = list(dict.fromkeys([leg for pair in pairs for leg in pair]))
    return names, pairs


def _score_table(
    frame: pd.DataFrame,
    *,
    names: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
    use_cointegration_gate: bool,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, Dict[int, dict]]:
    closes = load_closes(names)
    ranks = rank_lookup(list(names))
    panels = {(left, right): align_closes(closes[left], closes[right]) for left, right in pairs}
    z_cache = {}
    gates = {}
    for left, right in pairs:
        panel = panels[(left, right)]
        times = panel["open_time"].to_numpy(dtype=np.int64)
        ca = panel["close_a"].to_numpy(dtype=float)
        cb = panel["close_b"].to_numpy(dtype=float)
        for mode in BETA_MODES:
            for window in WINDOWS_H:
                z_cache[(left, right, mode, int(window))] = pair_zscore(times, ca, cb, int(window), mode)
        if use_cointegration_gate:
            for mode in ("fixed", "rolling_ols"):
                gates[(left, right, mode)] = passing_windows(fold_results(times, ca, cb, mode))

    test_start, test_end = test_window()
    days = daily_calendar(test_start, test_end)
    btc_daily = daily_close_returns(closes["BTCUSDT"])
    market_daily = basket_returns({name: closes[name] for name in names if name in closes})
    cost_model = CostModel.binance_spot_taker()
    daily = np.zeros((days.size, len(frame)), dtype=float)
    scored: Dict[int, dict] = {}

    for position, (idx, cell) in enumerate(frame.iterrows()):
        left, right = str(cell["symbol"]).split("|", 1)
        mode = str(cell["beta_mode"])
        window = int(cell["window_h"])
        panel = panels[(left, right)]
        z, beta = z_cache[(left, right, mode, window)]
        trades = simulate_pair(
            panel["open_time"].to_numpy(dtype=np.int64),
            panel["close_a"].to_numpy(dtype=float),
            panel["close_b"].to_numpy(dtype=float),
            z,
            beta,
            entry=float(cell["entry"]),
            exit_band=float(cell["exit_band"]),
            max_hold_bars=int(cell["window_h"]) * int(cell["hold_mult"]),
            symbol=str(cell["symbol"]),
            beta_mode=mode,
            window=window,
        )
        if use_cointegration_gate:
            trades = filter_trades(trades, gates[(left, right, mode)])
        trades = attach_costs(trades, left, right, ranks)
        priced = apply_costs(trades, cost_model)
        oos = oos_mask(priced, test_start, test_end)
        net_daily = compound_daily(oos, "ret_net", days)
        daily[:, position] = net_daily
        if mode == "rolling_ols":
            stats = score_configuration(oos, days, btc_daily, market_daily, n_samples=n_samples, seed=seed)
            stats.pop("daily_net", None)
            scored[int(idx)] = stats
        if position % 100 == 0:
            log.info(f"simulated {position + 1}/{len(frame)}")
    return daily, scored


def _apply_scores(
    frame: pd.DataFrame, scored: Dict[int, dict], daily: np.ndarray, rerun_at: str
) -> pd.DataFrame:
    out = frame.copy()
    fixed = out["beta_mode"] != "rolling_ols"
    fixed_snapshot = (
        out.loc[fixed, list(METRIC_COLUMNS)].copy() if set(METRIC_COLUMNS) <= set(out.columns) else None
    )
    if "prior_n_trades" not in out.columns:
        out["prior_n_trades"] = np.nan
        out["prior_status"] = None
        out["prior_reasons"] = None
        out["invalidation_reason"] = None
        out["rerun_at"] = None
    if "pbo_completed_grid" not in out.columns:
        out["pbo_completed_grid"] = np.nan

    sharpes = []
    for stats in scored.values():
        sr = stats.get("sharpe_daily_sr")
        if sr is not None:
            sharpes.append(float(sr))
    for _, row in out.loc[fixed].iterrows():
        sr = row.get("sharpe_daily_sr")
        if sr is not None and np.isfinite(sr):
            sharpes.append(float(sr))
    sr_variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    n_trials = int(len(out))
    pbo = pbo_cscv(daily, n_splits=16)

    for idx, stats in scored.items():
        prior = out.at[idx, "prior_status"]
        missing_prior = prior is None or prior == "" or (isinstance(prior, float) and np.isnan(prior))
        if missing_prior:
            out.at[idx, "prior_n_trades"] = out.at[idx, "n_trades"]
            out.at[idx, "prior_status"] = out.at[idx, "status"]
            out.at[idx, "prior_reasons"] = out.at[idx, "reasons"]
            out.at[idx, "invalidation_reason"] = INVALID_REASON
        sr = stats.get("sharpe_daily_sr")
        if sr is None:
            stats["dsr"] = None
            stats["expected_max_sr"] = None
        else:
            packed = deflated_sharpe_ratio(float(sr), n_trials, sr_variance, int(stats["n_days"]))
            stats["dsr"] = packed["dsr"]
            stats["expected_max_sr"] = packed["expected_max_sr"]
        stats["sr_variance"] = sr_variance
        stats["n_trials"] = n_trials
        verdict = judge(stats)
        stats["status"] = verdict["status"]
        stats["reasons"] = " | ".join(verdict["reasons"])
        stats["sharpe_ci_width"] = verdict["sharpe_ci_width"]
        for column in METRIC_COLUMNS:
            if column in stats and column in out.columns:
                out.at[idx, column] = stats[column]
        out.at[idx, "rerun_at"] = rerun_at
        out.at[idx, "pbo_completed_grid"] = pbo.get("pbo")

    if fixed_snapshot is not None:
        after = out.loc[fixed, list(METRIC_COLUMNS)]
        if not fixed_snapshot.astype(str).equals(after.astype(str)):
            raise RuntimeError("fixed-beta columns changed during the rolling-beta rerun")
    return out, pbo, sr_variance


def _patch_sqlite(prefix: str, frame: pd.DataFrame, rerun_at: str) -> int:
    conn = sqlite3.connect(paths.RESEARCH_DB)
    rows = conn.execute(
        "SELECT experiment_id, parameters_json FROM experiments WHERE name LIKE ?",
        (f"{prefix}/%",),
    ).fetchall()
    by_id = frame.set_index("experiment_id")
    updated = 0
    for experiment_id, raw in rows:
        blob = json.loads(raw)
        inner = blob.get("parameters", {}).get("parameters", {})
        if inner.get("beta_mode") != "rolling_ols":
            continue
        if int(experiment_id) not in by_id.index:
            continue
        row = by_id.loc[int(experiment_id)]
        metrics = blob.get("metrics", {})
        if "invalid_execution" not in blob:
            blob["invalid_execution"] = {
                "status": metrics.get("status"),
                "n_trades": metrics.get("n_trades"),
                "reasons": metrics.get("reasons"),
                "sharpe_net_ann": metrics.get("sharpe_net_ann"),
                "reason": INVALID_REASON,
                "preserved_at": rerun_at,
            }
        new_metrics = dict(metrics)
        for column in METRIC_COLUMNS:
            if column in frame.columns and column in row.index:
                value = row[column]
                if column == "reasons":
                    new_metrics[column] = (
                        []
                        if value is None or (isinstance(value, float) and np.isnan(value))
                        else [str(value)]
                    )
                elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                    new_metrics[column] = None
                else:
                    new_metrics[column] = value if not hasattr(value, "item") else value.item()
        new_metrics["validity"] = "rerun_after_rolling_zscore_fix"
        new_metrics["rerun_at"] = rerun_at
        blob["metrics"] = new_metrics
        blob["rerun_at"] = rerun_at
        conn.execute(
            "UPDATE experiments SET parameters_json = ? WHERE experiment_id = ?",
            (json.dumps(blob, sort_keys=True, default=str), int(experiment_id)),
        )
        updated += 1
    conn.commit()
    conn.close()
    return updated


def _patch_jsonl(name: str, frame: pd.DataFrame, rerun_at: str) -> None:
    path = output_dir() / name
    if not path.exists():
        return
    by_id = frame.set_index("experiment_id")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        inner = row.get("parameters", {}).get("parameters", {})
        experiment_id = row.get("experiment_id")
        if inner.get("beta_mode") == "rolling_ols" and experiment_id in by_id.index:
            current = by_id.loc[int(experiment_id)]
            metrics = row.get("metrics", {})
            if "invalid_execution" not in row:
                row["invalid_execution"] = {
                    "status": metrics.get("status"),
                    "n_trades": metrics.get("n_trades"),
                    "reasons": metrics.get("reasons"),
                    "reason": INVALID_REASON,
                    "preserved_at": rerun_at,
                }
            metrics["status"] = current["status"]
            metrics["n_trades"] = int(current["n_trades"])
            metrics["sharpe_net_ann"] = (
                None if pd.isna(current["sharpe_net_ann"]) else float(current["sharpe_net_ann"])
            )
            metrics["reasons"] = [str(current["reasons"])]
            metrics["validity"] = "rerun_after_rolling_zscore_fix"
            metrics["rerun_at"] = rerun_at
            row["metrics"] = metrics
            row["rerun_at"] = rerun_at
        lines.append(json.dumps(row, sort_keys=True, default=str))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rerun_surface(
    stem: str,
    ledger_prefix: str,
    jsonl_name: str,
    *,
    use_cointegration_gate: bool = False,
    n_samples: int = 1000,
    seed: int = 7,
) -> Dict[str, object]:
    path = output_dir() / f"{stem}.parquet"
    frame = pd.read_parquet(path).reset_index(drop=True)
    fixed_before = frame.loc[
        frame["beta_mode"] != "rolling_ols", ["n_trades", "status", "sharpe_net_ann"]
    ].copy()
    names, pairs = _names_and_pairs(frame)
    log.info(f"{stem}: {int((frame.beta_mode == 'rolling_ols').sum())} rolling cells")
    daily, scored = _score_table(
        frame,
        names=names,
        pairs=pairs,
        use_cointegration_gate=use_cointegration_gate,
        n_samples=n_samples,
        seed=seed,
    )
    rerun_at = datetime.now(UTC).isoformat()
    updated, pbo, sr_variance = _apply_scores(frame, scored, daily, rerun_at)
    fixed_after = updated.loc[updated["beta_mode"] != "rolling_ols", ["n_trades", "status", "sharpe_net_ann"]]
    if not fixed_before.astype(str).equals(fixed_after.astype(str)):
        raise RuntimeError(f"fixed-beta results changed in {stem}")
    updated.to_parquet(path, index=False)
    summary_path = output_dir() / f"{stem}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original_pbo = summary.get("pbo")
    counts = updated["status"].value_counts().to_dict()
    summary["counts"] = {str(key): int(value) for key, value in counts.items()}
    summary["pbo_with_invalid_rolling"] = original_pbo
    summary["pbo_after_rolling_beta_fix"] = pbo.get("pbo")
    summary["pbo_after_rolling_beta_fix_splits"] = pbo.get("n_splits")
    summary["pbo_after_rolling_beta_fix_combinations"] = pbo.get("n_combinations")
    summary["rolling_beta_rerun"] = {
        "rerun_at": rerun_at,
        "cells": int(len(scored)),
        "sr_variance": sr_variance,
        "n_trials": int(len(updated)),
        "invalidation_reason": INVALID_REASON,
        "fixed_beta_rows_unchanged": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    sqlite_n = _patch_sqlite(ledger_prefix, updated, rerun_at)
    _patch_jsonl(jsonl_name, updated, rerun_at)
    rolling = updated[updated["beta_mode"] == "rolling_ols"]
    log.info(
        f"{stem} rolling status {rolling['status'].value_counts().to_dict()} "
        f"sqlite {sqlite_n} pbo {pbo.get('pbo')}"
    )
    return summary


def main() -> None:
    rows = smoke_pairs()
    broken = [row for row in rows if row["beta_mode"] == "rolling_ols" and row["z_finite_pct"] <= 0.0]
    if broken:
        raise RuntimeError(f"rolling-beta z-score is still empty: {broken[:3]}")
    for row in rows:
        if row["beta_mode"] != "rolling_ols":
            continue
        log.info(
            f"{row['pair']} {row['window_h']}h z_finite={row['z_finite_pct']:.3f} "
            f"std={row['z_std']:.3f} cross_1.5={row['cross_1.5']}"
        )
    rerun_surface("relative_value_phase1", "mn_rv_phase1", "experiments.jsonl")
    rerun_surface("relative_value_tier2", "mn_rv_tier2", "tier2_experiments.jsonl")
    rerun_surface(
        "relative_value_cointegration",
        "rv_cointegration_filter_v1",
        "cointegration_experiments.jsonl",
        use_cointegration_gate=True,
    )


if __name__ == "__main__":
    main()
