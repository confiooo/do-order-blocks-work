"""Run the Family B phase-1 matrix and log every cell before judging."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.market_neutral.calendar import (
    DEV_END,
    DEV_START,
    UNIVERSE_BIAS_NOTE,
    test_window,
    walk_forward_folds,
)
from experiments.market_neutral.data.bars import HOUR_MS, development_1h
from experiments.market_neutral.data.universe import rank_lookup
from experiments.market_neutral.experiments.registry import log_experiment, open_ledger, output_dir
from experiments.market_neutral.features.spread import pair_zscore
from experiments.market_neutral.funding_basis.costs import bnb_net_adjustment, cost_model_record
from experiments.market_neutral.funding_basis.criteria import judge_funding
from experiments.market_neutral.funding_basis.loaders import load_funding, load_perpetual
from experiments.market_neutral.funding_basis.matrix import (
    BASIS_WINDOWS,
    PERSIST_WINDOWS,
    build_cells,
    expected_cells,
    search_space,
)
from experiments.market_neutral.funding_basis.signals import (
    development_funding_sample,
    distinct_thresholds,
    latest_funding_index,
    per_name_percentiles,
    perp_price_at_funding,
    rolling_funding_mean,
)
from experiments.market_neutral.funding_basis.simulate import simulate_basis, simulate_carry
from experiments.market_neutral.funding_basis.universe import FAMILY_B_NAMES, PEPE_EXCLUSION_NOTE
from experiments.market_neutral.validation.score import (
    basket_returns,
    compound_daily,
    daily_calendar,
    daily_close_returns,
    ols_beta_corr,
    oos_mask,
    score_configuration,
)
from hawk_research import constants
from hawk_research.data import paths
from hawk_research.evaluation.costs import CostModel, apply_costs
from hawk_research.stats.multiple_testing import deflated_sharpe_ratio, pbo_cscv

log = logging.getLogger(__name__)

LEDGER_PREFIX = "mn_fb_phase1"
PBO_SPLITS = 16


def _periods() -> Dict[str, object]:
    start_ms, end_ms = test_window()
    return {
        "training_period": {
            "start": DEV_START,
            "end_exclusive": DEV_END,
            "scheme": "180d walk-forward trains inside tier D. Thresholds are frozen from the tier-D funding distribution, not refit per fold.",
            "folds": walk_forward_folds(),
        },
        "validation_period": {
            "scheme": "5-day embargo between each train end and its test start. Not used for selection.",
        },
        "test_period": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "scheme": "Stitched fold test windows inside tier D. Entries before 2024-09-01 are impossible: spot bars start then.",
            "holdout": "2026-04-01 through 2026-08-31 is sealed and was not read.",
            "pre_2024": "Funding and perp history before 2024-09-01 is warmup for rolling signals only.",
        },
    }


def _ranks_for(open_time: np.ndarray, symbol: str, lookup: Dict[tuple, float]) -> np.ndarray:
    close = open_time.astype(np.int64) + HOUR_MS
    days = pd.to_datetime(close, unit="ms", utc=True).normalize()
    out = np.empty(len(days), dtype=float)
    for i, day in enumerate(days):
        out[i] = lookup.get((pd.Timestamp(day), symbol), np.nan)
    return out


def _prepare(symbol: str, lookup: Dict[tuple, float]) -> Dict[str, object]:
    spot = development_1h(symbol)[["open_time", "close"]].rename(columns={"close": "spot"})
    perp = load_perpetual(symbol)[["open_time", "close"]].rename(columns={"close": "perp"})
    merged = spot.merge(perp, on="open_time", how="inner").sort_values("open_time", kind="mergesort")
    funding = load_funding(symbol).sort_values("funding_time_ms", kind="mergesort")
    open_time = merged["open_time"].to_numpy(dtype=np.int64)
    spot_px = merged["spot"].to_numpy(dtype=float)
    perp_px = merged["perp"].to_numpy(dtype=float)
    fund_time = funding["funding_time_ms"].to_numpy(dtype=np.int64)
    fund_rate = funding["funding_rate"].to_numpy(dtype=float)
    close_time = open_time + HOUR_MS
    latest = latest_funding_index(close_time, fund_time)
    signals = {window: rolling_funding_mean(fund_rate, latest, window) for window in (1, *PERSIST_WINDOWS)}
    zscores = {}
    for window in BASIS_WINDOWS:
        z, _beta = pair_zscore(open_time, perp_px, spot_px, window, "fixed")
        zscores[window] = z
    return {
        "open_time": open_time,
        "spot": spot_px,
        "perp": perp_px,
        "fund_time": fund_time,
        "fund_rate": fund_rate,
        "perp_at_fund": perp_price_at_funding(open_time, perp_px, fund_time),
        "latest": latest,
        "signals": signals,
        "z": zscores,
        "ranks": _ranks_for(open_time, symbol, lookup),
        "spot_frame": merged.rename(columns={"spot": "close"})[["open_time", "close"]],
    }


def _run_cell(cell: Dict[str, object], panel: Dict[str, object]) -> pd.DataFrame:
    symbol = str(cell["symbol"])
    common = dict(
        open_time=panel["open_time"],
        spot=panel["spot"],
        perp=panel["perp"],
        fund_time=panel["fund_time"],
        fund_rate=panel["fund_rate"],
        perp_at_fund=panel["perp_at_fund"],
        latest=panel["latest"],
        symbol=symbol,
        rank=panel["ranks"],
    )
    if cell["stage"] == "B4":
        tag = f"B4|w{cell['window']}|e{cell['entry']}|x{cell['exit_band']}|f{int(cell['funding_filter'])}"
        return simulate_basis(
            **common,
            z=panel["z"][int(cell["window"])],
            entry=float(cell["entry"]),
            exit_band=float(cell["exit_band"]),
            funding_filter=bool(cell["funding_filter"]),
            tag=tag,
        )
    tag = f"{cell['stage']}|w{cell['window']}|{cell['threshold_label']}|{cell['exit_mode']}"
    return simulate_carry(
        **common,
        signal=panel["signals"][int(cell["window"])],
        threshold=float(cell["threshold"]),
        exit_mode=str(cell["exit_mode"]),
        tag=tag,
    )


def _spec(cell: Dict[str, object], space: Dict[str, object], periods: Dict[str, object]) -> Dict[str, object]:
    return {
        "family": "funding_basis",
        "strategy_name": cell["strategy_name"],
        "symbols": [cell["symbol"]],
        "timeframe": "1h",
        "parameters": {key: value for key, value in cell.items() if key != "strategy_name"},
        "feature_definitions": {
            "B2": "current settled funding rate, known at or before the bar close",
            "B3": "mean of the last 3, 9, or 21 settled rates, known at or before the bar close",
            "B4": "z-score of log(perp) - log(spot), fixed beta 1, same windows and bands as relative value",
        },
        "entry_rules": (
            "B2/B3: signal > threshold enters long spot and short perp. "
            "B4: z > entry only. funding_filter requires the latest settled rate to be positive."
        ),
        "exit_rules": (
            "Carry: funding turns negative, or 21 settlements, and 21 is always the cap. "
            "Basis: z back through 0, or inside 0.5σ. No hold-length sweep. "
            "A trade that cannot exit inside tier D is dropped."
        ),
        "cost_model": cost_model_record(),
        "training_period": periods["training_period"],
        "validation_period": periods["validation_period"],
        "test_period": periods["test_period"],
        "universe_bias_note": UNIVERSE_BIAS_NOTE,
        "family_b_note": PEPE_EXCLUSION_NOTE,
        "search_space": space,
    }


def _cagr(daily: np.ndarray) -> float | None:
    values = np.where(np.isfinite(daily), daily, 0.0)
    if values.size < 2:
        return None
    equity = float(np.prod(1.0 + values))
    if equity <= 0.0:
        return None
    years = values.size / constants.DAYS_PER_YEAR
    return float(equity ** (1.0 / years) - 1.0)


def run_phase1(*, n_samples: int = 1000, seed: int = 7) -> Dict[str, object]:
    names = list(FAMILY_B_NAMES)
    if len(names) != 11 or "PEPEUSDT" in names:
        raise RuntimeError("Family B phase 1 must be the 11-name book, without PEPEUSDT")
    funding_frames = {symbol: load_funding(symbol) for symbol in names}
    pooled = development_funding_sample(funding_frames.values())
    levels, threshold_report = distinct_thresholds(pooled)
    per_name = {symbol: per_name_percentiles(funding_frames[symbol]) for symbol in names}
    cells = build_cells(names, levels)
    expected = expected_cells(11, 3)
    if len(cells) != expected or expected != 792:
        raise RuntimeError(f"refusing to run: cell count {len(cells)} expected {expected}")
    space = search_space(levels, {**threshold_report, "per_name": per_name})
    log.info(
        f"{len(cells)} cells on {len(names)} names. Thresholds "
        + ", ".join(f"{label}={value:.8g}" for label, value in levels)
    )
    log.info(f"percentile report: {threshold_report}")

    ledger = open_ledger(
        name_prefix=LEDGER_PREFIX,
        jsonl_path=output_dir() / "funding_basis_experiments.jsonl",
        description="Family B phase 1, long spot / short perp. Holdout 2026-04-01..2026-08-31 sealed.",
    )
    if ledger.n_trials:
        raise RuntimeError(
            f"{ledger.name_prefix} already has {ledger.n_trials} trials. Refusing to append a second pass."
        )

    lookup = rank_lookup(names)
    panels = {symbol: _prepare(symbol, lookup) for symbol in names}
    test_start, test_end = test_window()
    days = daily_calendar(test_start, test_end)
    btc = daily_close_returns(development_1h("BTCUSDT"))
    market = basket_returns({symbol: panels[symbol]["spot_frame"] for symbol in names})
    underlying = {symbol: daily_close_returns(panels[symbol]["spot_frame"]) for symbol in names}
    cost_model = CostModel.binance_spot_taker()
    periods = _periods()

    daily = np.zeros((days.size, len(cells)), dtype=float)
    scored: List[tuple] = []
    for i, cell in enumerate(cells):
        trades = _run_cell(cell, panels[str(cell["symbol"])])
        priced = apply_costs(trades, cost_model)
        oos = oos_mask(priced, test_start, test_end)
        stats = score_configuration(oos, days, btc, market, n_samples=n_samples, seed=seed)
        net_daily = np.asarray(stats.pop("daily_net"), dtype=float)
        daily[:, i] = net_daily
        beta_u, _corr_u = ols_beta_corr(net_daily, _align(days, underlying[str(cell["symbol"])]))
        stats["beta_underlying"] = beta_u
        stats["capital_efficiency_cagr"] = _cagr(net_daily)
        stats["mean_ret_funding"] = float(oos["ret_funding"].mean()) if len(oos) else None
        stats["mean_ret_price"] = float(oos["ret_price"].mean()) if len(oos) else None
        stats["mean_dollar_imbalance"] = float(oos["dollar_imbalance"].mean()) if len(oos) else None
        stats["mean_abs_net_coin"] = float(oos["net_coin_exposure"].abs().mean()) if len(oos) else None
        bnb_daily = compound_daily(
            oos.assign(ret_net=oos["ret_net"] + oos["perp_fee"].map(bnb_net_adjustment)) if len(oos) else oos,
            "ret_net",
            days,
        )
        stats["sharpe_net_ann_bnb"] = _sharpe(bnb_daily)
        stats["capital_efficiency_cagr_bnb"] = _cagr(bnb_daily)
        scored.append((cell, stats))
        if i % 50 == 0:
            log.info(f"scored {i + 1}/{len(cells)}")

    finite = [
        float(stats["sharpe_daily_sr"]) for _, stats in scored if stats.get("sharpe_daily_sr") is not None
    ]
    sr_variance = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
    n_trials = len(scored)
    pbo = pbo_cscv(daily, n_splits=PBO_SPLITS)
    rows = []
    for i, (cell, stats) in enumerate(scored):
        sr = stats.get("sharpe_daily_sr")
        if sr is None:
            dsr_value = None
            expected_max = None
        else:
            packed = deflated_sharpe_ratio(float(sr), n_trials, sr_variance, int(stats["n_days"]))
            dsr_value = packed["dsr"]
            expected_max = packed["expected_max_sr"]
        stats["dsr"] = dsr_value
        stats["expected_max_sr"] = expected_max
        stats["sr_variance"] = sr_variance
        stats["pbo"] = pbo.get("pbo")
        stats["pbo_n_splits"] = pbo.get("n_splits")
        stats["pbo_n_combinations"] = pbo.get("n_combinations")
        stats["pbo_median_logit"] = pbo.get("median_logit")
        stats["n_trials"] = n_trials
        verdict = judge_funding(stats)
        stats["status"] = verdict["status"]
        stats["reasons"] = verdict["reasons"]
        stats["sharpe_ci_width"] = verdict["sharpe_ci_width"]
        metrics = {
            key: value
            for key, value in stats.items()
            if key != "daily_net" and not isinstance(value, np.ndarray)
        }
        if isinstance(metrics.get("reasons"), list):
            metrics["reasons"] = [str(item) for item in metrics["reasons"]]
        experiment_id = log_experiment(ledger, _spec(cell, space, periods), metrics, f"{i:04d}")
        flat = {
            "experiment_id": experiment_id,
            "stage": cell["stage"],
            "symbol": cell["symbol"],
            "window": cell.get("window"),
            "threshold_label": cell.get("threshold_label"),
            "threshold": cell.get("threshold"),
            "exit_mode": cell.get("exit_mode"),
            "entry": cell.get("entry"),
            "exit_band": cell.get("exit_band"),
            "funding_filter": cell.get("funding_filter"),
        }
        for key, value in metrics.items():
            if not isinstance(value, (list, dict)):
                flat[key] = value
        flat["reasons"] = " | ".join(verdict["reasons"])
        rows.append(flat)

    frame = pd.DataFrame(rows)
    folder = output_dir()
    surface = folder / "funding_basis_phase1.parquet"
    summary_path = folder / "funding_basis_phase1.json"
    frame.to_parquet(surface, index=False)
    counts = frame["status"].value_counts().to_dict() if len(frame) else {}
    summary = {
        "family": "funding_basis",
        "phase": 1,
        "n_cells": n_trials,
        "n_names": len(names),
        "names": names,
        "pepe": PEPE_EXCLUSION_NOTE,
        "thresholds": [{"label": label, "value": value} for label, value in levels],
        "threshold_report": threshold_report,
        "per_name_percentiles": per_name,
        "universe_bias_note": UNIVERSE_BIAS_NOTE,
        "holdout_touched": False,
        "pbo": pbo.get("pbo"),
        "pbo_n_splits": pbo.get("n_splits"),
        "pbo_n_combinations": pbo.get("n_combinations"),
        "pbo_median_logit": pbo.get("median_logit"),
        "sr_variance": sr_variance,
        "counts": {str(key): int(value) for key, value in counts.items()},
        "pass_experiment_ids": (
            frame.loc[frame["status"] == "pass", "experiment_id"].astype(int).tolist() if len(frame) else []
        ),
        "bnb_footnote": {
            "note": "Same trades with perp taker 4.5 bps instead of 5. Not additional trials.",
            "median_sharpe_primary": _median(frame, "sharpe_net_ann"),
            "median_sharpe_bnb": _median(frame, "sharpe_net_ann_bnb"),
        },
        "surface": paths.rel(surface),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log.info(f"status counts: {summary['counts']}")
    log.info(f"PBO={pbo.get('pbo')} splits={pbo.get('n_splits')} combinations={pbo.get('n_combinations')}")
    log.info(f"wrote {summary_path}")
    return summary


def _align(days: np.ndarray, series: pd.Series) -> np.ndarray:
    if series.empty:
        return np.full(days.size, np.nan)
    return series.reindex(days).to_numpy(dtype=float)


def _sharpe(daily: np.ndarray):
    from hawk_research.evaluation import metrics as metrics_mod

    return metrics_mod.sharpe(daily, constants.DAYS_PER_YEAR)


def _median(frame: pd.DataFrame, column: str):
    if column not in frame.columns or frame.empty:
        return None
    values = frame[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def main() -> None:
    run_phase1()


if __name__ == "__main__":
    main()
