"""Run the pre-registered relative-value matrix on tier D and log every cell.

The holdout (2026-04-01 through 2026-08-31) is not read. No cell is judged
until the full matrix has been scored, so the deflated Sharpe and PBO see
every trial.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.market_neutral.backtest.pair_costs import attach_costs, cost_model_record
from experiments.market_neutral.backtest.simulate import simulate_pair
from experiments.market_neutral.calendar import (
    DEV_END,
    DEV_START,
    UNIVERSE_BIAS_NOTE,
    test_window,
    walk_forward_folds,
)
from experiments.market_neutral.data.bars import align_closes, load_closes
from experiments.market_neutral.data.universe import (
    describe_universe,
    rank_lookup,
    select_names,
    select_pairs,
)
from experiments.market_neutral.experiments.registry import log_experiment, open_ledger, output_dir
from experiments.market_neutral.features.spread import pair_zscore
from experiments.market_neutral.strategies.relative_value import (
    BETA_MODES,
    WINDOWS_H,
    as_records,
    feature_definitions,
    matrix_size,
    search_space,
)
from experiments.market_neutral.validation.criteria import judge
from experiments.market_neutral.validation.score import (
    basket_returns,
    daily_calendar,
    daily_close_returns,
    oos_mask,
    score_configuration,
)
from hawk_research.data import paths
from hawk_research.evaluation.costs import CostModel, apply_costs
from hawk_research.stats.multiple_testing import deflated_sharpe_ratio, pbo_cscv

log = logging.getLogger(__name__)

Pair = Tuple[str, str]


def _periods() -> Dict[str, object]:
    start_ms, end_ms = test_window()
    return {
        "training_period": {
            "start": DEV_START,
            "end_exclusive": DEV_END,
            "scheme": "180d walk-forward trains inside tier D. Parameters are pre-registered, not refit.",
            "folds": walk_forward_folds(),
        },
        "validation_period": {
            "scheme": "5-day embargo between each train end and its test start. Not used for selection.",
        },
        "test_period": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "scheme": "Stitched fold test windows inside tier D.",
            "holdout": "2026-04-01 through 2026-08-31 is sealed and was not read.",
        },
    }


def _spec(
    cell: Dict[str, object], universe: Dict[str, object], periods: Dict[str, object]
) -> Dict[str, object]:
    spec = {
        "family": "relative_value",
        "strategy_name": "pair_zscore",
        "symbols": [cell["symbol_a"], cell["symbol_b"]],
        "universe": universe,
        "timeframe": "1h",
        "parameters": {
            "beta_mode": cell["beta_mode"],
            "window_h": cell["window_h"],
            "entry": cell["entry"],
            "exit_band": cell["exit_band"],
            "hold_mult": cell["hold_mult"],
            "max_hold_bars": cell["max_hold_bars"],
        },
        "feature_definitions": feature_definitions(cell),
        "entry_rules": (
            "On a completed 1h close, z > entry shorts A and longs B; "
            "z < -entry longs A and shorts B. |z| equal to entry does not enter."
        ),
        "exit_rules": (
            "exit_band 0: z crosses back through 0 from the entry side. "
            "exit_band 0.5: abs(z) <= 0.5. Otherwise exit at max_hold_bars. "
            "The next entry is the following bar."
        ),
        "cost_model": cost_model_record(),
        "training_period": periods["training_period"],
        "validation_period": periods["validation_period"],
        "test_period": periods["test_period"],
        "universe_bias_note": UNIVERSE_BIAS_NOTE,
        "search_space": search_space(),
    }
    if cell.get("filter") is not None:
        spec["cointegration_filter"] = cell["filter"]
    return spec


def _jsonable_metrics(stats: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in stats.items():
        if key == "daily_net":
            continue
        if isinstance(value, list):
            out[key] = [str(item) for item in value]
        else:
            out[key] = value
    return out


def run_phase1(
    *,
    n_names: int = 12,
    n_samples: int = 1000,
    seed: int = 7,
    names: List[str] | None = None,
    pairs: List[Pair] | None = None,
    ledger_prefix: str = "mn_rv_phase1",
    jsonl_name: str = "experiments.jsonl",
    summary_stem: str = "relative_value_phase1",
    phase_label: str = "1",
    description: str | None = None,
    extra_note: str | None = None,
    trade_gate=None,
    cell_note=None,
) -> Dict[str, object]:
    ledger = open_ledger(
        name_prefix=ledger_prefix,
        jsonl_path=output_dir() / jsonl_name,
        description=description,
    )
    if ledger.n_trials:
        raise RuntimeError(
            f"{ledger.name_prefix} already has {ledger.n_trials} trials in {ledger.db_path}. "
            "Refusing to append a second pass onto the deflated-Sharpe count."
        )

    if names is None:
        names = select_names(n_names)
    if pairs is None:
        pairs = select_pairs(names)
    cells = as_records(pairs)
    expected = matrix_size(len(pairs))
    if len(cells) != expected:
        raise RuntimeError(f"matrix length {len(cells)} != {expected}")
    log.info(
        f"relative value phase {phase_label}: {len(names)} names, {len(pairs)} pairs, "
        f"{len(cells)} cells. Holdout is not read."
    )
    log.info("names: " + ", ".join(names))

    price_names = list(dict.fromkeys([*names, *[leg for pair in pairs for leg in pair]]))
    closes = load_closes(price_names)
    ranks = rank_lookup(price_names)
    panels = {(left, right): align_closes(closes[left], closes[right]) for left, right in pairs}
    z_cache: Dict[tuple, tuple] = {}
    for (left, right), panel in panels.items():
        times = panel["open_time"].to_numpy(dtype=np.int64)
        ca = panel["close_a"].to_numpy(dtype=float)
        cb = panel["close_b"].to_numpy(dtype=float)
        for mode in BETA_MODES:
            for window in WINDOWS_H:
                z_cache[(left, right, mode, window)] = pair_zscore(times, ca, cb, int(window), mode)

    test_start, test_end = test_window()
    days = daily_calendar(test_start, test_end)
    btc_daily = daily_close_returns(closes["BTCUSDT"])
    market_daily = basket_returns({name: closes[name] for name in names})
    cost_model = CostModel.binance_spot_taker()
    universe = describe_universe(names, pairs)
    if extra_note:
        universe["tier_note"] = extra_note
    periods = _periods()

    daily = np.zeros((days.size, len(cells)), dtype=float)
    scored: List[tuple] = []
    for i, cell in enumerate(cells):
        key = (cell["symbol_a"], cell["symbol_b"], cell["beta_mode"], int(cell["window_h"]))
        panel = panels[(cell["symbol_a"], cell["symbol_b"])]
        z, beta = z_cache[key]
        trades = simulate_pair(
            panel["open_time"].to_numpy(dtype=np.int64),
            panel["close_a"].to_numpy(dtype=float),
            panel["close_b"].to_numpy(dtype=float),
            z,
            beta,
            entry=float(cell["entry"]),
            exit_band=float(cell["exit_band"]),
            max_hold_bars=int(cell["max_hold_bars"]),
            symbol=str(cell["symbol"]),
            beta_mode=str(cell["beta_mode"]),
            window=int(cell["window_h"]),
        )
        if trade_gate is not None:
            trades = trade_gate(cell, trades)
        if cell_note is not None:
            cell = {**cell, "filter": cell_note(cell)}
        trades = attach_costs(trades, str(cell["symbol_a"]), str(cell["symbol_b"]), ranks)
        priced = apply_costs(trades, cost_model)
        oos = oos_mask(priced, test_start, test_end)
        stats = score_configuration(oos, days, btc_daily, market_daily, n_samples=n_samples, seed=seed)
        daily[:, i] = np.asarray(stats.pop("daily_net"), dtype=float)
        scored.append((cell, stats))
        if i % 50 == 0:
            log.info(f"scored {i + 1}/{len(cells)}")

    finite = [
        float(stats["sharpe_daily_sr"]) for _, stats in scored if stats.get("sharpe_daily_sr") is not None
    ]
    sr_variance = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
    n_trials = len(scored)
    pbo = pbo_cscv(daily, n_splits=16)
    pbo_value = pbo.get("pbo")

    rows: List[Dict[str, object]] = []
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
        stats["pbo"] = pbo_value
        stats["pbo_n_splits"] = pbo.get("n_splits")
        stats["pbo_n_combinations"] = pbo.get("n_combinations")
        stats["pbo_median_logit"] = pbo.get("median_logit")
        stats["n_trials"] = n_trials
        verdict = judge(stats)
        stats["status"] = verdict["status"]
        stats["reasons"] = verdict["reasons"]
        stats["sharpe_ci_width"] = verdict["sharpe_ci_width"]
        metrics = _jsonable_metrics(stats)
        experiment_id = log_experiment(ledger, _spec(cell, universe, periods), metrics, f"{i:04d}")
        rows.append(
            {
                "experiment_id": experiment_id,
                "symbol": cell["symbol"],
                "beta_mode": cell["beta_mode"],
                "window_h": cell["window_h"],
                "entry": cell["entry"],
                "exit_band": cell["exit_band"],
                "hold_mult": cell["hold_mult"],
                **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                "reasons": " | ".join(verdict["reasons"]),
            }
        )

    frame = pd.DataFrame(rows)
    folder = output_dir()
    surface_path = folder / f"{summary_stem}.parquet"
    summary_path = folder / f"{summary_stem}.json"
    frame.to_parquet(surface_path, index=False)
    counts = frame["status"].value_counts().to_dict() if len(frame) else {}
    passes = frame.loc[frame["status"] == "pass", "experiment_id"].astype(int).tolist() if len(frame) else []
    summary = {
        "family": "relative_value",
        "phase": phase_label,
        "n_cells": n_trials,
        "n_names": len(names),
        "n_pairs": len(pairs),
        "names": names,
        "pairs": [list(pair) for pair in pairs],
        "universe_bias_note": UNIVERSE_BIAS_NOTE,
        "search_space": search_space(),
        "holdout_touched": False,
        "holdout": "2026-04-01 through 2026-08-31 sealed",
        "pbo": pbo_value,
        "pbo_n_splits": pbo.get("n_splits"),
        "pbo_n_combinations": pbo.get("n_combinations"),
        "pbo_median_logit": pbo.get("median_logit"),
        "sr_variance": sr_variance,
        "n_trials": n_trials,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "pass_experiment_ids": passes,
        "surface": paths.rel(surface_path),
        "note": extra_note
        or (
            "A failing or inconclusive matrix is a valid phase-1 result. "
            "No configuration was selected before this log was written."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log.info(f"status counts: {summary['counts']}")
    log.info(f"PBO={pbo_value} DSR trial count={n_trials}")
    log.info(f"wrote {summary_path}")
    return summary


def run_cointegration(*, n_samples: int = 1000, seed: int = 7) -> Dict[str, object]:
    """Phase-1 pairs and cells, with a walk-forward ADF gate on each fold.

    The 1,728 cell definitions are unchanged. The filter only drops test-window
    trades whose training fold did not reject a unit root at p < 0.05.
    """
    from experiments.market_neutral.cointegration import (
        filter_trades,
        fold_results,
        pass_fraction,
        passing_windows,
    )
    from experiments.market_neutral.data.bars import align_closes, load_closes

    names = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "BNBUSDT",
        "PEPEUSDT",
        "SUIUSDT",
        "TRXUSDT",
        "ADAUSDT",
        "ENAUSDT",
        "LINKUSDT",
    ]
    pairs = [
        ("ETHUSDT", "BTCUSDT"),
        ("SOLUSDT", "BTCUSDT"),
        ("XRPUSDT", "BTCUSDT"),
        ("DOGEUSDT", "BTCUSDT"),
        ("BNBUSDT", "BTCUSDT"),
        ("PEPEUSDT", "BTCUSDT"),
        ("SUIUSDT", "BTCUSDT"),
        ("TRXUSDT", "BTCUSDT"),
        ("ADAUSDT", "BTCUSDT"),
        ("ENAUSDT", "BTCUSDT"),
        ("LINKUSDT", "BTCUSDT"),
        ("SOLUSDT", "ETHUSDT"),
    ]
    closes = load_closes(names)
    tables: Dict[tuple, list] = {}
    fractions = []
    for left, right in pairs:
        panel = align_closes(closes[left], closes[right])
        times = panel["open_time"].to_numpy(dtype=np.int64)
        ca = panel["close_a"].to_numpy(dtype=float)
        cb = panel["close_b"].to_numpy(dtype=float)
        for mode in ("fixed", "rolling_ols"):
            rows = fold_results(times, ca, cb, mode)
            tables[(left, right, mode)] = rows
            summary = pass_fraction(rows)
            fractions.append({"symbol_a": left, "symbol_b": right, "beta_mode": mode, **summary})
            log.info(f"{left}|{right} {mode} folds {summary['folds_passed']}/{summary['folds_total']}")

    def trade_gate(cell, trades):
        windows = passing_windows(tables[(cell["symbol_a"], cell["symbol_b"], cell["beta_mode"])])
        return filter_trades(trades, windows)

    def cell_note(cell):
        rows = tables[(cell["symbol_a"], cell["symbol_b"], cell["beta_mode"])]
        summary = pass_fraction(rows)
        summary["procedure"] = rows[0]["procedure"]
        return summary

    note = (
        "Walk-forward ADF filter on the phase-1 12 pairs. Fixed beta is stationarity "
        "of log(A)-log(B), not Engle-Granger. Rolling OLS is Engle-Granger two-step "
        "with the ADF p-value on the residual. Alpha = 0.05, not swept. "
        "Holdout was not read."
    )
    result = run_phase1(
        names=names,
        pairs=pairs,
        n_samples=n_samples,
        seed=seed,
        ledger_prefix="rv_cointegration_filter_v1",
        jsonl_name="cointegration_experiments.jsonl",
        summary_stem="relative_value_cointegration",
        phase_label="cointegration_filter_v1",
        description="Phase-1 relative value with a walk-forward ADF eligibility filter. Holdout sealed.",
        extra_note=note,
        trade_gate=trade_gate,
        cell_note=cell_note,
    )
    result["fold_pass_fractions"] = fractions
    summary_path = output_dir() / "relative_value_cointegration.json"
    current = json.loads(summary_path.read_text(encoding="utf-8"))
    current["fold_pass_fractions"] = fractions
    summary_path.write_text(json.dumps(current, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return result


def main() -> None:
    run_phase1()


def run_tier2(*, n_samples: int = 1000, seed: int = 7) -> Dict[str, object]:
    """Tier-2 pairs versus BTC. Holdout is not read."""
    from experiments.market_neutral.data.tier2 import (
        TIER2_NOTE,
        ensure_spot_15m,
        tier2_names,
        tier2_pairs,
    )

    selected = tier2_names()
    ingestion = ensure_spot_15m(selected)
    usable = [name for name in selected if not ingestion.get(name, {}).get("excluded")]
    pairs = tier2_pairs(usable)
    log.info(
        f"tier 2 selected {len(selected)}, priced {len(usable)}, "
        f"cells {len(pairs) * 144}. Holdout is not read."
    )
    log.info("tier 2 names: " + ", ".join(usable))
    if len(pairs) * 144 > 15 * 144:
        raise RuntimeError("tier-2 grid is larger than 15 pairs times 144 cells")
    return run_phase1(
        names=usable,
        pairs=pairs,
        n_samples=n_samples,
        seed=seed,
        ledger_prefix="mn_rv_tier2",
        jsonl_name="tier2_experiments.jsonl",
        summary_stem="relative_value_tier2",
        phase_label="tier2",
        description="Tier-2 relative value versus BTC. Holdout 2026-04-01..2026-08-31 sealed.",
        extra_note=TIER2_NOTE + " Ingestion: " + json.dumps(ingestion, default=str),
    )


if __name__ == "__main__":
    main()
