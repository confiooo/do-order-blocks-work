"""Strategy-evaluation harness.

``evaluate`` turns a trade table into one nested result: costs applied, a
capacity-constrained equity curve, gross and net trade metrics, bootstrap
intervals that respect serial and cross-sectional dependence, and, given a regime
table, a per-regime breakdown. ``walk_forward_folds`` builds the purged,
embargoed folds.

Nothing here mutates its input. Splits are by decision time: a training trade whose
outcome window reaches into the test window is purged, and test trades inside the
embargo are dropped. Callers log every evaluated configuration to the trial ledger
(``hawk_research.ledger``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research import config as research_config
from hawk_research.constants import BAR_MS, MS_PER_DAY, MS_PER_HOUR
from hawk_research.evaluation import metrics as metrics_mod
from hawk_research.evaluation import portfolio as portfolio_mod
from hawk_research.evaluation import regimes as regimes_mod
from hawk_research.evaluation.costs import CostModel, apply_costs, cost_summary, validate_trades
from hawk_research.evaluation.splits import chronological_windows
from hawk_research.stats import resampling


@dataclass(frozen=True)
class HarnessConfig:
    """Evaluation settings. The defaults are the values every published stage used."""

    train_days: int = 90
    test_days: int = 30
    step_days: int = 30
    embargo_hours: int = 24
    max_outcome_hours: int = 48
    bootstrap_block_hours: int = 24
    bootstrap_samples: int = 1000
    max_concurrent: int = 5
    seed: int = 7
    holdout_fraction: float = 0.2
    capital: float = 1.0
    bar_ms: int = BAR_MS
    one_position_per_symbol: bool = True

    @classmethod
    def from_research(cls, **overrides: Any) -> "HarnessConfig":
        """Harness settings from the active research configuration, plus overrides."""
        cfg = research_config.get()
        ev = cfg.evaluation
        base: Dict[str, Any] = dict(
            train_days=ev.train_days,
            test_days=ev.test_days,
            step_days=ev.step_days,
            embargo_hours=ev.embargo_hours,
            max_outcome_hours=ev.max_outcome_hours,
            bootstrap_block_hours=ev.bootstrap_block_hours,
            bootstrap_samples=ev.bootstrap_samples,
            max_concurrent=ev.max_concurrent,
            seed=cfg.seeds.harness,
            holdout_fraction=ev.holdout_fraction,
        )
        base.update(overrides)
        return cls(**base)

    @property
    def max_outcome_ms(self) -> int:
        return int(self.max_outcome_hours) * MS_PER_HOUR

    @property
    def risk_fraction(self) -> float:
        return 1.0 / float(self.max_concurrent)


def _time_bounds(trades: pd.DataFrame, time_col: str = "decision_time_ms") -> tuple[int, int]:
    times = trades[time_col].to_numpy(dtype=np.int64)
    return int(times.min()), int(times.max())


def purged_split(
    trades: pd.DataFrame,
    train_start_ms: int,
    train_end_ms: int,
    test_start_ms: int,
    test_end_ms: int,
    embargo_ms: int,
    max_outcome_ms: int,
) -> Dict[str, Any]:
    """Purged, embargoed chronological split of the trade table.

    Membership is by ``decision_time_ms``, the instant the signal became knowable.

    Purging: a training trade is dropped when its outcome window reaches into the
    test period. The window ends at ``max(exit_time_ms, decision_time_ms +
    max_outcome_ms)``, so a position still open when the test period starts cannot
    contribute a label that overlaps test-period price action.

    Embargo: test trades whose decision falls within ``embargo_ms`` after
    ``train_end_ms`` are dropped, since prices immediately after the training cut are
    still serially dependent on it.

    Returns ``{'train','test','purged','embargoed','n_train','n_test','n_purged',
    'n_embargoed'}`` with DataFrame partitions.
    """
    validate_trades(trades)
    frame = trades.copy()
    if len(frame) == 0:
        empty = frame
        return {
            "train": empty,
            "test": empty,
            "purged": empty,
            "embargoed": empty,
            "n_train": 0,
            "n_test": 0,
            "n_purged": 0,
            "n_embargoed": 0,
        }

    decision = frame["decision_time_ms"].to_numpy(dtype=np.int64)
    exit_ms = frame["exit_time_ms"].to_numpy(dtype=np.int64)
    outcome_end = np.maximum(exit_ms, decision + int(max_outcome_ms))

    in_train = (decision >= int(train_start_ms)) & (decision < int(train_end_ms))
    in_test = (decision >= int(test_start_ms)) & (decision < int(test_end_ms))
    overlaps = outcome_end >= int(test_start_ms)
    embargo_cut = int(train_end_ms) + int(embargo_ms)
    embargoed = in_test & (decision < embargo_cut)

    train_mask = in_train & ~overlaps
    purged_mask = in_train & overlaps
    test_mask = in_test & ~embargoed

    return {
        "train": frame.loc[train_mask].copy(),
        "test": frame.loc[test_mask].copy(),
        "purged": frame.loc[purged_mask].copy(),
        "embargoed": frame.loc[embargoed].copy(),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_purged": int(purged_mask.sum()),
        "n_embargoed": int(embargoed.sum()),
        "train_start_ms": int(train_start_ms),
        "train_end_ms": int(train_end_ms),
        "test_start_ms": int(test_start_ms),
        "test_end_ms": int(test_end_ms),
        "embargo_ms": int(embargo_ms),
        "max_outcome_ms": int(max_outcome_ms),
    }


def walk_forward_folds(trades: pd.DataFrame, cfg: HarnessConfig) -> List[Dict[str, Any]]:
    """Purged walk-forward folds over the trade table.

    Window geometry comes from ``splits.chronological_windows`` so the fold
    boundaries are identical to the already-published zone-level validation, and each
    fold is then materialized with ``purged_split``.
    """
    validate_trades(trades)
    if len(trades) == 0:
        return []
    start_ms, end_ms = _time_bounds(trades)
    windows = chronological_windows(
        start_ms,
        end_ms,
        cfg.train_days,
        cfg.test_days,
        cfg.step_days,
        cfg.embargo_hours,
    )
    folds: List[Dict[str, Any]] = []
    for window in windows:
        split = purged_split(
            trades,
            window["train_start_ms"],
            window["train_end_ms"],
            window["test_start_ms"],
            window["test_end_ms"],
            window["embargo_ms"],
            cfg.max_outcome_ms,
        )
        folds.append({**window, **split})
    return folds


def _sharpe_per_trade(values: np.ndarray) -> Optional[float]:
    """Per-trade Sharpe (not annualized). This is the unit the DSR machinery expects."""
    if values.size < 2:
        return None
    sd = float(values.std(ddof=1))
    if sd <= 0.0:
        return None
    return float(values.mean() / sd)


def evaluate(
    trades: pd.DataFrame,
    cost_model: CostModel,
    cfg: HarnessConfig,
    illiquidity: Optional[pd.Series] = None,
    regime_table: Optional[pd.DataFrame] = None,
    label: str = "",
) -> Dict[str, Any]:
    """Costs, the net equity curve under the concurrency cap, trade metrics and intervals.

    The interval to quote is the cross-sectional one (see ``stats.resampling``).
    ``equity_metrics`` is on the 15m grid (use its drawdown); ``equity_metrics_daily``
    is a daily resample (use its Sharpe: annualising a mostly flat 15m curve means nothing).
    """
    validate_trades(trades)
    priced = apply_costs(trades, cost_model, illiquidity=illiquidity)
    if regime_table is not None and len(priced):
        priced = regimes_mod.assign_regimes(priced, regime_table, bar_ms=cfg.bar_ms)

    result: Dict[str, Any] = {
        "label": label,
        "n_trades": int(len(priced)),
        "config": asdict(cfg),
        "cost_model": cost_model.as_dict(),
        "costs": cost_summary(priced),
        "trades": priced,
    }
    if len(priced) == 0:
        result["note"] = "empty trade table; nothing evaluated"
        return result

    result["period"] = {
        "start_ms": int(priced["decision_time_ms"].min()),
        "end_ms": int(priced["exit_time_ms"].max()),
        "n_symbols": int(priced["symbol"].nunique()),
    }

    sim = portfolio_mod.build_equity_curve(
        priced,
        ret_col="ret_net",
        max_concurrent=cfg.max_concurrent,
        capital=cfg.capital,
        risk_fraction=cfg.risk_fraction,
        one_position_per_symbol=cfg.one_position_per_symbol,
        bar_ms=cfg.bar_ms,
    )
    net_values = priced["ret_net"].to_numpy(dtype=float)
    times = priced["entry_time_ms"].to_numpy(dtype=np.int64)
    net_stats = metrics_mod.trade_metrics(priced, ret_col="ret_net")
    net_stats["sharpe_per_trade"] = _sharpe_per_trade(net_values)
    block = resampling.block_bootstrap_ci(
        net_values,
        times,
        block_hours=cfg.bootstrap_block_hours,
        n_samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )
    cross = resampling.cross_sectional_bootstrap_ci(
        net_values,
        times,
        block_hours=cfg.bootstrap_block_hours,
        n_samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )
    result["portfolio"] = {"net": sim}
    result["equity_metrics"] = {"net": metrics_mod.equity_metrics(sim["equity"])}
    result["equity_metrics_daily"] = {
        "net": metrics_mod.equity_metrics(metrics_mod.resample_equity(sim["equity"], MS_PER_DAY))
    }
    result["trade_metrics"] = {
        "gross": metrics_mod.trade_metrics(priced, ret_col="ret_gross"),
        "net": net_stats,
    }
    result["bootstrap"] = {"net": {"block": block, "cross_sectional": cross}}
    result["effective_sample_size"] = resampling.effective_sample_size(times, cfg.bootstrap_block_hours)
    result["significant_net"] = resampling.ci_excludes_zero(cross)

    if regime_table is not None:
        result["regime_breakdown"] = regimes_mod.regime_breakdown(
            priced,
            ret_col="ret_net",
            block_hours=cfg.bootstrap_block_hours,
            n_samples=min(cfg.bootstrap_samples, 400),
            seed=cfg.seed,
        )
    return result


def flatten_evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    """One flat row of headline numbers from an ``evaluate`` result."""
    net = result.get("trade_metrics", {}).get("net", {})
    gross = result.get("trade_metrics", {}).get("gross", {})
    equity = result.get("equity_metrics", {}).get("net", {})
    equity_daily = result.get("equity_metrics_daily", {}).get("net", {})
    sim = result.get("portfolio", {}).get("net", {})
    cross = result.get("bootstrap", {}).get("net", {}).get("cross_sectional", {})
    block = result.get("bootstrap", {}).get("net", {}).get("block", {})
    return {
        "label": result.get("label", ""),
        "n_trades": result.get("n_trades", 0),
        "effective_sample_size": result.get("effective_sample_size"),
        "mean_ret_gross": gross.get("expectancy"),
        "mean_ret_net": net.get("expectancy"),
        "win_rate_net": net.get("win_rate"),
        "profit_factor_net": net.get("profit_factor"),
        "payoff_ratio_net": net.get("payoff_ratio"),
        "sharpe_per_trade_net": net.get("sharpe_per_trade"),
        "trades_per_day": net.get("trades_per_day"),
        "ci_low_block": block.get("ci_low"),
        "ci_high_block": block.get("ci_high"),
        "ci_low_cross": cross.get("ci_low"),
        "ci_high_cross": cross.get("ci_high"),
        "significant_net": result.get("significant_net"),
        "cagr_net": equity.get("cagr"),
        "sharpe_daily_net": equity_daily.get("sharpe"),
        "sortino_daily_net": equity_daily.get("sortino"),
        "max_drawdown_net": equity.get("max_drawdown"),
        "calmar_net": equity.get("calmar"),
        "accepted": sim.get("accepted"),
        "rejection_rate": sim.get("rejection_rate"),
        "capacity_utilization": sim.get("capacity_utilization"),
        "mean_total_cost": result.get("costs", {}).get("mean_total_cost"),
    }
