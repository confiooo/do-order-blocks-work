"""Experiment records for this node. The survivor-bias note is mandatory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from experiments.market_neutral.calendar import DEV_END, DEV_START, UNIVERSE_BIAS_NOTE
from hawk_research.data import paths
from hawk_research.ledger import TrialLedger

REQUIRED_FIELDS = (
    "family",
    "strategy_name",
    "symbols",
    "timeframe",
    "parameters",
    "feature_definitions",
    "entry_rules",
    "exit_rules",
    "cost_model",
    "training_period",
    "validation_period",
    "test_period",
    "universe_bias_note",
)

LEDGER_PREFIX = "mn_rv_phase1"


def output_dir() -> Path:
    path = paths.MARKET_NEUTRAL_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_ledger(
    *,
    db_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
    name_prefix: str = LEDGER_PREFIX,
    description: Optional[str] = None,
) -> TrialLedger:
    folder = output_dir() if db_path is None else Path(db_path).parent
    folder.mkdir(parents=True, exist_ok=True)
    if description is None:
        description = "Market-neutral relative-value phase 1. Holdout 2026-04-01..2026-08-31 sealed."
    return TrialLedger(
        name_prefix=name_prefix,
        description=description,
        dataset_start=DEV_START,
        dataset_end=DEV_END,
        db_path=db_path if db_path is not None else paths.RESEARCH_DB,
        jsonl_path=jsonl_path if jsonl_path is not None else folder / "experiments.jsonl",
    )


def validate_record(spec: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_FIELDS if key not in spec or spec[key] in (None, "")]
    if missing:
        raise ValueError(f"experiment spec missing required fields: {missing}")
    if spec["universe_bias_note"] != UNIVERSE_BIAS_NOTE:
        raise ValueError("universe_bias_note must be the canonical survivor-bias statement")


def log_experiment(
    ledger: TrialLedger,
    spec: Dict[str, Any],
    metrics: Dict[str, Any],
    label: str,
) -> int:
    validate_record(spec)
    return ledger.log_trial(parameters=spec, metrics=metrics, label=label)
