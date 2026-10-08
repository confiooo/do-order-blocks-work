"""Typed research configuration.

Defaults equal the values that produced the published results. ``config/research.yaml``
restates them so they are visible without reading code; ``--config other.yaml`` and
``--set section.key=value`` override them for a run, and every run writes the resolved
configuration next to its outputs.

Modules read the active configuration through :func:`get`. The CLI calls
:func:`activate` before importing any stage, so module-level constants derived from
the configuration reflect the overrides for that process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ReplayConfig:
    """Round-1 replay geometry, inherited from the live collector."""

    live_window: int = 400
    lookback_ratio: float = 0.15
    warmup_bars: int = 400
    btc_range_bars: int = 16


@dataclass(frozen=True)
class DetectorConfig:
    """Round-1 detector thresholds.

    The live bot's settings: loosened by hand on 2026-09-06 from 1.824 / 1.824 / 1.051,
    which an automatic tuner had produced from the code defaults 1.5 / 1.5 / 1.0. Not
    fitted to the study's history; see docs/PROVENANCE.md.
    """

    strength_threshold: float = 1.35
    wick_to_body_ratio_min: float = 1.2
    min_distance_from_high_low: float = 0.5


@dataclass(frozen=True)
class LiveGateConfig:
    """Live alert gates, reproduced as flags on replayed zones (not filters)."""

    btc_volatility_threshold: float = 0.045
    min_volume_ratio: float = 1.8
    max_distance_to_price: float = 0.45
    min_signal_interval_minutes: int = 15


@dataclass(frozen=True)
class SmcConfig:
    """Textbook order-block detector (round 2). Fixed rules, never tuned."""

    fractal_bars: int = 3
    max_ob_age_bars: int = 32
    max_ob_search_bars: int = 10
    volume_lookback: int = 60


@dataclass(frozen=True)
class LabelConfig:
    return_horizons: Dict[str, int] = field(
        default_factory=lambda: {"15m": 1, "1h": 4, "4h": 16, "8h": 32, "24h": 96, "48h": 192}
    )
    barrier_levels: Tuple[float, ...] = (0.005, 0.01, 0.015, 0.02, 0.03)
    barrier_horizons: Dict[str, int] = field(
        default_factory=lambda: {"1h": 4, "4h": 16, "8h": 32, "24h": 96, "48h": 192}
    )
    forward_vol_horizons: Dict[str, int] = field(default_factory=lambda: {"4h": 16, "24h": 96, "48h": 192})
    retest_timeout_hours: int = 24
    mfe_mae_horizons_hours: Tuple[int, ...] = (1, 4, 8, 24)


@dataclass(frozen=True)
class EvaluationConfig:
    """Walk-forward harness and holdout settings shared by s7-s12.

    Strategy grids (take-profit/stop-loss pairs, horizons, entry rules) are part of
    each stage's pre-specified scan and stay in the stage code; changing them changes
    the number of trials, which the ledger records.
    """

    holdout_fraction: float = 0.2
    bootstrap_samples: int = 600  # s8, s10, s12; s7, s9, s11 default to 400 (stage flags)
    max_concurrent: int = 5
    train_days: int = 90
    test_days: int = 30
    step_days: int = 30
    embargo_hours: int = 24
    max_outcome_hours: int = 48
    bootstrap_block_hours: int = 24


@dataclass(frozen=True)
class SeedConfig:
    """Default seeds. Each stage also accepts ``--seed`` after ``--``."""

    harness: int = 7  # walk-forward bootstrap inside the evaluation harness
    matched_controls: int = 3  # s8
    strategy_bootstrap: int = 11  # cross-sectional CIs in s10, s12
    leakage_audit: int = 11  # s6 (label truncation uses seed + 1)
    harness_check: int = 17  # s7
    location_control: int = 3  # s9


@dataclass(frozen=True)
class ResearchConfig:
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    live_gates: LiveGateConfig = field(default_factory=LiveGateConfig)
    smc: SmcConfig = field(default_factory=SmcConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce(section_cls: type, values: Dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(section_cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown keys in [{section_cls.__name__}]: {sorted(unknown)}")
    default = section_cls()
    out = {}
    for name, value in values.items():
        current = getattr(default, name)
        if isinstance(current, tuple):
            value = tuple(value)
        elif isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int) and not isinstance(value, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        out[name] = value
    return dataclasses.replace(default, **out)


def from_dict(data: Optional[Dict[str, Any]]) -> ResearchConfig:
    data = dict(data or {})
    sections = {f.name: f for f in dataclasses.fields(ResearchConfig)}
    unknown = set(data) - set(sections)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    built = {}
    for name in sections:
        cls = type(getattr(ResearchConfig(), name))
        built[name] = _coerce(cls, data.get(name) or {})
    return ResearchConfig(**built)


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """``["evaluation.holdout_fraction=0.25"]`` -> nested dict. Values parse as YAML."""
    out: Dict[str, Dict[str, Any]] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or "." not in key:
            raise ValueError(f"override must look like section.key=value, got {pair!r}")
        section, name = key.split(".", 1)
        out.setdefault(section, {})[name] = yaml.safe_load(raw)
    return out


def load(path: Optional[Path] = None, overrides: Optional[List[str]] = None) -> ResearchConfig:
    data: Dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping")
        data = loaded
    for section, values in parse_overrides(overrides or []).items():
        data.setdefault(section, {}).update(values)
    return from_dict(data)


_ENV = "HAWK_RESEARCH_RESOLVED_CONFIG"
_ACTIVE: ResearchConfig = (
    from_dict(json.loads(os.environ[_ENV])) if os.environ.get(_ENV) else ResearchConfig()
)


def activate(cfg: ResearchConfig) -> None:
    """Make ``cfg`` the active configuration for this process and its children.

    The resolved configuration is also exported in ``HAWK_RESEARCH_RESOLVED_CONFIG``
    so worker processes started with ``spawn`` see the same values.
    """
    global _ACTIVE
    _ACTIVE = cfg
    os.environ[_ENV] = json.dumps(cfg.to_dict(), sort_keys=True)


def get() -> ResearchConfig:
    return _ACTIVE
