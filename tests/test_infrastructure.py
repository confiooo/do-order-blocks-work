"""Configuration, provenance, ledger, universes and the stage registry."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hawk_research import config as research_config
from hawk_research import ledger as ledger_mod
from hawk_research import provenance
from hawk_research.data import universe
from hawk_research.stages import registry

REPO = Path(__file__).resolve().parents[1]

# The values that produced the published results. A change here must be deliberate:
# update this table, config/research.yaml and the docs together.
PUBLISHED_DEFAULTS = {
    "replay": {"live_window": 400, "lookback_ratio": 0.15, "warmup_bars": 400, "btc_range_bars": 16},
    "detector": {
        "strength_threshold": 1.35,
        "wick_to_body_ratio_min": 1.2,
        "min_distance_from_high_low": 0.5,
    },
    "live_gates": {
        "btc_volatility_threshold": 0.045,
        "min_volume_ratio": 1.8,
        "max_distance_to_price": 0.45,
        "min_signal_interval_minutes": 15,
    },
    "smc": {"fractal_bars": 3, "max_ob_age_bars": 32, "max_ob_search_bars": 10, "volume_lookback": 60},
    "evaluation": {
        "holdout_fraction": 0.2,
        "bootstrap_samples": 600,
        "max_concurrent": 5,
        "train_days": 90,
        "test_days": 30,
        "step_days": 30,
        "embargo_hours": 24,
        "max_outcome_hours": 48,
        "bootstrap_block_hours": 24,
    },
    "seeds": {
        "harness": 7,
        "matched_controls": 3,
        "strategy_bootstrap": 11,
        "leakage_audit": 11,
        "harness_check": 17,
        "location_control": 3,
    },
}


def test_config_defaults_equal_published_values():
    cfg = research_config.ResearchConfig().to_dict()
    for section, values in PUBLISHED_DEFAULTS.items():
        for key, value in values.items():
            assert cfg[section][key] == value, f"{section}.{key} drifted from the published value"
    assert cfg["labels"]["retest_timeout_hours"] == 24


def test_research_yaml_restates_the_defaults():
    """config/research.yaml must not silently differ from the code defaults."""
    loaded = research_config.load(REPO / "config" / "research.yaml")
    assert loaded == research_config.ResearchConfig()


def test_overrides_and_unknown_keys():
    cfg = research_config.load(
        overrides=["evaluation.holdout_fraction=0.25", "labels.barrier_levels=[0.01, 0.02]"]
    )
    assert cfg.evaluation.holdout_fraction == 0.25
    assert cfg.labels.barrier_levels == (0.01, 0.02)
    with pytest.raises(ValueError):
        research_config.load(overrides=["evaluation.nope=1"])
    with pytest.raises(ValueError):
        research_config.load(overrides=["no_dot=1"])


def test_config_file_round_trip(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump({"smc": {"fractal_bars": 5}}))
    cfg = research_config.load(path)
    assert cfg.smc.fractal_bars == 5
    assert cfg.hash() != research_config.ResearchConfig().hash()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_provenance_records_dirty_flag(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("x")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")
    assert provenance.git_dirty(tmp_path) is False
    sha = provenance.git_commit(tmp_path)
    assert len(sha) == 40
    (tmp_path / "a.txt").write_text("changed")
    assert provenance.git_dirty(tmp_path) is True
    monkeypatch.chdir(tmp_path)
    rec = provenance.record(seed=7)
    assert rec["git_commit"] == sha and rec["git_dirty"] is True and rec["seed"] == 7
    assert {"config_hash", "command", "packages", "config"} <= set(rec)
    with pytest.raises(SystemExit):
        provenance.require_clean()


def test_provenance_outside_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert provenance.git_commit() == "unknown"
    assert provenance.git_dirty() is None


def test_ledger_rows_carry_provenance(tmp_path):
    db = tmp_path / "ledger.sqlite"
    led = ledger_mod.TrialLedger("unit", db_path=db, jsonl_path=tmp_path / "l.jsonl")
    led.log_trial({"a": 1}, {"m": float("nan")})
    conn = ledger_mod.connect(db)
    row = dict(conn.execute("SELECT * FROM experiments").fetchone())
    conn.close()
    assert row["name"] == "unit/0000"
    assert "git_dirty" in row and row["config_hash"] == research_config.get().hash()
    assert json.loads(row["parameters_json"])["metrics"]["m"] is None  # NaN stored as null
    line = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert line["provenance"]["config_hash"] == row["config_hash"]
    assert ledger_mod.count_trials("unit", db_path=db) == 1
    assert ledger_mod.count_trials("other", db_path=db) == 0


def test_universes_are_explicit_and_consistent():
    r1 = universe.symbols("round1")
    zones = universe.symbols("round1_zone_symbols")
    assert len(r1) == 80 and len(zones) == 76 and set(zones) <= set(r1)
    assert set(r1) | set(universe.symbols("market_neutral_tier2_extra")) == set(
        universe.symbols("daily_fade")
    )
    with pytest.raises(KeyError):
        universe.symbols("nope")


def test_registry_resolves_dependencies_in_order():
    order = [s.name for s in registry.resolve(["s8"])]
    assert order == ["s3", "s4", "s5", "s6", "s8"]
    smc = [s.name for s in registry.resolve(["smc"])]
    assert smc.index("s3s") < smc.index("s5_smc") < smc.index("s6_smc") < smc.index("s12")
    everything = [s.name for s in registry.resolve(["all"])]
    assert not any(registry.BY_NAME[n].network for n in everything)
    assert len(everything) == len(set(everything))
    with pytest.raises(KeyError):
        registry.resolve(["s99"])


def test_every_stage_target_imports():
    for stage in registry.STAGES:
        assert callable(stage.load()), stage.name


def test_ledger_prior_trials_count_against_deflation(tmp_path):
    hist = tmp_path / "trial_history.json"
    hist.write_text(json.dumps({"ledgers": {"scan": {"prior_trials": 48}}}))
    assert ledger_mod.historical_prior("scan", hist) == 48
    assert ledger_mod.historical_prior("other", hist) == 0
    led = ledger_mod.TrialLedger(
        "scan", db_path=tmp_path / "l.sqlite", jsonl_path=tmp_path / "l.jsonl", prior_trials=48
    )
    for i in range(24):
        led.log_trial({"i": i})
    assert led.n_logged == 24 and led.n_trials == 72
    assert led.trials()[0]["name"] == "scan/0000"
    with pytest.raises(ValueError):
        ledger_mod.TrialLedger("scan", db_path=tmp_path / "l.sqlite", prior_trials=-1)


def test_published_s8_prior_reproduces_72_trials():
    """The published null_baselines.json deflated against 72 = 48 earlier + 24 trials."""
    assert ledger_mod.historical_prior("s8-null-baselines", REPO / "config" / "trial_history.json") == 48


def test_harness_defaults_come_from_the_research_config():
    from hawk_research.evaluation.harness import HarnessConfig

    h = HarnessConfig.from_research(bootstrap_samples=400)
    ev = research_config.get().evaluation
    assert (h.train_days, h.test_days, h.embargo_hours, h.holdout_fraction) == (
        ev.train_days,
        ev.test_days,
        ev.embargo_hours,
        ev.holdout_fraction,
    )
    assert h.bootstrap_samples == 400 and h.seed == research_config.get().seeds.harness
    # The published stages used the dataclass defaults; from_research must not move them.
    plain = HarnessConfig(bootstrap_samples=400, max_concurrent=5)
    assert h == plain


def test_activated_config_reaches_child_processes(monkeypatch):
    monkeypatch.delenv("HAWK_RESEARCH_RESOLVED_CONFIG", raising=False)
    previous = research_config.get()
    try:
        research_config.activate(research_config.load(overrides=["smc.fractal_bars=5"]))
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from hawk_research.detectors import smc_order_blocks as m; print(m.FRACTAL_BARS)",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == "5"
    finally:
        research_config.activate(previous)
        monkeypatch.delenv("HAWK_RESEARCH_RESOLVED_CONFIG", raising=False)


def test_mexc_api_scenario_costs_exactly_4bps_more_per_round_trip():
    from hawk_research.evaluation.costs import CostModel

    base, api = CostModel.mexc_futures_taker(), CostModel.mexc_futures_taker_api()
    assert base.taker_fee == 0.0004  # the value behind every published round-1/SMC net number
    assert abs((api.round_trip_fee() - base.round_trip_fee()) - 0.0004) < 1e-12
