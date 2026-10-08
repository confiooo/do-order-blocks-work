"""Provenance, configuration hashing and chronological splits."""

from __future__ import annotations

from hawk_research import config as research_config
from hawk_research.evaluation.splits import chronological_windows
from hawk_research.provenance import config_hash, git_commit


def test_config_hash_deterministic():
    a = config_hash()
    b = config_hash()
    assert a == b
    assert len(a) == 64
    changed = research_config.load(overrides=["evaluation.holdout_fraction=0.25"])
    assert changed.hash() != a


def test_git_commit_never_invented():
    sha = git_commit()
    assert sha == "unknown" or (len(sha) >= 7 and all(c in "0123456789abcdef" for c in sha.lower()[:7]))


def test_chronological_split_and_embargo():
    hour = 3600 * 1000
    day = 24 * hour
    windows = chronological_windows(
        start_ms=0,
        end_ms=200 * day,
        train_days=90,
        test_days=30,
        step_days=30,
        embargo_hours=24,
    )
    assert windows
    assert windows[0]["train_start_ms"] < windows[0]["test_start_ms"]
    assert windows[0]["test_start_ms"] - windows[0]["train_end_ms"] == 24 * hour
    for i in range(1, len(windows)):
        assert windows[i]["train_start_ms"] > windows[i - 1]["train_start_ms"]
