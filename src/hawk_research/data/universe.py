"""Explicit symbol universes from ``config/universe.json``.

The original code took "every parquet file in the data directory" as its
universe, so the same stage gave different results depending on what had been
downloaded since. Each stage now names the universe it uses.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

from hawk_research.data import paths


@lru_cache(maxsize=1)
def _load() -> Dict[str, dict]:
    path = paths.CONFIG_DIR / "universe.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run from the repository root or set HAWK_RESEARCH_CONFIG")
    return json.loads(path.read_text(encoding="utf-8"))


def names() -> List[str]:
    return [k for k, v in _load().items() if isinstance(v, dict) and "symbols" in v]


def symbols(universe: str) -> List[str]:
    """Sorted symbol list of a named universe (``round1``, ``daily_fade``, ...)."""
    data = _load()
    if universe not in data or not isinstance(data[universe], dict):
        raise KeyError(f"unknown universe {universe!r}; known: {names()}")
    return sorted(data[universe]["symbols"])


def available(universe: str, interval: str = "15m") -> List[str]:
    """Universe members whose OHLCV file exists, so a partial download still runs."""
    return [s for s in symbols(universe) if paths.ohlcv_path(s, interval).exists()]
