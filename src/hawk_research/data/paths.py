"""Canonical locations for research inputs and artifacts.

All data lives under one data directory, ``./data`` relative to the working
directory by default. Set ``HAWK_RESEARCH_DATA`` (or pass ``--data-dir`` to the
CLI, which sets it) to use another location. The variable is read once, at
import, so the CLI sets it before importing any stage.

Configuration files are read from ``./config`` (``HAWK_RESEARCH_CONFIG`` to
override). Run commands from the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "HAWK_RESEARCH_DATA"
CONFIG_ENV = "HAWK_RESEARCH_CONFIG"

RESEARCH_DIR = Path(os.environ.get(DATA_ENV) or Path.cwd() / "data").resolve()
CONFIG_DIR = Path(os.environ.get(CONFIG_ENV) or Path.cwd() / "config").resolve()

SNAPSHOT_DIR = RESEARCH_DIR / "snapshot"
OHLCV_DIR = RESEARCH_DIR / "ohlcv"
ZONES_DIR = RESEARCH_DIR / "zones"
FEATURES_DIR = RESEARCH_DIR / "features"
LABELS_DIR = RESEARCH_DIR / "labels"
UNIVERSE_DIR = RESEARCH_DIR / "universe"
REPORTS_DIR = RESEARCH_DIR / "reports"
DAILY_FADE_DIR = RESEARCH_DIR / "daily_fade"
MARKET_NEUTRAL_DIR = RESEARCH_DIR / "market_neutral"

# Optional copy of the live bot's database. Only the owner has it. Stages that
# compare the replay against the live book skip that comparison when it is absent.
LIVE_SNAPSHOT_DB = SNAPSHOT_DIR / "db.sqlite"

# Trial ledger. Every evaluated configuration is recorded here; the deflated
# Sharpe reads its trial count back from this file.
RESEARCH_DB = RESEARCH_DIR / "ledger.sqlite"

DATA_MANIFEST = RESEARCH_DIR / "ohlcv" / "coverage_manifest.json"

# Data window. Every stage clamps to SNAPSHOT_END_MS so re-running on a later
# day cannot silently change an already-published result.
SNAPSHOT_END_MS = 1789654500000  # 2026-09-17T14:15:00Z (open time of last bar)
HISTORY_START_MS = 1725148800000  # 2024-09-01T00:00:00Z


def ohlcv_path(symbol: str, interval: str) -> Path:
    return OHLCV_DIR / interval / f"{symbol}.parquet"


# Extract of the live system's ``zones`` table (1,543 zones, 2026-09-07 to 09-17), committed
# with the repository so the fidelity checks run for everyone. Market data only.
LIVE_ZONES_EXTRACT = CONFIG_DIR.parent / "snapshot" / "live_zones.parquet"


def has_live_snapshot() -> bool:
    """True if the live zones are available: the owner's database or the committed extract."""
    return LIVE_SNAPSHOT_DB.exists() or LIVE_ZONES_EXTRACT.exists()


def ensure_dirs() -> None:
    for d in (
        RESEARCH_DIR,
        OHLCV_DIR,
        ZONES_DIR,
        FEATURES_DIR,
        LABELS_DIR,
        UNIVERSE_DIR,
        REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    """``path`` relative to the data directory (POSIX form), for reports.

    Reports must not embed the absolute layout of whoever ran them.
    """
    p = Path(path).resolve()
    for root, prefix in ((RESEARCH_DIR.resolve(), ""), (CONFIG_DIR.resolve(), "config/")):
        try:
            return prefix + p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.name
