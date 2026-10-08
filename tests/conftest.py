"""Test session setup.

Every test session gets its own empty data directory, so tests can never read or
write a real ledger or dataset. The environment variables must be set before any
``hawk_research`` module is imported, which is why this happens at import time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_DATA = Path(tempfile.mkdtemp(prefix="hawk-research-tests-"))
os.environ["HAWK_RESEARCH_DATA"] = str(_DATA)
os.environ["HAWK_RESEARCH_CONFIG"] = str(Path(__file__).resolve().parents[1] / "config")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
