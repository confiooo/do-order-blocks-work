"""Who, what and from which code: provenance attached to every result.

Every report JSON and every ledger row carries :func:`record`. It includes the git
commit *and* whether the working tree had uncommitted changes, because a commit
alone can point at code that is not what actually ran.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from hawk_research import config as research_config
from hawk_research.data import paths

PACKAGES = ("hawk-research", "numpy", "pandas", "scipy", "statsmodels", "pyarrow", "pyyaml")


def _git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_commit(cwd: Optional[Path] = None) -> str:
    """HEAD commit, or ``"unknown"`` outside a git checkout. Never invented."""
    return _git("rev-parse", "HEAD", cwd=cwd) or "unknown"


def git_dirty(cwd: Optional[Path] = None) -> Optional[bool]:
    """True if tracked or untracked files differ from HEAD; None outside git."""
    out = _git("status", "--porcelain", cwd=cwd)
    if out is None:
        return None
    return bool(out)


def package_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def data_manifest_hash() -> Optional[str]:
    """sha256 of the downloaded-data manifest, if present."""
    path = paths.DATA_MANIFEST
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hash(cfg: Optional[research_config.ResearchConfig] = None) -> str:
    return (cfg or research_config.get()).hash()


def record(seed: Optional[int] = None, include_config: bool = True) -> Dict[str, Any]:
    """Provenance block for a report or ledger row."""
    cfg = research_config.get()
    out: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_hash": cfg.hash(),
        "seed": seed,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else "",
        "data_manifest_sha256": data_manifest_hash(),
        "packages": package_versions(),
    }
    if include_config:
        out["config"] = cfg.to_dict()
    return out


def require_clean() -> None:
    """Raise if the working tree is dirty: a publishable run must map to one commit."""
    dirty = git_dirty()
    if dirty:
        raise SystemExit("working tree has uncommitted changes; commit them or drop --require-clean")
