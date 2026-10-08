"""One-touch seal for the market-neutral holdout.

The protocol describes this gate; the order-block study does not implement it.
The first successful unseal for a family appends one line to the audit log and
returns the loader's data. A second unseal for that family raises and
does not call the loader.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from experiments.market_neutral.calendar import HOLDOUT_END, holdout_bounds
from hawk_research.data import paths
from hawk_research.provenance import config_hash, git_commit


class HoldoutSealed(RuntimeError):
    """The holdout for this family has already been read, or was never unsealed."""


def default_audit_path() -> Path:
    return paths.MARKET_NEUTRAL_DIR / "holdout_audit.jsonl"


def _family_touched(audit_path: Path, family: str) -> bool:
    if not audit_path.exists():
        return False
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("family") == family:
            return True
    return False


def unseal_holdout(
    family: str,
    reason: str,
    loader: Callable[[], Any],
    audit_path: Optional[Path] = None,
) -> Any:
    """Read holdout data once for ``family``.

    ``loader`` is called only when the audit log has no prior line for that
    family. The audit line is written before the loader runs, so a loader
    failure still consumes the touch.
    """
    if not family:
        raise ValueError("family is required")
    if not reason:
        raise ValueError("reason is required")
    path = default_audit_path() if audit_path is None else Path(audit_path)
    if _family_touched(path, family):
        raise HoldoutSealed(
            f"holdout for {family!r} was already unsealed; a second read is a protocol breach"
        )
    start_ms, end_ms = holdout_bounds()
    record: Dict[str, Any] = {
        "family": family,
        "reason": reason,
        "unsealed_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash(),
        "holdout_start_ms": start_ms,
        "holdout_end_ms": end_ms,
        "holdout_end_inclusive": "2026-08-31",
        "holdout_cap": HOLDOUT_END,
        "note": "September 2026 bars after this cap are not part of the holdout.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return loader()
