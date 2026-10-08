"""Append-only trial ledger in its own SQLite file (``paths.RESEARCH_DB``).

The deflated Sharpe depends on the trial count, so the count is kept in a database
rather than in anyone's memory. Every evaluated configuration is
recorded under a ``name_prefix``; :attr:`TrialLedger.n_trials` reads the count
back from the database, so earlier sessions of the same scan still count against
today's result.

Each row also stores the provenance of the run that produced it (commit, dirty
flag, config hash, command, data-manifest hash, package versions). A JSONL
sidecar mirrors every row so the count is auditable without opening SQLite.

The original project kept two ledgers: one inside the live bot's operational
database and one research-only file. Both are replaced by this module. Their
historical trial counts are preserved in ``config/trial_history.json``; a ledger
created with ``prior_trials`` counts those against every deflation.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hawk_research import provenance
from hawk_research.data import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    config_hash TEXT,
    command TEXT,
    data_manifest_sha256 TEXT,
    packages_json TEXT,
    dataset_start TEXT,
    dataset_end TEXT,
    parameters_json TEXT,
    parent_experiment_id INTEGER
);
"""


def jsonable(obj: Any) -> Any:
    """Coerce numpy scalars and non-finite floats so an insert cannot fail."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else paths.RESEARCH_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def count_trials(name_prefix: str, db_path: Optional[Path] = None) -> int:
    """Experiments logged under ``name_prefix``."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM experiments WHERE name LIKE ?;", (f"{name_prefix}/%",)
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"]) if row else 0


class TrialLedger:
    """Record every configuration evaluated under ``name_prefix``."""

    def __init__(
        self,
        name_prefix: str,
        description: str = "",
        dataset_start: Optional[str] = None,
        dataset_end: Optional[str] = None,
        db_path: Optional[Path] = None,
        jsonl_path: Optional[Path] = None,
        prior_trials: int = 0,
    ) -> None:
        if not name_prefix:
            raise ValueError("name_prefix is required so trials can be counted back")
        self.name_prefix = name_prefix
        self.description = description
        self.dataset_start = dataset_start
        self.dataset_end = dataset_end
        self.db_path = Path(db_path) if db_path is not None else paths.RESEARCH_DB
        self.jsonl_path = (
            Path(jsonl_path)
            if jsonl_path is not None
            else self.db_path.parent / "reports" / "experiments_ledger.jsonl"
        )
        if prior_trials < 0:
            raise ValueError("prior_trials must be non-negative")
        self.prior_trials = int(prior_trials)
        self.experiment_ids: List[int] = []
        self._provenance: Optional[Dict[str, Any]] = None
        connect(self.db_path).close()

    def _prov(self) -> Dict[str, Any]:
        if self._provenance is None:
            self._provenance = provenance.record(include_config=False)
        return self._provenance

    def log_trial(
        self,
        parameters: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        label: str = "",
    ) -> int:
        """Record one evaluated configuration and return its experiment_id."""
        prov = self._prov()
        payload = {
            "parameters": jsonable(parameters),
            "metrics": jsonable(metrics or {}),
            "ledger": self.name_prefix,
        }
        name = f"{self.name_prefix}/{label or f'{self.n_logged:04d}'}"
        conn = connect(self.db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO experiments (
                    name, description, created_at, git_commit, git_dirty, config_hash,
                    command, data_manifest_sha256, packages_json,
                    dataset_start, dataset_end, parameters_json, parent_experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    name,
                    self.description,
                    prov["created_at"],
                    prov["git_commit"],
                    None if prov["git_dirty"] is None else int(prov["git_dirty"]),
                    prov["config_hash"],
                    prov["command"],
                    prov["data_manifest_sha256"],
                    json.dumps(prov["packages"], sort_keys=True),
                    self.dataset_start,
                    self.dataset_end,
                    json.dumps(payload, sort_keys=True),
                    None,
                ),
            )
            conn.commit()
            experiment_id = int(cur.lastrowid or 0)
        finally:
            conn.close()
        self.experiment_ids.append(experiment_id)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            row = {"experiment_id": experiment_id, "name": name, **payload, "provenance": prov}
            fh.write(json.dumps(row, default=str) + "\n")
        return experiment_id

    @property
    def n_trials(self) -> int:
        """Trials recorded under this prefix across all sessions, plus ``prior_trials``."""
        return self.prior_trials + self.n_logged

    @property
    def n_logged(self) -> int:
        """Trials recorded under this prefix in the local database."""
        return count_trials(self.name_prefix, self.db_path)

    def trials(self) -> List[Dict[str, Any]]:
        """All recorded trials with their parsed parameters, oldest first."""
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT experiment_id, name, created_at, parameters_json FROM experiments "
                "WHERE name LIKE ? ORDER BY experiment_id;",
                (f"{self.name_prefix}/%",),
            ).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["parameters_json"] or "{}")
            except (TypeError, ValueError):
                payload = {}
            out.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "parameters": payload.get("parameters", {}),
                    "metrics": payload.get("metrics", {}),
                }
            )
        return out


def historical_prior(name_prefix: str, path: Optional[Path] = None) -> int:
    """``prior_trials`` for ``name_prefix`` from ``config/trial_history.json`` (0 if absent)."""
    f = Path(path) if path is not None else paths.CONFIG_DIR / "trial_history.json"
    if not f.exists():
        return 0
    entry = json.loads(f.read_text(encoding="utf-8")).get("ledgers", {}).get(name_prefix, {})
    return int(entry.get("prior_trials", 0))
