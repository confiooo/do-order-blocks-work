"""Compare the outputs of two data directories, file by file (owner regression check).

``hawk-research compare OLD_DIR NEW_DIR`` checks every report JSON and parquet table the
stages write: parquet row counts, columns and values (exact for integers and strings,
relative 1e-12 for floats by default), and JSON recursively. Keys that legitimately differ
between runs are ignored: timestamps, elapsed times, provenance blocks, file paths and
ledger ids.

This is the check used to show that restructuring the code did not change any result: run
the full pipeline into a fresh directory and compare it with a previous run. It is not
part of CI because it needs two full runs (about two hours each).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

OUTPUT_DIRS = ("zones", "features", "reports", "daily_fade", "market_neutral")
VOLATILE = {
    "_provenance",
    "created_at",
    "generated_at",
    "rerun_at",
    "elapsed_s",
    "git_commit",
    "path",
    "paths",
    "surface",
    "match_table",
    "feature_screen_files",
    "ledger_experiment_ids",
    "experiment_id",
}
KEY_COLS = ("symbol", "source_open_time", "open_time", "zone_uid", "name")


def _volatile(key: Any) -> bool:
    return isinstance(key, str) and (key in VOLATILE or key.endswith("_path"))


def _num_eq(a: Any, b: Any, rtol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-15)
    return bool(a == b)


def diff_json(
    a: Any, b: Any, rtol: float = 1e-12, path: str = "", out: Optional[List[str]] = None
) -> List[str]:
    """Every differing leaf as ``path: a vs b`` (volatile keys skipped)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if _volatile(k):
                continue
            if k not in a:
                out.append(f"{path}/{k}: only in new")
            elif k not in b:
                out.append(f"{path}/{k}: only in old")
            else:
                diff_json(a[k], b[k], rtol, f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, rtol, f"{path}[{i}]", out)
    elif not _num_eq(a, b, rtol):
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def diff_parquet(pa: Path, pb: Path, rtol: float = 1e-12) -> List[str]:
    """Row count, column set and per-column value differences, rows aligned on key columns."""
    a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    notes = []
    ca, cb = set(a.columns), set(b.columns)
    if len(a) != len(b):
        notes.append(f"rows {len(a)} vs {len(b)}")
    if ca - cb:
        notes.append(f"columns only in old: {sorted(ca - cb)}")
    if cb - ca:
        notes.append(f"columns only in new: {sorted(cb - ca)}")
    if len(a) != len(b):
        return notes
    keys = [k for k in KEY_COLS if k in ca & cb]
    if keys:
        a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
        b = b.sort_values(keys, kind="mergesort").reset_index(drop=True)
    bad = []
    for col in sorted(ca & cb):
        if _volatile(col):
            continue
        x, y = a[col], b[col]
        if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
            xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
            ok = np.isclose(xv, yv, rtol=rtol, atol=1e-15, equal_nan=True)
        else:
            both_na = (x.isna() & y.isna()).to_numpy()
            same = x.astype(str).to_numpy() == y.astype(str).to_numpy()
            ok = both_na | same
        if not ok.all():
            bad.append(f"{col}({int((~ok).sum())})")
    if bad:
        notes.append("value differences in " + ", ".join(bad))
    return notes


def output_files(root: Path) -> List[str]:
    """Report JSON and parquet tables the stages write, relative to ``root``."""
    found = []
    for d in OUTPUT_DIRS:
        base = root / d
        if base.exists():
            found += [
                p.relative_to(root).as_posix() for p in base.rglob("*") if p.suffix in (".json", ".parquet")
            ]
    return sorted(found)


def compare_dirs(
    old: Path, new: Path, files: Optional[Sequence[str]] = None, rtol: float = 1e-12
) -> List[Dict[str, Any]]:
    """One row per file: ``{"file", "status": SAME|DIFF|MISSING, "notes": [...]}``."""
    names = list(files) if files else sorted(set(output_files(old)) | set(output_files(new)))
    rows = []
    for rel in names:
        pa, pb = old / rel, new / rel
        if not pa.exists() or not pb.exists():
            where = "old" if pa.exists() else "new"
            rows.append({"file": rel, "status": "MISSING", "notes": [f"only in {where}"]})
            continue
        if pa.suffix == ".parquet":
            notes = diff_parquet(pa, pb, rtol)
        else:
            notes = diff_json(
                json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8")), rtol
            )
        rows.append({"file": rel, "status": "DIFF" if notes else "SAME", "notes": notes})
    return rows


def format_rows(rows: List[Dict[str, Any]], max_notes: int = 20) -> str:
    lines = []
    for r in rows:
        lines.append(f"{r['status']:<8}{r['file']}")
        lines += [f"          {n}" for n in r["notes"][:max_notes]]
        if len(r["notes"]) > max_notes:
            lines.append(f"          ... {len(r['notes']) - max_notes} more")
    n = {s: sum(1 for r in rows if r["status"] == s) for s in ("SAME", "DIFF", "MISSING")}
    lines.append(f"\n{n['SAME']} same, {n['DIFF']} differ, {n['MISSING']} only in one directory")
    return "\n".join(lines)
