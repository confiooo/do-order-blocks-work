"""Checksummed data manifest.

``config/data_manifest.json`` records, for every symbol and interval the
published results used, the row count, first/last bar and a sha256 of the bar
*content*. The hash is computed over the values in a fixed column order and
dtype, not over the parquet bytes, so it does not depend on the pyarrow version
or compression settings of the machine that wrote the file.

``hawk-research download`` and ``hawk-research verify-data`` compare local files
against it and report every mismatch. A mismatch is not hidden or repaired: the
upstream archive can change, and a reader needs to know if it did.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.data import paths

CONTENT_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_quote",
)
MANIFEST_PATH_NAME = "data_manifest.json"


def content_sha256(df: pd.DataFrame) -> str:
    """sha256 over bar values: open_time as int64, everything else as float64."""
    frame = df.sort_values("open_time", kind="mergesort")
    h = hashlib.sha256()
    for col in CONTENT_COLUMNS:
        if col not in frame.columns:
            h.update(f"{col}:absent;".encode())
            continue
        dtype = np.int64 if col == "open_time" else np.float64
        values = np.ascontiguousarray(frame[col].to_numpy(dtype=dtype)).astype(f"<{np.dtype(dtype).str[1:]}")
        h.update(f"{col}:".encode())
        h.update(values.tobytes())
    return h.hexdigest()


def describe(symbol: str, interval: str, df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "interval": interval,
        "rows": int(len(df)),
        "first_open_time": int(df["open_time"].min()) if len(df) else None,
        "last_open_time": int(df["open_time"].max()) if len(df) else None,
        "content_sha256": content_sha256(df) if len(df) else None,
    }


def manifest_path() -> Path:
    return paths.CONFIG_DIR / MANIFEST_PATH_NAME


def load_published() -> Dict[str, Any]:
    path = manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def entries_by_key(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {f"{e['symbol']}|{e['interval']}": e for e in manifest["entries"]}


def compare(local: Dict[str, Any], published: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Field-by-field differences between a local entry and the published one."""
    diffs = []
    for field in ("rows", "first_open_time", "last_open_time", "content_sha256"):
        if local.get(field) != published.get(field):
            diffs.append({"field": field, "local": local.get(field), "published": published.get(field)})
    return diffs


def check_local(
    keys: Optional[List[str]] = None,
    published: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare local files to the published manifest. Missing files are reported, not errors."""
    published = published or load_published()
    by_key = entries_by_key(published)
    keys = keys or sorted(by_key)
    out: Dict[str, Any] = {"matched": [], "mismatched": [], "missing": []}
    for key in keys:
        symbol, interval = key.split("|")
        path = paths.ohlcv_path(symbol, interval)
        if not path.exists():
            out["missing"].append(key)
            continue
        local = describe(symbol, interval, pd.read_parquet(path))
        diffs = compare(local, by_key[key])
        (out["mismatched"] if diffs else out["matched"]).append(
            {"key": key, "diffs": diffs} if diffs else key
        )
    return out
