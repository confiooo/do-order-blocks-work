"""The live system's zones table, for the replay-fidelity checks (s3, s3b, s5, s5b).

The owner's full snapshot database is used when present; otherwise the committed extract
``snapshot/live_zones.parquet``, which is that table exported unchanged.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from hawk_research.data import paths


def load_zones() -> pd.DataFrame:
    if paths.LIVE_SNAPSHOT_DB.exists():
        conn = sqlite3.connect(f"file:{paths.LIVE_SNAPSHOT_DB.as_posix()}?mode=ro", uri=True)
        try:
            return pd.read_sql_query("SELECT * FROM zones;", conn)
        finally:
            conn.close()
    if paths.LIVE_ZONES_EXTRACT.exists():
        return pd.read_parquet(paths.LIVE_ZONES_EXTRACT)
    raise FileNotFoundError(f"no live zones: neither {paths.LIVE_SNAPSHOT_DB} nor {paths.LIVE_ZONES_EXTRACT}")


def source() -> str:
    """Which copy :func:`load_zones` reads, for reports."""
    if paths.LIVE_SNAPSHOT_DB.exists():
        return "owner snapshot database"
    return "snapshot/live_zones.parquet"
