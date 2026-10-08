"""Stage 5b: re-run the live-vs-replay outcome cross-check on the built dataset.

Separated from stage 5 so the comparison can be re-run in seconds without
rebuilding the 152k-row dataset.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import pandas as pd

from hawk_research.data import paths
from hawk_research.stages.s5_zone_dataset import live_outcome_crosscheck

log = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    ds = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")
    result = live_outcome_crosscheck(ds)
    out = paths.FEATURES_DIR / "live_outcome_crosscheck.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
