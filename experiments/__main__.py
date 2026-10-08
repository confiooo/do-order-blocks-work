"""``python -m experiments``: the ``hawk-research`` command, on the side-study stages.

    python -m experiments stages
    python -m experiments --data-dir data run mn_ingest       # network: funding and perpetual bars
    python -m experiments --data-dir data run all              # every offline side-study stage
    python -m experiments --data-dir data verify               # their headline numbers

The zone trades read ``features/zone_dataset.parquet``: run ``hawk-research run s5`` first.
"""

from experiments.registry import METRICS, STAGES
from hawk_research import cli

if __name__ == "__main__":
    raise SystemExit(cli.main(stage_table=STAGES, metrics=METRICS, prog="python -m experiments"))
