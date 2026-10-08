"""Every runnable stage, its dependencies and what it needs.

``hawk-research stages`` prints this table; ``hawk-research run`` resolves the
dependency order from it. A stage's ``target`` is ``module:function`` and the
function is called with ``argv`` (a list of extra command-line arguments).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Stage:
    name: str
    target: str
    group: str
    summary: str
    deps: Tuple[str, ...] = ()
    argv: Tuple[str, ...] = ()
    network: bool = False
    live_snapshot: bool = False  # needs the live zones (snapshot/live_zones.parquet or the owner's database)
    workers: bool = False  # accepts --workers
    report: str = ""  # main output under data/reports (or another data subdirectory)

    def load(self) -> Callable[..., object]:
        module, func = self.target.split(":")
        return getattr(importlib.import_module(module), func)


S = "hawk_research.stages"
ST = "hawk_research.studies"

STAGES: Tuple[Stage, ...] = (
    # Data
    Stage(
        "s2b",
        f"{S}.s2b_pit_universe:main",
        "data",
        "Point-in-time volume ranks from the Binance archive listing",
        network=True,
        report="universe/pit_universe.parquet",
    ),
    # Round 1: the original HAWK detector
    Stage(
        "s3",
        f"{S}.s3_replay_zones:main",
        "round1",
        "Replay the round-1 detector over history",
        workers=True,
        report="zones/replay_zones.parquet",
    ),
    Stage(
        "s3b",
        f"{S}.s3b_fidelity_diagnostics:main",
        "round1",
        "Explain replay misses against the live book",
        deps=("s3",),
        live_snapshot=True,
    ),
    Stage(
        "s4",
        f"{S}.s4_bar_labels:main",
        "round1",
        "Per-bar forward labels",
        workers=True,
        report="labels/15m/",
    ),
    Stage(
        "s5",
        f"{S}.s5_zone_dataset:main",
        "round1",
        "Zone dataset: features, labels, lifecycle",
        deps=("s3", "s4"),
        workers=True,
        report="features/zone_dataset.parquet",
    ),
    Stage(
        "s5b",
        f"{S}.s5b_outcome_crosscheck:main",
        "round1",
        "Replay vs live outcome cross-check",
        deps=("s5",),
        live_snapshot=True,
    ),
    Stage(
        "s6",
        f"{S}.s6_leakage_audit:main",
        "round1",
        "Leakage and integrity audit (gates the rest)",
        deps=("s5",),
        report="reports/leakage_audit.json",
    ),
    # The published run used 60 replicates and the round-1 bootstrap size (600); see docs/deviations.md.
    Stage(
        "s7",
        f"{S}.s7_harness_check:main",
        "round1",
        "Harness calibration: null and positive control",
        deps=("s5",),
        argv=("--repeats", "60", "--bootstrap-samples", "600"),
        report="reports/harness_controls.json",
    ),
    Stage(
        "s8",
        f"{S}.s8_null_baselines:main",
        "round1",
        "Null baselines, matched controls, walk-forward, DSR",
        deps=("s6",),
        workers=True,
        report="reports/null_baselines.json",
    ),
    Stage(
        "s9",
        f"{S}.s9_location_control:main",
        "round1",
        "Retest rate vs matched control bars",
        deps=("s5",),
        report="reports/location_control.json",
    ),
    Stage(
        "s10",
        f"{S}.s10_bearish_only:main",
        "round1",
        "Bearish leg alone, after costs",
        deps=("s6",),
        report="reports/bearish_only.json",
    ),
    Stage(
        "s11",
        f"{S}.s11_capacity_check:main",
        "round1",
        "Results at the live concurrency cap",
        deps=("s6",),
        report="reports/capacity_check.json",
    ),
    # Family A (pre-registered S1/S5)
    Stage(
        "s3r",
        f"{S}.s3r_replay_pass_r:main",
        "family_a",
        "Pass R replay (swing floor off)",
        deps=("s5",),
        workers=True,
        report="zones/replay_zones_pass_r.parquet",
    ),
    Stage(
        "s1s5",
        f"{ST}.family_a.run:main",
        "family_a",
        "Pre-registered systems S1 and S5",
        deps=("s3r",),
        report="reports/s1_s5_results.json",
    ),
    # Round 2: textbook SMC order blocks
    Stage(
        "s3s",
        f"{S}.s3s_replay_smc:main",
        "smc",
        "Replay the textbook order-block detector",
        deps=("s3",),
        workers=True,
        report="zones/replay_zones_smc.parquet",
    ),
    Stage(
        "s5_smc",
        f"{S}.s5_zone_dataset:main",
        "smc",
        "SMC zone dataset",
        deps=("s3s", "s4"),
        argv=("--tag", "smc"),
        workers=True,
        report="features/zone_dataset_smc.parquet",
    ),
    Stage(
        "s6_smc",
        f"{S}.s6_leakage_audit:main",
        "smc",
        "SMC leakage audit",
        deps=("s5_smc",),
        argv=("--tag", "smc"),
        report="reports/leakage_audit_smc.json",
    ),
    Stage(
        "s8_smc",
        f"{S}.s8_null_baselines:main",
        "smc",
        "SMC null baselines",
        deps=("s6_smc",),
        argv=("--tag", "smc"),
        workers=True,
        report="reports/null_baselines_smc.json",
    ),
    Stage(
        "s12",
        f"{S}.s12_smc_eval:main",
        "smc",
        "SMC after-cost evaluation",
        deps=("s6_smc",),
        report="reports/smc_eval.json",
    ),
)

BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}
GROUPS: Dict[str, List[str]] = {}
for _s in STAGES:
    GROUPS.setdefault(_s.group, []).append(_s.name)


def resolve(
    names: Sequence[str], include_deps: bool = True, stages: Optional[Sequence[Stage]] = None
) -> List[Stage]:
    """Stages in dependency order. ``names`` may mix stage names, group names and ``all``.

    ``stages`` is the table to resolve against (the side studies have their own, in
    ``experiments.registry``). A dependency outside the table is not run from here.
    """
    from hawk_research.data import paths

    stages = STAGES if stages is None else stages
    by_name = {s.name: s for s in stages}
    groups: Dict[str, List[str]] = {}
    for s in stages:
        groups.setdefault(s.group, []).append(s.name)
    live = paths.has_live_snapshot()

    def runnable(s: Stage) -> bool:
        return not s.network and (live or not s.live_snapshot)

    wanted: List[str] = []
    for n in names:
        if n == "all":
            wanted.extend(s.name for s in stages if runnable(s))
        elif n in groups:
            wanted.extend(s for s in groups[n] if runnable(by_name[s]))
        elif n in by_name:
            wanted.append(n)
        else:
            raise KeyError(f"unknown stage or group {n!r}; see the `stages` command")
    order: List[str] = []

    def visit(name: str, chain: Tuple[str, ...] = ()) -> None:
        if name in chain:
            raise ValueError(f"dependency cycle: {' -> '.join(chain + (name,))}")
        if name in order:
            return
        if include_deps:
            for dep in by_name[name].deps:
                if dep not in by_name or by_name[dep].network:
                    continue  # data stages run through `download`; other tables run on their own
                visit(dep, chain + (name,))
        order.append(name)

    for n in wanted:
        visit(n)
    return [by_name[n] for n in order]
