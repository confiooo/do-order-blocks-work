"""The side-study stages, run with ``python -m experiments``.

Same ``Stage`` records as ``hawk_research.stages.registry``. Dependencies on main-pipeline
stages (the zone trades read the round-1 zone dataset from ``s5``) are not run from here:
run ``hawk-research run s5`` first.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from hawk_research.stages.registry import Stage
from hawk_research.verify import EXACT, RET, STAT, Metric

X = "experiments"

STAGES: Tuple[Stage, ...] = (
    # Side study: trade-management experiments on the round-1 alert box
    Stage(
        "zt_harvest",
        f"{X}.zone_trades.two_way_harvest:main",
        "zone_trades",
        "Two-way harvest at zone edges",
        deps=("s5",),
        report="reports/two_way_harvest.json",
    ),
    Stage(
        "zt_width",
        f"{X}.zone_trades.two_way_width_sweep:sweep",
        "zone_trades",
        "Synthetic straddle width sweep",
        deps=("s5",),
        report="reports/two_way_width_sweep.json",
    ),
    Stage(
        "zt_symsl",
        f"{X}.zone_trades.two_way_symmetric_sl:sweep",
        "zone_trades",
        "Straddle with symmetric remaining stop",
        deps=("s5",),
        report="reports/two_way_symmetric_sl.json",
    ),
    Stage(
        "zt_gainbuf",
        f"{X}.zone_trades.two_way_gain_buffer:sweep",
        "zone_trades",
        "Gain-buffer trailing stop",
        deps=("s5",),
        report="reports/two_way_gain_buffer.json",
    ),
    Stage(
        "zt_oco",
        f"{X}.zone_trades.two_way_oco_fast:sweep",
        "zone_trades",
        "Straddle with post-TP OCO",
        deps=("s5",),
        report="reports/two_way_oco.json",
    ),
    Stage(
        "zt_breakout",
        f"{X}.zone_trades.breakout_trail:main",
        "zone_trades",
        "Breakout from the box with a trailing stop",
        deps=("s5",),
        report="reports/breakout_trail.json",
    ),
    # Side study: daily fade
    Stage(
        "df_study",
        f"{X}.daily_fade.study:main",
        "daily_fade",
        "Daily-fade event study",
        report="daily_fade/summary.json",
    ),
    Stage(
        "df_zone_trade",
        f"{X}.daily_fade.zone_trade:main",
        "daily_fade",
        "Daily-fade rollback-zone rules",
        deps=("df_study",),
        report="daily_fade/zone_bot.json",
    ),
    # The original project produced tp_stops.json by calling run_tp_stops() by hand.
    Stage(
        "df_tp_stops",
        f"{X}.daily_fade.zone_trade:run_tp_stops",
        "daily_fade",
        "Daily-fade take-profit/stop grid",
        deps=("df_study",),
        report="daily_fade/tp_stops.json",
    ),
    # Side study: market neutral
    Stage(
        "mn_ingest",
        f"{X}.market_neutral.funding_basis.ingest:ingest",
        "market_neutral",
        "Ingest USD-M funding and 1h perpetual bars",
        network=True,
    ),
    Stage(
        "mn_rv_phase1",
        f"{X}.market_neutral.run:main",
        "market_neutral",
        "Relative-value phase-1 matrix",
        report="market_neutral/relative_value_phase1.json",
    ),
    Stage(
        "mn_tier2",
        f"{X}.market_neutral.run:run_tier2",
        "market_neutral",
        "Relative value, tier-2 names",
        deps=("mn_rv_phase1",),
        report="market_neutral/relative_value_tier2.json",
    ),
    Stage(
        "mn_coint",
        f"{X}.market_neutral.run:run_cointegration",
        "market_neutral",
        "Relative value with a cointegration filter",
        deps=("mn_rv_phase1",),
        report="market_neutral/relative_value_cointegration.json",
    ),
    Stage(
        "mn_fb_phase1",
        f"{X}.market_neutral.funding_basis.run:main",
        "market_neutral",
        "Family B funding/basis phase-1 matrix",
        deps=("mn_ingest",),
        report="market_neutral/funding_basis_phase1.json",
    ),
    Stage(
        "mn_rerun_rolling",
        f"{X}.market_neutral.rerun_rolling:main",
        "market_neutral",
        "Re-score rolling-beta cells with the corrected z-score",
        deps=("mn_rv_phase1", "mn_tier2", "mn_coint"),
    ),
)

BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}
GROUPS: Dict[str, List[str]] = {}
for _s in STAGES:
    GROUPS.setdefault(_s.group, []).append(_s.name)

# Headline numbers of the side studies, checked by ``python -m experiments verify``.
METRICS: List[Metric] = [
    (
        "zone trades: width sweep any profitable",
        "reports/two_way_width_sweep.json",
        ["any_profitable"],
        EXACT,
    ),
    ("zone trades: OCO any profitable", "reports/two_way_oco.json", ["any_profitable"], EXACT),
    (
        "zone trades: gain buffer any profitable",
        "reports/two_way_gain_buffer.json",
        ["any_profitable"],
        EXACT,
    ),
    (
        "zone trades: symmetric SL any profitable",
        "reports/two_way_symmetric_sl.json",
        ["any_profitable"],
        EXACT,
    ),
    ("daily fade events", "daily_fade/summary.json", ["n_events"], EXACT),
    ("daily fade symbols", "daily_fade/summary.json", ["n_symbols"], EXACT),
    ("daily fade zone rule test mean net", "daily_fade/zone_bot.json", ["chosen_test", "mean_net"], RET),
    ("market neutral RV trials", "market_neutral/relative_value_phase1.json", ["n_trials"], EXACT),
    (
        "market neutral RV PBO (after rolling fix)",
        "market_neutral/relative_value_phase1.json",
        ["pbo_after_rolling_beta_fix"],
        STAT,
    ),
    ("market neutral funding/basis cells", "market_neutral/funding_basis_phase1.json", ["n_cells"], EXACT),
    ("market neutral funding/basis PBO", "market_neutral/funding_basis_phase1.json", ["pbo"], STAT),
]
