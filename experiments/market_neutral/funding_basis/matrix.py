"""Family B phase-1 grid. Count is fixed before any cell is scored."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from experiments.market_neutral.funding_basis.universe import FAMILY_B_NAMES

CARRY_EXITS = ("funding_negative", "max_hold")
PERSIST_WINDOWS = (3, 9, 21)
BASIS_WINDOWS = (24, 72, 168)
BASIS_ENTRIES = (1.5, 2.0, 2.5, 3.0)
BASIS_EXITS = (0.0, 0.5)
MAX_SETTLEMENTS = 21

Level = Tuple[str, float]


def cells_per_name(n_thresholds: int = 3) -> Dict[str, int]:
    b2 = n_thresholds * len(CARRY_EXITS)
    b3 = len(PERSIST_WINDOWS) * n_thresholds * len(CARRY_EXITS)
    b4 = len(BASIS_WINDOWS) * len(BASIS_ENTRIES) * len(BASIS_EXITS) * 2
    return {"B2": b2, "B3": b3, "B4": b4, "total": b2 + b3 + b4}


def expected_cells(n_names: int, n_thresholds: int = 3) -> int:
    return n_names * cells_per_name(n_thresholds)["total"]


def build_cells(names: Sequence[str], levels: Sequence[Level]) -> List[Dict[str, object]]:
    """Deterministic list. B.2 is the current settlement. B.3 is the rolling windows."""
    if len(levels) != 3:
        raise ValueError("phase 1 uses exactly three funding thresholds")
    cells: List[Dict[str, object]] = []
    for symbol in names:
        for label, threshold in levels:
            for exit_mode in CARRY_EXITS:
                cells.append(
                    {
                        "stage": "B2",
                        "strategy_name": "funding_threshold",
                        "symbol": symbol,
                        "window": 1,
                        "threshold_label": label,
                        "threshold": float(threshold),
                        "exit_mode": exit_mode,
                        "funding_filter": False,
                    }
                )
        for window in PERSIST_WINDOWS:
            for label, threshold in levels:
                for exit_mode in CARRY_EXITS:
                    cells.append(
                        {
                            "stage": "B3",
                            "strategy_name": "funding_persistence",
                            "symbol": symbol,
                            "window": int(window),
                            "threshold_label": label,
                            "threshold": float(threshold),
                            "exit_mode": exit_mode,
                            "funding_filter": False,
                        }
                    )
        for window in BASIS_WINDOWS:
            for entry in BASIS_ENTRIES:
                for exit_band in BASIS_EXITS:
                    for funding_filter in (False, True):
                        cells.append(
                            {
                                "stage": "B4",
                                "strategy_name": "basis_zscore",
                                "symbol": symbol,
                                "window": int(window),
                                "entry": float(entry),
                                "exit_band": float(exit_band),
                                "funding_filter": bool(funding_filter),
                            }
                        )
    expected = expected_cells(len(names), len(levels))
    if len(cells) != expected:
        raise RuntimeError(f"cell count {len(cells)} != {expected}")
    return cells


def search_space(levels: Sequence[Level], report: Dict[str, object]) -> Dict[str, object]:
    counts = cells_per_name(len(levels))
    return {
        "names": list(FAMILY_B_NAMES),
        "n_names": len(FAMILY_B_NAMES),
        "structure": "long spot / short perpetual only",
        "thresholds": [{"label": label, "value": value} for label, value in levels],
        "threshold_report": report,
        "B2": {
            "signal": "current settlement > threshold",
            "exits": list(CARRY_EXITS),
            "max_settlements": MAX_SETTLEMENTS,
            "cells_per_name": counts["B2"],
        },
        "B3": {
            "windows_settlements": list(PERSIST_WINDOWS),
            "exits": list(CARRY_EXITS),
            "max_settlements": MAX_SETTLEMENTS,
            "cells_per_name": counts["B3"],
        },
        "B4": {
            "windows_hours": list(BASIS_WINDOWS),
            "entry_sigma": list(BASIS_ENTRIES),
            "exit_sigma": list(BASIS_EXITS),
            "funding_filter": [False, True],
            "side": "z > entry only; the cheap-perp side is not traded",
            "cells_per_name": counts["B4"],
        },
        "cells_per_name": counts["total"],
        "n_cells": expected_cells(len(FAMILY_B_NAMES), len(levels)),
    }
