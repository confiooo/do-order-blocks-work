"""Pre-registered relative-value search matrix. Every cell is one trial."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterator, List, Sequence, Tuple

BETA_MODES = ("fixed", "rolling_ols")
WINDOWS_H = (24, 72, 168)
ENTRIES = (1.5, 2.0, 2.5, 3.0)
EXITS = (0.0, 0.5)
HOLD_MULTIPLIERS = (1, 2, 4)

Pair = Tuple[str, str]


def cells_per_pair() -> int:
    return len(BETA_MODES) * len(WINDOWS_H) * len(ENTRIES) * len(EXITS) * len(HOLD_MULTIPLIERS)


def iter_cells(pairs: Sequence[Pair]) -> Iterator[Dict[str, object]]:
    """Deterministic cartesian product. Do not add cells after seeing results."""
    grid = product(BETA_MODES, WINDOWS_H, ENTRIES, EXITS, HOLD_MULTIPLIERS, pairs)
    for beta_mode, window_h, entry, exit_band, hold_mult, (symbol_a, symbol_b) in grid:
        yield {
            "beta_mode": beta_mode,
            "window_h": int(window_h),
            "entry": float(entry),
            "exit_band": float(exit_band),
            "hold_mult": int(hold_mult),
            "max_hold_bars": int(window_h) * int(hold_mult),
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "symbol": f"{symbol_a}|{symbol_b}",
            "timeframe": "1h",
            "family": "relative_value",
            "strategy_name": "pair_zscore",
        }


def matrix_size(n_pairs: int) -> int:
    return n_pairs * cells_per_pair()


def search_space() -> Dict[str, object]:
    return {
        "family": "relative_value",
        "strategy_name": "pair_zscore",
        "timeframe": "1h",
        "bar_source": "15m spot resampled to completed 1h bars",
        "beta_mode": list(BETA_MODES),
        "beta_note": "fixed means beta=1. rolling_ols is the trailing OLS slope of log A on log B over the same window as the z-score. No third beta variant.",
        "zscore_window_hours": list(WINDOWS_H),
        "entry_sigma": list(ENTRIES),
        "exit": {
            "0.0": "z crosses back through 0 from the entry side",
            "0.5": "abs(z) <= 0.5",
        },
        "max_hold_bars": "window_hours * {1, 2, 4}",
        "cells_per_pair": cells_per_pair(),
        "entry_rule": "z > entry shorts A and longs B; z < -entry longs A and shorts B. |z| == entry does not enter.",
        "exit_rule": "convergence as above, else max holding period. One position at a time. Next entry is the bar after the exit.",
        "neutrality": "equal notional on each leg (dollar neutral). Measured beta and BTC correlation are not assumed to be zero.",
        "keys": [
            "beta_mode",
            "window_h",
            "entry",
            "exit_band",
            "hold_mult",
            "symbol_a",
            "symbol_b",
        ],
    }


def feature_definitions(cell: Dict[str, object]) -> Dict[str, object]:
    window = int(cell["window_h"])
    return {
        "spread": "log(close_A) - beta * log(close_B)",
        "beta_mode": cell["beta_mode"],
        "beta_window_hours": window if cell["beta_mode"] == "rolling_ols" else None,
        "z": "(spread - trailing_mean) / trailing_sample_std",
        "z_window_hours": window,
        "z_includes_current_bar": True,
        "decision_price": "close of the signal hour, known at hour open + 1h",
        "no_future_bars": True,
    }


def as_records(pairs: Sequence[Pair]) -> List[Dict[str, object]]:
    return list(iter_cells(pairs))
