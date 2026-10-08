from __future__ import annotations

from typing import Dict, List


def chronological_windows(
    start_ms: int,
    end_ms: int,
    train_days: int,
    test_days: int,
    step_days: int,
    embargo_hours: int,
) -> List[Dict[str, int]]:
    """TRAIN → TEST folds in market time. No random split."""
    train_ms = train_days * 24 * 3600 * 1000
    test_ms = test_days * 24 * 3600 * 1000
    step_ms = step_days * 24 * 3600 * 1000
    embargo_ms = embargo_hours * 3600 * 1000
    folds: List[Dict[str, int]] = []
    cursor = start_ms
    fold = 0
    while cursor + train_ms + embargo_ms + test_ms <= end_ms + 1:
        train_start = cursor
        train_end = cursor + train_ms
        test_start = train_end + embargo_ms
        test_end = test_start + test_ms
        folds.append(
            {
                "fold": fold,
                "train_start_ms": train_start,
                "train_end_ms": train_end,
                "test_start_ms": test_start,
                "test_end_ms": test_end,
                "embargo_ms": embargo_ms,
            }
        )
        cursor += step_ms
        fold += 1
        if step_ms <= 0:
            break
    return folds
