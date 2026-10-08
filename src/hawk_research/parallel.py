"""The one process pool every stage uses.

Workers start with ``spawn`` on every platform. ``spawn`` is the only start method on
Windows and the default on macOS, so using it everywhere means a stage that runs on
Linux also runs there: worker functions and initializer arguments must be picklable
top-level objects, and no worker may rely on state copied from a forked parent.

Workers learn the data directory and the resolved research configuration from the
environment variables the CLI sets before any pool starts (``HAWK_RESEARCH_DATA``,
``HAWK_RESEARCH_CONFIG`` and ``HAWK_RESEARCH_RESOLVED_CONFIG``), because a spawned
process re-imports every module from scratch.

``HAWK_RESEARCH_MP_START=fork`` (or ``forkserver``) overrides the start method, for
example to save worker start-up time on Linux. Results do not depend on it.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Optional, Sequence

START_METHOD_ENV = "HAWK_RESEARCH_MP_START"


def start_method() -> str:
    return os.environ.get(START_METHOD_ENV) or "spawn"


def process_pool(
    max_workers: Optional[int],
    initializer: Optional[Callable[..., Any]] = None,
    initargs: Sequence[Any] = (),
) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` with the project's start method."""
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context(start_method()),
        initializer=initializer,
        initargs=tuple(initargs),
    )
