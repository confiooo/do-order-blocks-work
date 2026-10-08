"""Zone-trade study: the breakout simulator, checked by hand."""

from __future__ import annotations

import numpy as np
import pytest


def test_breakout_trail_simulate_by_hand():
    from experiments.zone_trades import breakout_trail as bt

    n = 120
    o = np.full(n, 101.4)
    h = np.full(n, 101.4)
    low = np.full(n, 101.4)
    c = np.full(n, 101.4)
    # bar 0 breaks the 100-101 box upward (path open, low, high, close): entry at the box top
    o[0], h[0], low[0], c[0] = 100.5, 101.5, 100.4, 101.4
    # bar 5 drops to 99: the 0.25% trail (stop 101.5 * 0.9975) is hit, the 5% trail is not
    o[5], h[5], low[5], c[5] = 101.4, 101.4, 99.0, 99.5
    o[6:], h[6:], low[6:], c[6:] = 99.5, 99.5, 99.5, 99.5
    trails = np.array([0.0025, 0.05])
    out = bt.simulate(o, h, low, c, 101.0, 100.0, trails)
    assert out is not None and out["side"] == 1 and out["wait_bars"] == 1
    assert out["stopped"][:, 0].all() and not out["stopped"][:, 1].any()
    assert out["gross"][0, 0] == pytest.approx((101.5 * 0.9975 - 101.0) / 101.0)
    assert out["gross"][:, 1] == pytest.approx([(99.5 - 101.0) / 101.0] * 3)
    assert out["valid"].all()

    # too few bars for the 24h snapshot while one trail is still open: no 24h observation at all
    short = bt.simulate(o[:40], h[:40], low[:40], c[:40], 101.0, 100.0, trails)
    assert short is not None
    assert short["valid"][:2].all() and not short["valid"][2].any()
