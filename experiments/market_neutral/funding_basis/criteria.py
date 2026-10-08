"""Family B phase-1 pass bar.

Same statistical gates as relative value. The hedge gate is the strategy's
beta to its own spot leg and its correlation to BTC, both inside 0.2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.market_neutral.validation.criteria import (
    ABS_BETA_MAX,
    ABS_BTC_CORR_MAX,
    DSR_MIN,
    MIN_TRADES,
    PBO_MAX,
)


def judge_funding(metrics: Dict[str, Any]) -> Dict[str, Any]:
    n = int(metrics.get("n_trades") or 0)
    width = _width(metrics.get("sharpe_ci_low"), metrics.get("sharpe_ci_high"))
    if n < MIN_TRADES:
        return {
            "status": "inconclusive",
            "reasons": [f"n_trades {n} < {MIN_TRADES}"],
            "n_trades": n,
            "sharpe_ci_width": width,
        }
    reasons: List[str] = []
    dsr = metrics.get("dsr")
    if dsr is None or not (float(dsr) >= DSR_MIN):
        reasons.append(f"dsr {dsr} < {DSR_MIN}")
    pbo = metrics.get("pbo")
    if pbo is None or not (float(pbo) < PBO_MAX):
        reasons.append(f"pbo {pbo} is not < {PBO_MAX}")
    sharpe = metrics.get("sharpe_net_ann")
    low = metrics.get("sharpe_ci_low")
    high = metrics.get("sharpe_ci_high")
    if sharpe is None or low is None or high is None or not (float(low) > 0.0):
        reasons.append(f"net sharpe CI does not lie above zero (sharpe={sharpe}, ci=[{low}, {high}])")
    beta = metrics.get("beta_underlying")
    if beta is None or not (abs(float(beta)) < ABS_BETA_MAX):
        reasons.append(f"|net directional exposure| {beta} is not < {ABS_BETA_MAX}")
    corr = metrics.get("corr_btc")
    if corr is None or not (abs(float(corr)) < ABS_BTC_CORR_MAX):
        reasons.append(f"|btc correlation| {corr} is not < {ABS_BTC_CORR_MAX}")
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "n_trades": n,
        "sharpe_ci_width": width,
    }


def _width(low: Optional[float], high: Optional[float]) -> Optional[float]:
    if low is None or high is None:
        return None
    return float(high) - float(low)
