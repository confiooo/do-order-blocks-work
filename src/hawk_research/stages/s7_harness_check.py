"""Stage 7: check that the intervals the stages quote are calibrated.

All controls use the round-1 trade table (direction rule, close entry, 4h exit,
no costs), so timing and cross-sectional clustering are those of the real data.

* **Randomised sides** (calibration): replace the side with a coin flip. The true
  mean is then zero, so the cross-sectional 95% interval should exclude zero in
  about 5% of replicates. Far more would mean the intervals are too narrow.
* **Within-block shuffle** (permutation test): shuffle unsigned returns inside 24h
  blocks. This null is not centred on zero (see ``main``), so the observed mean is
  compared with the replicates, not with zero.
* **Positive control**: add a known edge; the interval must exclude zero.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research import config as research_config
from hawk_research.data import paths
from hawk_research.evaluation import strategies
from hawk_research.evaluation.harness import HarnessConfig
from hawk_research.stats.resampling import block_bootstrap_ci, ci_excludes_zero, cross_sectional_bootstrap_ci

log = logging.getLogger(__name__)

BLOCK_MS = 24 * 3600 * 1000


def _permute_within_blocks(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def shuffle_labels_within_blocks(
    trades: pd.DataFrame, block_ms: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Shuffle unsigned returns within time blocks, then reapply each trade's side.

    Shuffling the signed return would leave the mean unchanged and the control would
    test nothing.
    """
    out = trades.copy()
    blocks = out["entry_time_ms"].to_numpy(dtype=np.int64) // block_ms
    side = out["side"].to_numpy(dtype=float)
    raw = out["ret_gross"].to_numpy(dtype=float) / side
    out["ret_gross"] = side * _permute_within_blocks(raw, blocks, rng)
    return out


def randomize_sides(trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace the signal with a coin flip, keeping timing and magnitudes.

    Only the thing under test, which way to bet, is destroyed; the trade population,
    its clustering and its return magnitudes are untouched. The true mean is zero.
    """
    out = trades.copy()
    side = out["side"].to_numpy(dtype=float)
    raw = out["ret_gross"].to_numpy(dtype=float) / side
    new_side = rng.choice(np.array([-1.0, 1.0]), size=len(out))
    out["side"] = new_side.astype(int)
    out["ret_gross"] = new_side * raw
    return out


def _block_ci(trades: pd.DataFrame, cfg: HarnessConfig) -> Dict:
    """Moving-block interval of the gross mean (serial dependence only)."""
    return block_bootstrap_ci(
        trades["ret_gross"].to_numpy(dtype=float),
        trades["entry_time_ms"].to_numpy(dtype=np.int64),
        block_hours=cfg.bootstrap_block_hours,
        n_samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )


def _cross_ci(trades: pd.DataFrame, cfg: HarnessConfig) -> Dict:
    """Cross-sectional interval: whole 24h blocks with every symbol inside them.

    This is the interval every stage quotes; it is the one that must be calibrated.
    """
    return cross_sectional_bootstrap_ci(
        trades["ret_gross"].to_numpy(dtype=float),
        trades["entry_time_ms"].to_numpy(dtype=np.int64),
        block_hours=cfg.bootstrap_block_hours,
        n_samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )


def _significant(ci: Dict) -> bool:
    return bool(ci_excludes_zero(ci))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _CFG = research_config.get()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--edge-bps", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=_CFG.seeds.harness_check)
    args = parser.parse_args(argv)

    ds = pd.read_parquet(paths.FEATURES_DIR / "zone_dataset.parquet")
    spec = strategies.TradeSpec(side_rule="direction", entry="close", exit="time", horizon="4h")
    trades = strategies.build_trades(ds, spec)
    log.info(f"trade table: {len(trades)} rows, {trades['symbol'].nunique()} symbols")

    cfg = HarnessConfig.from_research(bootstrap_samples=args.bootstrap_samples)
    rng = np.random.default_rng(args.seed)

    # Observed mean, for reference only; s8 is the actual test.
    observed = _block_ci(trades, cfg)
    observed_stat = observed["stat"]

    # Calibration: with random sides the true mean is zero, so the share of
    # intervals excluding zero is the false-positive rate (should be ~5%).
    log.info(f"negative control 'randomized_sides' x{args.repeats}")
    hits = 0
    rand_means: List[float] = []
    for _ in range(args.repeats):
        ci = _cross_ci(randomize_sides(trades, rng), cfg)
        rand_means.append(ci["stat"])
        if _significant(ci):
            hits += 1
    rand_rate = hits / max(1, args.repeats)
    spread = float(np.std(rand_means))

    # Permutation test. This null is not centred on zero: within a 24h block the
    # bullish/bearish mix correlates with the block's mean return. So the observed
    # mean is compared with the replicates, not with zero.
    log.info(f"permutation null 'shuffled_labels' x{args.repeats}")
    perm_means: List[float] = []
    for _ in range(args.repeats):
        perm_means.append(_block_ci(shuffle_labels_within_blocks(trades, BLOCK_MS, rng), cfg)["stat"])
    perm = np.asarray(perm_means, dtype=float)
    perm_sd = float(perm.std(ddof=1))
    z = float((observed_stat - perm.mean()) / perm_sd) if perm_sd > 0 else float("nan")
    p_perm = float((np.abs(perm - perm.mean()) >= abs(observed_stat - perm.mean())).mean())

    controls: Dict[str, Dict] = {
        "randomized_sides": {
            "purpose": "confidence-interval calibration under a true zero null",
            "repeats": args.repeats,
            "significant_at_95pct": hits,
            "false_positive_rate": rand_rate,
            "nominal_rate": 0.05,
            "mean_of_replicate_means": float(np.mean(rand_means)),
            "std_of_replicate_means": spread,
            "degenerate": spread < 1e-12,
            "verdict": (
                "FAIL - replicate statistic is invariant, the control tests nothing"
                if spread < 1e-12
                else "PASS"
                if rand_rate <= 0.15
                else "FAIL - confidence intervals are too narrow"
            ),
        },
        "shuffled_labels_permutation": {
            "purpose": "permutation test of the observed directional statistic",
            "repeats": args.repeats,
            "null_mean": float(perm.mean()),
            "null_std": perm_sd,
            "observed": observed_stat,
            "z_vs_null": z,
            "p_value_two_sided": p_perm,
            "null_is_centred_on_zero": bool(abs(perm.mean()) < 2 * perm_sd / np.sqrt(args.repeats)),
            "note": (
                "The null is displaced from zero because block-level bullish/bearish "
                "imbalance correlates with that block's mean return. Comparing each "
                "replicate's CI to zero would be a category error; the observed "
                "statistic is compared to this empirical null instead."
            ),
            "verdict": (
                "PASS - observed lies outside the permutation null"
                if p_perm < 0.05
                else "observed is inside the permutation null (no directional information)"
            ),
        },
    }
    log.info(f"  randomized_sides FPR {rand_rate:.3f} ({controls['randomized_sides']['verdict']})")
    log.info(
        f"  permutation null mean={perm.mean() * 1e4:+.2f}bps "
        f"sd={perm_sd * 1e4:.2f}bps observed={observed_stat * 1e4:+.2f}bps z={z:+.1f}"
    )

    log.info("positive control")
    edge = args.edge_bps / 10_000.0
    injected = trades.copy()
    injected["ret_gross"] = injected["ret_gross"] + edge
    pos = _cross_ci(injected, cfg)

    # Same edge added to random-side trades: it must still be found.
    shuffled = randomize_sides(trades, rng)
    shuffled["ret_gross"] = shuffled["ret_gross"] + edge
    pos_shuffled = _cross_ci(shuffled, cfg)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_trades": int(len(trades)),
        "spec": spec.describe(),
        "negative_controls": controls,
        "positive_control": {
            "injected_edge_bps": args.edge_bps,
            "detected": _significant(pos),
            "measured_mean": pos["stat"],
            "ci": [pos["ci_low"], pos["ci_high"]],
            "verdict": "PASS" if _significant(pos) else "FAIL - harness is blind to a real edge",
        },
        "positive_control_on_shuffled": {
            "detected": _significant(pos_shuffled),
            "measured_mean": pos_shuffled["stat"],
        },
        "observed_reference": {
            "mean_return_gross": observed["stat"],
            "ci": [observed["ci_low"], observed["ci_high"]],
        },
    }
    report["passed"] = (
        controls["randomized_sides"]["verdict"] == "PASS"
        and not controls["randomized_sides"]["degenerate"]
        and report["positive_control"]["verdict"] == "PASS"
    )
    out = paths.REPORTS_DIR / "harness_controls.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
