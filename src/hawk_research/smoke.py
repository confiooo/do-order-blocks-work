"""Five-minute end-to-end run on a deterministic synthetic market.

``hawk-research smoke`` builds a small 15m dataset (BTC plus four coins over six
months) from a seeded generator, runs the pipeline stages on it in a
temporary data directory, and compares a set of output fingerprints with
``tests/fixtures/smoke_expected.json``.

The generator plants a textbook order-block shape (a down candle, an impulse that
breaks the previous swing high, a retest) every few days, so both detectors have
something to find. The numbers say nothing about real markets; the run checks
that every stage works, feeds the next, and gives the same answer everywhere.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS

log = logging.getLogger(__name__)

SEED = 20240901
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
START_MS = 1735689600000  # 2025-01-01T00:00:00Z
N_BARS = 180 * 96  # 180 days of 15m bars: enough for walk-forward folds (90d train + 30d test)
STAGES = ("s3", "s4", "s5", "s6", "s8", "s10", "s3s", "s5_smc", "s6_smc", "s12")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = REPO_ROOT / "tests" / "fixtures" / "smoke_expected.json"


def synthetic_bars(symbol: str, seed: int, start_price: float) -> pd.DataFrame:
    """Random walk with volatility clustering and planted order-block shapes."""
    rng = np.random.default_rng(seed)
    vol = np.empty(N_BARS)
    v = 0.003
    for i in range(N_BARS):
        v = 0.0025 + 0.92 * (v - 0.0025) + 0.0006 * abs(rng.standard_normal())
        vol[i] = v
    rets = rng.standard_normal(N_BARS) * vol
    # Every ~3 days: a red candle, then a 6-bar impulse up, then a pullback.
    for k in range(200, N_BARS - 40, 290):
        rets[k] = -4 * vol[k]
        rets[k + 1 : k + 7] = 2.5 * vol[k + 1 : k + 7]
        rets[k + 7 : k + 14] = -1.2 * vol[k + 7 : k + 14]
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.standard_normal((2, N_BARS))) * vol * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = rng.lognormal(mean=6.0, sigma=0.6, size=N_BARS)
    volume[np.abs(rets) > 2 * vol] *= 3.0
    ot = START_MS + np.arange(N_BARS, dtype=np.int64) * BAR_MS
    return pd.DataFrame(
        {
            "open_time": ot,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": (volume / 3).astype(np.int64),
            "taker_buy_quote": volume * close * 0.5,
        }
    )


def build_dataset(data_dir: Path, config_dir: Path) -> None:
    """Write the synthetic bars and a matching config directory."""
    (data_dir / "ohlcv" / "15m").mkdir(parents=True, exist_ok=True)
    names = ("BTCUSDT",) + SYMBOLS
    for i, sym in enumerate(names):
        synthetic_bars(sym, SEED + i, 100.0 * (i + 1)).to_parquet(
            data_dir / "ohlcv" / "15m" / f"{sym}.parquet", index=False
        )
    config_dir.mkdir(parents=True, exist_ok=True)
    universe = {
        "round1": {"description": "synthetic smoke universe", "symbols": list(names)},
        "round1_zone_symbols": {"description": "synthetic smoke universe", "symbols": list(names)},
    }
    (config_dir / "universe.json").write_text(json.dumps(universe, indent=2), encoding="utf-8")


def fingerprint(data_dir: Path) -> Dict[str, Any]:
    """Small, stable numbers that change if any stage changes its behaviour."""

    def jload(rel: str) -> Dict[str, Any]:
        return json.loads((data_dir / rel).read_text(encoding="utf-8"))

    z = pd.read_parquet(data_dir / "zones" / "replay_zones.parquet")
    ds = pd.read_parquet(data_dir / "features" / "zone_dataset.parquet")
    smc = pd.read_parquet(data_dir / "zones" / "replay_zones_smc.parquet")
    nb = jload("reports/null_baselines.json")
    ev = jload("reports/smc_eval.json")
    key = "all|full|retest|time|-/-|mexc_futures_taker"
    return {
        "round1_zones": int(len(z)),
        "round1_bullish": int((z["direction"] == "bullish").sum()),
        "zone_dataset_rows": int(len(ds)),
        "zone_dataset_retested": int(ds["retested"].fillna(False).astype(bool).sum()),
        "leakage_audit_passed": bool(jload("reports/leakage_audit.json")["passed"]),
        "round1_direction_4h_gross": round(
            float(nb["directional"]["direction"]["4h|mexc_futures_taker"]["gross_mean"]), 10
        ),
        "round1_folds": int(nb["walk_forward"]["summary"]["n_folds"]),
        "bearish_strategies": len(jload("reports/bearish_only.json")["strategies"]),
        "smc_zones": int(len(smc)),
        "smc_with_fvg": int(smc["has_fvg"].sum()),
        "smc_leakage_audit_passed": bool(jload("reports/leakage_audit_smc.json")["passed"]),
        "smc_retest_net_mean": round(float(ev["strategies"][key]["mean_ret_net"]), 10)
        if ev["strategies"][key]["mean_ret_net"] is not None
        else None,
    }


def main(workdir: Optional[Path] = None, update_expected: bool = False) -> int:
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="hawk-smoke-"))
    data_dir, config_dir = tmp / "data", tmp / "config"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    build_dataset(data_dir, config_dir)
    log.info("smoke dataset in %s", tmp)
    env = dict(os.environ, HAWK_RESEARCH_DATA=str(data_dir), HAWK_RESEARCH_CONFIG=str(config_dir))
    # A fresh process, so the data directory is fixed before any stage is imported.
    # Fewer bootstrap resamples keep the run short; the fingerprint holds no interval bounds.
    cmd = [
        sys.executable,
        "-m",
        "hawk_research.cli",
        "--workers",
        "2",
        "--log-level",
        "warning",
        "--set",
        "evaluation.bootstrap_samples=100",
        "run",
        *STAGES,
        "--no-deps",
    ]
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc != 0:
        log.error("pipeline failed (exit %d)", rc)
        return rc
    got = fingerprint(data_dir)
    print(json.dumps(got, indent=2))
    if update_expected:
        EXPECTED.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED.write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
        log.warning("wrote %s", EXPECTED)
        return 0
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    bad = compare(got, expected)
    for line in bad:
        print("MISMATCH", line)
    print("smoke:", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


def compare(got: Dict[str, Any], expected: Dict[str, Any], tol: float = 1e-9) -> List[str]:
    bad = []
    for k, v in expected.items():
        g = got.get(k)
        if isinstance(v, float) and isinstance(g, float):
            if abs(g - v) > tol:
                bad.append(f"{k}: expected {v}, got {g}")
        elif g != v:
            bad.append(f"{k}: expected {v}, got {g}")
    return bad
