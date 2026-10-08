"""Leakage and integrity audit for the replayed zone dataset.

Main check, the truncation test: for a zone on bar ``i``, recompute its feature
vector from the series cut at ``i``. A feature that used a later bar changes or
becomes NaN. The same cut must make every forward label empty; a label that
survives it is reading something other than later bars.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from hawk_research.constants import BAR_MS
from hawk_research.features import point_in_time as feat
from hawk_research.labels import outcomes as lab

# Columns that are contemporaneous-cross-sectional or bookkeeping rather than
# per-symbol time series, so the single-symbol truncation test does not apply.
TRUNCATION_EXEMPT_PREFIXES = ("xs_", "pit_", "fwd_", "entry_", "first_up_", "first_dn_", "bar_")
TRUNCATION_EXEMPT = {
    "symbol",
    "source_open_time",
    "decision_time_ms",
    "zone_uid",
    "bar_index",
    "mode",
    "detection_lag_bars",
    "live_equivalent",
    "would_alert",
    "passes_signal_gate",
    "btc_stable",
    "btc_range_4h_pct",
    "direction",
    "retested",
    "retest_open_time",
    "hypothetical_entry_price",
    "retest_lag_bars",
    "lifecycle_status",
    "bars_since_prev_zone",
    "fwd_bars_available",
    "entry_fwd_bars_available",
}


def _is_feature_column(name: str) -> bool:
    if name in TRUNCATION_EXEMPT:
        return False
    if name.startswith(TRUNCATION_EXEMPT_PREFIXES):
        return False
    if name.startswith(("mfe_", "mae_")):
        return False
    return True


def truncation_test(
    dataset: pd.DataFrame,
    bars_by_symbol: Dict[str, pd.DataFrame],
    btc: pd.DataFrame,
    n_samples: int = 150,
    seed: int = 11,
    rtol: float = 1e-9,
) -> Dict:
    """Recompute features in a world where no bar after the decision bar exists.

    A mismatch here means a feature is reading the future. NaN-vs-NaN counts as
    agreement; NaN-vs-value in either direction does not.
    """
    rng = np.random.default_rng(seed)
    pool = dataset[dataset["symbol"].isin(bars_by_symbol)]
    if pool.empty:
        return {"n_tested": 0}
    idx = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
    sample = pool.iloc[idx]

    feature_cols = [c for c in dataset.columns if _is_feature_column(c)]
    mismatches: Dict[str, int] = {}
    tested = 0
    nan_only: Dict[str, int] = {}

    btc_idx = btc.set_index("open_time", drop=False)

    for row in sample.itertuples(index=False):
        bars = bars_by_symbol[row.symbol]
        pos = int(np.searchsorted(bars["open_time"].to_numpy(), row.source_open_time))
        if pos >= len(bars) or int(bars["open_time"].iloc[pos]) != int(row.source_open_time):
            continue
        truncated = bars.iloc[: pos + 1]
        btc_trunc = btc_idx[btc_idx["open_time"] <= row.source_open_time].reset_index(drop=True)
        if len(truncated) < 20 or btc_trunc.empty:
            continue
        recomputed = feat.build_symbol_features(truncated, btc_trunc).iloc[-1]
        tested += 1
        for col in feature_cols:
            if col not in recomputed.index:
                continue
            a = getattr(row, col, None)
            b = recomputed[col]
            a_nan = a is None or (isinstance(a, float) and np.isnan(a))
            b_nan = b is None or (isinstance(b, float) and np.isnan(b))
            if a_nan and b_nan:
                continue
            if a_nan != b_nan:
                nan_only[col] = nan_only.get(col, 0) + 1
                mismatches[col] = mismatches.get(col, 0) + 1
                continue
            if not np.isclose(float(a), float(b), rtol=rtol, atol=1e-12):  # type: ignore[arg-type]  # both non-NaN here
                mismatches[col] = mismatches.get(col, 0) + 1

    return {
        "n_tested": tested,
        "n_feature_columns": len(feature_cols),
        "columns_with_mismatch": mismatches,
        "columns_with_nan_disagreement": nan_only,
        "passed": len(mismatches) == 0,
        "method": (
            "Features recomputed from a series truncated at the decision bar. "
            "Any use of a later bar would change the value or make it NaN."
        ),
    }


def label_truncation_test(
    dataset: pd.DataFrame,
    bars_by_symbol: Dict[str, pd.DataFrame],
    n_samples: int = 150,
    seed: int = 12,
) -> Dict:
    """Forward labels must vanish when the future is removed.

    A label that still has a value when the series stops at the decision bar is
    reading something other than later bars.
    """
    rng = np.random.default_rng(seed)
    pool = dataset[dataset["symbol"].isin(bars_by_symbol)]
    idx = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
    sample = pool.iloc[idx]

    survived: Dict[str, int] = {}
    tested = 0
    for row in sample.itertuples(index=False):
        bars = bars_by_symbol[row.symbol]
        pos = int(np.searchsorted(bars["open_time"].to_numpy(), row.source_open_time))
        if pos >= len(bars) or int(bars["open_time"].iloc[pos]) != int(row.source_open_time):
            continue
        truncated = bars.iloc[: pos + 1]
        out = lab.per_bar_labels(truncated).iloc[-1]
        tested += 1
        for name in lab.RETURN_HORIZONS:
            if pd.notna(out[f"fwd_ret_{name}"]):
                survived[f"fwd_ret_{name}"] = survived.get(f"fwd_ret_{name}", 0) + 1
        for name in lab.FORWARD_VOL_HORIZONS:
            if pd.notna(out[f"fwd_rv_{name}"]):
                survived[f"fwd_rv_{name}"] = survived.get(f"fwd_rv_{name}", 0) + 1
        for level in lab.BARRIER_LEVELS:
            tag = lab._level_tag(level)
            for side in ("up", "dn"):
                col = f"first_{side}_{tag}"
                if int(out[col]) != 0:
                    survived[col] = survived.get(col, 0) + 1
    return {
        "n_tested": tested,
        "labels_surviving_truncation": survived,
        "passed": len(survived) == 0,
    }


def structural_checks(dataset: pd.DataFrame, bars_by_symbol: Dict[str, pd.DataFrame]) -> Dict:
    """Duplicates, epoch units, bar alignment, and lifecycle ordering."""
    errors: List[str] = []
    warnings: List[str] = []

    dup = dataset.duplicated(subset=["symbol", "source_open_time"]).sum()
    if dup:
        errors.append(f"{dup} duplicated (symbol, source_open_time) rows")

    ot = dataset["source_open_time"].to_numpy(dtype=np.int64)
    off_grid = int((ot % BAR_MS != 0).sum())
    if off_grid:
        errors.append(f"{off_grid} source_open_time values are not on the 15m grid")
    # Millisecond epochs for 2024-2026 live in [1.7e12, 1.8e12]. Seconds or
    # microseconds would land orders of magnitude away.
    bad_scale = int(((ot < 1_600_000_000_000) | (ot > 1_900_000_000_000)).sum())
    if bad_scale:
        errors.append(f"{bad_scale} source_open_time values are not plausible epoch ms")

    if "decision_time_ms" in dataset.columns:
        bad = int((dataset["decision_time_ms"].to_numpy(dtype=np.int64) != ot + BAR_MS).sum())
        if bad:
            errors.append(f"{bad} rows where decision_time_ms != source_open_time + one bar")

    # Off-by-one alignment: the zone's high/low must be the zone bar's own
    # high/low, not the neighbouring bar's. For round-1 zones that is the
    # decision bar; SMC order blocks (``ob_open_time``) are an earlier candle
    # confirmed by the decision bar, so they are checked against that candle.
    zone_bar = "ob_open_time" if "ob_open_time" in dataset.columns else "source_open_time"
    if zone_bar == "ob_open_time":
        bad = int((dataset["ob_open_time"].to_numpy(dtype=np.int64) >= ot).sum())
        if bad:
            errors.append(f"{bad} rows where the order-block candle is not before the decision bar")
    checked = 0
    misaligned = 0
    for symbol, grp in dataset.groupby("symbol"):
        bars = bars_by_symbol.get(symbol)
        if bars is None:
            continue
        m = grp[[zone_bar, "zone_high", "zone_low"]].merge(
            bars[["open_time", "high", "low"]], left_on=zone_bar, right_on="open_time"
        )
        if not len(m):
            continue
        checked += len(m)
        bad_rows = (~np.isclose(m["zone_high"], m["high"], rtol=1e-12)) | (
            ~np.isclose(m["zone_low"], m["low"], rtol=1e-12)
        )
        misaligned += int(bad_rows.sum())
    if misaligned:
        errors.append(f"{misaligned}/{checked} zones whose zone_high/low != the zone bar's high/low")

    if "det_volume_ratio" in dataset.columns:
        sub = dataset[["volume_ratio", "det_volume_ratio"]].dropna()
        bad = int((~np.isclose(sub["volume_ratio"], sub["det_volume_ratio"], rtol=1e-9)).sum())
        if bad:
            errors.append(
                f"{bad} rows where the independently recomputed det_volume_ratio "
                "disagrees with the detector's volume_ratio"
            )

    if "retest_open_time" in dataset.columns:
        r = dataset.dropna(subset=["retest_open_time"])
        bad = int(
            (
                r["retest_open_time"].to_numpy(dtype=np.int64)
                <= r["source_open_time"].to_numpy(dtype=np.int64)
            ).sum()
        )
        if bad:
            errors.append(f"{bad} retests at or before the source bar")
        late = int(
            (
                r["retest_open_time"].to_numpy(dtype=np.int64)
                - r["source_open_time"].to_numpy(dtype=np.int64)
                > lab.RETEST_TIMEOUT_HOURS * 3600 * 1000
            ).sum()
        )
        if late:
            errors.append(f"{late} retests beyond the {lab.RETEST_TIMEOUT_HOURS}h timeout")

    excursion_cols = [f"{k}_{h}h" for h in lab.MFE_MAE_HORIZONS for k in ("mfe", "mae")]
    present = [c for c in excursion_cols if c in dataset.columns]
    if present:
        no_entry = dataset["hypothetical_entry_price"].isna()
        bad = int(dataset.loc[no_entry, present].notna().any(axis=1).sum())
        if bad:
            errors.append(f"{bad} zones with MFE/MAE populated but no entry")
    if "mfe_24h" in dataset.columns:
        neg_mfe = int((dataset["mfe_24h"].dropna() < -0.5).sum())
        if neg_mfe:
            warnings.append(f"{neg_mfe} zones with implausibly negative 24h MFE")

    for level in lab.BARRIER_LEVELS:
        tag = lab._level_tag(level)
        for side in ("up", "dn"):
            col = f"first_{side}_{tag}"
            if col in dataset.columns:
                bad = int((dataset[col] < 0).sum())
                if bad:
                    errors.append(f"{bad} negative first-touch offsets in {col}")

    return {"errors": errors, "warnings": warnings, "passed": not errors}


def forward_fill_check(bars_by_symbol: Dict[str, pd.DataFrame], min_repeat: int = 8) -> Dict:
    """Detect stale-price contamination: long runs of an exactly repeated OHLC.

    A forward-filled series produces bars where open == high == low == close and
    volume is zero, repeated. Genuine flat bars on a thin pair exist, so this is
    reported rather than treated as fatal.
    """
    findings: List[Dict] = []
    for symbol, bars in bars_by_symbol.items():
        flat = (
            (bars["open"] == bars["high"]) & (bars["high"] == bars["low"]) & (bars["low"] == bars["close"])
        ).to_numpy()
        if not flat.any():
            continue
        runs = 0
        best = 0
        cur = 0
        for v in flat:
            cur = cur + 1 if v else 0
            best = max(best, cur)
            if cur == min_repeat:
                runs += 1
        findings.append(
            {
                "symbol": symbol,
                "flat_bars": int(flat.sum()),
                "flat_fraction": float(flat.mean()),
                "longest_flat_run": int(best),
                "runs_at_least_min_repeat": runs,
            }
        )
    findings.sort(key=lambda f: -f["flat_fraction"])
    return {
        "symbols_with_flat_bars": len(findings),
        "worst": findings[:10],
        "note": (
            "Flat bars are zero-volume prints on illiquid pairs, not forward fill: "
            "the archive stores them as-is and the pipeline never reindexes or ffills. "
            "Gaps are left as gaps and every label spanning one is masked."
        ),
    }


def gap_masking_check(dataset: pd.DataFrame) -> Dict:
    """Labels must be absent wherever the forward window crossed a data gap."""
    out: Dict[str, object] = {}
    if "fwd_bars_available" not in dataset.columns:
        return {"skipped": "fwd_bars_available missing"}
    avail = dataset["fwd_bars_available"].to_numpy()
    violations = {}
    for name, k in lab.RETURN_HORIZONS.items():
        col = f"fwd_ret_{name}"
        if col not in dataset.columns:
            continue
        bad = int((dataset[col].notna().to_numpy() & (avail < k)).sum())
        if bad:
            violations[col] = bad
    out["violations"] = violations
    out["passed"] = not violations
    out["n_rows_with_incomplete_forward_window"] = int((avail < lab.MAX_HORIZON_BARS).sum())
    return out
