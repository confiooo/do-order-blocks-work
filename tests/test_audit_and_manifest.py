"""The leakage audit's structural checks catch each planted fault; the data manifest check
classifies local files correctly; offline download reports missing files without the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hawk_research.audit import leakage
from hawk_research.constants import BAR_MS
from hawk_research.data import download, manifest, paths
from hawk_research.labels import outcomes as lab

T0 = 1_760_000_400_000  # 2025-10-09, on the 15m grid
HOUR = 3_600_000


def _bars(n=12):
    t = T0 + BAR_MS * np.arange(n, dtype=np.int64)
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open_time": t,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        }
    )


def _dataset(bars):
    rows = bars.iloc[[2, 5]]
    t = rows["open_time"].to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "symbol": "AAAUSDT",
            "source_open_time": t,
            "decision_time_ms": t + BAR_MS,
            "zone_high": rows["high"].to_numpy(),
            "zone_low": rows["low"].to_numpy(),
            "volume_ratio": [1.5, 2.0],
            "det_volume_ratio": [1.5, 2.0],
            "retest_open_time": [t[0] + 4 * BAR_MS, np.nan],
            "hypothetical_entry_price": [101.0, np.nan],
            "mfe_24h": [0.01, np.nan],
            "mae_24h": [0.005, np.nan],
            f"first_up_{lab._level_tag(lab.BARRIER_LEVELS[0])}": [3, 0],
        }
    )


def test_clean_dataset_passes_structural_checks():
    bars = _bars()
    out = leakage.structural_checks(_dataset(bars), {"AAAUSDT": bars})
    assert out == {"errors": [], "warnings": [], "passed": True}


@pytest.mark.parametrize(
    "plant, expected",
    [
        (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "duplicated (symbol, source_open_time)"),
        (lambda d: d.assign(source_open_time=d["source_open_time"] + 1), "not on the 15m grid"),
        (
            lambda d: d.assign(source_open_time=d["source_open_time"] // 1000 // 900 * 900),
            "not plausible epoch ms",
        ),
        (lambda d: d.assign(decision_time_ms=d["source_open_time"]), "decision_time_ms != source_open_time"),
        (lambda d: d.assign(zone_high=d["zone_high"] + 1.0), "zone_high/low != the zone bar"),
        (lambda d: d.assign(det_volume_ratio=[1.5, 2.5]), "det_volume_ratio"),
        (
            lambda d: d.assign(retest_open_time=[d["source_open_time"].iloc[0], np.nan]),
            "at or before the source bar",
        ),
        (
            lambda d: d.assign(retest_open_time=[d["source_open_time"].iloc[0] + 30 * HOUR, np.nan]),
            "beyond the",
        ),
        (lambda d: d.assign(mfe_24h=[0.01, 0.02]), "MFE/MAE populated but no entry"),
        (
            lambda d: d.assign(**{f"first_up_{lab._level_tag(lab.BARRIER_LEVELS[0])}": [3, -1]}),
            "negative first-touch offsets",
        ),
        (lambda d: d.assign(ob_open_time=d["source_open_time"]), "order-block candle is not before"),
    ],
)
def test_each_planted_fault_is_reported(plant, expected):
    bars = _bars()
    out = leakage.structural_checks(plant(_dataset(bars)), {"AAAUSDT": bars})
    assert not out["passed"]
    assert any(expected in e for e in out["errors"]), out["errors"]


def test_gap_masking_flags_a_label_across_a_gap():
    k = lab.RETURN_HORIZONS["4h"]
    ds = pd.DataFrame({"fwd_bars_available": [k, k - 1], "fwd_ret_4h": [0.01, 0.02]})
    out = leakage.gap_masking_check(ds)
    assert out["violations"] == {"fwd_ret_4h": 1} and not out["passed"]
    assert leakage.gap_masking_check(ds.assign(fwd_ret_4h=[0.01, np.nan]))["passed"]


def test_forward_fill_check_reports_flat_runs():
    bars = _bars(20)
    flat = bars.copy()
    flat.loc[5:14, ["open", "high", "low", "close"]] = 7.0
    out = leakage.forward_fill_check({"FLATUSDT": flat, "AAAUSDT": bars}, min_repeat=8)
    assert out["symbols_with_flat_bars"] == 1
    worst = out["worst"][0]
    assert (
        worst["symbol"],
        worst["flat_bars"],
        worst["longest_flat_run"],
        worst["runs_at_least_min_repeat"],
    ) == (
        "FLATUSDT",
        10,
        10,
        1,
    )


@pytest.fixture
def ohlcv_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OHLCV_DIR", tmp_path)
    (tmp_path / "15m").mkdir()
    return tmp_path


def test_manifest_check_classifies_matched_mismatched_and_missing(ohlcv_dir):
    good, changed = _bars(), _bars()
    changed.loc[3, "close"] += 0.01  # one revised bar
    good.to_parquet(ohlcv_dir / "15m" / "AAAUSDT.parquet", index=False)
    changed.to_parquet(ohlcv_dir / "15m" / "BBBUSDT.parquet", index=False)
    published = {
        "entries": [
            manifest.describe("AAAUSDT", "15m", good),
            manifest.describe("BBBUSDT", "15m", good),
            manifest.describe("CCCUSDT", "15m", good),
        ]
    }
    out = manifest.check_local(published=published)
    assert out["matched"] == ["AAAUSDT|15m"]
    assert out["missing"] == ["CCCUSDT|15m"]
    assert [m["key"] for m in out["mismatched"]] == ["BBBUSDT|15m"]
    assert [d["field"] for d in out["mismatched"][0]["diffs"]] == ["content_sha256"]


def test_offline_download_checks_local_files_only(ohlcv_dir, monkeypatch):
    bars = _bars()
    bars.to_parquet(ohlcv_dir / "15m" / "AAAUSDT.parquet", index=False)
    published = {"entries": [manifest.describe("AAAUSDT", "15m", bars)]}
    monkeypatch.setattr(download.manifest, "load_published", lambda: published)
    monkeypatch.setattr(paths, "DATA_MANIFEST", ohlcv_dir / "coverage_manifest.json")
    monkeypatch.setattr(
        download.backfill, "download_symbol", lambda *a, **k: pytest.fail("network used offline")
    )
    out = download.run([("AAAUSDT", "15m"), ("ZZZUSDT", "15m")], offline=True)
    t = out["totals"]
    assert (t["jobs"], t["ok"], t["manifest_match"]) == (2, 1, 1)
    assert t["errors"] == ["ZZZUSDT|15m: missing (offline)"]
    assert (ohlcv_dir / "coverage_manifest.json").exists()
