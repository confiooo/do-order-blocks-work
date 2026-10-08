"""CLI wiring, the results comparison and the OHLCV loader."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from hawk_research import cli, verify
from hawk_research.data import download, loader, paths


def _cli(*args, data_dir):
    env = dict(os.environ, HAWK_RESEARCH_DATA=str(data_dir))
    return subprocess.run(
        [sys.executable, "-m", "hawk_research.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_lists_stages_and_rejects_unknown_ones(tmp_path):
    out = _cli("stages", data_dir=tmp_path)
    assert out.returncode == 0 and "s8" in out.stdout and "s12" in out.stdout
    bad = _cli("run", "s99", data_dir=tmp_path)
    assert bad.returncode != 0 and "s99" in bad.stderr


def test_extra_arguments_need_a_single_stage():
    with pytest.raises(SystemExit):
        cli.main(["stages", "--", "--seed", "3"])


DUMMY_STAGE = """
import json, sys
from hawk_research.stages import registry
from hawk_research.data import paths

def dummy(argv):
    (paths.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "dummy.json").write_text(json.dumps({"argv": argv}))
    return 0

sys.modules["dummy_stage"] = sys.modules[__name__]
stage = registry.Stage("dummy", "dummy_stage:dummy", "test", "test stage", report="reports/dummy.json")
registry.STAGES = registry.STAGES + (stage,)
registry.BY_NAME["dummy"] = stage
from hawk_research import cli
raise SystemExit(cli.main(sys.argv[1:]))
"""


def test_run_writes_provenance_and_resolved_config(tmp_path):
    env = dict(os.environ, HAWK_RESEARCH_DATA=str(tmp_path))
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            DUMMY_STAGE,
            "--set",
            "evaluation.holdout_fraction=0.25",
            "run",
            "dummy",
            "--no-deps",
            "--",
            "--seed",
            "5",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    prov = json.loads((tmp_path / "runs" / "dummy.json").read_text())
    assert prov["config"]["evaluation"]["holdout_fraction"] == 0.25
    assert prov["argv"] == ["--seed", "5"] and prov["exit_code"] == 0
    assert {"git_commit", "git_dirty", "config_hash", "command", "packages", "data_manifest_sha256"} <= set(
        prov
    )
    assert "holdout_fraction: 0.25" in (tmp_path / "runs" / "dummy.config.yaml").read_text()
    report = json.loads((tmp_path / "reports" / "dummy.json").read_text())
    assert report["argv"] == ["--seed", "5"] and report["_provenance"]["config_hash"] == prov["config_hash"]


def _write_report(root, value):
    (root / "reports").mkdir(parents=True, exist_ok=True)
    doc = {"pbo": {"pbo": value}, "deflated_sharpe": {"n_trials": 72, "dsr": 0.01}}
    (root / "reports" / "null_baselines.json").write_text(json.dumps(doc))


def test_verify_compares_within_tolerance(tmp_path, monkeypatch):
    data, results = tmp_path / "data", tmp_path / "results"
    _write_report(data, 0.5)
    monkeypatch.setattr(paths, "RESEARCH_DIR", data)
    verify.write_committed(results)
    committed = json.loads((results / "headline_metrics.json").read_text())["metrics"]
    assert committed["round1 PBO"] == 0.5 and committed["round1 DSR trials"] == 72

    report = verify.run(results)
    rows = {r["label"]: r for r in report["rows"]}
    assert report["ok"] and rows["round1 PBO"]["status"] == "OK"
    assert rows["smc PBO"]["status"] == "MISSING"  # not regenerated: reported, not failed

    _write_report(data, 0.5 + 5e-5)  # inside the 1e-4 statistic tolerance
    assert verify.run(results)["ok"]
    _write_report(data, 0.6)
    bad = verify.run(results)
    assert not bad["ok"] and "DIFF" in verify.format_table(bad)


def test_loader_reads_sorts_and_reports_missing_files():
    path = paths.ohlcv_path("ZZZUSDT", "15m")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"open_time": [900_000, 0], "close": [2.0, 1.0]}).to_parquet(path, index=False)
    try:
        assert loader.load_ohlcv("ZZZUSDT", sort=True)["close"].tolist() == [1.0, 2.0]
        assert loader.load_ohlcv("ZZZUSDT", columns=["close"]).columns.tolist() == ["close"]
        assert list(loader.load_many(["ZZZUSDT", "NOPEUSDT"])) == ["ZZZUSDT"]
        with pytest.raises(loader.MissingOHLCV):
            loader.load_ohlcv("NOPEUSDT")
        assert loader.load_ohlcv("NOPEUSDT", missing_ok=True).empty
    finally:
        path.unlink()


def test_download_plan_uses_explicit_universes():
    jobs = download.plan()
    assert ("BTCUSDT", "1h") not in jobs  # unused by every stage; opt-in only
    assert len({s for s, iv in jobs if iv == "15m"}) == 85
    assert download.plan(symbols=["ETHUSDT"]) == [("ETHUSDT", "15m")]
    assert ("BTCUSDT", "1h") in download.plan(symbols=["ETHUSDT"], include_btc_1h=True)


def test_paths_rel_never_leaks_absolute_paths():
    assert paths.rel(paths.REPORTS_DIR / "x.json") == "reports/x.json"
    assert paths.rel(paths.CONFIG_DIR / "costs.json") == "config/costs.json"
    assert paths.rel(paths.Path("/somewhere/else/file.parquet")) == "file.parquet"


def test_custom_window_download_is_not_compared_and_not_taken_as_complete(tmp_path, monkeypatch):
    from hawk_research.constants import BAR_MS

    monkeypatch.setattr(paths, "OHLCV_DIR", tmp_path)
    start = paths.SNAPSHOT_END_MS - 9 * BAR_MS
    t = start + BAR_MS * pd.RangeIndex(10).to_numpy()
    bars = pd.DataFrame(
        {
            "open_time": t,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 5.0,
            "close_time": t + BAR_MS - 1,
        }
    )

    class _Cov:
        def as_dict(self):
            return {}

    monkeypatch.setattr(download.backfill, "download_symbol", lambda *a, **k: (bars, _Cov()))
    pub = {"ZZZUSDT|15m": {"symbol": "ZZZUSDT", "interval": "15m", "first_open_time": paths.HISTORY_START_MS}}
    rec = download.fetch_one("ZZZUSDT", "15m", start, paths.SNAPSHOT_END_MS, False, pub)
    assert rec["rows"] == 10 and rec["manifest"] == "not-compared (custom window)"
    # The same file is complete for the short window but not for the full one.
    assert download._complete("ZZZUSDT", "15m", start, paths.SNAPSHOT_END_MS, pub["ZZZUSDT|15m"])
    assert not download._complete(
        "ZZZUSDT", "15m", paths.HISTORY_START_MS, paths.SNAPSHOT_END_MS, pub["ZZZUSDT|15m"]
    )


def test_date_option_parsing():
    assert cli._date_ms("2024-09-01") == paths.HISTORY_START_MS


def test_live_zones_extract_is_shipped_and_loads():
    from hawk_research.data import live_snapshot

    assert paths.LIVE_ZONES_EXTRACT.exists() and paths.has_live_snapshot()
    zones = live_snapshot.load_zones()
    assert len(zones) == 1543 and zones[["symbol", "source_open_time"]].duplicated().sum() == 0
    # market data only: none of the live system's user, alert or order tables
    assert {"symbol", "source_open_time", "zone_high", "zone_low", "status"} <= set(zones.columns)
    assert not {"chat_id", "user_id", "details", "order_id"} & set(zones.columns)
