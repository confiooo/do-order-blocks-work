"""``hawk-research``: download data, run stages, verify results.

Global options come before the command::

    hawk-research --data-dir data --workers 8 run all
    hawk-research --set evaluation.holdout_fraction=0.25 run s12
    hawk-research download --universe round1 --intervals 15m
    hawk-research verify

The data directory and configuration are fixed for the process before any stage
module is imported, because stage modules read them at import time.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger("hawk_research")


def _setup(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.data_dir:
        os.environ["HAWK_RESEARCH_DATA"] = str(Path(args.data_dir).resolve())
    if args.config_dir:
        os.environ["HAWK_RESEARCH_CONFIG"] = str(Path(args.config_dir).resolve())
    from hawk_research import config as research_config

    cfg_file = Path(args.config) if args.config else None
    if cfg_file is None:
        default = Path(os.environ.get("HAWK_RESEARCH_CONFIG") or "config") / "research.yaml"
        cfg_file = default if default.exists() else None
    research_config.activate(research_config.load(cfg_file, args.set))


def _call(fn: Any, argv: List[str]) -> int:
    params = inspect.signature(fn).parameters
    if "argv" in params:
        rc = fn(argv)
    else:
        if argv:
            raise SystemExit(f"{fn.__module__}.{fn.__name__} takes no extra arguments, got {argv}")
        rc = fn()
    return int(rc) if isinstance(rc, int) else 0


def _stamp(report: Path, prov: Dict[str, Any]) -> None:
    """Attach provenance to a stage's JSON report (``_provenance`` key)."""
    if report.suffix != ".json" or not report.exists():
        return
    try:
        doc = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(doc, dict):
        doc["_provenance"] = prov
        report.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")


def _stage_table(args: argparse.Namespace) -> Any:
    from hawk_research.stages import registry

    return args.stage_table if args.stage_table is not None else registry.STAGES


def cmd_stages(args: argparse.Namespace) -> int:
    print(f"{'stage':<17}{'group':<15}{'depends on':<26}notes")
    for s in _stage_table(args):
        notes = []
        if s.network:
            notes.append("network")
        if s.live_snapshot:
            notes.append("live zones (snapshot/)")
        print(
            f"{s.name:<17}{s.group:<15}{','.join(s.deps) or '-':<26}{s.summary}"
            + (f" [{'; '.join(notes)}]" if notes else "")
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from hawk_research import provenance
    from hawk_research.data import paths
    from hawk_research.stages.registry import resolve

    if args.require_clean:
        provenance.require_clean()
    stages = resolve(args.stages, include_deps=not args.no_deps, stages=_stage_table(args))
    if args.skip_done:
        stages = [s for s in stages if not (s.report and (paths.RESEARCH_DIR / s.report).exists())]
    log.info("plan: %s", " -> ".join(s.name for s in stages) or "(nothing to do)")
    run_dir = paths.RESEARCH_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if stage.live_snapshot and not paths.has_live_snapshot():
            log.warning(
                "skip %s: needs the live zones at %s",
                stage.name,
                paths.LIVE_ZONES_EXTRACT,
            )
            continue
        argv = [*stage.argv]
        if args.workers is not None and stage.workers:
            argv += ["--workers", str(args.workers)]
        if args.extra:
            if len(stages) != 1:
                raise SystemExit("arguments after '--' need exactly one stage (use --no-deps)")
            argv += args.extra
        log.info("== %s: %s %s", stage.name, stage.target, " ".join(argv))
        t0 = time.time()
        rc = _call(stage.load(), argv)
        elapsed = round(time.time() - t0, 1)
        prov = provenance.record()
        prov.update({"stage": stage.name, "argv": argv, "elapsed_s": elapsed, "exit_code": rc})
        (run_dir / f"{stage.name}.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
        # The resolved configuration on its own, reusable as `--config runs/<stage>.config.yaml`.
        (run_dir / f"{stage.name}.config.yaml").write_text(
            yaml.safe_dump(prov["config"], sort_keys=False), encoding="utf-8"
        )
        if stage.report:
            _stamp(paths.RESEARCH_DIR / stage.report, prov)
        log.info("== %s done in %.1fs (exit %d)", stage.name, elapsed, rc)
        if rc != 0:
            log.error("%s failed; stopping", stage.name)
            return rc
    return 0


def _date_ms(text: str) -> int:
    """``YYYY-MM-DD`` (UTC midnight) as epoch milliseconds."""
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def cmd_download(args: argparse.Namespace) -> int:
    from hawk_research.data import download, paths

    jobs = download.plan(
        universes=args.universe or download.DEFAULT_UNIVERSES,
        symbols=args.symbols,
        intervals=args.intervals,
        include_btc_1h=args.btc_1h,
    )
    start_ms, end_ms = paths.HISTORY_START_MS, paths.SNAPSHOT_END_MS
    if args.since or args.until:
        if args.since:
            start_ms = max(start_ms, _date_ms(args.since))
        if args.until:
            end_ms = min(end_ms, _date_ms(args.until))
        if start_ms >= end_ms:
            log.error("empty window: --since must be before --until, inside 2024-09-01..2026-09-17")
            return 2
        log.warning(
            "custom window: files are not compared with the manifest and results will not match "
            "the published ones (fine for trying the pipeline; use the full window to reproduce)"
        )
    log.info("%d files to check/download (%s)", len(jobs), "offline" if args.offline else "online")
    out = download.run(
        jobs,
        workers=args.workers or 8,
        force=args.force,
        offline=args.offline,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    t = out["totals"]
    print(json.dumps(t, indent=2))
    if t["manifest_mismatch"]:
        log.warning(
            "%d files differ from the published manifest; results may not reproduce exactly. "
            "Details: data/ohlcv/coverage_manifest.json",
            len(t["manifest_mismatch"]),
        )
    if args.pit_universe and not args.offline:
        from hawk_research.stages import s2b_pit_universe

        s2b_pit_universe.main([])
    return 1 if t["errors"] and args.strict else 0


def cmd_verify(args: argparse.Namespace) -> int:
    from hawk_research import verify

    if args.write:
        print(verify.write_committed(Path(args.results), metrics=args.metrics))
        return 0
    report = verify.run(Path(args.results), only=args.only, metrics=args.metrics)
    print(verify.format_table(report))
    return 0 if report["ok"] else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from hawk_research import compare

    rows = compare.compare_dirs(Path(args.old), Path(args.new), files=args.files, rtol=args.rtol)
    print(compare.format_rows(rows))
    return 0 if all(r["status"] == "SAME" for r in rows) else 1


def cmd_verify_data(_args: argparse.Namespace) -> int:
    from hawk_research.data import manifest

    out = manifest.check_local()
    print(json.dumps({k: (len(v) if k == "matched" else v) for k, v in out.items()}, indent=2, default=str))
    return 0 if not out["mismatched"] else 1


def cmd_plots(args: argparse.Namespace) -> int:
    from hawk_research import plots

    written = plots.main(Path(args.results))
    for p in written:
        print(p)
    return 0 if written else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    from hawk_research import smoke

    return smoke.main(Path(args.workdir) if args.workdir else None, update_expected=args.update_expected)


def build_parser(prog: str = "hawk-research") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", help="data directory (default ./data, or $HAWK_RESEARCH_DATA)")
    p.add_argument("--config-dir", help="config directory (default ./config, or $HAWK_RESEARCH_CONFIG)")
    p.add_argument("--config", help="research YAML (default config/research.yaml)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value; repeatable",
    )
    p.add_argument("--workers", type=int, default=None, help="process/thread pool size")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="list stages and their dependencies").set_defaults(func=cmd_stages)

    r = sub.add_parser("run", help="run stages (with their dependencies)")
    r.add_argument("stages", nargs="+", help="stage names, group names (round1, smc, ...) or 'all'")
    r.add_argument("--no-deps", action="store_true", help="run only the named stages")
    r.add_argument("--skip-done", action="store_true", help="skip stages whose main output exists")
    r.add_argument("--require-clean", action="store_true", help="refuse to run on a dirty git tree")
    r.set_defaults(func=cmd_run, extra=[])

    d = sub.add_parser("download", help="download OHLCV from the public Binance archive")
    d.add_argument("--universe", action="append", help="universe name from config/universe.json; repeatable")
    d.add_argument("--symbols", nargs="+", help="explicit symbols instead of a universe")
    d.add_argument("--intervals", nargs="+", default=["15m"], help="15m (default), 1m, 1h")
    d.add_argument("--btc-1h", action="store_true", help="also fetch BTCUSDT 1h bars (unused by every stage)")
    d.add_argument("--force", action="store_true", help="re-download files that already exist")
    d.add_argument("--offline", action="store_true", help="no network: only check local files")
    d.add_argument("--pit-universe", action="store_true", help="also build the point-in-time universe (s2b)")
    d.add_argument("--since", help="YYYY-MM-DD: download only from this date (default: 2024-09-01)")
    d.add_argument("--until", help="YYYY-MM-DD: download only up to this date (default: 2026-09-17)")
    d.add_argument("--strict", action="store_true", help="exit non-zero if any file failed")
    d.set_defaults(func=cmd_download)

    v = sub.add_parser("verify", help="compare regenerated results to committed results/")
    v.add_argument("--results", default="results", help="committed results directory")
    v.add_argument("--only", nargs="+", help="check only these result files")
    v.add_argument(
        "--write",
        action="store_true",
        help="(maintainers) overwrite results/headline_metrics.json from this run",
    )
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("compare", help="compare every output of two data directories (regression check)")
    c.add_argument("old", help="data directory of the reference run")
    c.add_argument("new", help="data directory of the run to check")
    c.add_argument("--files", nargs="+", help="only these files (paths relative to the data directories)")
    c.add_argument("--rtol", type=float, default=1e-12, help="relative tolerance for floats (default 1e-12)")
    c.set_defaults(func=cmd_compare)

    sub.add_parser("verify-data", help="compare local OHLCV to config/data_manifest.json").set_defaults(
        func=cmd_verify_data
    )

    pl = sub.add_parser("plots", help="draw the figures in results/ from the stage reports")
    pl.add_argument("--results", default="results", help="output directory")
    pl.set_defaults(func=cmd_plots)

    s = sub.add_parser("smoke", help="5-minute end-to-end run on a synthetic dataset")
    s.add_argument("--workdir", help="where to build the synthetic data (default: a temp dir)")
    s.add_argument(
        "--update-expected", action="store_true", help="rewrite tests/fixtures/smoke_expected.json"
    )
    s.set_defaults(func=cmd_smoke)
    return p


def main(
    argv: Optional[List[str]] = None,
    stage_table: Any = None,
    metrics: Any = None,
    prog: str = "hawk-research",
) -> int:
    """Run the CLI. ``stage_table`` and ``metrics`` replace the main pipeline's (``python -m experiments``)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after a bare "--" is passed through to a single stage (``run s8 -- --seed 5``).
    extra: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    args = build_parser(prog).parse_args(argv)
    args.stage_table, args.metrics = stage_table, metrics
    if extra:
        if args.command != "run":
            raise SystemExit("extra arguments after '--' are only accepted by `run`")
        args.extra = extra
    _setup(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
