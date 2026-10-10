# Reproducing the results

Everything needed to rerun the study, change its settings and check it. For what the study
found, start with the [README](../README.md) or the two-page [overview](OVERVIEW.md).

## Full reproduction

```bash
hawk-research --data-dir data download                 # ~200 MB of 15m bars from Binance's public archive
hawk-research --data-dir data verify-data              # compare against config/data_manifest.json
hawk-research --data-dir data download --pit-universe  # point-in-time universe, for Family A
hawk-research --data-dir data run all                  # every offline stage in dependency order
hawk-research --data-dir data verify                   # compare headline numbers with results/
hawk-research --data-dir data plots                    # redraw the figures in results/
```

The side studies need their own inputs and run separately (`make experiments`):

```bash
python -m experiments --data-dir data run mn_ingest    # funding and 1h perpetual bars (network)
python -m experiments --data-dir data run all          # zone trades, daily fade, market neutral
python -m experiments --data-dir data verify           # their 11 headline numbers
```

Or `make download reproduce`. To try the real pipeline on less data first, download a few
symbols over a shorter window; the stages run on whatever is present, but the numbers will not
match the published ones:

```bash
hawk-research --data-dir data-small download --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT --since 2026-03-01
hawk-research --data-dir data-small run s3 s4 s5 s6
```

**Resources (measured).** The full offline run (`run all`) takes about 8 minutes on a 12-core laptop
CPU (Intel i5-12600HX, 16 threads, on mains power), and the side studies in `experiments/` about 7 more.
The longest stages are `s8_smc` and `s8` (under 2 min each) and `s1s5` (70 s); everything else takes
under 45 s. Each stage's time is in its `data/runs/<stage>.json`. Peak memory was 5.1 GB, in
`s5_smc` (resident memory summed over the main and worker processes; shared pages count once per
process, so this is an upper bound). Fewer workers use less memory.
Disk: about 1.4 GB (0.2 GB downloaded, about 1.2 GB generated).
`s3`, `s3r`, `s3s`, `s4`, `s5` and `s8` split work by symbol across worker processes
(default 8 workers; 12 for `s3r`); on a smaller machine pass fewer to all of them with
`hawk-research --data-dir data --workers 4 run all`. Every download is checked against content
hashes in `config/data_manifest.json`; if Binance has revised an archive file, `verify-data`
names it. The manifest also lists 1-minute bars (`download --intervals 1m`, 1.8 GB) that the
original project collected; no stage uses them. `verify` compares 40 headline numbers with
`results/headline_metrics.json`: counts must match exactly, means within 0.01 bps and interval
bounds within 1e-4.

Three stages check the replay against the live system it came from, using the live zones
table shipped in [`snapshot/`](../snapshot/README.md): `s3` (1,510 of 1,543 live zones reproduced
with every field identical), `s3b` (the cause of each of the 33 misses) and `s5b` (retest, entry
and excursions identical to the live tracker). `run all` includes them.

### Stages

`hawk-research stages` prints this table with dependencies. Run one stage with
`hawk-research run s8`, a group with `hawk-research run smc`, and pass extra arguments to a
single stage after `--`: `hawk-research run s8 --no-deps -- --seed 5`.

| Group | Stages | What they do |
|---|---|---|
| data | `s2b` | Point-in-time universe from the full Binance archive listing (network) |
| round1 | `s3` `s3b` `s4` `s5` `s5b` `s6` `s7` `s8` `s9` `s10` `s11` | Replay the detector, explain its misses against the live book, per-bar labels, zone dataset, live outcome cross-check, leakage audit, harness controls, null baselines and walk-forward, location control, bearish-only leg, capacity |
| family_a | `s3r` `s1s5` | Pre-registered systems S1 and S5 |
| smc | `s3s` `s5_smc` `s6_smc` `s8_smc` `s12` | Textbook order blocks through the same pipeline |

The side studies (zone trades, daily fade, market-neutral) are not part of this pipeline. Their code,
stages and 11 headline checks are in [`experiments/`](../experiments/README.md) and run with
`python -m experiments` from the repository root.

Each run writes `data/runs/<stage>.json` (git commit, dirty flag, resolved configuration,
seed, command, data-manifest hash, package versions) and `data/runs/<stage>.config.yaml`,
and stamps the same provenance into the stage's report. Every evaluated configuration is
appended to the trial ledger (`data/ledger.sqlite` plus a JSONL mirror). `run --require-clean`
refuses to run on a working tree with uncommitted changes.

## Configuration

`config/research.yaml` holds every tunable value at the setting that produced the published
results: replay geometry, detector thresholds, live gates, SMC rules, label horizons and
barriers, walk-forward geometry, bootstrap size and seeds. A test fails if the file and the
code defaults drift apart. Override for one run:

```bash
hawk-research --set evaluation.holdout_fraction=0.25 --set seeds.matched_controls=4 run s8
hawk-research --config my.yaml run smc
```

`config/costs.json` holds cost assumptions with sources, and records which fee produced each
published result. `config/universe.json` lists every symbol set explicitly.
`config/trial_history.json` carries the trial counts from the original ledgers.

## Repository layout

```
src/hawk_research/
  data/          download, validation, content-hash manifest, universes, the one OHLCV loader
  detectors/     round-1 detector (as used live) and the textbook SMC detector
  features/      point-in-time features
  labels/        forward returns, barrier first-touch, retest lifecycle
  replay/        bar-by-bar replay of the live detector and its gates
  evaluation/    trades, costs, capacity-limited portfolio, metrics, walk-forward harness
  stats/         matched controls, block bootstrap, deflated Sharpe, PBO, FDR
  audit/         leakage audit (truncation tests, structural checks)
  stages/        the numbered pipeline stages
  studies/       Family A (the pre-registered systems), with a README
  ledger.py      trial ledger;  provenance.py  run provenance;  config.py  typed configuration
config/          research.yaml, costs, universes, data manifest, trial history
docs/            OVERVIEW, methodology, deviations, PROVENANCE, archive/ (original documents)
results/         headline numbers and figures from the full run (small files only)
snapshot/        the live system's zones table, for the replay-fidelity checks
tests/           unit tests, the smoke integration test and its fixture
experiments/     the side studies: code, stages, tests (not installed; `python -m experiments`)
```

## Testing

```bash
pytest -m "not integration"                  # unit tests
pytest                                        # plus the smoke pipeline (~2 min)
pytest -m "not integration" --cov=hawk_research
ruff check src tests && ruff format --check src tests && mypy
```

The unit tests use hand-computed expectations on tiny inputs: feature truncation (no
look-ahead), trade construction including same-bar barrier ambiguity, label gap masking,
matched-control constraints, OHLCV validation, the content hash, the replay window, the
leakage audit catching a planted leak, the ledger's trial counting and provenance. CI runs them on
Linux, macOS and Windows with Python 3.11-3.14, plus the smoke run, lint and type checks.

### Regression check between two full runs (optional, not in CI)

`hawk-research compare OLD_DIR NEW_DIR` compares every report and table two data directories
contain: parquet rows and values (floats to a relative 1e-12) and report JSON recursively,
ignoring timestamps, paths, provenance and ledger ids. Use it after changing code that
should not change results: run the pipeline into a new directory and compare it with the
previous run. This is how the restructuring for publication was checked against the original
project's outputs ([`docs/deviations.md`](deviations.md#4-reproduction-against-the-published-reports)).
