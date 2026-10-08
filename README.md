# do-order-blocks-work

![Lines of Python](https://img.shields.io/badge/lines_of_Python-23k-blue)

**Do order blocks predict where crypto prices go next?** This repository tests that
question on two years of 15-minute Binance data for 76 altcoins, with leakage audits,
matched random controls, realistic trading costs and a trial ledger that counts every
configuration tried.

**Answer: not here.** Under two common definitions, the direction an order block points to
carries no more information than a matched random candle's, and no tested way of trading it
makes money after costs. Order blocks mark slightly more volatile moments, nothing more. The
negative result is the finding; what it does and does not cover is spelled out
[below](#what-is-proven-and-what-is-only-observed).

| | Round 1: the original detector | Round 2: textbook (SMC) order blocks |
|---|---|---|
| What is detected | High-volume, long-wick candles near a swing, direction from candle colour | Last opposite-coloured candle before a break of structure (3-bar fractals), direction from the break |
| Zones (76 symbols, Sep 2024 - Sep 2026) | 152,266 | 218,217 |
| Direction vs matched random candle, 4h | −0.31 bps [−3.38, +2.69] | +0.50 bps [−2.17, +3.63] |
| Forward volatility vs matched candle, 4h | +2.35 bps [+1.88, +2.87] | +0.77 bps [+0.34, +1.34] |
| Best net mean per trade (MEXC futures costs) | −8.86 bps (bearish only, retest entry, 2:1 barrier) | −7.6 bps (with fair-value gap, retest entry, 24h) |
| Walk-forward months net-positive | 0 of 21 | 3 of 21 |
| Deflated Sharpe ratio | 0.00 (72 trials) | 0.00 (24 trials) |
| Leakage audit | passed | passed |

Intervals are 95% block-bootstrap intervals. Full numbers:
[`docs/OVERVIEW.md`](docs/OVERVIEW.md) (two-page summary),
[`docs/archive/round1_findings.md`](docs/archive/round1_findings.md),
[`docs/archive/round2_smc_findings.md`](docs/archive/round2_smc_findings.md). The same
data also failed a pre-registered test of two more refined systems (Family A,
[`docs/archive/s1_s5_results.md`](docs/archive/s1_s5_results.md)) and three side
studies (zone straddles, a daily fade, market-neutral pairs; [`docs/OVERVIEW.md`](docs/OVERVIEW.md)),
whose code is kept separately in [`experiments/`](experiments/README.md).

The one real effect: zones mark slightly *more volatile* moments than matched candles. It
is small, it is not directional, and nothing here turns it into a profitable trade.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/zone_vs_matched_control-dark.png">
  <img alt="Zones vs matched control bars: no directional difference, slightly higher volatility" src="results/zone_vs_matched_control.png">
</picture>

## What is proven and what is only observed

- **Shown, within this data:** for these 76 symbols, 15m bars, Sep 2024 - Sep 2026 and these
  cost assumptions, neither detector's direction beats a matched random candle, and no tested
  configuration has positive expected return after costs. The pipeline demonstrably finds a
  planted +25 bps edge (positive control) and its intervals are calibrated (random sides
  flagged 4 of 60 times at a nominal 5%).
- **Not shown:** that order blocks never work. Other timeframes, higher-timeframe structure,
  confluence filters, other assets, and maker execution in rounds 1 and 2 were not tested
  ([`docs/archive/round2_smc_findings.md`](docs/archive/round2_smc_findings.md) §5).
- **Only observed:** the volatility effect and the retest-entry improvement are measured, not
  explained, and the retest effect has no matched control in round 2.
- Only Family A was pre-registered. Rounds 1 and 2 are, by the protocol's own rules,
  exploratory ([`docs/deviations.md`](docs/deviations.md)).

## Quick start (5 minutes, no download)

Requires Python 3.11-3.14 (tested on 3.11, 3.12, 3.13 and 3.14).

Clone this repository and, from its root directory (the command-line tool is called
`hawk-research`, after the alerting system the study started from):

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -m "not integration"     # unit tests, < 1 minute
hawk-research smoke              # the real pipeline on a synthetic market, ~2 minutes
```

`smoke` builds six months of synthetic 15m bars for five symbols with planted
order-block shapes, runs ten stages end to end (replay, labels, dataset, leakage audit,
null baselines, SMC detector and evaluation) and checks the result against
`tests/fixtures/smoke_expected.json`. The synthetic numbers mean nothing about markets; the
point is that every stage runs and gives the same answer on every machine. `uv.lock` pins
the exact versions the results were produced with (`uv sync --extra dev`).

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
table shipped in [`snapshot/`](snapshot/README.md): `s3` (1,510 of 1,543 live zones reproduced
with every field identical), `s3b` (the cause of each of the 33 misses) and `s5b` (retest, entry
and excursions identical to the live tracker). `run all` includes them.

### Stages

`hawk-research stages` prints this table with dependencies. Run one stage with
`hawk-research run s8`, a group with `hawk-research run smc`, and pass extra arguments to a
single stage after `--`: `hawk-research run s8 --no-deps -- --seed 5`.

| Group | Stages | What they do |
|---|---|---|
| data | `s2b` | Point-in-time universe from the full Binance archive listing (network) |
| round1 | `s3` `s4` `s5` `s6` `s7` `s8` `s9` `s10` `s11` | Replay the detector, per-bar labels, zone dataset, leakage audit, harness controls, null baselines and walk-forward, location control, bearish-only leg, capacity |
| family_a | `s3r` `s1s5` | Pre-registered systems S1 and S5 |
| smc | `s3s` `s5_smc` `s6_smc` `s8_smc` `s12` | Textbook order blocks through the same pipeline |

The side studies (zone trades, daily fade, market-neutral) are not part of this pipeline. Their code,
stages and 11 headline checks are in [`experiments/`](experiments/README.md) and run with
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
project's outputs ([`docs/deviations.md`](docs/deviations.md#4-reproduction-against-the-published-reports)).

## Limitations

- **Survivorship.** Rounds 1 and 2 use the 80 symbols the original live system tracked, a list
  chosen by recent volume. Because the result is negative, this bias works against the
  conclusion, not for it.
- **Costs are modelled, not measured.** No order-book data; slippage scales with a liquidity
  proxy. Spot bars are priced with futures costs and no funding. Both favour the strategy.
- **Same-bar barrier ambiguity** resolves as "down barrier first", which is pessimistic for
  longs and optimistic for shorts. Every barrier result reports how much it depends on this
  ([`docs/methodology.md`](docs/methodology.md#9-same-bar-barrier-ambiguity)).
- **One timeframe, one asset class, taker execution.**
- **Hand-picked thresholds.** The round-1 detector's thresholds are the live system's settings,
  loosened by hand to make it alert more often. They were not fitted to this data, but they
  are not principled either. An exploratory check with the stricter earlier settings finds the
  same result ([`docs/PROVENANCE.md`](docs/PROVENANCE.md#where-the-detector-thresholds-came-from)).
- Protocol items that were specified but not built (SPA, StepM and others) are listed in
  [`docs/deviations.md`](docs/deviations.md).

## Provenance

This began as a personal project that grew out of many ideas (a live alerting bot first, then
the question of whether its signals meant anything) and only gradually became the research
design described here. It is published as a clean snapshot, so the git history does not show
that development. The pre-registration is included as written; its timing cannot be verified
from this repository, and the trial counts are lower bounds (some early experiments were not
kept). AI was used as a tool in building this project. Details:
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## License

MIT; see [`LICENSE`](LICENSE).
