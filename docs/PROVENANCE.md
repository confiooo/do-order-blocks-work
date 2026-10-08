# Provenance and honesty

Plain statements about where this work came from and what cannot be checked. Each one
is something a careful reader would otherwise have to discover alone.

## A personal project, published as a snapshot

This started as a personal project, not as a planned study. It grew out of many ideas: a live
alerting system for order-block zones, then questions about whether those alerts meant
anything, then side experiments (zone straddles, a daily fade, market-neutral pairs), and only
gradually became the research design documented here. The protocol, the pre-registration and
the leakage controls were added along the way, not fixed at the start.

The research was developed locally over several weeks (September-October 2026) and is
published as a clean snapshot. **The git history of this repository does not show how the
research developed.** Before the snapshot, the research code was not under version control
at all: the original project's repository contains no research code in any commit before
2026-10-07. The development history that does exist is the trial ledger (every
configuration evaluated, with timestamps) and the dated documents in `docs/archive/`.

## Pre-registration timing cannot be verified here

[`archive/preregistration.md`](archive/preregistration.md) is included exactly as written.
Its own text says it was frozen before the Family A run (2026-09-17 16:21 UTC, the
timestamp of the two ledger rows). This repository cannot prove that:

- `archive/s1_s5_results.md` cites git SHA `2ee634d` as the pre-registration and code
  version. That commit exists in the original project but does **not** contain the
  pre-registration or any research code, so it proves nothing about timing.
- The file was first committed to any repository on 2026-10-07, three weeks after the run.

Given how the project evolved (above), treat the Family A pre-registration as the author's
statement of what was decided before that one run, not as independently verified, and treat
everything else as exploratory. Newly regenerated Family A reports record the SHA-256 of the
pre-registration file they ran against (`preregistration_sha256`), so future runs are
tied to its content.

## AI assistance

AI was used as a tool in building this project.

## Where the detector thresholds came from

The round-1 detector's thresholds (`strength_threshold` 1.35, `wick_to_body_ratio_min` 1.2,
`min_distance_from_high_low` 0.5) are the live alerting system's settings. The author did not
remember how they were chosen. The project's own records (git history, the live server's logs,
and saved development notes on the author's machine) show this:

1. **The original code defaults were 1.5 / 1.5 / 1.0.** They were hand-written in the first
   version of the bot, with no record of a reason.
2. **An automatic "tuner" then produced 1.824 / 1.824 / 1.051.** The live system had a tuner
   (`ai/tuner.py`, later disabled) that rewrote the config from recent alert outcomes. Each time
   fewer than half the alerts "succeeded", it raised the strength and wick thresholds by 5% and
   the swing distance by 2.5%. Four such steps from 1.5 give exactly 1.824 (1.5 → 1.575 → 1.654
   → 1.737 → 1.824), and two from 1.0 give 1.051. These values are already in the config template
   on 2025-10-19.
3. **On 2026-09-06 they were loosened by hand to 1.35 / 1.2 / 0.5**, together with a move from
   1h to 15m candles, because the bot had stopped sending alerts. The values were picked without
   analysing any data; the aim was to get alerts flowing again. The live server ran with these values from
   2026-09-07.

So the thresholds were **not fitted to the 2024-2026 history** this study tests. The 1.824 values
came from a short run of live alert outcomes (2025); the current values are round numbers chosen
to make the bot less quiet. They are not standard values from trading literature either: there
is no agreed numerical definition of an order block, which is why round 2 tests the textbook
structural definition, with fixed rules and no tuned thresholds.

### Exploratory check: the stricter old thresholds

*Exploratory and post-hoc (run 2026-10-08, after the published results): 12 extra trials, not
part of any published number.*

Could the loosened thresholds be why round 1 found nothing? Zones that also pass the old
1.824 / 1.824 / 1.051 settings are a subset of the zones tested (33,400 of 152,266, 22%), so
the same tests were run on that subset against its own matched controls, with the same seeds
and costs. The same script reproduces the published full-population numbers exactly.

| | All zones (1.35 / 1.2 / 0.5) | Stricter subset (1.824 / 1.824 / 1.051) |
|---|---|---|
| Direction vs matched control, 4h | −0.31 bps [−3.38, +2.69] | −3.50 bps [−10.04, +2.76] |
| Direction vs matched control, 24h | +4.07 bps [−2.81, +10.51] | +7.39 bps [−5.91, +20.89] |
| Forward volatility vs matched control, 4h | +2.35 bps [+1.88, +2.87] | +4.14 bps [+2.99, +5.30] |
| Configurations net-positive (MEXC costs) | 0 of 12 | 0 of 12 |
| Best net mean per trade | −10.5 bps | −9.1 bps [−16.0, −1.5] |
| Walk-forward months net-positive | 0 of 21 | 3 of 21 |

Stricter thresholds pick more volatile moments but carry no more directional information, and
every configuration still loses money after costs. The thresholds are not why round 1 failed.

## Checks against the live system

The published round-1 results were produced with checks against a snapshot of the
original live system's database (1,543 detected zones, 791 alerts, 80,343 bars from
2026-09-07 to 2026-09-17). The database itself is not distributed, but its `zones` table, the
only part checks 1-3 read, is: [`snapshot/live_zones.parquet`](../snapshot/README.md).

1. **Replay fidelity** (`s3`): 1,510 of 1,543 live detections reproduced with every compared
   field bit-identical.
2. **Miss diagnostics** (`s3b`): each of the 33 misses attributed to a cause (history shorter
   than the 400-bar window the detector needs).
3. **Outcome fidelity** (`s5b`, and a cross-check in `s5`): retest, entry price and MFE/MAE
   identical to the live tracker on the 1,510 overlapping zones.
4. **Bar fidelity**: the original download script compared the archive's bars with the live
   system's own bars on the overlap (80,343 bars, 0 mismatches). That script is replaced by
   `hawk-research download`, which checks bars against the published content hashes instead.

Checks 1-3 run for everyone from that file, and reproduce the published results exactly.
They verify that the replay is the live detector; they do not affect any other number.
Check 4 needs the live system's bars, which are not shipped (they are public Binance bars, and
the download is checked against content hashes instead). The original project's script that
pulled the snapshot from the live server is not included.

## Trial counts are preserved

The deflated Sharpe ratio depends on how many configurations were tried. The original
project kept two ledgers; their counts are in `config/trial_history.json`:

| Ledger | Trials | Note |
|---|---|---|
| Round 1 null baselines (s8) | 72 | The same 24-configuration grid, run three times on 2026-09-17. The published DSR deflates against all 72. |
| Round 1 bearish-only (s10) | 12 | |
| Round 2 SMC (s8 `--tag smc`, s12) | 24 + 24 | |
| Family A (S1, S5) | 2 | |
| Market neutral: relative value phase 1 / tier 2 / cointegration filter / funding-basis | 1,728 / 2,160 / 1,728 / 792 | |

The original ledgers held 6,544 rows in total (134 in the live system's database, where the
round-1 and round-2 trials were logged, including 2 infrastructure test rows; 6,410 in the
research ledger), and every row is accounted for in that file.

Two gaps, neither explained by any record:

- In the research ledger, experiment ids 3-1730 (1,728 ids, exactly the size of one
  relative-value phase-1 run) were allocated and later removed.
- In the live system's database, 44 ids (1-2, 5-22, 107-120, 169-178) were allocated and
  later removed.

The author recalls these as exploratory experiments trying different things, removed to save
time; their configurations were not kept. They are not in `trial_history.json` because nothing
about them can be counted reliably. **The trial counts used for the deflated Sharpe ratios are
therefore lower bounds.** More trials would lower every deflated Sharpe further, so no
conclusion depends on this: every result is already negative.

## Data

All market data comes from Binance's public archive (`data.binance.vision`) and public REST
API. `config/data_manifest.json` records a content hash, row count and first/last bar for every
file used. If Binance revises an archive, `hawk-research verify-data` reports the file as
different and results may not reproduce exactly. Re-downloading four of the 85 15m files
(BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT) for the full window matched their published checksums
exactly. One listed file cannot match: BTCUSDT 1h, whose last bar was captured while still
forming; no stage reads it (see [`deviations.md`](deviations.md)).
