# Deviations from the protocol

[`archive/validation_protocol.md`](archive/validation_protocol.md) and
[`archive/preregistration.md`](archive/preregistration.md) specify more than was built.
This page lists every protocol item that was specified but not implemented, and every
place where what ran differs from what was written. None of these changes a verdict:
every result is negative, and each deviation either leaves a negative result negative
or would make it more negative. They are listed so nobody has to find them.

## 1. Specified but not implemented

| Protocol item | Where specified | Status |
|---|---|---|
| Hansen SPA (consistent p-value) across the Family A candidate set | protocol §6.4(e), §8 Gate 4; pre-registration §"Multiplicity" | **Not implemented.** Only S1 and S5 of the five Family A systems were run, so SPA across five curves is undefined. `studies/family_a/run.py` records Gate 4 as failed with that reason. Rounds 1 and 2 use deflated Sharpe and BH-FDR instead. |
| Romano-Wolf StepM | protocol §6.2, §8 Gate 4 | **Not implemented**, for the same reason. The protocol only allows StepM after SPA passes. |
| Ledoit-Wolf studentised bootstrap SE on `n_eff` | protocol §8 Gate 4 | Not implemented. |
| Harvey-Liu haircut Sharpe | protocol §8 Gate 4, §10 | Not implemented. |
| Realised Li-Ji `N_eff` across trials | protocol §6.3 | Not implemented. Deflated Sharpe uses the raw ledger trial count. That over-counts in one way (three mechanical re-runs of the same s8 grid) and under-counts in another (early exploratory experiments whose ledger rows were removed; see [`PROVENANCE.md`](PROVENANCE.md#trial-counts-are-preserved)). Every result is already negative, so more trials could not change a conclusion. |
| Minimum track record length (MinTRL) | protocol §8 Gate 5 | Not implemented. |
| `+1`-bar shift ablation | protocol §8 Gate 1, §9.6 | Not implemented as a stage. The feature and label truncation tests in `s6` (and the unit test `test_features_unchanged_when_the_future_is_cut`) test the same property directly. |
| Sign check: mean vs 5%-trimmed mean ("if the sign flips, not a finding") | pre-registration, metric table | Not applied. Family A reports `trim5_net_bps`, which is a 5%/95% *winsorised* mean (clipped at full-sample quantiles, not trimmed), as a descriptive number only. |
| Calendar tiers D / F / H with a one-touch, audit-logged holdout read | protocol §2.1-2.2 | Family A and the market-neutral study use the calendar tiers and never read the holdout. The market-neutral study implements the one-touch seal (`experiments/market_neutral/holdout.py`: a second read for a family raises; tested), but no stage calls it. Rounds 1 and 2 use a 20% chronological holdout instead (below), with no seal. |
| Rolling vs anchored walk-forward sensitivity; purge/embargo sensitivity | protocol §2.3, §2.4 | Not run for rounds 1 and 2. |
| Stationary bootstrap with B = 10,000 | protocol §3.4 | Rounds 1 and 2 use a cross-sectional block bootstrap (24h blocks, all symbols in a block resampled together) with 600 resamples (400 in s9 and s11, set in those stages; the published s7 run used 600 with 60 repeats, which the stage registry reproduces). Family A uses the stationary bootstrap. The evaluation harness also computed a stationary-bootstrap interval for every round-1/round-2 configuration; no report or ledger row used it, and it was removed. |

## 2. Ran differently from the protocol

| Item | Protocol | What ran (rounds 1 and 2) | Effect |
|---|---|---|---|
| Walk-forward folds | 6 folds, train 180 d, gap 5 d, test 60 d | 21 monthly folds: train 90 d, test 30 d, step 30 d, embargo 24 h, purge by outcome window (max 48 h) | More, shorter folds. Every fold-count verdict (0/21 net-positive in round 1, 3/21 in round 2) clears the protocol's FALSIFIED line (≤ 2 of 6) by a wide margin either way. |
| Embargo | 72 h | 24 h | Shorter embargo can only leak information *into* training, which would flatter a strategy. The results are negative. |
| PBO | CSCV, S = 12 (924 partitions) | s8: CSCV with 10 splits (252 partitions) over the configuration grid; market-neutral: 16 splits | Different partition count. Reported as computed. |
| Holdout | Calendar tier H (2026-04-01 → 2026-08-31), touched once | Last 20% of decision times (from 2026-05-06 for both rounds), reported alongside development | The holdout was looked at in rounds 1 and 2. It never turned a negative result positive. |
| Pre-registration (Gate 0) | Required for a result to count as evidence | Only Family A (S1, S5) was pre-registered. Rounds 1 and 2 and the side studies were not. | Under §8.1 the round-1, round-2 and side-study verdicts are formally **EXPLORATORY**. An exploratory search that finds no edge still found no edge; it cannot be read as evidence *for* one. |
| Same-bar barrier ambiguity | §9.3: resolve pessimistically | Both barriers inside one 15m bar resolve as "down barrier first". That is the stop for a long but the **target for a short**, so the rule is pessimistic for longs and *optimistic* for shorts. | Reported per barrier strategy as `ambiguous_fraction`, with a sensitivity that books every ambiguous short as a loss. See [`methodology.md`](methodology.md#9-same-bar-barrier-ambiguity). The rule itself is unchanged. |

## 3. Fixed while publishing

Small inconsistencies found during the clean-up. Each is recorded with its effect on
reported numbers.

| Issue | Fix | Effect on published numbers |
|---|---|---|
| MEXC taker fee written three ways (4 bps in code, 6 bps in the cost config, "0.02%" in round 1's findings) | The code value (4 bps) stays the default for the results it produced. 6 bps is an explicit scenario, `CostModel.mexc_futures_taker_api()`. The findings text has a marked erratum. | None. Under the 6 bps scenario every MEXC net mean is exactly 4 bps lower. |
| Family A annualised with 365 days, everything else with 365.25 | One constant, `constants.DAYS_PER_YEAR = 365.25` | Family A annualised Sharpe ratios scale by √(365.25/365) = 1.00034; see the regenerated `s1_s5_results.json`. |
| s8's deflated Sharpe counted trials from the local ledger only; a fresh run counted 24, the published report 72 (three sessions of the same grid) | `config/trial_history.json` carries the 48 earlier trials into a fresh ledger | A fresh reproduction now deflates against 72, as published. |
| Replay universe taken from "whatever files are in the folder" | Explicit universes in `config/universe.json` | None on the published data. A folder with extra files used to change round 1 silently (5 later downloads moved the zone count from 152,266 to 162,483). |
| Detector test fixtures used old thresholds while claiming to match live config | Tests read the research config, and a test fails if config defaults drift | None. |
| Report field `feature_screen_files.signed_ret_4h` in s8 named a file that does not exist (`feature_screen_signed_ret_4h.csv`); the CSV is written as `feature_screen_signed_direction_4h.csv` | The field now names the file actually written | None on any number. |
| The published data manifest's BTCUSDT 1h checksum cannot be reproduced: its last bar (2026-09-17 14:00 UTC) was still forming when the original project downloaded it, and Binance now serves the completed bar (found by the isolated verification). No stage reads that file; the original project fetched it for the live system's stability check | `download` no longer fetches it by default (`--btc-1h` to opt in); the manifest entry carries a note explaining the mismatch | None: no stage reads the file. |

## 4. Reproduction against the published reports

A clean run of every offline stage from raw inputs (`hawk-research run all`, no live-system
snapshot) was compared with the reports the original project published. Counts, verdicts
and every headline number in `results/headline_metrics.json` match. The differences:

| Report | Published | Regenerated | Cause |
|---|---|---|---|
| Market-neutral PBO, top-level `pbo` field (phase 1 / tier 2 / cointegration filter) | 0.775 / 0.933 / 0.495 | 0.524 / 0.650 / 0.747 | The published reports carry two PBO fields. `pbo` was computed while the 864 (tier 2: 1,080) rolling-beta cells were still invalid (a NaN in the beta warm-up made every z-score NaN); `pbo_after_rolling_beta_fix` was computed after `mn_rerun_rolling` re-ran them, and is 0.524 / 0.650 / 0.747. The current code never produces the invalid state, so a fresh run writes the corrected value to both fields. Every corrected number matches; cell statuses (fail / inconclusive) are identical and no cell passes in either version. No document quotes the pre-fix values. |
| Round-1 harness check (s7) interval bounds | e.g. observed reference upper bound −1.29 bps | −1.61 bps | Bounds differ by at most 0.32 bps. Repeats (60), bootstrap size (600), the false-positive count (4 of 60) and every mean match; the published bounds could not be reproduced with 400, 500, 600 or 1000 resamples. Unexplained; no verdict depends on it. |
| SMC leakage audit, number of columns checked | 342 | 343 | The original project's current code also gives 343; the published report came from an earlier version. The audit passes in both. |
| Zone dataset features (`rv_4h`, `beta_btc_24h`, `corr_btc_24h`) | — | — | Floating-point differences of about 1e-11 in 8 values across 152,266 zones (38 values in the Pass-R dataset); the published run was on Windows. |
| Family A annualised Sharpe ratios | S1 −17.134 | S1 −17.140 | Annualisation now uses 365.25 days (section 3): every Family A Sharpe ratio and its standard error scales by 1.00034. Verdicts unchanged. |
| Round-1 deflated Sharpe, trial count | 72 | 72 | Reproduced through `config/trial_history.json` (section 3). Without it a fresh ledger counts 24. |

Not regenerated, by design: the checks against the live system's database (replay fidelity,
the live outcome cross-check, the live performance report), the market-neutral ingestion log
(network stage `mn_ingest`), and two daily-fade files that no remaining code writes. Both are leftovers of interactive
analysis and carry nothing new. `y2026_trades.parquet` is the 516 2026 test trades of the chosen
daily-fade rule, row for row the same trades and net returns as `zone_bot_test.parquet` (mean
−0.95%). `rollback_features.parquet` is a per-event feature table for the 1,478 events in
`events.parquet` (same rows, same `max_rollback`), with derived columns such as close location
and volume ratio. No published number comes from either.
