# Market neutral (side study)

Stages: `python -m experiments run market_neutral` (`mn_rv_phase1`, `mn_tier2`, `mn_coint`, `mn_fb_phase1`,
`mn_rerun_rolling`; `mn_ingest` downloads funding and perpetual data). Unrelated to order blocks.

**Question.** Is there a market-neutral edge in the same universe: relative-value pairs (long one
coin, short another, hedged by a fixed or rolling beta), and funding/basis (spot long, perpetual
short, collect funding)?

| Matrix | Cells (trials) | Cells passing every gate |
|---|---|---|
| Relative value, 11 large caps vs BTC plus SOL vs ETH (12 pairs) | 1,728 | 0 |
| Relative value, tier 2 (15 mid caps vs BTC) | 2,160 | 0 |
| Relative value, phase-1 pairs with a cointegration filter | 1,728 | 0 |
| Funding/basis, 11 names | 792 | 0 |

Every cell is logged to the trial ledger before any is judged, and the deflated Sharpe and PBO are
computed over the whole matrix. The holdout (2026-04-01 to 2026-08-31) is sealed behind a one-touch
loader (`holdout.py`) and was never read.

**Reproduction note.** Today's code reproduces every cell's final status (for example phase 1:
1,534 fail, 194 inconclusive), the zero passes, and the corrected PBO values the published reports
give as `pbo_after_rolling_beta_fix` (phase 1 0.524, tier 2 0.650, cointegration filter 0.747).
The published reports also have a top-level `pbo` field with a different value (0.775, 0.933,
0.495): it was computed when the rolling-beta cells had first been run with a bug (a NaN in the
beta warm-up made every z-score NaN), before `mn_rerun_rolling` re-executed them. The fixed code
never goes through the buggy state, so in a fresh run both fields hold the corrected value. See
[`docs/deviations.md`](../../docs/deviations.md).

The code describes the matrices as pre-registered, but no pre-registration document for this study
exists in the original project.
