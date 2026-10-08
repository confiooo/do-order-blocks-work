# Archive: the original research documents

These are the documents written while the research was done, kept as written so the
record is not rewritten after the fact. The only edits are a marked **erratum** in
`round1_findings.md` (a fee value) and the removal of a stray tool-inserted string in
`validation_protocol.md` (see below). Line endings were normalised from CRLF to LF;
the text is otherwise byte-identical to the originals. For a short version, read
[`../OVERVIEW.md`](../OVERVIEW.md) and [`../methodology.md`](../methodology.md) instead.

| File | What it is | Read it for |
|---|---|---|
| [`validation_protocol.md`](validation_protocol.md) | The statistical protocol: splits, purge/embargo, bootstrap, effective sample size, power, multiple-testing budget, pass/fail gates (§8), leakage checklist (§9). | The rules every result was judged against. |
| [`preregistration.md`](preregistration.md) | Pre-registration of Family A (systems S1-S5): hypotheses, priors, frozen parameters, gates, stop rules. | What was promised before the S1/S5 run. Its timing cannot be verified from this repository; see [`../PROVENANCE.md`](../PROVENANCE.md). |
| [`evidence_signals.md`](evidence_signals.md) | Literature review: what published evidence says about wick/volume/order-block style signals. | Priors and why each system was expected to fail or not. |
| [`evidence_context_costs.md`](evidence_context_costs.md) | Literature and fee-schedule review: spreads, slippage, venue fees, funding. Source of `config/costs.json`. | Where every cost number comes from. |
| [`round1_findings.md`](round1_findings.md) | Round 1: dataset, replay fidelity, leakage audit, harness controls, first null/baseline results for the original (wick-candle) detector. | The main negative result. |
| [`s1_s5_results.md`](s1_s5_results.md) | Family A: the two pre-registered trials that were run (S1, S5), both FALSIFIED. | The pre-registered test. |
| [`round2_smc_findings.md`](round2_smc_findings.md) | Round 2: textbook (SMC) order blocks through the same harness. | Whether the negative result holds for "real" order blocks. |

## Reading the old paths

The documents refer to the layout of the original project. The mapping is:

| In the archive | In this repository |
|---|---|
| `src/hawk_eye/research/*.py` | `src/hawk_research/{data,features,labels,replay,evaluation,stats,audit}/` |
| `src/hawk_eye/logic/order_blocks.py` | `src/hawk_research/detectors/order_blocks.py` |
| `src/hawk_eye/logic/smc_order_blocks.py` | `src/hawk_research/detectors/smc_order_blocks.py` |
| `scripts/research/sN_*.py` | `src/hawk_research/stages/sN_*.py`, run as `hawk-research run sN` |
| `scripts/research/run_s1_s5.py`, `src/hawk_eye/research/family_a.py` | `src/hawk_research/studies/family_a/` (`hawk-research run s1s5`) |
| `config/research_costs.json` | `config/costs.json` |
| `data/research/...` | `data/...` (the data directory; `--data-dir`) |
| `data/research/research.sqlite`, the bot's `db.sqlite` `experiments` table | `data/ledger.sqlite` (one research-only ledger; historical counts in `config/trial_history.json`) |
| `s1_snapshot_live.py`, `data/research/snapshot/` | not distributed: it pulled the owner's live-bot database. The checks that used it are optional and skipped without it. |

## Known errors in these documents

- `round1_findings.md` §4 says "MEXC futures taker 0.02%/side". The code that produced every
  round-1 number used **0.04%/side plus 3 bps base slippage per side** (14 bps round trip before
  the illiquidity add-on). Marked in place as an erratum. `config/costs.json` → `result_scenarios`
  records which fee produced which result.
- `s1_s5_results.md` names git SHA `2ee634d` as the pre-registration and code version. That
  commit does not contain the pre-registration or the research code; see `../PROVENANCE.md`.
- `round2_smc_findings.md` refers to `config/research_costs.json`; here it is `config/costs.json`.
- `validation_protocol.md` §10, item 1 contained a stray string an editing tool had inserted in
  place of "SHA"; it now reads "git commit SHA of the code".
