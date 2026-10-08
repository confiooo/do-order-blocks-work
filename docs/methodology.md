# Methodology

How the numbers in this repository are produced, and which mistakes the pipeline is
built to prevent. This is the short version; the long rules are in
[`archive/validation_protocol.md`](archive/validation_protocol.md) (cited as "protocol §n").
Where the code departs from the protocol, [`deviations.md`](deviations.md) says so.

## 1. Data

- **Bars:** Binance spot klines from the public archive (`data.binance.vision`), 15-minute
  bars, 2024-09-01 to 2026-09-17. (The original project also collected BTCUSDT 1h and
  1-minute bars; no stage uses them.) `hawk-research download` fetches the bars; `config/data_manifest.json` holds a content hash
  per file, so two people can confirm they have identical inputs.
- **Universe:** fixed lists in `config/universe.json`. Round 1 and round 2 use the 80
  symbols the original live system tracked (76 of which produce zones). This list was
  chosen by *recent* volume, which is survivorship bias; a point-in-time universe built from
  the full archive listing (`s2b`, 597 pairs including 115 delisted) is used by Family A and
  the market-neutral study. Because every result is negative, the bias works against the
  conclusion, not for it (protocol §9.1).
- **Validation:** every file is checked on download (`data/validation.py`). Duplicate or
  non-increasing timestamps, `high < low`, non-positive prices and open/close outside the
  bar's range are errors; nothing is repaired silently. Zero-volume bars and gaps are real
  in exchange data, so they are counted and recorded, not dropped.

## 2. The detectors

- **Round 1** (`detectors/order_blocks.py`) is the original live detector, unchanged. It
  flags a candle with high relative volume and a long wick near a recent swing, and gives it
  a direction from the candle's colour. It is replayed bar by bar through a 400-bar rolling
  window so its internal lookback (60 bars) equals what the live system used. The replay
  reproduced 1,510 of 1,543 live detections with every compared field bit-identical; all 33
  misses are explained (round 1 findings §2). Its thresholds are the live system's settings,
  picked by hand to make the bot alert more often, not fitted to this data ([`PROVENANCE.md`](PROVENANCE.md)).
- **Round 2** (`detectors/smc_order_blocks.py`) is a textbook "smart money concepts" order
  block: 3-bar fractal swings, a close beyond the latest swing (break of structure), and the
  last opposite-coloured candle before the impulse as the zone. Fixed rules, no parameter
  search. The decision time is the break-of-structure bar, never the earlier order-block
  candle.

## 3. Point in time: no look-ahead

Every number attached to a zone is either a **feature** (known at the decision bar's close)
or a **label** (needs strictly later bars). Four mechanisms keep them apart:

1. **Feature truncation test** (`audit/leakage.py`, stage `s6`): for a random sample of zones,
   recompute every feature from a series cut at the decision bar. Any feature that reads the
   future changes value or turns NaN. Published runs: zero mismatches.
2. **Label truncation test:** the mirror image. With the future cut off, every label must be
   empty. Published runs: zero survivors.
3. **Gap masking:** a forward return or barrier label is only defined where the following bars
   are contiguous (`labels/outcomes.py: contiguous_forward_bars`), so no label silently spans
   missing data. Missing BTC bars give NaN features, never forward-filled ones.
4. **Entry timing:** trades enter at the close of the decision bar (or, for a retest entry, at
   the close of the bar after the retest bar, because the live tracker excluded the fill bar).

## 4. Controls: is the zone special?

A strategy can lose money for reasons unrelated to the zone. The comparison that matters is
zone vs non-zone bars that look the same:

- **Matched controls** (`stats/nulls.py: sample_matched_controls`): for each zone, up to two
  non-zone bars on the same symbol within ±24h whose volume ratio and 24h realised volatility
  are within tolerance (log-difference 0.15 and 0.25). Matching uses only values known at the
  control bar's own close, never its outcome (a unit test enforces this).
- **Harness controls** (stage `s7`): randomised sides (a null with true mean zero, used to
  check that 95% intervals exclude zero about 5% of the time), a block-shuffled permutation
  null, and a positive control (a synthetic +25 bps edge injected into the same pipeline must
  be detected).
- **Location control** (stage `s9`): does price come back to a zone more often than to a
  matched random level?

## 5. Splits, purge and embargo

- **Holdout:** the last 20% of decision times is held out and reported separately.
- **Walk-forward** (`evaluation/harness.py`): chronological folds of 90 days training and
  30 days test, stepping 30 days. A training trade whose outcome window reaches into the test
  window is purged, and test trades inside a 24h embargo are dropped.
- **Family A** uses the protocol's calendar tiers (development, firewall, holdout) and never
  read its holdout.

## 6. Uncertainty

Trades are not independent: zones cluster in time, and one market move fires the detector
on dozens of correlated symbols at once. Every interval comes from a **cross-sectional block
bootstrap** (`stats/resampling.py`): resample whole 24-hour blocks with every symbol inside
a block kept together. That widens intervals to reflect the much smaller number of independent
observations. Seeds are fixed (`config/research.yaml` → `seeds`), so intervals reproduce.

## 7. Multiple testing

Every configuration evaluated is written to the trial ledger (`ledger.py`), with the commit,
a dirty-tree flag, the config hash and the command that ran it. The ledger is what the
corrections count, so trials cannot be forgotten:

- **Deflated Sharpe ratio** (Bailey and López de Prado) for the best configuration, against
  the number of trials in the ledger. Round 1 deflates against 72 trials: three sessions of the
  same 24-configuration grid (`config/trial_history.json`).
- **Probability of backtest overfitting** (CSCV) across the configuration grid.
- **Benjamini-Hochberg FDR** across the feature screen, in-sample and out-of-sample.

Hansen's SPA and Romano-Wolf StepM, which the protocol specifies for Family A, were not
implemented ([`deviations.md`](deviations.md)).

## 8. Costs

Every result is reported gross and net. Cost values and the result each one produced are in
`config/costs.json` (`result_scenarios`):

- **Rounds 1 and 2, MEXC futures:** 4 bps fee per side plus 3 bps base slippage per side,
  about 14 bps round trip before an illiquidity add-on that grows with the symbol's volume
  rank. MEXC bills API futures trades at 6 bps; that is the named alternative scenario
  `CostModel.mexc_futures_taker_api()`, which lowers every net mean by exactly 4 bps.
- **Binance spot:** 10 bps fee per side plus 2 bps base slippage per side.
- **Family A:** maker entry 4 bps, taker exit 6 bps, a spread by volume rank, slippage, funding
  per settlement and a tick-size charge.
- Side studies use the values listed for them in `config/costs.json`.

Spot bars are priced with futures costs in rounds 1 and 2; funding is not charged there
because no funding history was collected for that universe. Both simplifications favour the
strategy.

## 9. Same-bar barrier ambiguity

With a take-profit and a stop-loss, both can fall inside one 15-minute bar's high-low range,
and 15-minute data cannot tell which was hit first. `labels/outcomes.py: barrier_label` flags
these bars and resolves them as **"down barrier first"**. For a long that is the stop
(pessimistic). For a short the down barrier is the *target*, so the same rule books an
ambiguous short as a **win**. The rule is applied identically to zones and controls and has not
been changed. Every barrier strategy now reports `ambiguous_fraction`, and the net mean with
every ambiguous short booked as a loss (`net_mean_if_ambiguous_shorts_lose`), so a reader can
see how much a short result depends on it.

## 10. Verdicts

The protocol's pass/fail gates (§8) are applied mechanically: economic viability (net
expectancy above zero with the lower bound above zero), walk-forward consistency, overfitting
(PBO, DSR) and sample adequacy. A result is FALSIFIED, INCONCLUSIVE, VOID (leakage) or, without
a pre-registration, EXPLORATORY. Only Family A was pre-registered.
