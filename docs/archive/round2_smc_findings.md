# Round 2 — Textbook (SMC) Order Blocks

**Compiled:** 2026-10-07.
**Detector:** `src/hawk_eye/logic/smc_order_blocks.py` (new; the round-1 detector is unchanged).
**Pipeline:** `s3s_replay_smc.py` → `s5_zone_dataset.py --tag smc` → `s6_leakage_audit.py --tag smc`
→ `s8_null_baselines.py --tag smc` → `s12_smc_eval.py`.
**Reports:** `data/research/reports/{null_baselines_smc,smc_eval,leakage_audit_smc}.json`,
`data/research/zones/replay_smc_summary.json`.

---

## 0. Headline

Round 1 did not test order blocks. Its detector flags high-volume, long-wick candles and labels
them by colour; only **4.2%** of textbook order blocks are candles it ever flagged. Round 2 tests the
textbook definition through the same harness, data, costs and holdout.

**Textbook order blocks show no directional edge either.** Their direction is no more informative than
a volume-and-volatility-matched random candle's, every one of 24 tradeable configurations is net
negative, 3 of 21 walk-forward months are net positive, and the deflated Sharpe is 0.00. The round-1
negative result now holds for the thing traders actually mean by an order block.

---

## 1. What was tested

Fixed rules, no parameter search (`smc_order_blocks.py` docstring):

1. Swing points: 3-bar fractals, known 3 bars after the pivot.
2. Break of structure (BOS): a close beyond the latest confirmed, unbroken swing.
3. Order block: the last opposite-coloured candle at or before the impulse leg's extreme
   (within 10 bars of it, within 32 bars of the BOS). Zone = its high–low. Direction = BOS direction.
4. FVG flag: the leg from OB to BOS left a 3-candle gap. Tested as the only second variant.
5. Dropped if price left the zone and returned to it before the BOS bar.

The decision bar is the **BOS bar**, never the OB candle. Same 76 symbols, 15m bars,
2024-09-05 → 2026-09-17, 400-bar warmup, same 20% chronological holdout (from 2026-05-06).

| | Round 1 (wick candles) | Round 2 (SMC order blocks) |
|---|---|---|
| Zones | 152,266 | 218,217 (59% with FVG) |
| Per symbol per day | 2.70 | 3.87 |
| SMC OB candles also flagged by round 1 | — | **4.2%** |

## 2. Leakage audit — PASSED

All hard checks in `s6_leakage_audit.py --tag smc` pass: structural (including OB candle strictly
before the decision bar and zone = OB candle's range), gap masking, feature truncation (103 features)
and label truncation. Detector-level truncation invariance is a unit test
(`test_smc_order_blocks.py`). Visual spot-check of random zones matched the textbook pattern.

## 3. Results

Costs as round 1 (`CostModel.mexc_futures_taker`: 4 bps fee per side plus 3 bps base slippage per
side, 14 bps round trip before the illiquidity add-on; Binance spot taker higher). Note that
`config/research_costs.json` lists a 6 bps MEXC API taker fee, so these net figures are, if anything,
optimistic.
Intervals are cross-sectional time-block bootstraps (600 resamples).

### 3.1 Direction vs matched random candles

| | Round 1 | **Round 2** |
|---|---|---|
| Signed 4h return, zone − matched control | −0.31 bps [−3.38, +2.69] | **+0.50 bps [−2.17, +3.63]** |
| Signed 24h return, zone − matched control | +4.07 bps [−2.81, +10.51] | **−2.63 bps [−9.34, +5.11]** |
| Win rate 1h / 4h / 24h (enter at close) | 47.3 / 48.7 / 49.4% | **45.0 / 46.8 / 48.3%** |
| ±1% first touch, bullish up-first / bearish down-first (4h) | 47.0 / 51.5% | **47.8 / 51.2%** |

The BOS direction carries no information beyond a matched candle. Entering at the BOS close is
slightly *worse* than a coin flip at short horizons, consistent with breakouts partially reverting.

### 3.2 Volatility vs matched random candles (option 2, applied to the real detector)

| Forward realised vol 4h, zone − control | Round 1 | **Round 2** |
|---|---|---|
| Difference | +2.35 bps [+1.88, +2.87] | **+0.77 bps [+0.34, +1.34]** |
| Absolute 1h return difference | +5.46 bps [+3.64, +7.26] | **+2.63 bps [+1.69, +3.60]** |

Order blocks still precede slightly more movement than matched candles, but the effect is about a
third of round 1's. The round-1 wick-candle filter is the better volatility marker.

### 3.3 After costs (`s12_smc_eval.py`, 24 trials logged)

Net mean per trade, MEXC futures taker, 24h time exit:

| Variant | Entry | Full sample | Development | Holdout |
|---|---|---|---|---|
| all | BOS close | −23.4 [−31.0, −15.6] | −25.7 | −14.1 |
| all | retest | −9.7 [−14.7, −5.6] | −12.0 | −0.1 [−8.1, +8.6] |
| fvg | BOS close | −22.0 [−30.5, −13.9] | −24.6 | −11.9 |
| fvg | retest | **−7.6 [−13.3, −2.0]** | −9.1 | −1.6 [−11.8, +8.5] |

With a 2:1 barrier exit every configuration is −8.8 to −28 bps net with intervals excluding zero.
Binance spot costs make every row 8+ bps worse. **No configuration has a positive net mean, in any
period, under either venue.**

Retest entry is again the largest improvement (+9.4 bps gross on all zones, +11.6 with FVG), as in
round 1 (+8.76). This has not been tested against a matched-control retest, so it may be a generic
property of resting limit entries rather than of order blocks; either way it covers about half of
costs.

### 3.4 Walk-forward, PBO, DSR (`null_baselines_smc.json`)

- 21 purged monthly folds: **10 gross-positive, 3 net-positive**. Train→test rank correlation −0.37.
- PBO = 0.27. Deflated Sharpe = **0.00**; best configuration by net Sharpe is
  `fade|close|time|24h` at −1.04, i.e. even the best rule loses.

## 4. Verdict

Under `validation_protocol.md` §8: Gate 2 (net expectancy > 0 with lower bound > 0) fails for every
configuration, and Gate 3 fails with 3/21 net-positive folds. **FALSIFIED** for directional trading of
textbook 15m order blocks on this universe and cost structure.

> Order blocks, as defined by SMC/ICT and detected without look-ahead, mark slightly more volatile
> moments but carry no information about which way price goes next. The negative result is the finding.

## 5. Limitations

- **Internal structure.** 3-bar fractals on 15m pick up minor swings inside consolidations. HTF-only
  structure (e.g. 1h/4h swings, or a minimum impulse size) is a different, untested variant.
- **One timeframe, one asset class.** Mid-cap crypto, 15m, 2 years, taker costs.
- **No confluence filters.** Premium/discount, liquidity sweeps, session timing and HTF bias are part
  of many traders' OB playbooks and were not tested. Each would be a new trial against the same data.
- **Retest control missing.** See §3.3. Running `s9_location_control.py`-style controls for the SMC
  population is the obvious next measurement.
- **Holdout rows are less negative than development** (retest: −0.1 vs −12.0). Per protocol §8 Gate 6
  a better holdout is treated as noise to be explained, not a lead; its intervals span zero and the
  development period is clearly negative.
