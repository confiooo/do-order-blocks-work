# Round 1 — Dataset, Harness, and First Null/Baseline Results

**Compiled:** 2026-09-17.
**Detector:** `src/hawk_eye/logic/order_blocks.py` — unmodified. Every zone in this dataset comes
from the production function, called through a rolling window that reproduces the live effective
lookback of 60 bars.
**Governing protocol:** `docs/research/validation_protocol.md`.
**Pipeline:** `scripts/research/s1`…`s10`, modules in `src/hawk_eye/research/`.

---

## 0. Headline

Over two years and 152,266 replayed zones, **the order-block detector shows no directional edge that
survives costs, and no directional edge at all against a properly matched control.** The single
best of 84 logged configurations makes +10.4 bps gross per trade and needs ~19 bps to trade. The
detector does carry one real, repeatable signal — a small but highly significant **volatility**
signal — and that is the only role the data supports.

This is a negative result on the direction question. It is reported as such.

---

## 1. Dataset

| Artifact | Path | Contents |
|---|---|---|
| Live snapshot | `data/research/snapshot/db.sqlite` | 1,543 zones, 791 alerts, 80,343 OHLCV rows, 80 tokens. sha256 `cce281a4…21ddc` |
| MEXC paper trades | `data/research/snapshot/mexc_trades.sqlite` | 712 rows. sha256 `9c75d35e…4962` |
| Historical OHLCV | `data/research/ohlcv/15m/*.parquet`, `1h/BTCUSDT.parquet` | 80 symbols, 4,513,751 bars, 2024-09-01 → 2026-09-17 |
| Replayed zones | `data/research/zones/replay_zones.parquet` | 152,266 zones, of which 146,978 are live-equivalent (pass `signal_gate` + interval control) |
| Per-bar labels | `data/research/labels/15m/*.parquet` | forward returns, realized vol, first-touch offsets for every bar |
| Full zone dataset | `data/research/features/zone_dataset.parquet` | 152,266 × 347 columns, features + labels + outcomes |
| PIT universe | `data/research/universe/pit_universe.parquet` | 597 USDT pairs × 747 dates, incl. 115 delisted before the snapshot |

The 15m archive was validated against the live snapshot on the overlap: **80,343 bars compared,
0 mismatches** on open/high/low/close/volume. 12 duplicate rows dropped, 0 gaps, 0 fabricated bars.
51 of 80 symbols have the full 2 years; the other 29 record their genuine listing date (BNCBUSDT
starts 2026-09-14 with 298 bars).

### Survivorship bias

The live 80-symbol list is chosen by *today's* volume, which is look-ahead when applied to 2023–2026.
`s2b_pit_universe.py` reconstructs membership from Binance's S3 archive listing rather than from the
current `tokens` table, recomputing trailing-30d quote-volume ranks as of each date across every USDT
pair Binance ever archived — 597 pairs with data, including 115 delisted before the snapshot, median
416 listed per day. Strategy research should filter on `in_topN_*` from that table, not on `tokens`.

**Residual bias that cannot be removed from this source:** pairs delisted early enough that Binance
purged their archives entirely are unrecoverable. All headline numbers below are computed on the
*current* 80-symbol universe and therefore still carry survivorship bias. Since the headline result
is negative, this bias works against the conclusion, not for it — a survivorship-free universe would
be expected to look worse, not better.

---

## 2. Replay fidelity

`detect_order_blocks` is called with a 400-bar rolling window so that
`lookback = int(400 × 0.15) = 60` matches production exactly.

**1,510 of 1,543 live zones reproduced — 97.86% recall, with all nine compared fields bit-identical**
(`zone_high`, `zone_low`, `zone_midpoint`, `zone_size_pct`, `volume_ratio`, `wick_ratio`,
`distance_to_swing`, `distance_to_price`, `direction`; `max_abs_diff = 0.0` on every numeric field).

All 33 misses are explained, none unexplained:

| Symbol | Missed | Bars available | Implied live lookback | Cause |
|---|---|---|---|---|
| 牛来USDT | 19 | 768 | 60 | inside the 400-bar warm-up |
| MARSCOINUSDT | 11 | 1,254 | 60 | inside the 400-bar warm-up |
| BNCBUSDT | 3 | 298 | 44 | history shorter than the 400-bar window |

BNCBUSDT is the instructive case: with only 298 bars the live engine itself ran a 44-bar lookback,
so it is a *different detector* on that symbol. The replay refuses to guess and declines those bars.

The replay produces 1,365 extra zones in the live window. This is expected and not a defect: live
scans only the top-40 symbols by 24h quote volume each cycle, while the replay evaluates every
backfilled symbol on every bar. 181 of the extras are on symbols live never scanned at all.

**Live filtering.** The `zones` table stores every detection; `signal_gate`
(`min_volume_ratio` 1.8, `max_distance_to_price` 0.45) and
`signal_control.min_signal_interval_per_pair` gate *alerting*, not persistence. Both the pre-gate
population and the `passes_signal_gate` / `live_equivalent` / `would_alert` flags are kept, so
research can work either side of the gate.

**Outcome fidelity.** `labels.zone_lifecycle` imports the live `find_first_retest`, `_entry_price`,
`candles_after_fill` and `_compute_excursions` unchanged, so the MAE sign convention cannot drift.
On the 1,510 overlapping zones: retest flag agreement 99.19%, retest bar identical 100%, entry price
identical 100%, and **MFE/MAE at 1h/4h/8h/24h identical to the last bit** (`max_abs_diff = 0.0`).

---

## 3. Leakage audit — PASSED

`scripts/research/s6_leakage_audit.py`, over 152,266 zones × 341 columns:

- **Feature truncation test.** 400 sampled zones, 105 feature columns recomputed from a series
  truncated at the decision bar. Any feature touching a later bar would change value or go NaN.
  **Zero mismatches, zero NaN disagreements.**
- **Label truncation test.** The mirror image: **zero labels survive truncation**, confirming every
  outcome genuinely requires strictly-later bars.
- **Structural checks.** No duplicate zones, no epoch-unit errors, no off-by-one bar alignment, zone
  bands consistent with their source bar. No errors, no warnings.
- **Gap masking.** No violations. 690 rows correctly flagged as having an incomplete forward window.
- **Forward-fill contamination.** 46 symbols contain flat bars (worst: LSKUSDT, 17.9%). These are
  genuine zero-volume prints on illiquid pairs, stored as-is by the archive; the pipeline never
  reindexes or forward-fills, and every label spanning a gap is masked.

### Negative controls — the harness is calibrated

Two controls, because the obvious one is subtly wrong (`s7_harness_check.py`):

1. **Randomized sides** — the side is re-drawn at random, giving a null whose true expectation is
   exactly zero. Across 60 replicates the confidence interval excluded zero **4 times, a 6.7%
   false-positive rate against a nominal 5%**. The interval is honestly calibrated.
2. **Shuffled labels within 24h blocks** — this does *not* give a zero-centred null, and treating it
   as one is a category error. Inside a 24h block the bullish/bearish mix is itself correlated with
   that block's mean return (rising markets produce more bullish zones), which displaces the null to
   **+3.05 bps (sd 0.82)**. Scored as a permutation test instead, the observed statistic of
   −3.42 bps sits **z = −7.9** from the null, p < 0.017. The direction label carries real information
   — *of the wrong sign*.
3. **Positive control.** A synthetic +25 bps edge injected into the same pipeline is detected at
   +21.6 bps with an interval excluding zero. The harness is not blind.

---

## 4. First-round results

All intervals are cross-sectional bootstrap (600 resamples) that respect the fact that ~80 correlated
symbols fire simultaneously. Costs: Binance spot taker 0.10%/side, MEXC futures taker 0.02%/side [**erratum:** the code that produced these numbers used 0.04%/side plus 3 bps slippage per side; see `config/costs.json` → `result_scenarios`],
plus a slippage model scaling with symbol illiquidity.

### 4.1 Unconditional — a zone predicts nothing

Zone bars vs volume-and-volatility-matched non-zone bars on the same symbol in the same ±24h window:

| Horizon | Zone | Matched control | Difference | 95% CI |
|---|---|---|---|---|
| 1h | +1.74 bps | +1.78 bps | **−0.03 bps** | [−2.07, +2.18] |
| 4h | +4.00 bps | +3.53 bps | **+0.47 bps** | [−3.09, +4.41] |
| 24h | +13.52 bps | +16.37 bps | **−2.85 bps** | [−10.33, +4.47] |
| 48h | +16.58 bps | +16.30 bps | **+0.27 bps** | [−7.76, +8.20] |

Every interval contains zero. Against *all* bars (unmatched) zones look better (+1.59 to +13.43 bps),
but that difference vanishes entirely once you match on volume and volatility — it was a selection
effect for busy bars, not a property of the detector.

### 4.2 Directional — the `direction` label does not work

| Rule | Horizon | n | Gross | 95% CI | Win rate |
|---|---|---|---|---|---|
| direction | 1h | 152,245 | **−2.48 bps** | [−3.88, −0.97] | 47.30% |
| direction | 4h | 152,206 | **−3.42 bps** | [−5.89, −0.80] | 48.67% |
| direction | 24h | 151,983 | −2.20 bps | [−7.01, +2.75] | 49.40% |
| direction | 48h | 151,576 | −3.40 bps | [−10.14, +3.29] | 49.76% |

The nominal direction is *significantly negative* at 1h and 4h before any costs. First-touch at ±1%
barriers tells the same story with a sharp asymmetry:

| | 4h | 24h |
|---|---|---|
| bullish zones, up first | **47.03%** | 47.24% |
| bearish zones, down first | **51.55%** | 51.41% |
| pooled direction accuracy | 49.32% | 49.36% |

Ambiguous bars (both barriers inside one 15m bar's range) are 2.4% of resolved cases and are always
resolved pessimistically as stop-first.

**Per regime** (BTC trend × vol) the direction rule is net-negative in **all six cells**, win rate
44.1–45.4%, and **all six** intervals exclude zero. There is no regime where it works.

**Per liquidity tier** (top20 / 21–50 / 51–100 / 100+) gross returns run −5.08 to +0.99 bps with
intervals spanning zero in three of four tiers. No tier works.

### 4.3 Reversed / fade — also does not work

Fading is the exact negative of the above, so it is gross-positive by construction (+2.48 bps at 1h,
+3.42 bps at 4h) — and every one of those figures is smaller than the cheapest round-trip cost.
Net, fade loses 15.8 to 36.3 bps per trade depending on horizon and venue.

### 4.4 The best thing in the study, and why it still fails

The pooled direction rule averages a bearish leg that is right 51.4% of the time with a bullish leg
that is right 47.0% of the time. Testing the bearish leg alone (`s10_bearish_only.py`):

| Configuration | n | Gross | Net (MEXC fut) | Net (Binance spot) |
|---|---|---|---|---|
| bearish, retest entry, 2:1 barrier | 61,171 | **+10.37 bps** [+7.39, +13.80] | **−8.86** | −17.12 |
| bearish, close entry, 2:1 barrier | 77,410 | +3.80 bps [+0.61, +7.10] | −15.45 | −23.70 |

The gross edge is real and its interval excludes zero. It is also **roughly half of what it costs to
trade**, and it does not improve out of sample (bearish 4h: −7.12 bps development, −8.03 bps holdout).

### 4.5 Walk-forward, PBO, DSR

- 21 purged, embargoed monthly folds: **8 of 21 gross-positive, 0 of 21 net-positive.** Mean fold
  test return −2.65 bps, sd 6.33 bps.
- **Train→test rank correlation −0.34.** Folds that looked good in training did *worse* in test —
  the signature of noise, not edge.
- **PBO = 0.357** over the 8-configuration grid (252 CSCV combinations).
- **Deflated Sharpe = 0.00.** Best configuration by net portfolio Sharpe is
  `direction|retest|time|24h|mexc_futures_taker` at **−0.30** annualized (deflated against 72 trials).
  84 strategy trials in total are logged in the `experiments` table; none of them is positive.

**Every one of the 24 headline configurations is significantly net-negative**, with maximum drawdowns
of 87–100%.

### 4.6 Capacity — the constraint nobody costed

`s11_capacity_check.py` re-prices the best configurations at the concurrency the account actually
has. Live `mexc.max_open_positions` is **3**, but the harness default is 5, so every equity metric
above assumed capacity that does not exist.

Per-trade expectancy is invariant to the cap, so **no directional conclusion changes**. What changes
is everything about the equity curve, and it changes against the strategy:

| Configuration | cap | Net/trade | Daily Sharpe | Max DD | **Signals rejected** |
|---|---|---|---|---|---|
| direction, retest, 24h | 3 | −15.69 bps | −0.04 | 0.795 | **98.1%** |
| direction, retest, 24h | 5 | −15.69 bps | −0.30 | 0.872 | 96.9% |
| bearish, retest, 2:1 barrier | 3 | −8.86 bps | −1.23 | 0.892 | **74.0%** |
| bearish, retest, 2:1 barrier | 10 | −8.86 bps | −2.75 | 0.936 | 35.6% |

Sharpe is negative at every cap and every configuration, and drawdowns run 73–99%.

The rejection rate is the number to take away. The detector fires ~2.7 zones per symbol per day
across 76 symbols, roughly 205 signals a day; a 3-position account can act on about **2% of them**.
Even if an edge existed, *which* 2% to take would dominate the result — and §6 found no feature that
ranks them out of sample. Any future work has to solve selection, not just direction.

---

## 5. What the detector is actually for

Each claim is a number, not an opinion.

| Candidate role | Test | Verdict |
|---|---|---|
| **Directional signal** | Zone-direction signed return vs matched control signed the same way: **−0.31 bps** [−3.38, +2.69] at 4h | **No.** A zone bar's direction is no more informative than a matched ordinary bar's direction. |
| **Location / context filter** | Zone retest rate **78.1%** [77.7, 78.7] vs matched control bars **81.6%** [81.3, 82.0]; median lag 8 bars vs 6 | **No — worse than nothing.** Price returns to an *ordinary* bar's band more often and faster. The 78% retest rate is a property of 15m oscillation, not of the detector. |
| **Volatility / regime filter** | Forward realized vol 4h: **0.00613 vs 0.00589**, diff +0.000235 [+0.000188, +0.000287]; absolute forward return +5.46 bps [+3.64, +7.26] at 1h | **Yes — the one thing that works.** ~4% more forward volatility than a volume-and-vol-matched bar, robustly significant at 1h/4h/24h/48h. Small, but it is the only signal that survives matching. |
| **Entry trigger** | Waiting for a retest turns a −3.42 bps close-entry into **+8.76 bps** at 4h | **Partly.** The resting limit order is genuinely causal and adds ~12 bps. Not enough to reach break-even, but it is the largest single improvement found. |
| **Trade-management reference** | MFE vs MAE after fill: 1.070% vs 0.921% at 1h, 4.875% vs 4.644% at 24h. P(price never trades through entry) falls 12.4% → 2.8% | **No.** MFE/MAE are essentially symmetric (ratio 1.16 falling to 1.05). There is no favourable asymmetry to manage around. |
| **Event marker** | 152,266 zones, 2.70 per symbol per day | **This is what it is.** A high-frequency marker of unusually active bars. |

**Conclusion: the detector is a volatility/activity marker, not a direction signal.** The honest
framing for any future work is that it identifies *when* something is happening, and carries no
usable information about *which way*. A direction-selection layer would have to source its edge
from somewhere other than the zone itself.

One caveat stated plainly: the entry-trigger comparison in row 4 has a selection subtlety. The
+32.43 bps close-entry figure quoted on the *filled* subset conditions on a future event (whether a
retest occurred within 24h) and is **not tradeable**. The tradeable numbers are the ones in the
table: −3.42 bps for close entry on all zones, +8.76 bps for retest entry on the zones that fill.

---

## 6. Feature screening

118 point-in-time features, rank IC with Newey-West-corrected significance, in-sample on the
development period and out-of-sample on a genuinely held-out final period (2026-05-04 onward,
30,456 zones), Benjamini-Hochberg FDR across the whole feature set, plus a sign-agreement
requirement. Files: `data/research/reports/feature_screen_*.csv`.

### Against raw forward return (`fwd_ret_24h`): 29 survive

Top survivors by out-of-sample IC: `dist_to_range_low_7d_pct` (IS −0.156, **OOS −0.129**),
`trend_ret_7d` (−0.177 / **−0.121**), `dist_to_range_low_4d_pct` (−0.118 / **−0.114**),
`corr_btc_7d` (+0.087 / **+0.110**), `close_vs_sma_7d` (−0.203 / **−0.107**), `btc_ret_7d`
(−0.218 / **−0.090**), `corr_btc_24h` (+0.121 / **+0.086**), `dist_to_range_low_24h_pct`
(−0.074 / **−0.084**), `dist_to_swing_low_pct` (−0.077 / **−0.083**), `rv_4d` (−0.043 / **−0.080**).

**Read this result carefully.** Every survivor is a generic multi-day mean-reversion or
market-beta feature. Not one of them is a zone feature. Their AUC for direction is 0.47–0.52, i.e.
noise. And because the IC is pooled across symbols on each date, much of it is "the whole market
reverted today", which is not per-zone alpha without a market-neutral construction.

### Against the actual tradeable quantity (`signed_direction_4h`): **0 survive**

29 features clear FDR in-sample. **None clears out-of-sample.** The casualty list is precisely the
set anyone would have bet on:

| Feature | IS IC | IS q | OOS IC | OOS q |
|---|---|---|---|---|
| `confidence` (the detector's own score) | −0.0181 | 0.009 | −0.0338 | 0.232 |
| `wick_asymmetry` (upper-vs-lower, the new feature) | −0.0134 | 0.008 | −0.0052 | 0.702 |
| `lower_wick_frac` | +0.0150 | 0.002 | +0.0052 | 0.702 |
| `upper_wick_frac` | −0.0117 | 0.023 | −0.0053 | 0.702 |
| `zone_size_pct` | −0.0172 | 0.001 | −0.0251 | 0.309 |
| `rv_24h` | −0.0164 | 0.000 | −0.0147 | 0.702 |
| `atr_48_pct` | −0.0155 | 0.000 | −0.0130 | 0.702 |

The wick-asymmetry hypothesis — that `wick_ratio` throws away which side the wick is on and that
this matters — was worth testing and **is not supported**. Splitting the wick by side produces an
in-sample IC of −0.013 that does not replicate.

Also worth naming, from the raw-return screen, the features that looked strong in-sample and died:
`dist_to_range_high_7d_pct` (IS −0.121 → OOS +0.002), `btc_ret_24h` (−0.138 → −0.024),
`btc_rv_4h` (+0.066 → +0.011), `btc_range_4h_pct` (+0.062 → +0.020), `dist_to_swing_high_pct`
(+0.060 → +0.010). Note the pattern: the *low*-side range distances replicate and the *high*-side
ones do not.

---

## 7. Assumptions and limitations

1. **Survivorship.** Headline numbers use the current 80-symbol universe. The PIT universe exists
   and should be used going forward; pairs whose archives Binance purged are unrecoverable.
2. **Ambiguous barrier bars** (2.4%) are always resolved as stop-first. This is pessimistic by
   design and is applied identically to zones and controls.
3. **No order book data.** Slippage is modelled from a liquidity proxy built from trailing dollar
   volume, not from measured spreads. Real costs on tier-4 symbols are probably worse than modelled.
4. **Spot data, perp costs.** The MEXC futures cost ladder is applied to Binance *spot* bars. Perp
   funding is parameterised but not populated — no funding history was collected.
5. **Live comparison window is 10 days.** Fidelity and outcome cross-checks are strong (bit-exact)
   but confined to 2026-09-07 → 2026-09-17. Nothing verifies that the 2024 detector configuration
   matched today's; `config_hash` shows two distinct configs even within the 10-day live window.
6. **Four of the 80 backfilled symbols produce no zones** (76 in the final dataset): BNCBUSDT has
   only 298 bars, below the 400-bar window; USDCUSDT and RLUSDUSDT are stablecoin pairs with no
   qualifying wick/volume structure; UUSDT likewise yields none.
7. **The DSR trial count is 84 and append-only**, spanning three re-runs of the same grid. A
   mechanical re-run is not a new hypothesis, so this over-counts; since DSR is 0.00 either way, the
   conclusion is unaffected.
