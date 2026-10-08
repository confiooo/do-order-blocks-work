# Family A — S1 and S5 results (tier D only)

**Run:** 2026-09-17T16:21:25Z
**Pre-registration:** `docs/research/preregistration.md` (git SHA `2ee634de2f7709e02256c346329a5d94e6282d6a`)
**Code SHA:** `2ee634de2f7709e02256c346329a5d94e6282d6a`
**Holdout (tier H):** not read. Firewall F not used as evidence.
**Trials logged:** 2 (S1, S5) into `data/research/research.sqlite`. S2/S3/S4 were not run.
**Detector:** unmodified. Pass R only changed the *call* `min_distance_from_high_low = 0.0` and flagged `passes_swing_gate`.

Nothing is VOID.

A clean FALSIFIED on both trials is the result. S3 is nonetheless authorized by the letter of §9.4, because S5's `Δ_zone` CI is not contained in ±5 bps. That is not a claim that S3 will work.

---

## Verdicts

| Item | Result |
|---|---|
| **S1** | **FALSIFIED** |
| **S5** | **FALSIFIED** |
| Event-marker stop (§13.2) | **not fired** |
| S3 authorized (§9.4) | **yes** (only because `Δ_zone` CI is not ⊂ [−5, +5] bps) |
| S4 / S2 | unauthorized |
| H | sealed |

---

## A. Pass R (required for S1)

Replay of `detect_order_blocks` on the same 15m history and 400-bar rolling window, with `min_distance_from_high_low = 0.0`. Every other live detector parameter unchanged (`strength_threshold=1.35`, `wick_to_body_ratio_min=1.2`, `min_volume_ratio=1.8`, `max_distance_to_price=0.45`, `btc_volatility_threshold=0.045`, `min_signal_interval_minutes=15`). `passes_swing_gate = (distance_to_swing >= 0.5)`.

| Population | n |
|---|---|
| Pass P (existing replay) | 152,266 |
| Pass R (swing floor off) | 294,187 |
| Pass R ∩ `passes_swing_gate` | 152,266 |
| Pass R failing the swing gate | 141,921 |
| **n_BREAK** (full Pass R, ungated) | **86,219** |
| **n_NEAR** | **84,874** |
| **n_FAR** | **123,018** |
| SKIP (broke both swings) | 76 |

**Bit-identity (VOID check).** Pass R ∩ `passes_swing_gate` vs Pass P: 152,266 overlapping `(symbol, source_open_time)` rows; 0 Pass-P-only; 0 Pass-R-gate-only; `zone_high`, `zone_low`, `volume_ratio`, `wick_ratio`, `distance_to_swing` bit-identical (max abs diff 0). **S1 is not VOID on identity.**

Causal features/labels were attached from the existing `zone_dataset.parquet` on the overlap and computed for the new NEAR/BREAK-eligible rows. Written to `data/research/zones/replay_zones_pass_r.parquet` and `data/research/features/zone_dataset_pass_r.parquet`.

---

## B. S1 — Level-mechanism split (Osler)

Maker/limit at the far zone edge, 8 h horizon, `T = 1.0 σ̂_8h`, `L = 0.5 σ̂_8h`, costs from `config/research_costs.json`, `K_max = 5`, queue by `pit_rank_prev_1d` then `zone_uid`. FAR is scored, not traded. Inference on residual net (`ret_resid_net`) for the cell deltas; portfolio gates on net with the cost ladder.

### Sample (tier D, common gates)

| | n |
|---|---|
| Pass R events on D | 207,252 |
| Pass common gates | 95,873 |
| skip vol (`σ_8h` outside [0.60%, 4.00%]) | 41,155 |
| skip dislocation (BTC 4h range > expanding p99) | 4,845 |
| skip integrity (`fwd_bars_available` < 64) | 0 |
| skip universe (`pit_rank` not in 1–80 or listed < 30d) | 83,818 |
| gated D cells | BREAK 30,748 / NEAR 27,074 / FAR 38,041 / SKIP 10 |
| filled, tradeable (BREAK+NEAR) | 45,988 |
| BREAK fill rate | 79.6% (not unreachable) |
| ambiguous first-touch | 0.046% (Gate 1 pass) |

### Cell deltas (residual net, bps)

| Statistic | Point | 95% CI | vs 0 |
|---|---|---|---|
| **Δ_B** (BREAK − FAR) | **+4.07** | **[−2.23, +10.03]** | includes 0 |
| **Δ_N** (NEAR − FAR) | **−0.48** | **[−4.01, +3.11]** | includes 0 |
| FAR mean residual net | −29.76 | [−32.98, −26.65] | excludes 0 (negative) |

BREAK is 4 bps less bad than FAR. The CI includes 0 and does not sit inside ±5 bps (upper bound +10). NEAR is indistinguishable from FAR. Osler's continuation-at-break / reversion-at-untouched-level split is **not detected** at this horizon and cost.

Per-cell filled net (ladder, not residual): BREAK −30.5 bps, NEAR −37.0, FAR −36.0. Gross: BREAK +1.6, NEAR −4.9, FAR −2.7.

### Traded book (BREAK+NEAR, K_max=5)

| Metric | Value |
|---|---|
| Gross expectancy | −1.44 bps/trade, CI [−3.68, +0.68] |
| **Net expectancy (ladder)** | **−33.50 bps/trade, CI [−35.80, −31.33]** |
| Flat 22 / 32 / 52 bps | −23.4 / −33.4 / −53.4 |
| Hit rate (target / n) | 26.14% (12,020 / 27,594 / 6,374 target/stop/timeout) |
| Wilson on n_eff | [25.71%, 26.57%] — upper bound **below** p* = 42% and below p₀ = 33.3% |
| n_eff (measured ρ̄ = 0.0088) | 40,921 |
| n_eff at ρ̄ = 0.70 (pessimistic) | 5,246 |
| Σū | 3,806 |
| Rolling folds positive | **0 / 6** |
| CPCV frozen-rule paths positive | **0 / 11** |
| Ann. Sharpe (daily equity) | −17.1 |
| DSR | ~0 |
| Rejection rate at K_max=5 | 68.4% |
| K_max ∈ {2,5,10,20} sign | stable (all negative) |

Measured ρ̄ ≈ 0.009 is far below the protocol's 0.35–0.70 forecast; it is a same-day residual correlation, not BTC beta. Gate 7 still passes at the pessimistic ρ̄. Sample size is not why S1 fails.

### Gates 0–7 (D only)

| Gate | Result | Why |
|---|---|---|
| 0 pre-registration | pass | this file is the contract |
| 1 leakage | pass | identity bit-exact; ambiguity 0.05% |
| **2 economic** | **FALSIFIED** | net32 < 0; net CI LB < 0; net52 < 0; Wilson hit UB 26.6% ≤ p* ; also net22 = −23.4 ≤ −10 |
| **3 walk-forward** | **FALSIFIED** | 0 of 6 rolling folds positive |
| 4 multiplicity | cannot pass | Hansen SPA across five Family-A curves is undefined with only S1 and S5 run |
| **5 overfitting** | **FALSIFIED** | DSR ≈ 0; CPCV median path ≤ 0 (0/11 positive) |
| 6 holdout | not run | H sealed |
| 7 sample | pass | n_eff ≫ 300 |
| S1 Δ requirement | fail | Δ_B CI includes 0; not Δ_B > Δ_N > 0 |

**S1 verdict: FALSIFIED.** Independent grounds: Gate 2, Gate 3, Gate 5. The level-mechanism split does not produce a tradeable book and does not separate BREAK from FAR at 95%.

---

## C. S5 — Zone vs matched control (S3 rule on both arms)

S3's signed rule, verbatim, on both arms: skip `|resid_ret_4h| < 1 σ̂_4h`; T1 (pit ≤ 15) momentum, else fade; maker at the bar's far edge; 1.0σ / 0.5σ; 8 h. Zone arm capacity-capped at K_max=5. Control arm unconstrained, as pre-registered. Matching: up to 3 controls per zone, ±48 h, not Pass R, fail volume or wick gate, exact bins on BTC-range tercile / ATR decile / pit tier / 6h block / residual sign / |resid| decile, Euclidean NN without replacement.

### Construction

| | n |
|---|---|
| S3-signalled Pass P zones on gated D | 8,592 |
| Unmatched drops | 2,236 |
| Zone arm after match + S3-skip alignment | 5,696 |
| Control arm (after S3 magnitude skip) | 11,878 |
| Matching table (pre-S3 skip) | 14,320 |
| Covariate SMD < 0.10 on every key | **yes** (max \|SMD\| = 0.043 on `resid_abs`) |
| Near-gate controls (within 10% of 1.35 vol or 1.2 wick) | 15.7% (< 40% threshold) |

Matching is a match. S5 is not INCONCLUSIVE on construction.

### Primary differences

| Statistic | Point | 95% CI |
|---|---|---|
| **Δ_zone** (net bps, zone − control) | **+7.44** | **[+1.36, +13.40]** |
| Zone mean net | −33.18 | [−39.39, −26.54] |
| Control mean net | −40.61 | — |
| **Δ_retest** (pp) | **+0.17** | **[−1.36, +1.77]** ⊂ [−3, +3] |
| Zone retest rate | 78.84% | |
| Control retest rate | 78.67% | |
| Δ_fill (pp) | +1.22 | (79.4% vs 78.2%) |
| **Δ_rv** (`fwd_rv_4h`, absolute) | **+9.41×10⁻⁵** | **[+5.01×10⁻⁵, +1.38×10⁻⁴]** |

`Δ_zone` CI **excludes 0** and is **not** contained in ±5 bps. The zone arm loses 7 bps *less* than the matched control after costs. Both arms lose: zone −33 bps, control −41 bps. This is not an edge. It is a statistically detectable difference in the depth of a hole.

`Δ_retest` CI is inside ±3 pp. The ~79% retest rate equals the control's. Under §8 item 12 this is geometry, not order flow, and may not be cited as evidence of absorption.

`Δ_rv` CI excludes 0 (zone slightly higher forward 4h RV: 0.532% vs 0.523%). Same sign as Round 1's unsigned test, smaller magnitude (Round 1: +0.000235). It does not rescue the book.

Walk-forward on **Δ_zone** (not on zone-arm PnL): +11.6, +8.7, +13.3, +6.9, −7.8, −3.0 bps. **4 of 6** folds have Δ_zone > 0. Zone-arm *PnL* folds positive: **0 / 6**.

### Gates 0–7 on the zone arm

| Gate | Result | Why |
|---|---|---|
| 0 | pass | |
| 1 | pass | ambiguity 0.04% |
| **2 economic** | **FALSIFIED** | net32 = −31.5 < 0; Wilson hit UB 28.5% < p* |
| **3 walk-forward** | **FALSIFIED** | 0 of 6 zone-arm folds positive (the Δ_zone-fold rule of ≤2/6 is *not* hit: 4/6) |
| 4 | cannot pass | SPA not defined on two trials |
| 5 | fail (not independently falsified by CPCV-median rule in this implementation's S5 gate) | DSR ≈ 0; 0/11 CPCV paths positive |
| 6 | not run | H sealed |
| 7 | pass | n_eff = 3,347 |

**S5 verdict: FALSIFIED.** Independent grounds: (i) `Δ_retest` CI ⊂ [−3, +3] pp (geometry); (ii) Gate 2, zone arm; (iii) Gate 3, zone-arm folds. The headline event-marker falsification (`Δ_zone` CI ⊂ [−5, +5] bps) **did not fire**.

Round 1's unsigned close-to-close zone-vs-matched test had a CI covering 0 and a significant +Δ_rv. This trial is not that test. S5 uses S3's signed maker rule. The signed difference is +7 bps net, CI excluding 0; the retest rate is not.

---

## D. Earn-the-next-trial (§9.4 / §13.2)

Pre-committed rule, applied as written:

- S3 runs only if S5's `Δ_zone` CI is **not** contained in ±5 bps **or** S1's `Δ_B` CI excludes zero.
- Event-marker stop (§13.2) fires if `Δ_zone` CI ⊂ [−5, +5] bps.

**Facts used:**

- S5 `Δ_zone` 95% CI = **[+1.36, +13.40] bps**, which is **not** a subset of [−5, +5].
- S1 `Δ_B` 95% CI = **[−2.23, +10.03] bps**, which **includes 0**.

**Decision: AUTHORIZE S3 only.** S4 and S2 remain unauthorized until S3 is at least INCONCLUSIVE with a positive point estimate. Holdout remains sealed.

**Event-marker stop: not fired.** The zone is not classified as a pure event marker under the ±5 bps containment rule. That does not mean the zone is a trading edge. Both S5 arms lose money after costs; S1 is FALSIFIED; Gate 4 cannot pass on this round.

This authorization is mechanical. It is not a forecast that S3 will pass. S3 tests the residual fade/momentum rule on Pass P, which S5 already showed loses ~33 bps net on the zone arm and ~41 bps on matched non-zones.

---

## E. What this is not / limitations

- Round 1's 84-config screen was not rerun.
- Round 1's unsigned zone-vs-matched close-to-close result is not S5.
- No parameter variants were added. One horizon, one barrier pair, one entry, one cost ladder.
- Production/homelab DB (`data/db/db.sqlite`) was not opened. Ledger is `data/research/research.sqlite`.
- **Survivorship (declared, Round 1).** `pit_universe` ranks 597 pairs including 115 delisted-before-snapshot names. The 15m OHLCV/label panel is the current ~80-symbol research set. Ranking is PIT over the full listed cross-section; *outcomes* are only on surviving names with 15m history. Round 1 noted the residual bias cannot be removed from this source. The trading verdicts are negative, so that bias does not create a false *positive* edge. It could inflate `Δ_zone` (the one positive difference); that difference is already not being treated as evidence of an edge.
- Funding history is not in the research store. Funding is 1 bp × actual 8 h UTC settlements crossed (00/08/16), per the cost file, not a live funding series.
- Gate 4 (SPA/StepM across five systems) cannot pass until S2/S3/S4 have been run or formally abandoned. It is not a reason to keep searching.

---

## Artifacts

- Pass R zones: `data/research/zones/replay_zones_pass_r.parquet`
- Pass R dataset: `data/research/features/zone_dataset_pass_r.parquet`
- Pass R summary (identity + cell counts): `data/research/zones/replay_pass_r_summary.json`
- S5 matching table: `data/research/features/s5_control_matches.parquet`
- Machine-readable report: `data/research/reports/s1_s5_results.json`
- S1 checkpoint: `data/research/reports/s1_checkpoint.json`
- This writeup: `docs/research/s1_s5_results.md`
- Ledger sqlite: `data/research/research.sqlite` (experiment_id 1 = S1, 2 = S5)
- Ledger jsonl: `data/research/reports/experiments_ledger.jsonl`
