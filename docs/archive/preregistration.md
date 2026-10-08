# HAWK Pre-Registration — Family A, Round 1

**Status:** pre-registration. Written *before* any Family-A test touches the development window.
**Family:** `A1` (order-block event stream, directional decision layer).
**Compiled:** 2026-09-17.
**Governing protocol:** `docs/research/validation_protocol.md` (§6.5 items 1–7 are all answered below).
**Evidence base:** `docs/research/evidence_signals.md`, `docs/research/evidence_context_costs.md`, `config/research_costs.json`.
**Detector:** `src/hawk_eye/logic/order_blocks.py` — **FIXED**. Not redesigned by any system here.

This file must be committed to git and its SHA recorded before the first read of tier D for
family `A1`. After that commit it may not be edited. Amendments are permitted only where this
document explicitly authorises them (§3.4 barrier calibration, §3.11 threshold recomputation), must
be committed as a separate file referencing this SHA, and must be made before any outcome statistic
is examined.

---

## 0. Executive summary of what is being locked down

**Five systems, one configuration each, 5 raw trials, forecast `N_eff` 4.2–5.0 against a budget of
7. All five fit.** No system has a parameter grid. There is exactly one outcome horizon (8 h), one
barrier ratio (2:1, volatility-scaled), one entry convention (resting maker limit at the far zone
edge) and one cost ladder across all five. Every remaining knob is a *joint robustness requirement*
whose worst value must hold, not a menu to select from.

**The single arithmetic fact that drives the whole design.** For a first-touch barrier strategy with
target `T`, stop `L` and round-trip cost `c`, the break-even hit rate is `p* = (L + c)/(T + L)` but
the driftless-martingale null hit rate is `p₀ = L/(T + L)`, **not 0.50**. So the edge a design
actually has to manufacture is

```
required edge above the null  =  p* - p₀  =  c / (T + L)
```

The `evidence_context_costs.md` §5.6 table's "excess over coin flip" column compares `p*` to 0.50,
which is only the right null for symmetric barriers. Corrected:

| Design | `T`/`L` (bps) | `c` (bps) | `p*` | null `p₀` | **required edge above null** |
|---|---|---|---|---|---|
| Current HAWK: ±1%, taker/taker, rank 26–45 | 100 / 100 | 34 | 67.0% | 50.0% | **17.0 pp** |
| ±1% with maker entry | 100 / 100 | 22 | 61.0% | 50.0% | 11.0 pp |
| ±1%, maker, 2:1 (the cost doc's "48%" row) | 100 / 50 | 22 | 48.0% | 33.3% | 14.7 pp |
| **This pre-registration: 1.0σ̂/0.5σ̂ at 8 h, maker entry** | **170 / 85** | **22** | **42.0%** | **33.3%** | **8.63 pp** |

Two consequences that are easy to get wrong and are therefore stated up front:

1. **Asymmetry alone buys nothing.** Moving from 100/100 to 100/50 at the same cost *raises* the
   required edge above the null from 11.0 pp to 14.7 pp, because it shrinks the barrier span from
   200 bps to 150 bps and the required edge is `c / span`. The cost doc's "2:1 R:R is worth −11 pp
   of required win rate" is an artefact of comparing against 0.50. **The real levers are (a) cutting
   `c` and (b) widening the total span `T + L`.** Asymmetry is still worth having — it raises span
   for a given target and makes timeouts cheaper — but only in combination with a wider span.
2. So the design pairs maker entry (`c` 34 → 22 bps) with a **wider, volatility-scaled span**
   (255 bps rather than 200) obtained by moving the horizon out to 8 h. That takes the required
   edge above the null from **17.0 pp to 8.63 pp — a 49% reduction.** That, not "67% → 48%", is the
   honest measure of what this redesign buys.

**Where power now binds.** With an 8 h outcome, an 8 h entry window (max label life 16 h) and
BTC-residualised outcomes (`ρ̄` forecast 0.35), `n_eff` on the 546-day development window is
**≈ 2,200**, not 720 — a 3.1× gain over the protocol's baseline, entirely from shortening `H_max`
and residualising. The hit-rate test can then detect `p ≥ 45.0%` at 80% power. But the Deflated
Sharpe gate is calendar-bound and **no design change can relax it**: it still requires an annualised
net Sharpe of ≈ 1.90, which on this trade frequency corresponds to `p ≈ 45.6%`. **Gate 5 binds, by a
hair, and it is the gate that will decide every one of these five systems.**

**Bottom line requirement, stated once, numerically.** Each system must deliver a barrier-hit rate
of **≈ 45.6%** against a break-even of **42.0%** and a driftless null of **33.3%** — equivalently a
**gross** directional edge of ≈ 31 bps per trade, an information coefficient of ≈ 0.18 against the
8 h volatility scale. Published gross short-horizon crypto reversal effects are of that order;
every cost-adjusted replication of them has come in far lower. **My honest joint prior that at least
one of the five reaches "EVIDENCE OF AN EDGE" is 12–18%. The modal outcome is INCONCLUSIVE across
all five, and the pre-committed response to that is in §9, not more searching.**

---

## 1. Deviations from `validation_protocol.md`, declared up front

| Protocol item | Protocol value | This pre-registration | Justification |
|---|---|---|---|
| `H_max` [A5] | 48 h | **16 h** (8 h entry window + 8 h hold) | §4.4(a): the highest-leverage change available. Raises `n_eff` from 720 to ≈ 2,200. |
| Purge | 48 h | **16 h** = `H_max` | Derived from `H_max` by the §2.3 formula, not chosen. |
| Embargo | 72 h | **40 h** = `H_max` + `L_feat` (24 h) | `L_feat` = 24 h because every feature used by any of the five rules has a lookback ≤ 96 bars. The 2,880-bar (30 d) dollar-volume features are **excluded from all five rules** precisely so the embargo stays short; liquidity tiering uses a trailing-**24 h** rank instead. Dead zone 56 h × 10 boundaries = 23 of 546 days = **4.2%** of D (protocol baseline was 6.8%). |
| Null hit rate | `p₀ = 0.50` | **`p₀ = L/(T+L) = 0.3333`**; the gate is against `p̄* = 0.420` | §0. A 2:1 barrier has a 33.3% martingale null. Testing against 0.50 would be testing the wrong hypothesis in the *favourable* direction, which is worse than testing the wrong hypothesis. |
| Fold structure | 6 folds, 180/5/60 d | **unchanged** | Operational, pre-registered, not tuned. |
| CPCV | N=12, k=2, 11 paths | **unchanged** | |
| PBO | CSCV S=12, 924 partitions | **unchanged** | |
| `K_max` | 5 | **unchanged** | |
| Trial budget | `N_eff` ≤ 7 | **unchanged**; `M_raw` = 5, forecast `N_eff` 4.2–5.0 | §7 |

Required joint sensitivity (the verdict must hold at the **worst** value of each axis, per §3.12):
`(purge, embargo) ∈ {(16,40), (48,72)}`.

---

## 2. Population definition (identical for all five systems)

### 2.1 The detector is a fixed event generator

`detect_order_blocks` is used unmodified, via `research/replay.py`, pinned to
`LIVE_WINDOW = 400` / `LOOKBACK_RATIO = 0.15` (effective lookback 60 bars), with
`strength_threshold = 1.35`, `wick_to_body_ratio_min = 1.2`. Nothing below changes the volume gate
or the wick gate.

Two replay passes are required:

- **Pass P (passing population):** `min_distance_from_high_low = 0.5`. This is the live-equivalent
  event stream. Used by S2, S3, S4, S5.
- **Pass R (relaxed-swing population):** `min_distance_from_high_low = 0.0`, every row tagged
  `passes_swing_gate = (det_distance_to_swing >= 0.5)`. Used by S1 only.
  **Hard assertion:** the subset of Pass R with `passes_swing_gate == 1` must be bit-identical to
  Pass P on `(symbol, source_open_time, zone_high, zone_low, volume_ratio, wick_ratio,
  distance_to_swing)`. If it is not, the two populations are not comparable and S1 is VOID.

Turning the swing gate into a continuous conditioning variable is prescribed by **both** evidence
reviews (`evidence_signals.md` Topic 1 "Implication for HAWK", first item;
`evidence_context_costs.md` §4.1 final paragraph). The detector's other two gates are untouched, and
the passing subpopulation is unchanged, so all five systems remain mutually comparable.

### 2.2 Universe, point-in-time

- `pit_rank_prev_1d` ≤ 80, where `pit_rank_prev_1d` is the integer rank of the symbol by
  **trailing-24 h quote volume as of the last completed UTC day strictly before the decision bar**,
  computed over **every symbol listed at that time**, not over today's top 80.
- `listed_at + 30 d ≤ decision_time < delisted_at`, from an exchange listing calendar, not from
  today's `exchangeInfo`.
- The universe must contain at least one symbol that was delisted or went to near-zero volume
  inside the window (protocol §9.1 item 2).

### 2.3 Gates applied identically to every system and to every control

Fixed values, no alternatives:

1. **Volatility scale gate.** `0.60% ≤ σ̂_8h ≤ 4.00%`, where `σ̂_8h ≡ rv_24h × sqrt(32)` and `rv_24h`
   is `features.volatility_features`' std of 15 m log returns over the trailing 96 bars, as-of the
   decision bar. Outside that band the zone is SKIPped and counted. Purpose: keep `c / span` bounded
   (≤ 15.3%, so `p*` ≤ 46.6%) and exclude data-error wicks on broken symbols.
2. **Dislocation gate.** SKIP if `btc_range_4h_pct` at the decision bar exceeds its **expanding,
   point-in-time 99th percentile**. Rationale: `evidence_context_costs.md` §4.2 — quoted price
   impact blows out 3.2–9.1× and depth collapses in cascades, so the cost model is *invalid* in
   those minutes; and 80 correlated positions become one. This is a risk/validity filter with a
   cited mechanism, not a tuned parameter.
3. **Data-integrity gate.** SKIP if the `[decision, decision + 16 h]` window contains any gap of
   more than 3 consecutive missing bars, or if the bar spacing is not exactly 900,000 ms
   (`labels.contiguous_forward_bars`).
4. **`btc_stable` is NOT a filter.** The full Pass-P / Pass-R population is used regardless of
   `btc_stable`; `btc_stable` and `btc_range_4h_pct` terciles are *reported stratifications* only.
   Justification: `evidence_context_costs.md` §2.3 — the boolean is a lossy encoding of a
   continuous, asymmetric relationship, and what it predicts is an interaction (signal-to-noise),
   which needs far more data than a main effect.
5. **`signal_gate` and `min_signal_interval_per_pair` are NOT applied.** Per `replay.py`'s docstring
   these govern Telegram paging only and do not shape the zone population. The research population
   is the replay rows, not the `would_alert` rows.

### 2.4 Calendar tiers

Unchanged from protocol §2.1. Development **D** = 2024-09-01 → 2026-02-28 (546 d). Firewall **F** =
2026-03-01 → 2026-03-31, discarded. Holdout **H** = 2026-04-01 → 2026-08-31 (153 d), **one touch
ever**, allocation rule in §7.4. Live tier **L** is not evidence and is not cited anywhere in this
document (protocol §4.4(b): ~10 effective observations).

---

## 3. Common specification (referenced by all five systems)

### 3.1 Barriers — stated in both volatility and percentage terms

Per zone, from the decision bar's own `rv_24h`:

```
sigma_hat_8h = rv_24h * sqrt(32)            # 8 h return scale, causal
T  = 1.0 * sigma_hat_8h                     # target
L  = 0.5 * sigma_hat_8h                     # stop   -> 2:1 reward:risk
```

| Quantity | Volatility units | ATR units (cross-check) | Percentage at assumed median **[B1]** |
|---|---|---|---|
| Target `T` | `1.0 × σ̂_8h` = `5.657 × rv_24h` | ≈ `3.5 × atr_48_pct` | **1.70%** |
| Stop `L` | `0.5 × σ̂_8h` = `2.828 × rv_24h` | ≈ `1.77 × atr_48_pct` | **0.85%** |
| Span `T + L` | `1.5 × σ̂_8h` | — | **2.55%** (255 bps) |

**[B1]** Assumed median `rv_24h = 0.30%` per 15 m bar for `pit_rank_prev_1d ≤ 80`, giving
`σ̂_8h = 1.70%`. The ATR-unit column assumes `atr_48_pct ≈ 1.7 × rv_24h`; both mappings must be
measured and published. The **multipliers 1.0 and 0.5 are fixed and are not amendable.**

A descriptive readout on the already-computed fixed grid — `first_up_2` / `first_dn_1`, i.e. the
`(0.02, 0.01)` pair at horizon `8h`, label `bar_2_1_8h` in `research/labels.py` — is reported for
every system. It **may not establish or overturn any verdict** (§3.12).

### 3.2 Entry — resting maker limit at the far zone edge

Uniform across all five systems. No system gets its own entry convention.

| Element | Specification |
|---|---|
| Order type | Resting limit, **maker**, placed at the decision bar's close |
| Price, LONG | `limit = zone_low` |
| Price, SHORT | `limit = zone_high` |
| Passivity check | Not placed if `abs(limit - close)/close < 0.0005` (5 bps). Recorded `fill_status = 'not_placed'`. A limit inside 5 bps of the last trade cannot be assumed passive. |
| Time validity | 32 bars (8 h) from the decision bar's close. Cancel-on-expiry. |
| Fill condition | Price must trade **through**, not merely touch: LONG fills iff some bar in the window has `low <= limit * (1 - 1e-4)`; SHORT iff `high >= limit * (1 + 1e-4)`. Protocol §9.5 item 16 / cost doc §5.6 hygiene. |
| Fill price | Exactly `limit`. No improvement assumed. |
| Fill time | Close of the first bar satisfying the fill condition. The fill bar is **excluded** from the outcome window (`labels.candles_after_fill` convention). |
| On no-fill | `fill_status = 'expired_no_fill'`. **A shadow record is written**: the identical barriers measured from `limit` over the identical 8 h window starting at the expiry bar. Adverse selection is then the *measured* quantity `E[ret_gross | filled] - E[ret_gross | shadow]`, not an assumption. |
| Recorded fields | `limit_price`, `fill_status ∈ {filled, expired_no_fill, not_placed}`, `fill_open_time`, `traded_through`, `fill_lag_bars`, and the full `shadow_*` outcome block. |

Why `zone_low`/`zone_high` and not HAWK's current `hypothetical_entry_price`: the live convention
(`_entry_price` returns `zone_high` for bullish) requires price to first establish *above* the zone
and then return, which is a conditional event and prices the entry at the *near* edge. Resting at
the far edge is unconditionally passive at the moment of placement (the decision-bar close sits
inside the zone by construction), guarantees maker execution, and is a strictly better price. The
cost is fill rate, and fill rate is measured rather than assumed.

### 3.3 Exit — fixed, zero free parameters

1. **Target:** first bar after the fill bar whose extreme reaches `fill ± T` in the favourable
   direction. Filled at `T`.
2. **Stop:** first bar after the fill bar whose extreme reaches `fill ∓ L`. Filled at `L`
   (optimistic; the ambiguity convention in item 4 pushes the other way).
3. **Time stop:** 32 bars (8 h) after the fill bar. Exit at that bar's close, **taker**.
4. **Same-bar ambiguity:** if a single 15 m bar's range spans both barriers, resolve
   **pessimistically as the stop** (`labels.barrier_label` already does this). Report the ambiguous
   fraction and the optimistic-resolution upper bound. **Hard stop at 10%** (protocol §9.3).
5. **No trade management.** No trailing stop, no break-even move, no partial exits, no scaling. Each
   would be a free parameter and therefore a trial.

Outcomes are reported in **three** categories — target, stop, timeout — never as a two-way win rate.

### 3.4 Two outcome variables, one configuration

| | Definition | Used for |
|---|---|---|
| **Economic** | Raw-price first-touch on intrabar `high`/`low` from `limit`, as §3.3. Net of costs. | Gates 2, 3, 5, 6. The number that matters. |
| **Scientific** | BTC-β-hedged **residual close path**: `p_res(k) = close(k)/fill - beta_btc_24h * (btc_close(k)/btc_close(0) - 1)`, evaluated at 15 m **closes only**, with the same `±T/−L` barriers. | Hit-rate test, `ρ̄` measurement, power. |

The residual barrier is evaluated at closes because the high/low of a difference of two series is
not computable from the two series' OHLC. This is *conservative* (it misses intrabar touches) and it
eliminates same-bar ambiguity from the scientific label entirely. Both outcomes come from one rule,
so this is **one trial, two measurements** — protocol §5.4 explicitly prescribes doing the scientific
test on the hit rate and the economic test on P&L.

**Pre-registered interpretation rule:** if a system passes on the residual outcome but fails on the
raw outcome, the conclusion is *"the edge exists only market-neutral and requires a BTC hedge leg,
which doubles costs and is not representable at HAWK's $40 notional"* — i.e. **not tradeable**, a
negative result, not a partial success.

### 3.5 Portfolio construction

`research/portfolio.build_equity_curve` with `max_concurrent = 5`, `one_position_per_symbol = True`,
`risk_fraction = 0.2`, `ret_col = 'ret_net'`. Queue priority when capacity is short: ascending
`pit_rank_prev_1d` (most liquid first — the cheapest to trade, decided on cost grounds, never on
outcome), tie-broken by `zone_uid` ascending. Rejections are logged; `rejection_rate`,
`capacity_utilization` and `concurrency_histogram` are reported for every system.

### 3.6 Cost model

`config/research_costs.json`, **maker in / taker out**, per-symbol by `pit_rank_prev_1d`:

| Tier | Fees (4 maker + 6 taker) | Spread (maker entry pays 0, taker exit pays `S`) | Slippage `0.25·S·tod·stress` | Funding (1–2 settlements) | **Round trip** |
|---|---|---|---|---|---|
| 1–10 | 10 | 4 | 1 | 2 | **17 bps** |
| 11–25 | 10 | 8 | 2 | 2 | **22 bps** |
| **26–45 (base)** | **10** | **16** | **4** | **2** | **32 bps** |
| 46–65 | 10 | 28 | 7 | 2 | **47 bps** |
| 66–80 | 10 | 45 | 11 | 2 | **68 bps** |

The headline `c = 22 bps` used in §0 and §3.11 is the **notional-weighted mean** across the realised
tier mix, and is pinned to the cost doc's `maker_in_taker_out_rank_26_45 = 22` row for the base
case. Because the ladder is per-symbol, the gate is on **net expectancy**, which handles the
per-zone `p*` automatically. `time_of_day_mult` ∈ {1.0 US/EU overlap, 1.3 thin Asian-evening/early
UTC, 1.5 weekend}; `stress_mult` = 3.0 is moot because §2.3(2) skips those bars. `tick_rounding_bps
= 1` added where tick size exceeds 1 bp of price. No square-root impact term (§5.6: negligible at
$40).

Every metric is reported **gross and net**, and net at **three cost levels: 22 / 32 / 52 bps**, plus
the full per-tier ladder. Funding is charged per **actual settlements crossed**, using the
per-symbol MEXC funding interval pulled from the venue API — not an assumed 8 h.

### 3.7 Inference machinery

- **All inference on the portfolio daily net return series.** Trade-level statistics are descriptive
  and may never gate (protocol §7.2).
- **Bootstrap:** Gonçalves (2011) time-slice stationary panel bootstrap —
  `research/resampling.stationary_bootstrap_ci` / `paired_block_bootstrap_diff_ci`, resampling the
  time index and carrying the whole cross-section. `b = 10 d`, `B = 10,000` for CIs, `B = 5,000` for
  SPA/StepM, **shared index sequences across all five trials** so cross-trial dependence is
  preserved. Percentile-t (studentised) intervals.
- **Sharpe:** Lo (2002) SE with Newey–West HAC **and** Ledoit–Wolf (2008) studentised
  circular-block-bootstrap interval; if they disagree, the wider is used.
- **Multiplicity:** Hansen SPA (`studentize=True`, `bootstrap="stationary"`, `block_size=10`,
  `reps=5000`) as omnibus pre-test across all five curves against **both** a zero-net benchmark and
  a passive equal-weight-altcoin benchmark; then Romano–Wolf StepM for per-system claims. If SPA's
  consistent p > 0.05, **StepM is not reported and no individual claim may be made**.
- **Overfitting:** `multiple_testing.deflated_sharpe_ratio` at the declared `N_eff`;
  `pbo_cscv(S=12)` over the 11 CPCV paths; `MinTRL`; Harvey–Liu haircut Sharpe under Bonferroni,
  Holm and BHY.
- **Sample weights:** AFML §4.4 average uniqueness `ū_n`, recomputed **within each training fold**,
  passed as `sample_weight` to every estimator, with sequential bootstrapping inside any bagged
  ensemble. `Σū` and `n_eff` appear in **every** table.

### 3.8 Metric suite (identical for all five)

Every metric with (i) point estimate, (ii) 95% interval by the method below, (iii) `n_raw`,
(iv) `Σū` / `n_eff`, (v) gross **and** net at 22/32/52 bps.

| Metric | CI method | Gate-eligible? |
|---|---|---|
| **Expectancy** (mean net bps/trade) + 5%-trimmed mean + median | stationary panel bootstrap, percentile-t, `b=10 d`, `B=10,000` | **Yes.** If the sign flips between mean and 5%-trimmed mean, the result rests on ≤5% of trades and is **not a finding**. |
| **Profit factor** (net) | bootstrap percentile only | No. Not quoted unless ≥ 100 losing trades. |
| **Win rate** (3-category: target / stop / timeout) | **Wilson score on `n_eff`**, Clopper–Pearson as conservative cross-check | Yes, but only as "barrier-hit rate vs `p̄*`" |
| **Avg win / avg loss** | bootstrap, reported only as the triple `(hit rate, avg win, avg loss)` | No |
| **Trade frequency** (zones/day, signals/day, fills/day, by month and by `rv_24h` tercile) | Poisson / bootstrap on daily counts | No |
| **MFE / MAE** | bootstrap quantile intervals, full deciles, in bps **and** normalised by `atr_48_pct`; measured on the **un-barriered** 8 h path *and* the barriered path | No |
| **Max drawdown** | **bootstrap distribution: median, 5th, 95th percentile**, plus time-under-water. Realised single number reported but never gated. | Yes — the **95th percentile** only |
| **Sharpe** (annualised ×√365 on daily net) | Lo HAC + Ledoit–Wolf studentised bootstrap, wider of the two; **DSR reported in the same cell** | **Yes** |
| **Sortino, Calmar** | same bootstrap | No |
| **Turnover / avg holding period** | bootstrap | No — reported as the cost-sensitivity multiplier |
| **Fill rate & adverse selection** `E[ret|filled] − E[ret|shadow]` | paired block bootstrap | Yes, as a leakage/realism gate (§3.10) |

### 3.9 Objective-test skeleton

- **Sample:** tier D, 2024-09-01 → 2026-02-28, population per §2, system-specific rule per §4–§8.
- **Null H0:** mean net daily portfolio return = 0, *and* residual 8 h barrier-hit rate = `p̄*`.
- **Benchmarks, all three required:** (i) zero-net-return; (ii) passive equal-weight altcoin basket;
  (iii) **random-signal placebo** — a fair coin at the identical signal timestamps, identical fill
  simulation, identical costs, run on the same 10,000 bootstrap index sequences so the comparison is
  paired. The placebo's bootstrap distribution is the operative null for Gate 2.
- **Ablations (mandatory, not trials):** label shuffle within fold (expect AUC 0.50, expectancy
  ≈ −c); feature `+1`-bar shift (edge must collapse into the CI of zero); feature `−1`-bar shift
  (deliberate leak — edge must *improve markedly*; if it does not, the original leaks).

### 3.10 Pass criteria (identical thresholds, all must hold)

| Gate | Requirement |
|---|---|
| 0 | This file committed, SHA predates first read of D. |
| 1 | All 12 protocol §9 assertions pass; same-bar ambiguity ≤ 10%; `+1`-bar ablation destroys the edge; assumed fill rate < 100% and measured; adverse selection reported. |
| 2 | Net expectancy > 0 at `c = 32 bps` with **bootstrap 95% lower bound > 0**; point estimate > 0 at `c = 52 bps`; residual 8 h barrier-hit rate 95% **lower** bound > `p̄* = 42.0%`; beats the random-signal placebo's 95th percentile. |
| 3 | Net expectancy positive in **≥ 5 of 6** folds; stitched 360-d OOS net Sharpe > 0; no fold > 50% of OOS P&L; no symbol > 25%; no month > 35%; rolling and anchored agree in sign; conclusion unchanged across `K_max ∈ {2,5,10,20}` and `b ∈ {4,10,20,40}`. |
| 4 | Hansen SPA consistent p ≤ 0.05 across all five curves; Romano–Wolf StepM adjusted p ≤ 0.05 for this system; **\|t\| ≥ 2.71** on the OOS portfolio curve with a Ledoit–Wolf studentised SE on `n_eff`; realised Li–Ji `N_eff` ≤ 7; haircut Sharpe (all three adjustments) above the economically required level. |
| 5 | **DSR ≥ 0.95** at the declared `N_eff` (⇒ annualised net SR ≥ **1.90**); **PBO ≤ 0.25**; CPCV median OOS Sharpe ≥ 0.50 × IS Sharpe; **≥ 8 of 11** CPCV paths with positive net expectancy; track length ≥ MinTRL. |
| 6 | Holdout, one shot, frozen: net expectancy > 0; annualised net Sharpe ≥ 1.0; H's point estimate inside D's bootstrap 95% interval (a *higher* H result triggers a leakage re-audit); sign agreement with D on expectancy, hit rate and Sharpe. |
| 7 | `n_eff` ≥ **300** in the evaluated window; ≥ 100 losing trades before any profit-factor or avg-loss figure; bootstrap 95th-percentile MaxDD ≤ **25%** of equity; `Σū` and `n_eff` in every table. |

### 3.11 The numbers behind those thresholds

With `T = 170`, `L = 85`, `c = 22` bps (base case), `net_exp_bps = 255·p − 107`:

| Quantity | Value |
|---|---|
| Driftless-martingale null `p₀` | **33.33%** |
| Break-even `p̄*` | **42.0%** (recomputed once from the measured `(T, L, c)` distribution and published before outcomes are examined — §0 amendment clause) |
| Hit-rate MDE at `n_eff ≈ 2,200`, 80% power, two-sided α = 0.05, against `p̄* = 0.420` | **45.0%** (⇒ net +7.7 bps/trade) |
| DSR-binding hit rate (annualised net SR = 1.90) | **45.6%** (⇒ net +9.3 bps/trade, gross ≈ 31 bps, IC ≈ 0.18) |
| **Operative requirement** | **`p ≥ 45.6%`** |

Sharpe arithmetic, so the DSR figure is checkable rather than asserted. Per-trade net sd
≈ `sqrt(p(1−p)) × 255` ≈ 126 bps. With `f` fills/day at `risk_fraction = 0.2` and within-day
cross-sectional correlation 0.5:

```
mean_daily = f * 0.2 * m
sd_daily   = 0.2 * 126 * sqrt(f + f*(f-1)*0.5)
SR_ann     = (mean_daily / sd_daily) * sqrt(365)
```

| `f` (fills/day) | `sd_daily` (bps) | `SR_ann` per bps of `m` | `m` needed for `SR_ann = 1.90` |
|---|---|---|---|
| 3 | 61.7 | 0.186 | 10.2 bps (`p` = 45.9%) |
| 10 | 186.9 | 0.204 | **9.3 bps (`p` = 45.6%)** |
| 18 | 329.5 | 0.209 | 9.1 bps (`p` = 45.6%) |

**Note what this table says: a 6× increase in trade frequency moves the required per-trade edge by
1.1 bps.** Within-day cross-sectional correlation caps the diversification benefit almost
completely. This independently confirms protocol §4.4(a) — more events buy essentially nothing — and
it is why no system here is tuned for event count.

`n_eff` derivation, per system in §4–§8, from:

```
concurrency  = signals_per_day * 16h / 24h
Sigma_u      = n_raw / concurrency          ( = calendar_hours / 16, the protocol identity)
breadth      = concurrency / (1 + (concurrency - 1) * rho_bar)
n_eff        = Sigma_u * breadth
```

with `ρ̄` forecast **0.35** on the residualised outcome matrix (down from the protocol's [A2] 0.50,
because residualising against BTC is exactly the lever §4.4 identifies). Reported at
`ρ̄ ∈ {0.35, 0.50, 0.70}` as bounds. `ρ̄` must be **measured** on the per-symbol signed residual
barrier outcomes bucketed into 16 h windows, separately for BTC-up / flat / down regimes, and the
**highest** used.

### 3.12 What is *not* a trial, and the rule that keeps it that way

The following are **joint robustness requirements**. The published verdict must hold at the **worst**
value of every axis simultaneously. Selecting the best value of any axis is a protocol breach and
retroactively converts that axis into `k` trials.

`c ∈ {22, 32, 52}` bps and the full tier ladder · `K_max ∈ {2, 5, 10, 20}` · `b ∈ {4, 10, 20, 40}` d ·
`(purge, embargo) ∈ {(16,40), (48,72)}` · rolling **and** anchored walk-forward · pessimistic **and**
optimistic same-bar resolution (conclusion must hold under pessimistic) · the fixed-`%`
`bar_2_1_8h` readout · per-fold, per-symbol, per-month, per-tier, per-cell and per-decile breakdowns
*within* a pre-registered rule · MFE/MAE deciles · the three benchmarks and three ablations of §3.9.

Also not trials: the Family-B univariate screens (separate family, BH-FDR `q = 0.10`, cap 40,
**may never be cited as evidence**), and dataset-construction steps (Pass R, control matching, ATR
barrier labels).

---

## 4. System S1 — Level-Mechanism Split (Osler)

> **Rank 1 of 5. Prior that it works: 20%.**

### 1. Hypothesis

*Within the volume-and-wick-qualified candidate population, forward 8 h residual outcome asymmetry
depends on the decision bar's geometry relative to the prior-60-bar swing extremes, with opposite
signs on either side of the extreme, and the population HAWK currently discards (`distance_to_swing
< 0.5%`) carries the reversal branch while the population that *broke* the extreme carries the
stronger continuation branch.*

Falsifiable, directional, with magnitude: the pre-registered rule of item 3 achieves a residual 8 h
barrier-hit rate ≥ **45.6%** (net expectancy ≥ +9.3 bps/trade at `c = 22 bps`), **and** the ordered
contrast `E[resid | BREAK] > E[resid | NEAR] > E[resid | FAR]` holds with the `BREAK − FAR`
difference's 95% CI excluding zero.

**Mechanism.** Osler (2003, *JF* 58(5), 1791–1819): take-profit orders cluster **at** round
numbers/levels (counter-trend limit orders ⇒ negative feedback ⇒ reversal), stop-loss orders cluster
**just beyond** them (trend-following market orders ⇒ positive feedback ⇒ acceleration). Osler
(2005, *JIMF* 24(2), 219–241): the stop-loss response is **larger and longer-lasting** than the
take-profit response, and "most results are statistically significant for hours, although not for
days" — which brackets an 8 h horizon exactly. Osler (2000, *FRBNY EPR*): published levels stopped
intraday trends 56.2% of the time versus arbitrary levels.

**Prior: 20%.** `evidence_context_costs.md` §6 gives 50% to "liquidity / stop-hunt / cascade
microstructure conditioning materially improves results", and calls Osler the single strongest
citation in the family. Marked down to 20% because: the asset class is FX spot, not crypto perps;
the trading-rule versions of level rules died on costs (Curcio, Goodhart, Guillaume & Payne 1997 —
profitable only in trending sub-periods, gone after spreads) and out-of-sample (Sullivan,
Timmermann & White 1999 — "results completely reversed" 1987–96); and Osler's results are statements
about conditional *paths*, not about net-of-cost profit, which she does not claim.

### 2. Why the zone information specifically could provide an edge

The zone contributes two things a random bar does not. (a) The volume-and-wick gate selects bars
where a large amount of one-sided flow met resting liquidity — in Osler's terms, bars where an order
cluster was *consumed*. (b) The zone's high–low range is the empirical footprint of where that
liquidity sat, and therefore a non-arbitrary resting price for a passive re-entry. The swing
geometry is what says **which** cluster was consumed: take-profit (⇒ reversal) or stop-loss
(⇒ continuation). Neither piece is available on a bar chosen at random. This is the only one of the
five systems where the zone plausibly supplies *directional* information rather than merely marking
an event.

### 3. Exact direction rule

All inputs from the decision bar's close or earlier. Computed from `research/features.py`
`swing_features` and `detector_features` over the same prior-60-bar window the detector uses
(`shift(1).rolling(60)`).

```
if broke_swing_high == 1 and broke_swing_low == 1:      side = SKIP   # outside bar, ambiguous
elif broke_swing_high == 1:                             side = LONG   # stop cascade above
elif broke_swing_low  == 1:                             side = SHORT  # stop cascade below
elif det_distance_to_swing < 0.5:                       # NEAR: HAWK currently discards these
        if dist_to_swing_high_pct < dist_to_swing_low_pct:  side = SHORT  # TP cluster at resistance
        else:                                               side = LONG   # TP cluster at support
else:                                                   # FAR: what HAWK currently trades
        side = REFERENCE_ONLY                           # no capital; scored, never traded
```

Tradeable population = `BREAK ∪ NEAR`. `FAR` is scored with the NEAR sign rule as the internal
reference cell and receives no capital. The three-cell ordering is the scientific statistic;
monotonicity across three pre-specified cells is far harder to produce by chance than a single
threshold result (`evidence_signals.md` Topic 4 implication 3).

### 4. Exact entry rule

§3.2 verbatim. LONG rests at `zone_low`, SHORT at `zone_high`, 8 h validity, trade-through fill,
shadow record on no-fill.

**S1-specific realism note:** `BREAK` trades will have systematically **lower** fill rates than
`NEAR` trades, because a bar that broke a swing extreme is unlikely to retrace to its far edge
within 8 h. Fill rate is therefore reported separately per cell, and the `BREAK` shadow arm is the
primary evidence on whether the edge survives a realistic entry. The obvious alternative — resting
at the **broken level** `swing_high_prior` / `swing_low_prior` rather than the zone edge — is
mechanically better motivated by Osler and is expected to fill more often. **It is not authorised
here**; it would be a second entry convention and therefore a sixth trial. It is the first candidate
for Round 2 if S1 shows edge in the shadow arm but not the filled arm.

### 5. Exact exit rules

§3.1 and §3.3 verbatim: `T = 1.0 σ̂_8h` ≈ **1.70%** ≈ 3.5 × ATR(48), `L = 0.5 σ̂_8h` ≈ **0.85%**
≈ 1.77 × ATR(48), time stop 32 bars after fill, taker exit, ambiguity → stop, no trade management.

### 6. Required existing HAWK fields

From the `zones` schema (`src/hawk_eye/db.py`) / Pass-R replay rows, named exactly:
`symbol`, `source_open_time`, `direction`, `zone_high`, `zone_low`, `zone_midpoint`,
`zone_size_pct`, `volume_ratio`, `wick_ratio`, `distance_to_swing`, `btc_range_4h_pct`,
`btc_stable`, `universe_quote_volume`.

Derived from stored 15 m OHLCV via `research/features.py`:
`det_volume_ratio`, `det_wick_ratio`, `det_distance_to_swing`, `det_zone_size_pct`,
`dist_to_swing_high_pct`, `dist_to_swing_low_pct`, `broke_swing_high`, `broke_swing_low`,
`swing_high_prior`, `swing_low_prior`, `rv_24h`, `atr_48_pct`, `beta_btc_24h`, `btc_ret_4h`,
`hour_of_day_utc`, `is_weekend`.

Deliberately **not used**: `distance_to_price` (algebraically a restatement of bar geometry, per
`evidence_signals.md` Topic 1 note 3 — collinear with `wick_ratio`, not information), `confidence`,
`symbol_return_*` (replaced by BTC-residualised equivalents).

### 7. Additional data required

- **Pass R** (relaxed-swing replay). Recomputed from already-stored bars. **Zero download cost.**
- ATR-normalised first-touch labels from the fill price (§11 item 4). New label family, computed
  from stored bars.
- Per-symbol MEXC funding interval + realised funding history (cheap API pull).
- **Explicitly excluded:** the round-number-grid test motivated by Osler's round-number clustering.
  It is a genuine test but the level grid is scale-dependent across an 80-coin universe
  (BTC at 10⁴ vs a coin at 10⁻⁴), it is a multiple-comparison trap, and it would cost a trial. It is
  deferred to Round 2 with the grid definition to be pre-registered then.
- **No order-book or trade-level data.** Nothing here needs it.

### 8. Market condition / behaviour exploited

Concentration of resting conditional orders at obvious reference prices. Just beyond a swing extreme
sit stop-losses, whose triggering is a market order that consumes depth and mechanically accelerates
price — a positive-feedback cascade with a documented half-life of hours and 88% of forced flow
completing inside 30 minutes. At the extreme itself sit take-profits, which are counter-trend limit
orders and mechanically provide liquidity, damping and reversing the move. Economically the trade is
**providing liquidity into an inventory-driven dislocation** (NEAR branch) or **front-running the
remaining stop queue** (BREAK branch). HAWK's `min_distance_from_high_low = 0.5` gate currently
excludes the first branch and, by requiring distance from the extreme, is biased against the second.

### 9. The objective test

- **H0 (economic):** mean net daily portfolio return of the `BREAK ∪ NEAR` rule = 0.
- **H0 (scientific):** residual 8 h barrier-hit rate = `p̄* = 42.0%`; and
  `Δ_B ≡ E[resid net | BREAK] − E[resid net | FAR] = 0`, `Δ_N ≡ E[resid net | NEAR] − E[resid net |
  FAR] = 0`.
- **Tests:** §3.7 machinery. `Δ_B` and `Δ_N` via `paired_block_bootstrap_diff_ci` with the shared
  index sequence, one-sided at 5% in the pre-registered direction, *inside the single trial*. Ordered
  alternative tested with a Jonckheere–Terpstra trend statistic across (FAR, NEAR, BREAK),
  bootstrapped by the same time-slice scheme.
- **Control / benchmark:** the **FAR cell** — literally the population HAWK trades today — plus the
  §3.9 random-signal placebo and the passive equal-weight-altcoin basket.
- **Sample:** tier D, Pass R, §2.2–2.3 gates.
- **Outcome variables:** §3.4 (economic = `ret_net`; scientific = residual close-path barrier).

### 10. Metrics

§3.8 in full, gross and net at 22/32/52 bps and per tier, **additionally disaggregated by cell**
(BREAK / NEAR / FAR) and by fill status (filled / shadow). Per-cell fill rate and per-cell adverse
selection are mandatory.

### 11. Evidence of edge (pass)

Gates 0–7 of §3.10 all pass, **plus** the S1-specific requirement: `Δ_B` 95% CI excludes zero with
`Δ_B > 0`, and the point estimates satisfy `Δ_B > Δ_N > 0`. Concretely: DSR ≥ 0.95 (annualised net
SR ≥ 1.90), PBO ≤ 0.25, |t| ≥ 2.71, ≥ 5/6 folds positive, ≥ 8/11 CPCV paths positive,
`n_eff` ≥ 300, hit-rate 95% lower bound > 42.0%.

### 12. Falsification (any one suffices)

- Residual hit-rate 95% **upper** bound ≤ **42.0%**.
- ≤ 2 of 6 folds with positive net expectancy.
- PBO > 0.50, or CPCV median OOS Sharpe ≤ 0.
- Net expectancy point estimate ≤ **−10 bps**/trade at `c = 22 bps`.
- **Both** `Δ_B` and `Δ_N` 95% CIs contained inside **±5 bps** of expectancy. This is the specific
  kill shot for the whole "the swing gate is inverted" hypothesis: it says swing geometry carries no
  information in *either* direction, and therefore neither relaxing nor tightening the gate can
  help.
- `BREAK` fill rate < 15% **and** the `BREAK` shadow arm's expectancy 95% CI contains zero — the
  branch is unreachable and there is nothing there to reach.
- Same-bar ambiguity > 10% and 1 m data unavailable ⇒ VOID, then falsified on the barrier
  formulation.

### 13. Execution realism

Assumptions tagged **[B*]**, to be replaced by measurements.

| Quantity | Forecast |
|---|---|
| Pass-P zone rate **[B2]** | 30,000 / 730 d ≈ **41/day** |
| Pass-R inflation from dropping the swing gate **[B3]** | ×1.8 ⇒ ≈ **74/day** candidates |
| BREAK share **[B4]** ≈ 25%, NEAR share **[B5]** ≈ 30% | ≈ **41/day** signalled |
| Fill rate within 8 h at the far edge **[B6]** ≈ 45% (NEAR ~55%, BREAK ~25%) | ≈ **18 fills/day** |
| `E[n_open]` = 18 × 16/24 (entry window + hold) | ≈ **12** vs `K_max = 5` ⇒ **cap binds** |
| Forecast `rejection_rate` | **50–60%** — the highest of the five. Reported, and the `K_max ∈ {2,5,10,20}` sensitivity is load-bearing here. |
| Cost | maker in / taker out; realised tier mix expected to centre on 26–45 ⇒ **32 bps**; headline base case 22 bps; ladder applied per symbol |
| Funding settlements crossed | 1–2 at an 8 h interval ⇒ 2 bps; **verified per symbol**, not assumed |
| `n_eff` | concurrency = 41 × 16/24 = 27.3; `Σū` = (41×546)/27.3 = 819; breadth = 27.3/(1+26.3×0.35) = 2.68; **`n_eff` ≈ 2,190** (1,620 at `ρ̄`=0.5; 1,170 at 0.7) |

A rejection rate above 50% means the measured edge is only partly reachable at `K_max = 5`; the
queue-priority rule (§3.5) is therefore part of the pre-registration, and the gap between mean
per-trade return and portfolio CAGR is itself a reported result.

### 14. Look-ahead and leakage audit

| Risk | Specific check |
|---|---|
| **Point-in-time universe / survivorship** | `pit_rank_prev_1d` built from a trailing-24 h window over *all listed symbols* as of the prior UTC day. Assert `listed_at + 30 d ≤ decision_time < delisted_at` for every event; report the excluded count; assert ≥ 1 delisted symbol present. Re-run with today's top-80 list: if expectancy moves > 20%, survivorship was material and is reported as such. |
| **Features crossing the decision bar** | `research/audit.truncation_test` on 20 randomly sampled events: recompute every feature from a bar series hard-truncated at the decision bar's close and assert **exact** equality. Grep assertions: no `center=True`, no `bfill`, no `shift(-k)`, explicit `min_periods` on every rolling/EWM. |
| **S1-specific: swing reference contaminated by its own bar** | `swing_high_prior` / `swing_low_prior` must be `shift(1).rolling(60)` — the decision bar excluded from its own swing window. Assert by hand-reconstructing 20 zones from a truncated series; `broke_swing_high` must be `high > prior_high` with `prior_high` from bars strictly before. |
| **S1-specific: Pass R ≠ Pass P on the overlap** | Bit-identity assertion of §2.1. A mismatch means the relaxed pass changed the detector, not just its gate. |
| **Same-bar barrier ambiguity** | Report fraction of resolved outcomes on an ambiguous bar and on the fill bar; pessimistic resolution; optimistic bound reported; hard stop at 10%. Forecast **< 3%** because a single 15 m bar spanning 255 bps is ≈ 5–6 σ at these volatilities, but this must be measured. |
| **Outcome windows overlapping training data** | Label life = decision → `min(fill + 32 bars, decision + 64 bars)` ⇒ `H_max = 16 h`. `assert_no_leak(purge=16h, embargo=40h)` run per walk-forward fold **and** per each of the 66 CPCV splits. `Σū` recomputed inside each training fold; assert `sample_weight is not None`. |
| **Timestamp convention** | Bars keyed on `open_time`; `decision_time_ms = open_time + 900_000`; assert the detector, feature builder, labeller and trade table all agree. Run the `+1` / `−1` bar-shift ablations. |
| **Fill optimism** | `traded_through` with the 1 bp margin; `fill_status` distribution published; no 100%-fill assumption; adverse selection measured against the shadow arm. |
| **Cost look-ahead** | Spread from a per-symbol, per-month Abdi–Ranaldo proxy computed on trailing 15 m OHLCV **excluding signal bars** (Corwin–Schultz on the signal bar is circular here — HAWK's own wick gate mechanically inflates it, `evidence_context_costs.md` §4.4), cross-checked against Amihud and the Harvard Dataverse Binance spread dataset. Never today's spread. |
| **Cost-proxy / volatility confound** | Any "wide-spread zones do better" finding must be re-run with `rv_24h` in the same specification; if the spread proxy loses significance, it was measuring volatility (*Entropy* 22(7):783). |

---

## 5. System S3 — Idiosyncratic Overreaction Fade with a Liquidity Sign Flip

> **Rank 2 of 5. Prior that it works: 18%.** (Presented second because §10's ranking is by evidence
> strength; the numbering S1/S2/S3/S4/S5 is stable identity, the ordering here is rank order.)

### 1. Hypothesis

*Conditional on a zone event, the sign of the forward 8 h residual return is predicted by the
**negative** of the sign of the symbol's BTC-residualised trailing 4 h return, and the strength of
that reversal increases monotonically with illiquidity, flipping to continuation in the liquid
head (`pit_rank_prev_1d ≤ 15`).*

Magnitude: the tier-conditional rule of item 3 achieves a residual 8 h barrier-hit rate ≥ **45.6%**
(net ≥ +9.3 bps/trade at `c = 22 bps`), **and** the tier-ordered contrast
`E[fade | T3] > E[fade | T2] > E[fade | T1]` holds with a Jonckheere–Terpstra trend p ≤ 0.05.

**Mechanism.** Wen, Bouri, Xu & Zhao (2022, *NAJEF* 62): on half-hourly BTC, among pairs with
significant in-sample predictability *and* positive OOS R², **21 reversal vs 12 momentum** — and the
reversal component is attributed to overreaction to **non-fundamental** information, which is
exactly what a high-volume long-wick 15 m bar is. Borgards & Czudaj (2020, *JIFMIM* 65): price
overreactions highly prevalent at hourly frequency across 12 coins, contrarian returns significantly
larger than in US stocks, with **overreaction magnitude the dominant driver of profitability**.
Kozlowski, Puleo & Zhou (2021, *AEL* 28(11)): loser-minus-winner spread 7.37% daily (t = 24.23),
**most pronounced among smaller-cap and less liquid coins**. Zaremba et al. (2021, *IRFA* 78): last
day's return a powerful negative cross-sectional predictor, but the effect is
liquidity-conditional — "the handful of largest and most tradeable coins exhibit daily momentum
rather than a reversal." Liu, Tsyvinski & Wu (2022, *JF* 77(2)): crypto momentum is **stronger among
large coins**, reconciling the flip.

**Prior: 18%.** `evidence_signals.md` §Closing gives family 4 the highest prior of the five at 32%.
Marked down because: the sign at exactly the 1-day horizon is genuinely contested (Caporale & Plastun
2020 find *continuation* after abnormal returns, with exploitable profits); all the published
spreads are gross on universes far broader and thinner than HAWK's liquid head; and Kozlowski's
7.37%/day is not a number any execution model survives.

### 2. Why the zone information specifically could provide an edge

Honestly: **little, and possibly nothing directionally.** The reversal literature is about the
*size of a recent move*, not about bar shape. What the zone plausibly adds is (a) a volatility and
attention marker — `volume_ratio ≥ 1.35` selects moments of crowding, and Lee & Swaminathan's
mechanism is that volume proxies investor interest and therefore the *imminence* of reversal — and
(b) a non-arbitrary passive limit price. Note the contrary evidence: Moser & Brauneis (2026) report
their surviving crypto patterns are "unaffected by volume", so volume conditioning may add nothing
here either. **S5 is the test of exactly this question, on exactly this rule.**

### 3. Exact direction rule

All inputs as-of the decision bar close.

```
resid_ret_4h = trend_ret_4h - beta_btc_24h * btc_ret_4h        # features.py, both causal
move_scale   = rv_4h * sqrt(16)                                 # 4 h return scale
if abs(resid_ret_4h) < 1.0 * move_scale:  side = SKIP           # not an overreaction
tier = T1 if pit_rank_prev_1d <= 15 else (T2 if <= 45 else T3)   # <= 80 by universe rule
side = +sign(resid_ret_4h)  if tier == T1   else  -sign(resid_ret_4h)
```

The `1.0 σ` magnitude gate is the natural unit and requires no tuning; monotonicity of the effect
across |`resid_ret_4h`| **deciles** is reported as the robustness check (Borgards & Czudaj:
magnitude is the dominant driver, and decile monotonicity is much harder to fake than a threshold).
The `T1 ≤ 15` cut implements Zaremba et al.'s "handful of largest coins" literally — the momentum
branch is deliberately narrow and the reversal branch broad. `T2` is assigned to the reversal branch
because the reversal evidence covers the whole non-head cross-section while the momentum evidence is
explicitly confined to the very top.

### 4. Exact entry rule

§3.2 verbatim.

### 5. Exact exit rules

§3.1 and §3.3 verbatim.

### 6. Required existing HAWK fields

Pass-P replay / `zones`: `symbol`, `source_open_time`, `zone_high`, `zone_low`, `zone_size_pct`,
`volume_ratio`, `wick_ratio`, `distance_to_swing`, `btc_range_4h_pct`, `universe_rank`,
`universe_quote_volume`.
Derived: `trend_ret_4h`, `btc_ret_4h`, `beta_btc_24h`, `corr_btc_24h`, `rv_4h`, `rv_24h`,
`atr_48_pct`, `pit_rank_prev_1d`, `quote_volume_24h`, `xs_rank_quote_volume_24h`,
`hour_of_day_utc`, `is_weekend`.

**Not used:** `symbol_return_1h/4h/24h` as stored. They mix market beta and idiosyncratic move,
which `evidence_signals.md` Topic 5 identifies as a sufficient explanation for AUC ≈ 0.50 on its
own. They are replaced by `resid_ret_4h`.

### 7. Additional data required

Same as S1 minus Pass R (S3 uses Pass P). Plus `pit_rank_prev_1d` over the **full listed universe**,
which is the one genuinely expensive item — see §11 item 3.

### 8. Market condition / behaviour exploited

Retail-dominated, 24/7, thin-book markets overreact to non-fundamental order flow, and the resulting
inventory imbalance is unwound over hours by liquidity providers who are compensated for absorbing
it. That compensation is the return being harvested, and its size scales with how costly the
inventory is to carry — which is why the effect strengthens as liquidity falls. In the very largest
names, market-making capital is deep enough that the residual effect is instead slow information
diffusion, which shows up as momentum. The trade is therefore **selling insurance on a
non-fundamental move, at a price set by how thin the book is.**

### 9. The objective test

- **H0:** as §3.9, plus `Δ_tier ≡ E[fade net | T3] − E[fade net | T1] = 0`.
- **Tests:** §3.7. Tier ordering via Jonckheere–Terpstra on the three tier-level residual
  expectancies, bootstrapped by the time-slice scheme; `Δ_tier` by
  `paired_block_bootstrap_diff_ci`.
- **Control / benchmark:** §3.9's three benchmarks, plus the **same rule with the tier interaction
  removed** (pooled fade across all 80 symbols) as the internal reference. If pooled ≈ 0 while the
  tiers differ, that confirms `evidence_signals.md`'s diagnosis that pooling has been averaging two
  opposite-signed effects toward zero.
- **Sample:** tier D, Pass P, §2.2–2.3 gates, `|resid_ret_4h| ≥ 1.0 σ`.
- **Outcome variables:** §3.4.

### 10. Metrics

§3.8 in full, additionally disaggregated by tier (T1/T2/T3), by |`resid_ret_4h`| decile, and by fill
status. **Net performance must be reported at the tier's own cost rung** — 17 bps for T1, 32 bps for
T2, 47–68 bps for T3 — never at a universe average. `evidence_context_costs.md` §5.5 is explicit
that crypto ML alphas concentrate in exactly the illiquid names whose costs a universe-average model
hides, so this is the single most important reporting requirement for S3.

### 11. Evidence of edge (pass)

Gates 0–7, **plus** `Δ_tier` 95% CI excludes zero with `Δ_tier > 0`, **plus** the T3 branch's net
expectancy 95% lower bound > 0 **at T3's own cost rung (47 bps)**. A T3 edge that exists at 22 bps
and vanishes at 47 bps is not an edge; it is the illiquid-coin trap.

### 12. Falsification (any one suffices)

- Residual hit-rate 95% upper bound ≤ 42.0% in **all three** tiers.
- `Δ_tier` 95% CI contained inside ±5 bps **and** the pooled rule's CI contained inside ±5 bps ⇒ the
  liquidity interaction does not exist and the pooling explanation for AUC ≈ 0.50 is wrong.
- Sign of `Δ_tier` **negative** with its 95% CI excluding zero — i.e. momentum in the tail and
  reversal in the head, the opposite of every paper cited. This would falsify the hypothesis as
  stated; it may **not** be re-badged as a discovery, and trading it would require a fresh
  pre-registration and a fresh trial.
- T3 net expectancy point estimate < 0 at 47 bps while T3 gross > 0 ⇒ **the effect is real and
  untradeable**, which is a falsification of the *system*, and is the outcome I consider most likely
  for S3.
- ≤ 2 of 6 folds positive; PBO > 0.50; CPCV median OOS Sharpe ≤ 0.

### 13. Execution realism

| Quantity | Forecast |
|---|---|
| Pass-P zones **[B2]** | ≈ 41/day |
| Survive `\|resid_ret_4h\| ≥ 1.0 σ` **[B7]** ≈ 32% (normal-approximation two-sided tail) | ≈ **13/day** signalled |
| Fill rate **[B6]** ≈ 45% | ≈ **6 fills/day** |
| `E[n_open]` = 13 × 16/24 | ≈ **8.7** ⇒ cap binds; forecast `rejection_rate` **35–45%** |
| Cost | tier-specific: T1 17, T2 32, T3 47–68 bps. Expected mix ≈ 30% T1 / 45% T2 / 25% T3 ⇒ weighted ≈ **35 bps** |
| Funding | 1–2 settlements ⇒ 2 bps, verified per symbol |
| `n_eff` | concurrency = 8.67; `Σū` = (13×546)/8.67 = 819; breadth = 8.67/(1+7.67×0.35) = 2.35; **`n_eff` ≈ 1,927** (1,410 at `ρ̄`=0.5) |

Note the tension this table exposes and does not hide: the literature says the edge lives in T3,
where the cost rung is 47–68 bps against a 255 bps span, i.e. `c/span` = 18–27% and `p*` = 51–60%.
**T3 is close to untradeable by construction.** T3 is nonetheless tested, at its own cost, because
skipping it would be testing the hypothesis where the literature says it is weakest.

### 14. Look-ahead and leakage audit

All of S1's §14 rows apply, plus:

| Risk | Specific check |
|---|---|
| **S3-specific: tiering is the largest survivorship surface in the study** | `pit_rank_prev_1d` must be a trailing-24 h rank over **all symbols listed at that time**, computed from the prior completed UTC day. Full-sample liquidity ranking is a look-ahead that preferentially selects coins that *later* became popular — a momentum signal in disguise (protocol §9.1 item 3). Assert the ranking function never reads bars at or after the decision bar, and report monthly universe churn; a static 80-symbol list over 2 years fails this check by inspection. |
| **S3-specific: `beta_btc_24h` is a fitted quantity** | Assert it is estimated on a strictly trailing 96-bar window with `min_periods = 96`, never on the full sample, and never re-fitted inside the test window. `features.btc_features` aligns BTC on `open_time` and yields NaN rather than forward-filling a stale price — assert no `ffill` was introduced. |
| **S3-specific: residual outcome uses forward BTC** | The residual close path uses BTC bars *after* the decision, which is legitimate (it is part of the label, not a feature). Assert `beta_btc_24h` is frozen at the decision bar and not re-estimated over the outcome window. |
| **S3-specific: global normalisation of the magnitude gate** | `move_scale` is per-zone from `rv_4h`; assert no full-sample z-scoring, quantile binning or scaler is fitted outside the training fold (protocol §9.2 item 7). |

---

## 6. System S4 — Absorbed-Flow Reversal (signed taker imbalance vs close location)

> **Rank 3 of 5. Prior that it works: 12%.**

### 1. Hypothesis

*On a zone bar where aggressive (taker) flow and price disagree — heavy taker **buying** into a bar
that closes in its **lower** third, or heavy taker **selling** into a bar that closes in its
**upper** third — the aggressive side was absorbed by resting liquidity, and the forward 8 h
residual return follows the price, not the flow.*

Magnitude: residual 8 h barrier-hit rate ≥ **45.6%** (net ≥ +9.3 bps/trade at `c = 22 bps`), with
the effect **absent** (95% CI inside ±5 bps) in the agreeing-flow cells, which is the discriminating
comparison.

**Mechanism.** Cont, Kukanov & Stoikov (2013, *JFE* 12(1), 47–88): mid-price changes are linear in
**order-flow imbalance** with average R² ≈ 65% (35–60% excluding price-changing events), the slope
inversely proportional to depth, while "the relation between price changes and **trade volume** is
noisy and less robust." Jones, Kaul & Lipson (1994, *RFS* 7(4)): the informative activity variable is
the **number of transactions**, not trade size — so the event should be a trade-*count* spike.
Osler (2003/2005) supplies the inventory mechanism that makes an absorbed imbalance persist for
hours rather than seconds. HAWK currently infers order flow from candle shape while Binance
publishes a direct measurement of it: `evidence_context_costs.md` §4.3 calls this "the highest-
value-per-unit-effort data addition available to HAWK."

**Prior: 12%.** The microstructure evidence is strong but it is **contemporaneous explanatory power,
not prediction**; cross-asset lagged OFI predictability "decays quickly" and manifests "up to
several minutes"; and a crypto replication on 1-second BTCUSDT got direction AUC 0.66–0.69 but a
**validation R² of −0.06** on returns, underperforming a zero-return baseline. A 15-minute bar
aggregate of taker volume is a coarse proxy for true OFI, and asking it to predict 8 hours ahead is
two orders of magnitude beyond where the evidence lives. The 12% is mostly the inventory/absorption
channel, not the OFI channel.

### 2. Why the zone information specifically could provide an edge

This is the one system where the zone's *geometry* is load-bearing for a reason other than
convenience. The detector's wick gate guarantees a bar whose range is large relative to its body —
i.e. a bar where price went somewhere and came back. Combined with signed taker flow, that is a
direct, measurable signature of absorption: someone lifted offers aggressively and price ended at
the bottom of the range, which means resting sell liquidity won. Without the wick gate this cell is
rare. The zone also supplies the two prices (`zone_high`, `zone_low`) at which the winning liquidity
demonstrably sat, which is where a passive re-entry belongs.

### 3. Exact direction rule

All inputs as-of the decision bar close.

```
taker_buy_share = taker_buy_quote / quote_volume                   # decision bar
flow_imb        = 2 * taker_buy_share - 1                          # in [-1, +1]
trade_count_ratio = trades / mean(trades over prior 60 bars)       # prior window, bar excluded

require trade_count_ratio >= 1.35                 # JKL: count, not size. Reuses the detector's
                                                  # own multiplier, so no new constant.
require abs(flow_imb) >= 0.10                     # a 55/45 taker split
require closed_upper_third == 1 or closed_lower_third == 1   # features.py, |clv_signed| >= 1/3

if flow_imb > 0 and closed_lower_third == 1:  side = SHORT   # aggressive buyers absorbed
elif flow_imb < 0 and closed_upper_third == 1: side = LONG   # aggressive sellers absorbed
else:                                          side = SKIP   # flow and price agree -> reference cell
```

The agreeing cells (`flow_imb > 0` with `closed_upper_third`, `flow_imb < 0` with
`closed_lower_third`) are scored as the **internal reference cell** with the same sign rule and
receive no capital. Every constant here is either taken from the detector (1.35) or from an existing
`features.py` definition (`closed_upper_third`/`closed_lower_third` at 1/3); the only new number is
the 0.10 flow threshold, fixed and not amendable.

### 4. Exact entry rule

§3.2 verbatim.

### 5. Exact exit rules

§3.1 and §3.3 verbatim.

### 6. Required existing HAWK fields

Pass-P replay / `zones`: `symbol`, `source_open_time`, `zone_high`, `zone_low`, `zone_size_pct`,
`volume_ratio`, `wick_ratio`, `btc_range_4h_pct`, `universe_quote_volume`.
Derived: `close_position_in_range`, `closed_upper_third`, `closed_lower_third`, `upper_wick_frac`,
`lower_wick_frac`, `body_to_range`, `rv_24h`, `atr_48_pct`, `beta_btc_24h`, `btc_ret_4h`,
`pit_rank_prev_1d`, `trades_24h`.

### 7. Additional data required

**This is the system with a real data dependency, and it is cheap.**

- `taker_buy_quote`, `quote_volume`, `trades` per 15 m bar. **Already captured** — they are in
  `KEEP_COLUMNS` in `src/hawk_eye/research/backfill.py`. **Zero extra download cost.**
- `taker_buy_base` is parsed in `KLINE_COLUMNS` but **dropped from `KEEP_COLUMNS`**. It must be
  retained, for cross-checking `taker_buy_quote / quote_volume` against
  `taker_buy_base / volume` (they should agree to within intrabar VWAP drift; a disagreement flags a
  units or microsecond-timestamp bug of exactly the kind `_normalise_epoch` exists to catch).
  **This is free at backfill time and a full re-download otherwise — capture it now.**
- **No order-book data, no trade-level (tick) data.** True OFI as Cont et al. define it requires
  level-1 event data, which would be a per-symbol, per-day paid feed running to tens of GB for an
  80-symbol × 2-year panel. It is **not justified** at this stage: the prior on the whole family is
  12%, and the free bar-level proxy is the correct first test. If and only if S4 produces a
  significant result on the proxy does order-book data become worth costing.

### 8. Market condition / behaviour exploited

Liquidity provision against uninformed aggression. A taker imbalance that fails to move the close in
its own direction is, by definition, flow that was absorbed by resting limit orders at a price. The
absorbing party now holds inventory and, over the following hours, either earns the reversal
(because the aggression was uninformed) or is run over (because it was informed). The trade is a bet
on the **first** case, and the wick geometry plus the failure of the close to follow the flow is the
observable that discriminates them. Economically this is the same negative-feedback channel as
Osler's take-profit clustering, measured directly from flow rather than inferred from level
proximity.

### 9. The objective test

- **H0:** as §3.9, plus `Δ_abs ≡ E[net | disagreeing cells] − E[net | agreeing cells] = 0`.
- **Tests:** §3.7. `Δ_abs` by `paired_block_bootstrap_diff_ci`, shared index sequence, one-sided at
  5% in the pre-registered direction.
- **Control / benchmark:** the **agreeing-flow cells** (same bars, same machinery, flow and price
  aligned) plus §3.9's three benchmarks. This internal control is what separates "absorption" from
  "any flow signal".
- **Sample:** tier D, Pass P, §2.2–2.3 gates, plus the three item-3 requirements.
- **Outcome variables:** §3.4.

### 10. Metrics

§3.8 in full, additionally disaggregated by `flow_imb` quintile, by cell (agreeing / disagreeing),
and by tier. Also reported: the Family-B-style descriptive correlation between `flow_imb` and
`fwd_ret_15m` / `fwd_ret_1h` — the horizons at which the OFI literature says the effect should be
strongest. If `flow_imb` predicts nothing even at 15 m, the 8 h hypothesis is implausible and that
should be visible in the report even though it cannot, by itself, gate.

### 11. Evidence of edge (pass)

Gates 0–7, **plus** `Δ_abs` 95% CI excludes zero with `Δ_abs > 0`, **plus** the agreeing-cell
expectancy 95% CI containing zero. Both conditions are required: an effect present in *both* cells
is a flow-magnitude or volatility effect, not absorption, and does not pass.

### 12. Falsification (any one suffices)

- Residual hit-rate 95% upper bound ≤ 42.0% in the disagreeing cells.
- `Δ_abs` 95% CI contained inside ±5 bps ⇒ signed taker flow carries no conditional directional
  information at this horizon, and the "HAWK should capture order flow" recommendation is answered
  in the negative for the 15 m bar aggregate. This would be a genuinely useful negative: it closes
  off the order-book-data spend before it is made.
- The effect is equally large in the agreeing cells (both CIs exclude zero, `|Δ_abs| < 5 bps`) ⇒ it
  is not absorption.
- `taker_buy_quote / quote_volume` and `taker_buy_base / volume` disagree by more than 5% of bars
  beyond a 2% tolerance ⇒ **VOID** on data integrity until resolved.
- ≤ 2 of 6 folds positive; PBO > 0.50; CPCV median OOS Sharpe ≤ 0.

### 13. Execution realism

| Quantity | Forecast |
|---|---|
| Pass-P zones **[B2]** | ≈ 41/day |
| Survive trade-count spike **[B8]** ≈ 55% (correlated with the volume gate already applied) | ≈ 23/day |
| Survive `\|flow_imb\| ≥ 0.10` **[B9]** ≈ 60% | ≈ 14/day |
| In the outer third **and** disagreeing **[B10]** ≈ 40% × 50% | ≈ **2.8/day** signalled |
| Fill rate **[B6]** ≈ 45% | ≈ **1.3 fills/day** |
| `E[n_open]` = 2.8 × 16/24 | ≈ **1.9** ⇒ cap rarely binds; forecast `rejection_rate` **< 10%** |
| Cost | as §3.6; base case 32 bps at the expected tier mix |
| `n_eff` | concurrency = 1.87; `Σū` = (2.8×546)/1.87 = 819; breadth = 1.87/(1+0.87×0.35) = 1.43; **`n_eff` ≈ 1,173** (950 at `ρ̄`=0.5) |

The low frequency is fine: §3.11's table shows that dropping from 10 to 3 fills/day raises the
required per-trade net edge only from 9.3 to 10.0 bps. But `n_eff ≈ 1,160` is the lowest of the five,
so the Gate-7 floor of 300 is the one to watch if any further sub-setting is attempted — and it may
not be, because sub-setting would be a new trial.

### 14. Look-ahead and leakage audit

All of S1's §14 rows apply, plus:

| Risk | Specific check |
|---|---|
| **S4-specific: `taker_buy_quote` timestamp alignment** | The taker fields come from the same kline row as the OHLC, so they are known at the bar's close by construction — but `_normalise_epoch` in `backfill.py` handles a ms/µs archive switch, and a mis-normalised row would attach flow from the wrong bar. Assert `open_time` spacing is exactly 900,000 ms across every symbol, and spot-check 200 random bars against a REST `/api/v3/klines` pull. |
| **S4-specific: trade-count baseline contaminated** | `mean(trades over prior 60 bars)` must be `shift(1).rolling(60)` — the decision bar excluded from its own baseline, mirroring the detector's `prior = candles[i-lookback:i]`. Assert by truncation test. |
| **S4-specific: `quote_volume` zero or NaN** | `_parse_kline_csv` fills missing `quote_volume`/`trades` with NaN. Assert `taker_buy_share` is NaN-propagating, not zero-filled, and that NaN rows are **dropped and counted**, never imputed. A zero-filled `taker_buy_share` would read as maximal selling pressure. |
| **S4-specific: turn-of-the-candle artefact** | The "turn-of-the-candle effect" (positive returns concentrated at minutes 0/15/30/45, t > 9 across seven exchanges) means entries and decisions at 15 m boundaries sit in minutes with a distinct return and liquidity profile. Report expectancy split by whether the fill bar is the first bar after the decision versus later, as a diagnostic. |
| **S4-specific: venue mismatch on flow** | Flow is measured on **Binance spot** (the `backfill.py` archive source) while the trade is executed on **MEXC perp**. Assert and report the Binance-spot vs MEXC-perp 15 m close correlation per symbol for at least one sample month; a correlation below 0.99 means the flow measurement and the traded instrument are not the same market and S4's mechanism does not transfer. |

---

## 7. System S2 — Signed Wick Side, Conditional on the Residual Trend

> **Rank 4 of 5. Prior that it works: 8%.**

### 1. Hypothesis

*The **side** of the dominant wick on a zone bar carries directional information for the forward
8 h residual return over and above the sign of the BTC-residualised prior 4 h move, in the direction
"fade the wick" (a dominant lower wick is bullish, a dominant upper wick is bearish).*

Magnitude: the rule `side = −wick_side` achieves a residual 8 h barrier-hit rate ≥ **45.6%**
(net ≥ +9.3 bps/trade at `c = 22 bps`), **and** in the *crossed* cells — where `−wick_side`
disagrees with `−sign(resid_ret_4h)` — the wick-based sign beats the trend-based sign with a 95% CI
on the difference excluding zero.

**Mechanism, and an important correction to the obvious reading.** The candlestick taxonomy in
`evidence_signals.md` Topic 3c is:

| Pattern | Wick side | Prior trend | Predicted sign |
|---|---|---|---|
| Hammer | lower | down | bullish |
| Hanging Man | lower | up | **bearish** |
| Inverted Hammer | upper | down | bullish |
| Shooting Star | upper | up | **bearish** |

Read the last column against the third: **the predicted sign is a function of the prior trend alone,
and is identical for both wick sides.** The four-cell table collapses to "fade the prior trend",
which is *exactly S3's rule*. Testing the four-cell table as stated would therefore consume a trial
to re-test S3. The only content unique to the wick is whether its **side** adds sign information
*conditional on* the trend — which is what the crossed cells measure, because that is precisely
where `−wick_side` and `−sign(resid_ret_4h)` disagree. **S2 is therefore specified as the
wick-side-versus-trend contrast, not as the four-cell table.** This is a design finding, not a
result, and it is one reason §7 concludes the honest design space is narrower than five.

Supporting evidence: Moser & Brauneis (2026, *IREF* 108) — hourly OHLC, 07/2018–01/2022, ~400 coins,
2,000 pairs, 36 exchanges, **200 million observations**, 55 TA-Lib reversal patterns, profitability
assessed under the **stepwise Superior Predictive Ability** test. Bullish Harami and Hikkake precede
significant gains; **Bearish Harami and Hanging Man** precede significant declines; results stable
across time, regimes, coins and exchanges. **Hanging Man — long lower wick, small body, after an
uptrend — is the only single-bar survivor, and it is a *crossed* cell.** That is the specific,
published reason to expect the crossed cells to carry information.

**Prior: 8%.** Against Moser & Brauneis stands the strongest methodology in the family: Marshall,
Young & Rose (2006, *JBF* 30(8)) tested 28 candlestick rules on 35 DJIA stocks 1992–2002 under a
bootstrap that resamples **open, high, low and close jointly** — the correct null for a rule defined
on intrabar geometry — and found **22 of 28 rules significant on zero stocks**, a maximum of 3/35 for
any rule (consistent with noise at 980 tests), and **mean profit after bullish single lines higher on
the random series than on the real one**. Horton (2009) concurs. Lo, Mamaysky & Wang (2000) found
distributional information but explicitly not profit, and their patterns are multi-bar. And Moser &
Brauneis report their patterns are **"unaffected by volume"**, which means HAWK's volume gate adds
nothing to them. `evidence_signals.md` puts the family at 14%, rising to ~22% *if* the wick is signed
and the trend conditioned — which S2 does — but the 8% here additionally discounts for the cost
hurdle and for the fact that the one positive result is a single 2026 paper against two
well-identified negatives.

### 2. Why the zone information specifically could provide an edge

**Possibly nothing, and that is part of what this tests.** Moser & Brauneis found their patterns
unaffected by volume, so conditioning on `volume_ratio ≥ 1.35` is expected to be neutral at best.
What the zone does contribute is that `wick_ratio ≥ 1.2` guarantees a small body relative to the
dominant wick — the geometry the taxonomy requires — so the zone population is an efficient sampler
of the pattern rather than a source of extra information. If S2 passes and S5 shows no zone-vs-
control difference on this rule, the correct conclusion is **"the wick side is informative and the
zone is an event marker."**

A second, sharper reason to run S2: HAWK's own `wick_ratio = max(upper, lower) / |close − open|`
discards the wick side entirely, and `evidence_signals.md` names this as a sufficient mechanistic
explanation for the observed 47–51% hit rate and AUC ≈ 0.50. S2 is the cheapest possible test of
whether the discarded dimension was ever worth anything.

### 3. Exact direction rule

```
# features.py bar_shape_features, all from the decision bar's OHLC
dominant = 'upper' if upper_wick_frac > lower_wick_frac else 'lower'
require max(upper_wick_frac, lower_wick_frac) >= 2 * min(upper_wick_frac, lower_wick_frac)
                                                   # wick side must be unambiguous; scale-free,
                                                   # single-valued, no new tunable constant
side = SHORT if dominant == 'upper' else LONG      # i.e. side = -wick_side

# reported cell assignment for the scientific contrast (no effect on the traded side)
resid_ret_4h = trend_ret_4h - beta_btc_24h * btc_ret_4h
named   = (dominant == 'upper' and resid_ret_4h > 0) or (dominant == 'lower' and resid_ret_4h < 0)
crossed = not named
```

`named` cells are Shooting Star / Hammer (wick rule and trend rule agree). `crossed` cells are
Inverted Hammer / **Hanging Man** (they disagree) — the discriminating sample.

### 4. Exact entry rule

§3.2 verbatim.

### 5. Exact exit rules

§3.1 and §3.3 verbatim.

### 6. Required existing HAWK fields

Pass-P replay / `zones`: `symbol`, `source_open_time`, `zone_high`, `zone_low`, `zone_size_pct`,
`volume_ratio`, `wick_ratio`, `direction`, `btc_range_4h_pct`.
Derived: `upper_wick_frac`, `lower_wick_frac`, `wick_side`, `wick_asymmetry`, `body_to_range`,
`close_position_in_range`, `trend_ret_4h`, `btc_ret_4h`, `beta_btc_24h`, `rv_24h`, `atr_48_pct`,
`pit_rank_prev_1d`.

Note `wick_ratio` itself is recorded but **not used** — S2 exists precisely because the max()
collapse makes it unusable for sign. `upper_wick_frac`/`lower_wick_frac` normalise by `range`, not
by body, which also removes the unbounded blow-up as `|close − open| → 0` that
`evidence_signals.md` flags as heavy-tailed garbage in the current feature matrix.

### 7. Additional data required

None beyond §3 and §11's shared items. This system is free.

### 8. Market condition / behaviour exploited

Intrabar rejection: within 15 minutes, price probed in one direction, found no follow-through, and
returned. In a thin, retail-dominated book, that footprint is the visible residue of a liquidity
sweep that failed — a stop run with no continuation. The claim being tested is that the failure is
informative about the next several hours because the aggressor's inventory must still be unwound.
The competing explanation, which the equity literature supports, is that the footprint is a
statistical property of OHLC geometry under a random walk and carries nothing.

### 9. The objective test

- **H0:** as §3.9, plus `Δ_wick ≡ E[net | crossed, side = −wick_side] − E[net | crossed,
  side = −sign(resid_ret_4h)] = 0`.
- **Tests:** §3.7. `Δ_wick` by `paired_block_bootstrap_diff_ci` on the identical bar set with the
  two opposing sides, shared index sequence.
- **Control / benchmark:** §3.9's three benchmarks, **with one substitution that matters**: the
  random-signal placebo for S2 is not a coin flip but a **joint OHLC bootstrap** in the
  Marshall/Young/Rose sense — resample open, high, low and close together so the null preserves
  intrabar geometry statistics while destroying pattern information. A close-only random walk will
  make any geometry rule look significant, and adopting the MYR null is named in
  `evidence_signals.md` as "the single most important methodological borrow available." This is the
  only system whose placebo differs, and the difference is required, not optional.
- **Sample:** tier D, Pass P, §2.2–2.3 gates, dominant-wick requirement satisfied.
- **Outcome variables:** §3.4.

### 10. Metrics

§3.8 in full, disaggregated by cell (`named` / `crossed`), by wick side, and by tier. Additionally
report the four-cell table of residual expectancies for completeness, flagged as descriptive.

### 11. Evidence of edge (pass)

Gates 0–7, **plus** `Δ_wick` 95% CI excludes zero with `Δ_wick > 0`, **plus** Gate 2 satisfied
against the **joint-OHLC bootstrap** null rather than a close-only null.

### 12. Falsification (any one suffices)

- Residual hit-rate 95% upper bound ≤ 42.0%.
- `Δ_wick` 95% CI contained inside ±5 bps ⇒ **wick side carries no sign information conditional on
  the trend.** This is the clean, decisive answer to the single cheapest open question about HAWK's
  feature set, and it would justify dropping signed-wick work permanently.
- Mean net expectancy on the **joint-OHLC bootstrap** series ≥ the point estimate on the real series
  (the Marshall/Young/Rose outcome — the point estimate going the wrong way).
- ≤ 2 of 6 folds positive; PBO > 0.50; CPCV median OOS Sharpe ≤ 0.

### 13. Execution realism

| Quantity | Forecast |
|---|---|
| Pass-P zones **[B2]** | ≈ 41/day |
| Dominant-wick requirement (≥ 2× the other) **[B11]** ≈ 70% | ≈ **29/day** signalled |
| Fill rate **[B6]** ≈ 45% | ≈ **13 fills/day** |
| `E[n_open]` = 29 × 16/24 | ≈ **19** ⇒ cap binds hard; forecast `rejection_rate` **60–70%** |
| Cost | as §3.6; base 32 bps at the expected tier mix |
| `n_eff` | concurrency = 19.33; `Σū` = (29×546)/19.33 = 819; breadth = 19.33/(1+18.33×0.35) = 2.61; **`n_eff` ≈ 2,135** |

S2 has the highest signal rate and therefore the most binding concurrency cap. Its
`K_max ∈ {2,5,10,20}` sensitivity is load-bearing, and a conclusion that holds only at `K_max = 20`
is, per protocol §3.2, evidence of beta exposure rather than edge.

### 14. Look-ahead and leakage audit

All of S1's §14 rows apply, plus:

| Risk | Specific check |
|---|---|
| **S2-specific: the wrong null** | Assert the placebo used for Gate 2 is the joint-OHLC bootstrap, not a close-only random walk, and publish both so the difference is visible. Using the close-only null would manufacture significance for a geometry rule. |
| **S2-specific: `wick_side` ties and degenerate bars** | Assert `upper_wick_frac` and `lower_wick_frac` are NaN-propagating when `high == low`, that such bars are dropped and counted, and that the 2× dominance test is evaluated on the fractions of `range`, never on the body-normalised `wick_ratio` (which is unbounded as the body → 0). |
| **S2-specific: the trend used for cell assignment must not leak** | `trend_ret_4h` is `close/close.shift(16) − 1`, as-of the decision bar; assert by truncation test that it reads no bar after the decision bar, and that `beta_btc_24h` is the frozen trailing estimate. |

---

## 8. System S5 — Zone versus Matched Non-Zone Control (the necessity test)

> **Rank 5 of 5. Prior that the zone adds value: 7%.** Ranked last on probability of passing and
> **first on information value.** These are different quantities and §10 keeps them separate.

### 1. Hypothesis

*The HAWK zone population is not merely a sampler. Applying one fixed direction rule (S3's) to zone
events versus to matched non-zone bars on the same symbols, in the same conditions, produces a
strictly better net outcome for the zone arm.*

Falsifiable with a direction and a magnitude: the paired difference
`Δ_zone ≡ E[net | zone] − E[net | matched control]` is **> +5 bps per trade** with a 95% CI
excluding zero; and the zone arm independently satisfies Gates 2–5.

**Mechanism from the literature — or rather, the absence of one.** There is **no peer-reviewed study
validating order blocks, fair value gaps, break of structure or the SMC/ICT framework** as
predictive. Two independent mechanical grey-literature tests came back null: **0 of 648** ICT
variants beat buy-and-hold across SPY/QQQ/DIA/IWM, with the best order-block effect on SPY at
+0.121% over 5 days on 736 events and **t = 1.22**; and **0 of 54** FVG-after-sweep variants
profitable after a 0.5-pip cost on 2.55 M one-minute EURUSD bars. `evidence_signals.md` gives the
construct **4%**. Critically, HAWK's own ~83% retest rate is *not* evidence: a zone defined as a
recent bar's high–low range, near the current price, will be retested at a high rate under a random
walk at 15 m altcoin volatility. The review is explicit — "before it can mean anything it needs a
matched control … If a random band is also retested ~83% of the time, the statistic is measuring
15 m crypto volatility, not order flow." **S5 is that control, and it is the one test in this
document that the literature says should be run regardless of what anyone hopes to find.**

**Prior: 7%.** The 4% construct prior, adjusted slightly upward because S5's null is weaker than
"order blocks predict direction" — the zone only has to *improve* a rule that works elsewhere, e.g.
by selecting higher-volatility, higher-attention moments, which is a much lower bar than supplying
direction.

### 2. Why the zone information specifically could provide an edge

**The honest answer is "possibly nothing, and that is exactly what this tests."** Three candidate
contributions, all of which S5 measures separately: (a) the volume gate may select moments of
elevated realised volatility, which widens the achievable barrier span relative to a fixed cost and
is therefore worth something *even with zero directional content*; (b) the wick gate may select
moments of failed aggression, per S4's mechanism; (c) the zone's edges supply two objective passive
limit prices. If `Δ_zone` is zero while the *fill rate* and *adverse selection* are materially better
on the zone arm, the verdict is "the detector is an execution device, not a signal" — a specific,
useful, and pre-registered possible outcome.

### 3. Exact direction rule

**Identical to S3's, applied unchanged to both arms.** This is deliberate: the comparison is only
interpretable if the rule is held fixed, and reusing S3's rule also keeps the two trials highly
correlated, which is what keeps Li–Ji `N_eff` low.

```
resid_ret_4h = trend_ret_4h - beta_btc_24h * btc_ret_4h
if abs(resid_ret_4h) < 1.0 * rv_4h * sqrt(16):  side = SKIP
tier = T1 if pit_rank_prev_1d <= 15 else (T2 if <= 45 else T3)
side = +sign(resid_ret_4h) if tier == T1 else -sign(resid_ret_4h)
```

**Control population construction (the whole content of S5).** For each zone event `e` on symbol `s`
at decision bar `t`, draw up to **3** control bars from symbol `s` with `open_time` within **±48 h**
of `t` that:

1. are **not** detector hits under Pass R (so not even relaxed-swing candidates);
2. **fail at least one** of the volume gate (`det_volume_ratio < 1.35`) or the wick gate
   (`det_wick_ratio < 1.2`);
3. pass all of §2.2–2.3's universe, volatility, dislocation and data-integrity gates;
4. **match `e` exactly on** the point-in-time `btc_range_4h_pct` tercile, the `atr_48_pct` decile,
   the `pit_rank_prev_1d` tier (T1/T2/T3), the 6-hour UTC block, the sign of `resid_ret_4h`, and the
   |`resid_ret_4h`| decile;
5. are selected **without replacement**, nearest first by Euclidean distance on the standardised
   matching keys, ties broken by ascending `open_time`.

If no control satisfies (1)–(4), the zone is **dropped from the paired test** and counted. The
"zone" for a control bar is that bar's own `high`/`low`, and `research/baseline.py`'s methodology
statement already encodes this convention — every control outcome is computed from strictly later
closed bars, and no future MFE/MAE/retest is used to select controls.

### 4. Exact entry rule

§3.2 verbatim, applied identically to both arms: control LONG rests at the control bar's `low`,
control SHORT at its `high`. Both arms record `fill_status` and shadow outcomes. **The fill-rate
comparison between arms is a primary result**, because it is the direct test of whether HAWK's ~83%
retest rate is anything but geometry.

### 5. Exact exit rules

§3.1 and §3.3 verbatim, identical for both arms, with `σ̂_8h` computed from each bar's own `rv_24h`.

### 6. Required existing HAWK fields

All of S3's, plus, for every bar in the panel (not just zone bars): `det_volume_ratio`,
`det_wick_ratio`, `det_distance_to_swing`, `det_zone_size_pct`, `det_direction`. `features.py`'s
`detector_features` already computes these for **every** bar — its docstring states that this is
exactly so the comparison population can be matched on them. From the `zones` schema the paired keys
are `symbol`, `source_open_time`, `zone_high`, `zone_low`, `status`, `retest_at`,
`hypothetical_entry_price`, `mfe_1h/4h/8h/24h`, `mae_1h/4h/8h/24h`.

### 7. Additional data required

- Per-bar labels and per-bar detector features across the **whole** panel, not just zone bars.
  `research/labels.per_bar_labels` already does this by design ("Computing these for all bars (not
  just zone bars) is what makes the null tests possible"). Cost: storage and compute, roughly
  80 symbols × 70,000 bars × ~60 columns. No download.
- The control matching table itself, persisted with the matching keys and distances so the match is
  auditable and reproducible.
- No new market data.

### 8. Market condition / behaviour exploited

None, by construction. S5 is not designed to exploit a market condition; it is designed to determine
whether HAWK's event definition selects moments that behave differently from matched ordinary
moments. Its economic content is a prior question: whether the detector's three gates isolate a
population where the compensation for providing liquidity into a non-fundamental move is larger than
average.

### 9. The objective test

- **H0 (primary):** `Δ_zone = 0` — the zone contributes nothing beyond the matched control.
- **H0 (secondary, all reported):** `Δ_fill = 0` (fill rates equal); `Δ_adverse = 0` (adverse
  selection equal); `Δ_retest = 0` (the ~83% retest rate equals the control's);
  `Δ_rv = 0` (forward realised volatility `fwd_rv_4h` equal — this is the "zone as volatility
  filter" claim, expected to be the *only* one that survives).
- **Test:** `paired_block_bootstrap_diff_ci` on the matched pairs, Gonçalves time-slice resampling,
  `b = 10 d`, `B = 10,000`, percentile-t, **shared bootstrap index sequences with all other systems**
  so every comparison is paired.
- **Control / benchmark:** the matched non-zone arm *is* the benchmark. It is not a separate trial —
  it has no independent claim attached to it, and the statistic tested is the difference.
- **Sample:** tier D, Pass P zone arm plus its matched controls.
- **Outcome variables:** §3.4, both arms.

### 10. Metrics

§3.8 in full **for both arms side by side**, plus the four secondary differences of item 9, plus the
matched-pair count, the drop count from failed matching, and the covariate-balance table (mean and
SD of every matching key in both arms; standardised mean difference must be < 0.10 on every key, or
the match is not a match and S5 is INCONCLUSIVE on construction grounds).

### 11. Evidence of edge (pass)

Gates 0–7 on the **zone arm**, **plus** `Δ_zone > +5 bps` with its 95% CI excluding zero, **plus**
covariate balance < 0.10 standardised mean difference on every matching key.

### 12. Falsification (any one suffices)

- **`Δ_zone` 95% CI contained inside [−5 bps, +5 bps]** ⇒ **the zone is an event marker.** This is
  the headline falsification and it has a specific, pre-committed consequence in §9: no further
  directional trials may be spent on the order-block event stream, and any continued use of the
  detector must be justified as execution or volatility machinery only.
- `Δ_zone` negative with its 95% CI excluding zero ⇒ the detector selects **worse**-than-average
  moments, i.e. the three gates are actively harmful. Same consequence, more strongly.
- `Δ_retest` 95% CI inside ±3 pp ⇒ the ~83% retest rate is geometry, and no statistic derived from
  it may be cited again.
- Covariate imbalance > 0.10 on any matching key that cannot be fixed without changing the matching
  rule ⇒ INCONCLUSIVE (changing the matching rule after seeing the imbalance would be a new trial).
- ≤ 2 of 6 folds with `Δ_zone > 0`; PBO > 0.50 on the zone arm.

### 13. Execution realism

| Quantity | Forecast |
|---|---|
| Zone arm | identical to S3: ≈ 13/day signalled, ≈ 6 fills/day |
| Control arm | up to 3× the zone arm ⇒ ≈ 39/day signalled. **The control arm is simulated on an unconstrained portfolio**, paired by time, because imposing `K_max = 5` on it would make the comparison depend on the queue-priority rule rather than on the zone. The zone arm's capacity-constrained curve is reported separately for Gates 2–5. This asymmetry is deliberate and is declared here so it cannot be re-litigated later. |
| Cost | identical ladder on both arms, per-symbol by tier. Controls are **not** given cheaper execution. |
| Funding | identical treatment |
| `n_eff` (zone arm) | ≈ **1,927**, as S3 |
| `n_eff` (paired difference) | governed by the zone arm; forecast ≈ **1,900** |

### 14. Look-ahead and leakage audit

All of S1's and S3's §14 rows apply, plus the four risks unique to a matched-control design:

| Risk | Specific check |
|---|---|
| **Control selection using the future** | Assert the control-selection function's inputs contain **no** column whose name matches `fwd_*`, `entry_*`, `first_up_*`, `first_dn_*`, `mfe_*`, `mae_*`, `retested`, or `lifecycle_status` (`research/audit._is_feature_column` already provides the classifier). Selecting controls on anything in `(t0, t1]` would be the most damaging possible leak here, because it would manufacture `Δ_zone` in whichever direction the selection happened to favour. |
| **Matching on a post-decision quantity** | Every matching key must be point-in-time at the **control bar's own** close, not at the zone's. Assert by truncation test on 20 sampled control bars. |
| **Overlapping label windows across arms** | A control bar within ±48 h of its zone shares most of its outcome path with the zone's. The pairing is therefore intentional and handled by the **paired** bootstrap (identical index sequences), not by assuming independence. Assert the paired bootstrap is used and that no unpaired t-test appears anywhere in the S5 report. `Σū` must be computed on the **union** of both arms' label windows, because concurrency is what the two arms share. |
| **Control contamination by near-misses** | A bar that fails the volume gate at 1.34× is not a meaningful control for one that passes at 1.36×. Report the distribution of `det_volume_ratio` and `det_wick_ratio` in the control arm and assert the arm is **not** concentrated in the 1.2–1.35 volume band; if more than 40% of controls sit within 10% of a gate threshold, the control is a near-miss population and the test is INCONCLUSIVE on construction grounds. |
| **Purge/embargo across arms** | `assert_no_leak(purge=16h, embargo=40h)` must be evaluated on the union of both arms' `[t0, t1]` spans, per fold and per CPCV split. A control bar sitting inside the purge gap of a test-fold zone is a leak even though it is "only" a control. |

---

## 9. Trial budget allocation

### 9.1 The budget

Bailey, Borwein, López de Prado & Zhu (2014): with 2 years of daily data, no more than **~7
independent strategy configurations** may be tried before an in-sample Sharpe of 1.0 is the
*expected* outcome under zero skill. Verifying, as protocol §6.1 does:

```
bracket(7) = 0.4228 * 1.068 + 0.5772 * 1.621 = 1.387
MinBTL     = (1.387 / 1.0)^2 = 1.92 years   <= 2.0 available    OK
```

**Family-A budget: `N_eff` ≤ 7. Raw-test cap: 20.**

### 9.2 Every trial this pre-registration consumes

| # | System | Direction rule | Horizon | Barrier pair | Entry | Exit | Configs | **Raw trials** |
|---|---|---|---|---|---|---|---|---|
| 1 | **S1** Level-mechanism split | BREAK/NEAR, 1 fixed rule | 8 h | `1.0σ̂ / 0.5σ̂` | maker @ far edge | first-touch + 8 h time stop | 1 | **1** |
| 2 | **S3** Idiosyncratic overreaction fade | tier-conditional fade, 1 fixed rule | 8 h | `1.0σ̂ / 0.5σ̂` | maker @ far edge | same | 1 | **1** |
| 3 | **S4** Absorbed-flow reversal | 1 fixed rule | 8 h | `1.0σ̂ / 0.5σ̂` | maker @ far edge | same | 1 | **1** |
| 4 | **S2** Signed wick side | `side = −wick_side`, 1 fixed rule | 8 h | `1.0σ̂ / 0.5σ̂` | maker @ far edge | same | 1 | **1** |
| 5 | **S5** Zone vs matched control | S3's rule, unchanged, on two populations | 8 h | `1.0σ̂ / 0.5σ̂` | maker @ far edge | same | 1 | **1** |
| | | | | | | | **`M_raw`** | **5** |

Consumes **0** additional trials, with the enforcement rule of §3.12 (verdict must hold at the worst
value of every axis; selecting the best value retroactively converts the axis into `k` trials):

| Item | Count if it were a trial | Why it is not |
|---|---|---|
| Cost levels 22 / 32 / 52 bps + 5-rung tier ladder | 8 | Joint requirement; Gate 2 demands the lower bound at 32 and the point estimate at 52 |
| `K_max ∈ {2,5,10,20}` | 4 | Protocol §3.2: if the conclusion depends on it, the edge is beta exposure |
| Block length `b ∈ {4,10,20,40}` d | 4 | Protocol §3.4: report the widest interval as the headline |
| `(purge, embargo) ∈ {(16,40),(48,72)}` | 2 | Protocol §2.3 sensitivity; sign change ⇒ inconclusive, not positive |
| Rolling **and** anchored walk-forward | 2 | Protocol §2.4: both reported, neither selected |
| Pessimistic **and** optimistic ambiguity resolution | 2 | Protocol §9.3: must hold under pessimistic |
| Fixed-`%` `bar_2_1_8h` readout | 1 | Descriptive only; cannot establish or overturn a verdict |
| Within-rule breakdowns: cells, tiers, deciles, months, symbols, fill status | ~40 | All *inside* a pre-registered rule; none can be promoted to a claim |
| Three benchmarks + three ablations | 6 | Nulls and integrity checks, not candidate strategies |
| Family-B univariate screens | ≤ 40 | Separate family, BH-FDR `q = 0.10`; **may never be cited as evidence**; promotion costs Family-A budget |

### 9.3 Forecast `N_eff` (Li–Ji), and whether five fit

The five trials share one event population, one horizon, one barrier pair, one entry convention, one
cost model and one portfolio constructor. S5 uses S3's rule verbatim; S2's rule agrees with S3's on
the `named` cells; S1 operates on a superset of the same bars. Their daily net return series will
therefore be strongly correlated. Forecast under an equicorrelated 5×5 trial-correlation matrix:

| Assumed `ρ_trial` | Eigenvalues | **Li–Ji `N_eff`** | Nyholt `M_eff` | Reported (larger) |
|---|---|---|---|---|
| 0.5 | 3.0, 0.5×4 | 3.00 | 4.20 | **4.20** |
| 0.3 | 2.2, 0.7×4 | 4.00 | 4.71 | **4.71** |
| 0.1 | 1.4, 0.9×4 | 5.00 | 4.97 | **5.00** |

**Forecast `N_eff` = 4.2 – 5.0 against a budget of 7. All five systems fit, with 2.0–2.8 effective
trials of headroom.** That headroom is reserved, in this order: (i) one Gate-1 VOID re-run (protocol
§8 Gate 1: a re-run costs one trial); (ii) one Family-B promotion into a future, freshly
pre-registered round.

The resulting thresholds, from protocol §6.4, are applied at the **conservative end of the range**:
DSR computed at the declared realised `N_eff` but **gated at the `N_eff = 7` value of annualised net
SR ≥ 1.90**, and `|t| ≥ 2.71`. If the realised Li–Ji `N_eff` comes in above 15 — which would mean
pipeline re-runs consumed the headroom — the hurdle rises to `|t| ≥ 3.00` automatically, and the
budget is enforced by the harsher threshold rather than by pretending the extra trials never
happened.

### 9.4 Staged fallback, if the budget is breached in practice

If realised `N_eff` exceeds 7 before all five are evaluated (the realistic cause is bug-fix re-runs
against D, which protocol §6.3 requires be counted), the round is truncated to a staged sequence, in
this order, with an explicit earn-the-next-trial rule:

1. **S1** and **S5** run together. They are the two tests worth running unconditionally: S1 because
   it is the highest-prior mechanism and tests a population currently thrown away; S5 because it is
   what makes any S1 result interpretable.
2. **S3** runs only if S5's `Δ_zone` CI is not contained in ±5 bps **or** S1's `Δ_B` CI excludes
   zero. Rationale: if the zone adds nothing and swing geometry carries nothing, S3 is a test of
   generic crypto reversal on an arbitrary event sample, which belongs in a different project.
3. **S4** runs only if S3 reaches at least INCONCLUSIVE-with-positive-point-estimate. Its mechanism
   (absorption) is a refinement of S3's (overreaction), and refining a dead effect is not a good use
   of a trial.
4. **S2** runs last, or not at all. Its only unique deliverable is the `Δ_wick` contrast, which is
   cheap and decisive but has an 8% prior and is dominated by the negative equity evidence under the
   correct null.

### 9.5 The holdout allocation rule, fixed now

Tier H may be read **exactly once, ever**, for family `A1`. Therefore: **among the systems that pass
Gates 0–5 on D, the holdout is spent on the one with the lowest pre-registered rank number (S1 → S3
→ S4 → S2 → S5).** If **no** system passes Gates 0–5, **H is not touched at all** and remains
available for a future round. This is mechanical, rank-based, fixed before any result exists, and it
prevents the single most valuable asset in the study from being spent on whichever system happened to
look best.

---

## 10. Ranking by strength and robustness of prior evidence

Ranked by evidence quality, **not** by how attractive the idea is and **not** by expected return.

| Rank | System | Prior | Evidence it rests on | Why it ranks here |
|---|---|---|---|---|
| **1** | **S1** Level-mechanism split | **20%** | Osler 2003 *JF* (first order-book data on stop/take-profit clustering; two mechanisms with **opposite** signs); Osler 2005 *JIMF* (stop response larger and longer-lasting, "significant for hours, not days"); Osler 2000 FRBNY (levels stopped trends 56.2% vs arbitrary levels). `evidence_context_costs.md` §6 assigns the family **50%** — the highest directional prior in either review — and names Osler "the single strongest citation in this whole family." | The only one of the five with a **microstructural mechanism established on order-level data**, and the only one whose separating variable (distance to swing) is already recorded and already known to be used with the wrong sign. Discounted from 50% to 20% because: FX spot, not crypto perps; the trading-rule versions died on costs (Curcio/Goodhart/Guillaume/Payne 1997) and out-of-sample (Sullivan/Timmermann/White 1999 — "completely reversed"); Osler claims conditional paths, not net profit. |
| **2** | **S3** Idiosyncratic overreaction fade | **18%** | Wen/Bouri/Xu/Zhao 2022 *NAJEF* (21 reversal vs 12 momentum among OOS-positive half-hour pairs; reversal attributed to overreaction to non-fundamental information); Borgards & Czudaj 2020 *JIFMIM* (overreactions prevalent at hourly frequency, contrarian returns larger than in equities, magnitude the dominant driver); Kozlowski/Puleo/Zhou 2021 *AEL* (7.37% daily, t = 24.23, strongest in illiquid coins); Zaremba et al. 2021 *IRFA* (>3,600 coins; sign flips with liquidity); Liu/Tsyvinski/Wu 2022 *JF* (momentum stronger among large coins). | **The largest and most replicated body of peer-reviewed evidence at HAWK's horizon** — four independent studies, one *JF*. Ranked below S1 only because the sign at exactly the 1-day horizon is contested (Caporale & Plastun 2020 find continuation with exploitable profits), the spreads are all gross, and HAWK's universe is the liquid head where the literature says reversal **weakens or flips**. |
| **3** | **S4** Absorbed-flow reversal | **12%** | Cont/Kukanov/Stoikov 2013 *JFE* (OFI explains ~65% of mid-price variance, 35–60% excluding price-changing events; volume-based impact "noisy and less robust"); Jones/Kaul/Lipson 1994 *RFS* (trade **count**, not size, carries the information); Osler for the inventory persistence. | The underlying microstructure result is among the **strongest effect sizes anywhere in this document** and the data is free. But it is **contemporaneous explanatory power, not prediction**: cross-asset lagged OFI predictability decays within minutes, and a crypto replication produced direction AUC 0.66–0.69 alongside a **negative validation R²** on returns. Asking a 15 m bar aggregate to forecast 8 h is far outside the evidence's range. |
| **4** | **S2** Signed wick side | **8%** | For: Moser & Brauneis 2026 *IREF* — hourly, ~400 coins, 200 M observations, 55 patterns, **stepwise SPA** data-snooping control; Hanging Man the one single-bar survivor. Against: Marshall/Young/Rose 2006 *JBF* — **22 of 28 rules significant on zero of 35 stocks**, max 3/35, and bullish patterns did **better on the random OHLC bootstrap than on real data**; Horton 2009 concurs; Lo/Mamaysky/Wang 2000 found distributional information but explicitly not profit, on multi-bar patterns. | One credible positive against two well-identified negatives, the negatives using the **correct null** (joint OHLC bootstrap) that this system is obliged to adopt. Moser & Brauneis also report the patterns are "unaffected by volume", which means HAWK's volume gate adds nothing. `evidence_signals.md` puts the family at 14%, rising to ~22% if the wick is signed and the trend conditioned; 8% here after the cost hurdle. |
| **5** | **S5** Zone vs matched control | **7%** | Against: zero peer-reviewed validation of order blocks / FVG / BOS / SMC; **0 of 648** ICT variants beat buy-and-hold (best order-block effect t = 1.22); **0 of 54** FVG variants profitable after 0.5 pip on 2.55 M EURUSD bars; `evidence_signals.md` assigns the construct **4%**. For: nothing published. | Last on **probability of passing**, because the construct it tests has no supporting evidence at all. **First on information value**, because it is the only test whose *negative* result changes what the project is allowed to do next (§8 item 12 → §11 stopping rule), and because both evidence reviews state that no statistic about HAWK's zones — the retest rate above all — means anything until it has a matched control. |

**What I expect to fail, stated plainly.**

- **S2 and S5 I expect to fail**, and I would not be surprised if they fail cleanly enough to be
  FALSIFIED rather than INCONCLUSIVE. S2 because the correct null (joint OHLC bootstrap) has killed
  this family twice on better data. S5 because the construct has a 4% prior and two independent
  mechanical tests returning exactly zero.
- **S3 I expect to produce the most frustrating outcome**: a real gross effect concentrated in T3,
  the illiquid tier, that does not survive T3's own 47–68 bps cost rung. That is the documented
  pattern in the crypto ML literature (`evidence_context_costs.md` §5.5: alphas "concentrated in
  hard-to-trade assets … harvesting extreme returns on small, illiquid, volatile coins"), and it is
  a falsification of the *system* even where the *effect* is real.
- **S1 is the one I would bet on if forced**, at 20%.
- **Most likely overall outcome: INCONCLUSIVE on all five.** Protocol §8.1 says this explicitly for
  any true edge below ~55% hit rate or Sharpe ~2, and it must be reported as such rather than
  escalated. My joint probability that at least one reaches "EVIDENCE OF AN EDGE" is **12–18%**.

---

## 11. The role of the zone detector — a concrete, testable answer

### 11.1 The six candidate roles and my pre-registered position

| Role | My pre-registered position | Prior |
|---|---|---|
| **(a) Directional signal** | **No.** `direction = close > open` is the sign of a body that `wick_ratio ≥ 1.2` guarantees is small, so it is close to a coin flip on microstructure noise — a sufficient mechanistic explanation for the observed 47–51% hit rate and AUC ≈ 0.50. No system here uses it. | 4% |
| **(b) Location / context filter** | **Probably yes, but with the sign inverted.** The `distance_to_swing ≥ 0.5%` gate excludes exactly the locations where the one documented mechanism operates. This is the single most likely way the detector is currently destroying value. | 25% |
| **(c) Volatility / regime filter** | **Almost certainly yes, and almost certainly not an edge.** `volume_ratio` is a volatility forecaster, not a direction forecaster (Karpoff 1987; Jones/Kaul/Lipson 1994). Useful for barrier scaling and sizing; Cederburg et al. 2020 (72 of 103 strategies worse out-of-sample) is a direct warning against using it as a directional gate. | 85% that it forecasts `fwd_rv_*`; ~15% that gating on it improves risk-adjusted returns |
| **(d) Entry trigger / passive price reference** | **Yes, and this is the most likely way HAWK survives.** The zone supplies two objective, non-arbitrary passive limit prices. Maker entry is worth **12 bps of a 34 bps base case** — a third of the cost, available with **zero directional content**. | 70% |
| **(e) Trade-management reference** | **Untested, and deliberately not allocated a trial.** Whether `zone_size_pct`-scaled stops beat `atr_48_pct`-scaled stops is a real question, but it is a stop-placement parameter search and would cost trials this budget does not have. Deferred to Round 2. | not estimated |
| **(f) Mere event marker** | **My modal position.** The zone samples moments of elevated volume and unusual geometry; another signal, computed from data the zone does not supply, does the directional work. | 55% |

**Synthesis:** the detector is most defensibly an **event marker (f) whose edges are a useful passive
execution reference (d), with a real but non-tradeable volatility component (c), and whose location
gate (b) is currently pointed the wrong way.** That combination has a concrete implication: HAWK's
value, if it has any, is more likely to come from *paying less* and *knowing where to rest an order*
than from knowing which way price will go.

### 11.2 Which systems discriminate between the roles, and what result favours each

| Discriminating test | Result favouring each interpretation |
|---|---|
| **S5, `Δ_zone`** — the decisive test for (a)/(f) | `Δ_zone > +5 bps`, CI excludes zero ⇒ the zone contributes beyond a matched control: **(a) or (b) or (c)** is live. `Δ_zone` CI inside ±5 bps ⇒ **(f): event marker**, and by pre-commitment no further directional trials may be spent on the event stream. `Δ_zone < −5 bps`, CI excludes zero ⇒ the gates are **actively harmful**. |
| **S5, `Δ_retest`** | CI inside ±3 pp ⇒ the ~83% retest rate is **geometry, not order flow**, and may never be cited again. |
| **S5, `Δ_rv`** (forward `fwd_rv_4h`) | Zone arm materially higher, CI excludes zero, while `Δ_zone` ≈ 0 ⇒ **(c) volatility filter and nothing more.** I expect exactly this. |
| **S1, `Δ_B` and `Δ_N` vs the FAR cell** — the decisive test for (b) | `Δ_B > 0` and `Δ_N > 0` with `Δ_FAR ≈ 0` ⇒ **(b) location filter, with HAWK's gate inverted**: the information lives in the populations the 0.5% gate excludes or biases against. Both CIs inside ±5 bps ⇒ swing geometry carries nothing either way and (b) is dead. `Δ_B > 0 > Δ_N` ⇒ only the stop-cascade branch is real, which is Osler's stronger effect and would be the cleanest possible confirmation. |
| **S4, `Δ_abs`** | Effect confined to the disagreeing cells ⇒ the zone's **wick geometry** is doing real work in combination with flow, which is a specific, narrow version of **(a)**. Effect present in both cells ⇒ it is a flow or volatility effect and the zone is **(f)**. |
| **S2, `Δ_wick`** | `Δ_wick > 0`, CI excludes zero ⇒ wick **side** carries sign information, so HAWK's `max(upper, lower)` collapse was destroying a real feature — a narrow **(a)**. CI inside ±5 bps ⇒ it was not, and signed-wick work stops permanently. |
| **Fill rate + adverse selection, reported in all five** — the test for (d) | Maker fill rate ≥ 50% **and** measured adverse selection `E[ret\|filled] − E[ret\|shadow]` better than **−12 bps** ⇒ the zone edges are a **net-positive execution device even with zero directional content**, and (d) is confirmed independent of every other verdict. Adverse selection worse than −12 bps ⇒ the maker saving is illusory, the entry must be taker, and the cost base reverts to 34 bps, which raises the required edge above the null from 8.63 pp back to 13.3 pp and makes all five systems substantially harder. |

That last row is worth emphasising: **the adverse-selection measurement is the single most
consequential number in this study**, because it either preserves or destroys the 12 bps that the
whole design is built on, and it is measurable without any directional edge existing at all.

---

## 12. Hard data requirements

Consolidated checklist. **Bold = must be captured at backfill time or it is a full re-download.**

| # | Requirement | Status / source | Cost if deferred |
|---|---|---|---|
| 1 | 15 m OHLCV, ~80 symbols, ≥ 2 years, `open_time` spacing exactly 900,000 ms, gaps identified not interpolated | `research/backfill.py` (data.binance.vision monthly/daily ZIPs, REST fallback) | — |
| 2 | **`quote_volume`, `trades`, `taker_buy_quote` per bar** | Already in `KEEP_COLUMNS`. **Verify present for every symbol-month, including pre-2025 headerless archives.** | Free at backfill; full re-download otherwise |
| 3 | **`taker_buy_base` retained** — currently parsed in `KLINE_COLUMNS` but **dropped from `KEEP_COLUMNS`** | **ACTION REQUIRED.** Needed to cross-check `taker_buy_quote/quote_volume` against `taker_buy_base/volume`; a mismatch is the signature of exactly the ms/µs timestamp bug `_normalise_epoch` exists to catch. | Free now; full re-download otherwise. **S4 is VOID without the cross-check.** |
| 4 | **Point-in-time universe: `(symbol, listed_at, delisted_at)` and bars for symbols that are NOT in today's top 80**, including delisted and near-dead pairs | **ACTION REQUIRED — the single largest re-download risk.** If the backfill was scoped to today's top-80 list, the panel is survivorship-filtered and `pit_rank_prev_1d` cannot be computed over the correct cross-section. Protocol §9.1 items 1–4. | Full re-download of the excluded symbols. **S3 and S5 are VOID without it; all five are biased.** |
| 5 | `pit_rank_prev_1d`: integer rank by **trailing-24 h** quote volume as of the prior completed UTC day, over all then-listed symbols | Computable from (1)+(4). Note: **trailing-24 h, not 30 d**, chosen so `L_feat = 24 h` and the embargo stays at 40 h. | — |
| 6 | Pass R — second replay with `min_distance_from_high_low = 0.0`, `passes_swing_gate` flag, bit-identity assertion against Pass P on the overlap | `research/replay.py` with `DetectorParams(min_distance_from_high_low=0.0)` | Recompute only. **S1 is impossible without it.** |
| 7 | Per-bar detector features and per-bar labels across the **whole** panel, not just zone bars | `features.detector_features` and `labels.per_bar_labels` already do this by design | Compute + storage. **S5 is impossible without it.** |
| 8 | **ATR/volatility-normalised first-touch labels measured from the limit fill price**: `entry_first_up_sig1p0`, `entry_first_dn_sig0p5`, with `σ̂_8h = rv_24h × sqrt(32)` frozen at the decision bar | **NEW label family.** `labels.BARRIER_LEVELS` is a fixed fractional grid; per-zone volatility-scaled barriers need a new scan. Computable from stored bars. | Compute only. All five systems gate on it. |
| 9 | **Limit-order fill simulation fields**: `limit_price`, `fill_status`, `fill_open_time`, `traded_through`, `fill_lag_bars`, and a full `shadow_*` outcome block for unfilled signals | **NEW.** `strategies.TradeSpec` currently supports `entry ∈ {close, retest}` and `side_rule ∈ {direction, fade, long, short}`; it needs `entry = 'limit_far_edge'`, an externally supplied `side` column, and a volatility-scaled barrier exit mode. | Compute only. Without the shadow arm, adverse selection is an assumption and Gate 1 fails. |
| 10 | BTC 15 m series aligned on `open_time` with **no forward-fill**, plus trailing `beta_btc_24h` (96 bars, `min_periods = 96`) | `features.btc_features` already NaN-propagates rather than forward-filling | — |
| 11 | Residual close-path barrier outcomes (§3.4 scientific outcome) | Computable from (1)+(10). Closes only, by construction. | Compute only |
| 12 | **Per-symbol MEXC funding interval and realised funding-rate history** | **API pull required.** Do **not** assume 8 h across 80 symbols — many altcoin perps fund hourly or 4-hourly, and Coin Metrics distinguishes `period` from `interval` for exactly this reason. Record settlements crossed and realised funding per trade. | Cheap API pull; but an assumed interval mis-charges funding on a meaningful subset |
| 13 | Per-symbol, per-month spread proxy: **Abdi–Ranaldo on trailing 15 m OHLCV excluding signal bars**, cross-checked against Amihud, `quote_volume_24h`, and the Harvard Dataverse Binance spread dataset (`doi:10.7910/dvn/9ava2b`) | Computable from (1) + one dataset download. **Corwin–Schultz on the signal bar is circular** — HAWK's wick gate mechanically inflates it. | Cheap; but a constant or current spread is a look-ahead (protocol §9.5 item 13) |
| 14 | **MEXC perp 15 m klines for a sample month, per traded symbol** | **ACTION REQUIRED.** `backfill.py` sources `data.binance.vision/data/spot` — **Binance spot** — while execution is on **MEXC perp**. The price path being labelled and the instrument being traded are different markets. Minimum: verify per-symbol 15 m close correlation ≥ 0.99 and calibrate the spread ladder against MEXC's own book for one month. | Cheap for a sample; if the correlation fails, the entire cost model and every fill assumption are measured on the wrong instrument |
| 15 | **1 m klines for a 10% random symbol-month subsample** | **CAPTURE AT BACKFILL.** Protocol §9.3 makes same-bar barrier ambiguity > 10% a **hard stop** requiring 1 m or tick resolution. Forecast ambiguity is < 3% (a 255 bps single-bar range is ≈ 5–6 σ here), but it must be *measured*, and measuring it after the fact means a second archive pass. A 10% subsample makes the check cheap now. | Free at backfill; a second archive pass otherwise. If ambiguity > 10% and 1 m is unavailable ⇒ **VOID** on the barrier formulation |
| 16 | Control matching table with keys, distances and covariate-balance statistics, persisted and auditable | **NEW.** Per §8 item 3. | Compute only. **S5 is not auditable without it.** |
| 17 | Trial ledger: every configuration ever run against D, including abandoned runs and post-bug-fix re-runs | `multiple_testing.TrialLedger` + `research/experiments.create_experiment` | Protocol §6.3 — the most commonly violated rule in the literature. Without it the study is unassessable. |
| 18 | Holdout access script with an append-only audit log (UTC timestamp, repo git SHA, frozen config hash, this file's SHA), refusing a second read for family `A1` | Protocol §2.2 | Without it the one-touch rule is aspirational, not enforceable |

**The four items that must be actioned before or during backfill, or else require a re-download:**
**#3** (`taker_buy_base`), **#4** (non-top-80 and delisted symbols — the big one), **#14** (MEXC perp
sample), **#15** (1 m subsample). Everything else is compute.

---

## 13. What would make us abandon the project

A pre-committed stopping rule, so that a negative outcome is a **legitimate conclusion** rather than
a prompt to keep searching. All thresholds are numeric and evaluated on tier D after the single
Family-A round.

### 13.1 Primary abandonment condition (all three must hold)

Abandon — conclude there is no tradeable edge in the HAWK order-block event stream — if, after the
five trials:

1. Hansen's SPA consistent p-value **> 0.05** across all five curves against both benchmarks
   (⇒ protocol §6.4(e): StepM is not reported and **no individual claim may be made**); **and**
2. **no** system has net expectancy with a bootstrap 95% lower bound > 0 at `c = 32 bps`; **and**
3. the best system's residual 8 h barrier-hit-rate 95% **upper** bound is **< 43.5%** — i.e. even the
   optimistic end of the interval sits below break-even (42.0%) plus half the study's own 3.0 pp
   detection resolution (`p̄*` 42.0% → MDE 45.0%).

Consequence: all five recorded **FALSIFIED**. **Tier H is not touched.** No further Family-A trials
are authorised against the order-block event stream. The result is published in full, losers
included, per protocol §10.

### 13.2 Unconditional stop triggers (any one suffices, regardless of everything else)

| Trigger | Numeric condition | Consequence |
|---|---|---|
| **The zone adds nothing** | S5's `Δ_zone` 95% CI contained in **[−5, +5] bps** | The detector is an **event marker**. No further **directional** trials on the event stream, ever. Continued use of the detector must be justified solely as an execution reference (role d) or a volatility marker (role c), and any such use requires its own pre-registration. |
| **The zone is harmful** | S5's `Δ_zone` < −5 bps with the 95% CI excluding zero | Stronger form of the above: the three gates select **worse**-than-average moments. Stop using the detector for selection at all. |
| **Broad walk-forward failure** | **≥ 4 of 5** systems with **≤ 2 of 6** folds positive | Abandon. Five independent mechanisms failing the same consistency test is not five unlucky draws. |
| **Selection is worse than random** | PBO **> 0.50** for the best system | Abandon per protocol §6.4(b): the in-sample-optimal configuration is more likely than not below the OOS median. |
| **CPCV collapse** | CPCV median OOS Sharpe **≤ 0** for the best system | FALSIFIED per protocol §8 Gate 5. |
| **The label is not computable** | Same-bar ambiguity **> 10%** of resolved outcomes **and** 1 m data unavailable | VOID, then abandon the first-touch barrier formulation entirely. The outcome variable does not exist at 15 m resolution. |
| **The maker saving is illusory** | Measured adverse selection `E[ret\|filled] − E[ret\|shadow]` worse than **−12 bps**, with the 95% CI excluding −12 | The 12 bps the design is built on does not exist. Entry reverts to taker (`c` = 34 bps), the required edge above the null rises from **8.63 pp to 13.3 pp**, and none of the five is plausible at that hurdle. Abandon unless a system clears the taker hurdle outright, which none is expected to. |
| **Underpowered by construction** | `n_eff` **< 300** in the evaluated window for a system | That system is INCONCLUSIVE regardless of every other gate (protocol §8 Gate 7). An underpowered pass is not a pass. |
| **Budget exhausted** | Realised Li–Ji `N_eff` **> 15** before all five are evaluated | Hurdle rises to `\|t\| ≥ 3.00`; if the round cannot be completed inside `N_eff ≤ 20`, the entire round is downgraded to **EXPLORATORY**, no capital may be allocated on its basis, and a fresh pre-registration on fresh data is required. |
| **Holdout failure** | Gate 6: net expectancy < 0 on H with the 95% interval excluding D's point estimate | FALSIFIED. **Abandon.** H has been spent and may not be read again for family `A1`. |

### 13.3 What earns the right to continue

- **One system passes Gates 0–5** ⇒ spend the single H touch on it per §9.5. Pass Gate 6 ⇒
  "EVIDENCE OF AN EDGE", proceed to sized paper trading with the frozen configuration and continuous
  tier-L monitoring. Fail Gate 6 ⇒ FALSIFIED ⇒ abandon.
- **INCONCLUSIVE across the board with positive point estimates** (the modal outcome) ⇒ **do not
  touch H, do not spend more trials, do not re-run with adjusted parameters.** The only legitimate
  next action is to **acquire more independent information**: extend the calendar history (2 y → 4 y
  roughly doubles `n_eff` and halves the required Sharpe), or wait for tier L to accumulate. Protocol
  §5.5 item 5 is explicit — with this design small true edges *will* be missed, and the correct
  response is a better design or more data, **not looser thresholds**.
- **A Family-B screen produces a hit** ⇒ it is a **lead, never a finding**. Promotion into a future
  Family-A round costs Family-A budget and requires a new pre-registration file.

### 13.4 The pre-committed statement of a negative result

If §13.1 or any of §13.2 fires, the published conclusion is, verbatim:

> *The HAWK order-block detector, treated as a fixed event generator over ~80 Binance 15 m USDT
> pairs, does not identify a population whose forward 8 h price behaviour supports a directional
> strategy that survives realistic MEXC perpetual round-trip costs, under a pre-registered protocol
> with a 5-trial budget, `n_eff` of approximately 2,000–2,300 effectively independent observations,
> and a Deflated Sharpe threshold of 0.95. Five mechanisms with independent literature support were
> tested and all failed. The negative result is the finding.*

**A credible negative result is the valuable outcome here, and it is worth more than a marginal
positive that would not replicate.**

---

## 14. Assumption register for this pre-registration

To be measured and published **before** any outcome statistic is examined. `[A*]` tags refer to
`validation_protocol.md` §11.

| Tag | Assumed value | How it is replaced |
|---|---|---|
| **[B1]** | Median `rv_24h` = 0.30% per 15 m bar ⇒ `σ̂_8h` = 1.70%, `T` = 1.70%, `L` = 0.85%, span = 255 bps | Measure the median and full distribution of `rv_24h` over the D population. If the median `σ̂_8h` differs from 1.70% by more than **±25%**, recompute `p̄*` and the required hit rate **once**, publish as a committed amendment referencing this SHA, before examining outcomes. The multipliers 1.0 and 0.5 are **not** amendable. |
| **[B2]** | Pass-P zone rate ≈ 41/day (30,000 / 730 d, protocol [A4]) | Count after the replay. `n_eff` is nearly invariant to it (protocol §4.4(a)). |
| **[B3]** | Pass R inflates the candidate count ×1.8 | Count. |
| **[B4]/[B5]** | BREAK share 25%, NEAR share 30% of Pass R | Count. |
| **[B6]** | Maker fill rate within 8 h at the far edge ≈ 45% (NEAR ~55%, BREAK ~25%) | **The single most important measurement in the study.** Measure per system, per cell, per tier. Together with adverse selection it determines whether the 12 bps maker saving is real. |
| **[B7]** | 32% of zones satisfy `\|resid_ret_4h\| ≥ 1.0 σ` | Count. |
| **[B8]/[B9]/[B10]** | Trade-count spike 55%, `\|flow_imb\| ≥ 0.10` 60%, outer-third-and-disagreeing 20% | Count. |
| **[B11]** | 70% of zones have an unambiguous dominant wick (≥ 2× the other) | Count. |
| **[B12]** | `ρ̄` = 0.35 on the residualised outcome matrix (down from protocol [A2] = 0.50) | Measure on per-symbol signed residual barrier outcomes in 16 h buckets, separately for BTC-up / flat / down regimes; **use the highest**. Cross-check with Li–Ji `M_eff` and the first eigenvalue's variance share. Report `n_eff` at 0.35 / 0.50 / 0.70. |
| **[B13]** | Within-day cross-sectional trade correlation 0.5, used in the §3.11 Sharpe arithmetic | Measure from the realised trade table; re-derive the required per-trade edge and publish before outcomes. |
| **[B14]** | Same-bar barrier ambiguity < 3% | Measure. Hard stop at 10%. |
| **[B15]** | `atr_48_pct ≈ 1.7 × rv_24h` (the ATR-unit column of §3.1) | Measure and publish the realised mapping so the ATR-normalised and percentage statements of the barriers are reconcilable. |
| **[A7]** | Round trip 22 bps maker-in/taker-out base case; 32 bps at the expected tier mix; 52 bps pessimistic | Per-symbol, per-period Abdi–Ranaldo spread proxy + realised MEXC API fee tier + measured slippage from live fills. |
| **[A8]** | `γ₃ = −0.5`, `γ₄ = 6` for DSR and MinTRL | Measure on the candidate's daily net portfolio returns. A 2:1 barrier strategy is negatively skewed and leptokurtic, which **raises** MinTRL and **lowers** DSR — do not leave these Gaussian. |
| **[A9]** | Cross-trial annualised Sharpe dispersion `sqrt(V[SR])` = 0.50 | Compute the SD of annualised Sharpes across the five realised trials. Never leave this an assumption in the final report. |

---

## 15. Pre-registration completeness check (protocol §6.5 items 1–7)

| # | Requirement | Where |
|---|---|---|
| 1 | Exact entry rule, exit rule, barrier levels, horizon, universe filter — precise enough for an independent implementer to produce identical trades | §2, §3.1–3.3, and items 3/4/5 of each of §4–§8 |
| 2 | Complete list of parameter values to be tried, `M_raw`, forecast `N_eff` | §9.2 (`M_raw` = 5), §9.3 (forecast `N_eff` 4.2–5.0). **Values not on this list may not be tried against D.** |
| 3 | Cost model, `K_max`, sizing rule, queue-priority rule | §3.5, §3.6 |
| 4 | Primary metric, CI method, block length | §3.7, §3.8 (`b = 10 d` primary, `{4,10,20,40}` joint sensitivity) |
| 5 | **Numeric** pass / fail / inconclusive thresholds | §3.10, §3.11, item 11 and item 12 of each of §4–§8, §13 |
| 6 | Predicted effect direction and an economic reason to expect it | item 1 and item 8 of each of §4–§8 |
| 7 | Family-B leads being promoted and the budget they consume | **None.** No Family-B lead is promoted in this round; the 2.0–2.8 trials of headroom in §9.3 are reserved for one VOID re-run and one future promotion. |

**Ledger to open before the first read of D:** `multiple_testing.TrialLedger`, seeded with the five
configurations of §9.2, the forecast `N_eff` of §9.3, this file's git SHA, and the repo commit SHA.
