# HAWK Statistical Validation Protocol

**Status:** prescriptive protocol. Read sections 5, 6 and 8 before running any test.
**Compiled:** 2026-09-17.
**Scope:** the decision layer that turns detected order-block zones into long / short / skip, evaluated on ~80 USDT altcoin pairs, 15-minute bars, ~2 years of point-in-time replayed history, ~10,000–60,000 zone events, maximum outcome horizon 48 hours.

Every assumed number in this document is tagged **[A#]** and listed in §11 with the measurement that must replace it.

---

## 0. The one-paragraph verdict

With ~80 symbols whose returns are dominated by BTC beta, 30,000 zone events over 2 years are worth roughly **720 effectively independent observations**, not 30,000 — a shrinkage of ~42x, which inflates any naive t-statistic by ~6.5x. At that effective sample size and 80% power, the study can detect a barrier-hit rate of **55.2%** or an annualised Sharpe of **~2.0**, and nothing smaller. It therefore **cannot** distinguish 51%, 52% or 53% from noise, and cannot detect a Sharpe of 0.3, 0.5 or 1.0. It is, however, comfortably powered for the *economic* question, because after ~30 bps of round-trip cost a symmetric ±1% barrier strategy needs a **65% hit rate** to break even, and 65% requires only ~85 effectively independent observations to detect. The study is well-powered to answer "is this tradeable?" and badly underpowered to answer "is there a small true edge?". Design the study around the first question.

---

## 1. Evidence summary

### 1.1 Multiple testing and data snooping

| Work | Claim used here |
|---|---|
| White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5), 1097–1126. [PDF](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf) | Tests H0: the *best* of K strategies does not beat a benchmark, using a block/stationary bootstrap with recentring under the null. Controls FWER across all K jointly. Test statistic is `max_k mean(d_k)`, unstudentised. |
| Hansen, P.R. (2005). "A Test for Superior Predictive Ability." *JBES* 23(4), 365–380. [DOI](https://doi.org/10.1198/073500105000000063) | Studentises each loss differential by its HAC standard deviation and uses a sample-dependent null. Strictly more powerful than White's Reality Check and far less sensitive to padding the candidate set with bad variants — which matters here because HAWK will test many near-duplicate parameter cells. Available as `arch.bootstrap.SPA`. |
| Romano, J.P. & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized Data Snooping." *Econometrica* 73(4), 1237–1282. [PDF](http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/MultipleComparision/RomanoWolf05.pdf) | StepM: a step-down procedure that identifies *which* strategies beat the benchmark, not merely whether one does, while still controlling FWER and implicitly capturing the joint dependence of the test statistics. This is the right tool for HAWK's Family A because the team needs to know which concept survived, not just that something did. |
| Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *JRSS-B* 57(1), 289–300. [PDF](https://www.dcscience.net/Benjamini-Hochberg-1995-FDR.pdf) | Controls the expected *proportion* of false rejections. Valid under independence or positive regression dependence; under arbitrary dependence use Benjamini–Yekutieli (2001). Appropriate for exploratory screens whose outputs will be re-tested, not for confirmatory claims. |
| Harvey, C.R., Liu, Y. & Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *RFS* 29(1), 5–68. [Link](https://academic.oup.com/rfs/article/29/1/5/1843824) | After accounting for the factor-mining history, a new factor needs **t > 3.0**, not 2.0. Their BHY-calibrated hurdle reached 3.39 and Bonferroni 3.78 by the early 2010s. "Most claimed research findings in financial economics are likely false." |
| Harvey, C.R. & Liu, Y. (2015). "Backtesting." *JPM* 42(1), 13–28. [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF) | The haircut-Sharpe method: convert SR to a t-ratio, adjust the p-value for the number and correlation of trials, convert back. The haircut is *non-linear* — marginal Sharpes are penalised brutally, high Sharpes only mildly. A flat 50% haircut is explicitly "a serious mistake." Matlab `Haircut_SR.m` / `Profit_Hurdle.m` at [charvey/backtesting](https://people.duke.edu/~charvey/backtesting/). |
| Harvey, C.R. & Liu, Y. (2020). "False (and Missed) Discoveries in Financial Economics." *JF* 75(5), 2503–2553. [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P143_False_and_missed.pdf) | A data-specific double bootstrap that yields the t-hurdle for a *target* FDR, and a second hurdle for an acceptable ratio of misses to false discoveries. Existing methods "lack power to detect outperforming managers" — the Type II side is not free. |
| Sullivan, R., Timmermann, A. & White, H. (1999). "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap." *JF* 54(5), 1647–1691. [PDF](https://www.fmg.ac.uk/sites/default/files/2020-11/dp303.pdf) | The direct precedent: 7,846 technical rules on 100 years of DJIA. The magnitude of the snooping penalty is set by the *dependence structure of the whole universe the winner was drawn from*, so the universe must be specified honestly and up front. |

### 1.2 Backtest overfitting

| Work | Claim used here |
|---|---|
| Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *JPM* 40(5), 94–107. [PDF](https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) | `E[max SR] ≈ E[SR] + sqrt(V[SR])·((1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne)))`, γ = 0.5772. Under the zero-skill null this is `SR_0`, and `DSR = Φ( (SR - SR_0)·sqrt(T-1) / sqrt(1 - γ3·SR + ((γ4-1)/4)·SR²) )`. N must be the number of *independent* trials. |
| Bailey, D.H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk* 15(2). [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643) | PSR and Minimum Track Record Length: `MinTRL = 1 + (1 - γ3·SR + ((γ4-1)/4)·SR²)·(Φ⁻¹(p)/(SR - SR*))²`. Tells you whether a track record is even long enough to support the claim. |
| Bailey, Borwein, López de Prado & Zhu (2014). "Pseudo-Mathematics and Financial Charlatanism." *Notices of the AMS* 61(5), 458–471. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659) | Minimum Backtest Length. **With only 2 years of daily data, no more than ~7 independent strategy variations should be tried**; at 7 trials the expected maximum in-sample Sharpe is 1.0 with an expected out-of-sample Sharpe of 0.0. (5 years supports ~45.) Under memory effects, overfit strategies lose money out of sample rather than merely earning zero. A backtest that does not report N cannot be assessed. |
| Bailey, Borwein, López de Prado & Zhu (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4), 39–69. [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | PBO via **combinatorially symmetric cross-validation (CSCV)**: split the trial-return matrix into S blocks, form all C(S, S/2) IS/OOS partitions, and estimate the probability that the IS-optimal configuration lands below the OOS median. Model-free and non-parametric. Hold-out alone is "unreliable and inaccurate" in a backtest context. R package `pbo`. |
| Arnott, R., Harvey, C.R. & Markowitz, H. (2019). "A Backtesting Protocol in the Era of Machine Learning." *JFDS* 1(1), 64–74. [PDF](https://www.pm-research.com/content/iijjfds/1/1/64.full.pdf) | Research protocols, checklist-style, should be *de rigueur*. Their explicit warning: ML methods need far more data than finance supplies, so choose the application before the tool. Also the reminder that the goal is not to reject everything — Type II errors have a cost too. |

### 1.3 Financial cross-validation

**López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN 978-1-119-48208-6.** [Publisher TOC](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/advances-in-financial-machine-learning-978-1-119-48208-6)

- **Ch. 4.4, Average uniqueness of a label.** With overlapping outcome windows, compute concurrency `c_t` (how many labels are live at time t), give each label `u_t = 1/c_t`, and average over its lifespan: `ū_n = Σ_t u_{t,n} / Σ_t 1_{t,n}`. `Σ_n ū_n` is the number of non-redundant observations. Feeds sample weights and sequential bootstrapping so that bagged models produce honest out-of-bag estimates.
- **Ch. 7.4, Purged K-Fold CV.** Standard k-fold leaks whenever a training label's outcome window `[t0, t1]` overlaps a test label's window, because the two share the same future price path — the model is scored on information it was trained on. **Purging** removes those training observations. **Embargo** additionally drops observations immediately following the test set, to kill residual serial dependence (features and volatility carry across the boundary even when labels do not). Embargo is applied first, then purging.
- **Ch. 12.4, Combinatorial Purged CV.** Standard walk-forward yields exactly **one** out-of-sample path, so there is no distribution to test against. CPCV splits history into N blocks, tests all C(N, k) combinations of k test blocks with purge+embargo, and recombines them into `(k/N)·C(N,k)` distinct OOS equity paths. That distribution is what DSR and PBO should be computed against.

Available in `skfolio` as `CombinatorialPurgedCV(n_folds, n_test_folds, purged_size, embargo_size)`; sample weights in `mlfinpy`.

### 1.4 Walk-forward analysis and its pitfalls

- **Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*, 2nd ed. Wiley,** Ch. 11. The original treatment: optimise on an in-sample window, freeze, trade the immediately following unseen window, roll, stitch the disjoint out-of-sample segments into one curve. That stitched curve is the record; the in-sample fit is not evidence. Common practice sets in-sample length at 4–6x the out-of-sample period.
- **Anchored vs rolling.** Anchored (expanding) training uses all history from a fixed start and gives more stable estimates when the relationship is structurally stable; rolling (fixed-width) adapts to regime drift but is noisier and discards long-run information. Neither is correct a priori. The honest treatment is to fix the geometry from operational arguments, then report *both* as a pre-registered sensitivity — never to pick the one that scored better.
- **The critique that matters most for HAWK:** walk-forward is itself routinely overfit, by two distinct mechanisms. (i) *Geometry selection* — choosing train/test/step lengths because that combination produced the best composite is selection bias in a lab coat. (ii) *Repeated reconfiguration* — every time the pipeline is re-run after adjusting the feature set, the labelling, the cost model or the fold layout, another trial is consumed, and almost nobody counts them. Under Bailey et al.'s MinBTL arithmetic, 2 years of data is exhausted after **7** independent trials, and a team that has iterated a walk-forward twenty times has already spent its entire budget before the first formal test.
- **Fold count.** More folds is not more validation. Overlapping test windows inflate the apparent fold count while adding almost no independent information. Keep `step = test_length` so OOS windows are disjoint, and keep each test window long enough to contain a meaningful number of *effectively independent* observations (§4). Practical implementations cap folds in the mid single digits for exactly this reason.

### 1.5 Resampling under dependence

| Work | Claim used here |
|---|---|
| Künsch, H.R. (1989). *Annals of Statistics* 17(3), 1217–1241. [DOI](https://doi.org/10.1214/aos/1176347265) | Moving-block bootstrap (MBB): resample overlapping blocks of `ℓ` consecutive observations. Preserves dependence within blocks, destroys it across them. |
| Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap." *JASA* 89(428), 1303–1313. [Link](https://www.semanticscholar.org/paper/4cf01507070ef68e7bbf5c9421b21d4ede10feac) | Geometric (random) block lengths with mean `b = 1/p`. The resampled series is *stationary*, unlike MBB/CBB, which removes the edge artefacts that bias block bootstraps at short T. This is the default for HAWK. |
| Politis, D.N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." *Econometric Reviews* 23(1), 53–70. [PDF](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf) | Data-driven optimal `b` via flat-top lag-window spectral estimation, adaptive to the correlogram. |
| Patton, A., Politis, D.N. & White, H. (2009). "Correction to…" [PDF](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf) | The stationary-bootstrap variance constant is `D_SB = 2ĝ²(0)`. **Use the corrected version**; the 2004 formula as published gives the wrong `b`. |
| **Gonçalves, S. (2011). "The Moving Blocks Bootstrap for Panel Linear Regression Models with Individual Fixed Effects." *Econometric Theory* 27(5), 1048–1082.** [PDF](https://cireqmontreal.com/wp-content/uploads/2012/04/goncalves.pdf) | **This is the answer to "how do I resample when observations are correlated across time *and* across the cross-section simultaneously."** Apply the block bootstrap to the *vector* `Z_t = (z_{1t}, …, z_{nt})` containing all n cross-sectional units at time t, resampling whole time-slices. Because individual units are never resampled separately, the method is robust to **arbitrary, unknown, and even strong** contemporaneous cross-sectional dependence, while blocks handle weak serial dependence. Coverage is good "even when the degree of serial and cross sectional correlation is large, provided the block size is appropriately chosen." |
| Ledoit, O. & Wolf, M. (2008). "Robust performance hypothesis testing with the Sharpe ratio." *JEF* 15(5), 850–859. [PDF](http://www.ledoit.net/jef_2008pdf.pdf) | Jobson–Korkie/Memmel Sharpe tests are invalid under heavy tails or serial dependence. Use a **studentised circular-block-bootstrap confidence interval** for the Sharpe (difference) and reject if it excludes zero. HAC standard errors alone are not enough in finite samples. R: `PeerPerformance::sharpeTesting(type = 2)`. |

### 1.6 Power, breadth and effective sample size

| Work | Claim used here |
|---|---|
| Grinold, R.C. (1989). "The Fundamental Law of Active Management." *JPM* 15(3), 30–37. [DOI](https://doi.org/10.3905/jpm.1989.409211) | `IR = IC · sqrt(BR)`, where BR is the number of **independent** bets. Correlated bets do not count. |
| Clarke, R., de Silva, H. & Thorley, S. (2002). "Portfolio Constraints and the Fundamental Law of Active Management." *FAJ* 58(5), 48–66. [PDF](https://people.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf) | `IR = TC · IC · sqrt(BR)`. The transfer coefficient TC is the cross-sectional correlation between risk-adjusted forecasts and realised active weights. TC = 1.0 only without binding constraints; **in practice 0.3–0.8**. Equal-sized bets, a concurrency cap, and no conviction weighting all push TC down — which is exactly HAWK's design. |
| Li, J. & Ji, L. (2005). "Adjusting multiple testing in multilocus analyses using the eigenvalues of a correlation matrix." *Heredity* 95(3), 221–227. [Link](https://www.nature.com/articles/6800717) | Effective number of independent tests from the eigenvalues of the test-statistic correlation matrix: `M_eff = Σ_i [ 1(|λ_i| ≥ 1) + (|λ_i| - floor(|λ_i|)) ]`. Use `M_eff` in place of M in Bonferroni/Šidák. More accurate than Cheverud (2001), which is over-conservative for many moderately correlated tests. Nyholt (2004): `M_eff = 1 + (k-1)(1 - Var(λ)/k)`. R: `poolr::meff()`. |
| Two Sigma (2021). "Using Factors to Explain Risk in Crypto Assets." [PDF](https://www.twosigma.com/wp-content/uploads/2021/07/Using-Factors-to-Explain-Risk-in-Crypto-Assets-3.pdf) | Average pairwise return correlation across the ten largest coins ≈ **0.48**; BTC–ETH ≈ 0.74. First principal component ("crypto beta") explains ~47% of variance, first two ~69%. Basis for assumption **[A2]**. |
| Bariviera et al. (2020). "Are cryptocurrencies becoming more interconnected?" [Link](https://ar5iv.labs.arxiv.org/html/2009.14561) | Pesaran CD test rejects weak cross-sectional dependence in favour of **strong** cross-sectional dependence in all cases tested, with average pairwise correlations trending up over time. Justifies treating HAWK's panel as strongly cross-sectionally dependent, i.e. Gonçalves-style resampling rather than per-symbol resampling. |
| Jahanshahloo et al. (2023), *Journal of Money & Economy*. [PDF](https://jme.mbri.ac.ir/article-1-650-en.pdf) | Average pairwise crypto correlation is regime-dependent: 0.49 in a bullish sub-period, rising to **0.78** after the market fell. Correlations spike exactly when it hurts, so `ρ̄` must be measured per regime and the pessimistic value used for power. |

### 1.7 Metric definitions and their pitfalls

| Work | Claim used here |
|---|---|
| Lo, A.W. (2002). "The Statistics of Sharpe Ratios." *FAJ* 58(4), 36–52. [DOI](https://doi.org/10.2469/faj.v58.n4.2453) | Under IID, `SE(SR) ≈ sqrt((1 + SR²/2)/T)`. Under serial dependence the scale factor changes and **monthly Sharpes cannot be annualised by sqrt(12)** except in special cases; use the general stationary-return formula with a Newey–West HAC estimate. A hedge fund's annual Sharpe was overstated by up to **65%** through serial correlation alone, and rankings changed materially once corrected. Practical corollary: the SE of an annualised Sharpe is ≈ `1/sqrt(N_years)`, so distinguishing a true Sharpe of 1.0 from zero at 95% needs ~3.8 years. |
| Bailey, D.H. & López de Prado, M. (2015). "Stop-outs under serial correlation and the triple penance rule." *Journal of Risk* 18(2), 61–93. [Link](https://www.risk.net/journal-of-risk/2435335/stop-outs-under-serial-correlation-and-the-triple-penance-rule) | Closed-form maximum quantile loss and time-under-water without assuming IID normal. Ignoring serial correlation **underestimates downside potential by as much as 70%**, and firms using naive Sharpe-based stop-outs fire up to 3x more skilful managers than intended. The "triple penance" result: recovery takes ~3x the time taken to accumulate the maximum quantile loss. |

---

## 2. Prescribed data-splitting scheme

### 2.1 Calendar boundaries

Assume the backfill covers **2024-09-01 → 2026-08-31** (730 days) **[A1]**. Adjust the anchor dates if the actual backfill start differs, but preserve the *proportions* and the ordering.

| Tier | Dates | Days | Purpose | Times it may be touched |
|---|---|---|---|---|
| **D — Development** | 2024-09-01 → 2026-02-28 | 546 | Feature engineering, labelling decisions, walk-forward, CPCV, all parameter search, all model selection | Unlimited, but every distinct configuration counts against the trial budget (§6) |
| **F — Firewall** | 2026-03-01 → 2026-03-31 | 31 | Nothing. Deliberately discarded. | Never |
| **H — Holdout** | 2026-04-01 → 2026-08-31 | 153 | Single confirmatory evaluation of the one pre-registered configuration | **Exactly once, ever** |
| **L — Live forward** | 2026-09-01 → ongoing | accumulating | Continuous confirmation; the existing ~1,065-zone / 10-day sample lives here | Monitored continuously, but see §4.4 — 10 days is ~10 effectively independent observations and is nearly uninformative |

The 31-day firewall is far larger than the 5-day statistical minimum derived below. It is deliberately generous, because its job is not only to purge label overlap but to absorb the informal contamination that accumulates when a team looks at recent charts, recent live results, and recent zone behaviour during exploration.

### 2.2 The one-touch rule for H

Mechanically enforceable, not aspirational:

1. H is stored in a separate location with read access gated behind a script that **appends an immutable audit line** (UTC timestamp, git commit SHA of the whole repo, the frozen configuration hash, the pre-registration document hash) before returning any data.
2. The script refuses to run if the audit log already contains an entry for the same strategy family.
3. Any second read for a given family is a **protocol breach**: H is retired permanently, and the study must wait for new forward data (tier L) to accumulate a fresh confirmatory window.
4. Cost-model parameters, position sizing, and the concurrency cap must be frozen in the pre-registration too. "Re-running the holdout with realistic fees" is a second touch.
5. No summary statistic from H — not even "it was positive" — may be used to guide further work on D.

### 2.3 Purge and embargo, justified from the horizon

Let `H_max = 48 h` be the maximum outcome horizon and `L_feat` the longest feature lookback (trailing swing high/low window; assume 96 bars = 24 h **[A3]**).

- **Purge = 48 h.** Drop every training observation whose outcome window `[t0, t1]` intersects the span of any test observation's outcome window. Since `t1 - t0 ≤ 48 h`, this removes at most 48 h of training labels adjacent to each test boundary. Rationale (AFML §7.4.1): a training label whose future price path overlaps a test label's path is scored on shared information.
- **Embargo = 72 h.** Applied *after* the test window. Composed of 48 h (the horizon, so a test label's outcome cannot reach into subsequent training data) + 24 h (`L_feat`, so post-test features are not computed from bars inside the test window). Rounded up to cover volatility-cluster spillover, which AFML notes persists past the label horizon.
- **Total dead zone per fold boundary = 5 days.** At 10 boundaries across the fold structure below, this costs ~50 of 546 development days, i.e. **6.8% of the sample** — an acceptable price.
- Sensitivity, pre-registered: repeat with (purge, embargo) = (48 h, 48 h) and (48 h, 240 h). If the sign of the conclusion changes, the result is **inconclusive**, not positive.

```python
# Purge + embargo as a hard assertion, run per fold. A failure must abort the run.
def assert_no_leak(train, test, purge_h=48, embargo_h=72):
    # train/test: DataFrames with columns t0 (decision time) and t1 (outcome resolution time)
    lo, hi = test.t0.min(), test.t1.max()
    before = train[train.t1 <  lo]
    after  = train[train.t0 >  hi]
    assert len(before) + len(after) == len(train), "purge failed: label windows overlap test span"
    assert (lo - before.t1.max()) >= pd.Timedelta(hours=purge_h),   "purge gap too small"
    assert (after.t0.min() - hi)  >= pd.Timedelta(hours=embargo_h), "embargo gap too small"
```

### 2.4 Walk-forward fold structure inside D

Fixed from operational arguments before seeing results, per §1.4:

- **Scheme:** rolling (fixed-width) primary, anchored (expanding) as a pre-registered sensitivity. Both reported; neither selected on the basis of its result.
- **Train window:** 180 days. Rationale: Pardo's 4–6x guidance against a 60-day test window gives 240–360 days, but crypto regime persistence is short and 180 days already contains ~90 effectively independent observations (§4), so 180/60 = 3x is the defensible compromise. This is a *choice*, pre-registered, not tuned.
- **Gap:** 5 days (48 h purge + 72 h embargo).
- **Test window:** 60 days.
- **Step:** 60 days — equal to the test length, so **out-of-sample windows are strictly disjoint**.
- **Folds:** 6.

| Fold | Train (days) | Gap | Test (days) |
|---|---|---|---|
| 1 | 1–180 | 181–185 | 186–245 |
| 2 | 61–240 | 241–245 | 246–305 |
| 3 | 121–300 | 301–305 | 306–365 |
| 4 | 181–360 | 361–365 | 366–425 |
| 5 | 241–420 | 421–425 | 426–485 |
| 6 | 301–480 | 481–485 | 486–545 |

Total stitched out-of-sample: **360 days**, containing roughly `360/730 × 720 ≈ 355` effectively independent observations. Six folds is the maximum that keeps test windows both disjoint and long enough to be individually interpretable; do not increase it to manufacture apparent replication.

### 2.5 Combinatorial purged CV inside D

Walk-forward gives one OOS path, which is not a distribution. Run CPCV in parallel on the same development window:

- **Blocks:** N = 12, each ~45 days (12 × 45 = 540 ≈ 546).
- **Test blocks per split:** k = 2.
- **Splits:** C(12, 2) = **66**.
- **Distinct OOS equity paths:** (k/N)·C(N,k) = (2/12)·66 = **11**.
- Purge 48 h and embargo 72 h at every block boundary within every split.

The 11 paths are the input to the Deflated Sharpe Ratio and to the path-dependence diagnostics. For PBO, use CSCV with S = 12 blocks, giving C(12, 6) = **924** in-sample/out-of-sample partitions.

---

## 3. Handling overlapping and cross-correlated trades

### 3.1 Why per-trade independence is fatal here

Zones fire on many symbols at once. Two zone events on SOLUSDT and AVAXUSDT that open within the same hour share (a) the same 48-hour BTC path, which is most of both outcomes, and (b) the same volatility regime. Treating them as two observations rather than roughly one is the single largest source of overstated significance in this study. §4 quantifies it: the inflation factor on a naive t-statistic is about **6.5x**. A "highly significant" naive t of 4.0 is a true t of about 0.6.

### 3.2 Step 1 — aggregate to a portfolio equity curve

**All inference is performed on a portfolio return series, never on a pool of independent trades.** Trade-level statistics are descriptive only.

Construction:

| Element | Prescription |
|---|---|
| Capital model | Fixed total equity, `1/K_max` of equity per position, no compounding within a fold |
| Concurrency cap `K_max` | **5** simultaneous positions |
| Per-symbol cap | 1 open position per symbol |
| Queue discipline | When at cap, new zones are **rejected** (not queued and back-filled later). Rejections are logged and counted; the rejection rate is a reported diagnostic. |
| Selection when several zones fire in the same bar and capacity is short | Deterministic, pre-registered rank (e.g. descending zone score), tie-broken by symbol name. Never by outcome. |
| Aggregation frequency | Mark to market every 15 m; **resample to daily returns** for all Sharpe-family inference |
| Costs | Applied per fill at the historical spread proxy, not a constant (see `evidence_context_costs.md`) |

`K_max = 5` is deliberately close to the effective cross-sectional breadth of ~2 (§4.2). Holding 40 simultaneous altcoin longs does not give 40 bets; it gives ~2 bets, 40 lots of fees, and a portfolio that is a leveraged BTC proxy. Pre-register a sensitivity at `K_max ∈ {2, 5, 10, 20}` and report all four; if the conclusion depends on `K_max`, the "edge" is a beta exposure.

### 3.3 Step 2 — sample-uniqueness weighting

For any model fitted on zone-level labels (the meta-labeller, the univariate screen, the feature-importance work), apply AFML §4.4:

1. Build the indicator matrix `1_{t,n} = 1` if label n is live at bar t.
2. Concurrency `c_t = Σ_n 1_{t,n}`.
3. `u_{t,n} = 1_{t,n} / c_t`; average uniqueness `ū_n = Σ_t u_{t,n} / Σ_t 1_{t,n}`.
4. Pass `ū_n` as `sample_weight` to every estimator, and use **sequential bootstrapping** (draw with probability proportional to uniqueness recomputed after each draw) inside any bagged ensemble, so out-of-bag estimates are honest.
5. Report `Σ_n ū_n` alongside the raw event count in **every** table. A table that says "n = 30,000" without also saying "Σū = 365" is misleading and should be rejected in review.

### 3.4 Step 3 — cross-sectionally aware block bootstrap

Follow **Gonçalves (2011)**: block-resample the *time index*, carrying the entire cross-section with it. Never resample symbols or trades individually.

**Procedure (the canonical one; all confidence intervals in this protocol come from it):**

1. Form the daily panel `Z_t = (r_t^{port}, r_{1t}, …, r_{80,t}, x_t)` for `t = 1…T`, where `r_t^{port}` is the portfolio return, `r_{it}` the per-symbol contribution, and `x_t` any auxiliary series (BTC return, realised vol, zone count). `T = 546` for D, `T = 153` for H.
2. Draw geometric block lengths `ℓ ~ Geom(p)` with `p = 1/b`, `b = 10` days (**stationary bootstrap**, Politis–Romano 1994). Draw start indices uniformly on `1…T`, wrapping circularly.
3. Concatenate whole time-slices `Z_t` until the pseudo-series has length `T`. **Every symbol is carried by the same index sequence**, which is what preserves contemporaneous cross-sectional dependence of arbitrary strength.
4. Recompute the statistic of interest on each pseudo-series. `B = 10,000` replications for confidence intervals; `B = 5,000` for SPA / StepM (Davidson–MacKinnon 2000 on bootstrap counts).
5. Report **studentised (percentile-t)** intervals, not raw percentile intervals — Gonçalves shows percentile-t has the good coverage properties even under large serial and cross-sectional correlation. For Sharpe specifically, use the Ledoit–Wolf (2008) studentised construction.

**Block length `b`, justified rather than assumed:**

| Consideration | Implied `b` |
|---|---|
| Must exceed 2 x max outcome horizon to break label overlap | ≥ 4 days |
| `T^{1/3}` rule of thumb at `T = 546` | ≈ 8 days |
| Crypto volatility clustering / regime persistence | ~10–20 days |
| Politis–White (2004) automatic selection with the Patton–Politis–White (2009) `D_SB = 2ĝ²(0)` correction | **compute and report** |

**Prescription: `b = 10` days for the primary result.** Also compute the automatic `b`; if it exceeds 20 days, use the larger value, because under-sizing blocks understates variance and over-sizing merely costs power. Pre-registered sensitivity: `b ∈ {4, 10, 20, 40}`. Report the widest interval across the four as the headline.

**Do not bootstrap at the 15-minute level.** `T = 70,080` gives `T^{1/3} ≈ 41` bars ≈ 10 hours, which is shorter than the 48-hour horizon and would silently break the very dependence you are trying to preserve. Aggregate to daily first.

```python
def stationary_panel_bootstrap(Z, b=10, B=10000, rng=None):
    """Z: (T, k) daily panel. Resamples TIME SLICES, carrying the whole
    cross-section, per Goncalves (2011). Geometric blocks per Politis-Romano (1994)."""
    rng = rng or np.random.default_rng(0)
    T = len(Z); p = 1.0 / b
    for _ in range(B):
        idx, t = [], rng.integers(T)
        while len(idx) < T:
            idx.append(t)
            t = rng.integers(T) if rng.random() < p else (t + 1) % T
        yield Z[np.asarray(idx[:T])]          # all columns move together
```

---

## 4. Effective sample size

### 4.1 Temporal collapse via average uniqueness

Assumptions: 30,000 zone events **[A4]**, span 730 days = 17,520 h **[A1]**, max horizon 48 h **[A5]**.

```
total label-time      = 30,000 events x 48 h            = 1,440,000 symbol-hours
calendar time         = 730 days x 24 h                 =    17,520 hours
average concurrency   = 1,440,000 / 17,520              =      82.2 labels live at any instant
average uniqueness    = 1 / 82.2                        =    0.0122
sum of uniqueness     = 30,000 x 0.0122                 =       365 non-redundant observations
```

Cross-check by a completely different route: the number of **non-overlapping** 48-hour windows in the sample is `17,520 / 48 = 365`. The two agree exactly, which is reassuring — the horizon and the calendar, not the event count, set the temporal information content.

### 4.2 Cross-sectional collapse via the design effect

Within one instant, ~82 labels are live across ~80 symbols. For `K` observations with average pairwise correlation `ρ̄`, the variance of their mean is `(σ²/K)·(1 + (K-1)ρ̄)`, so the Kish effective count is:

```
n_eff,cluster = K / (1 + (K - 1) · ρ̄)
```

With `K = 82` and `ρ̄ = 0.50` **[A2]**:

```
n_eff,cluster = 82 / (1 + 81 x 0.50) = 82 / 41.5 = 1.98
```

**The ceiling is `1/ρ̄ = 2.00` as `K → ∞`.** This is the crucial structural fact: at ρ̄ = 0.5, eighty correlated altcoins are worth **two** independent bets, and no amount of universe expansion changes that. The 80-symbol universe is a cost-and-capacity decision, not a statistical-power decision.

### 4.3 Combining

```
n_eff  ≈  (non-overlapping time blocks) x (effective cross-sectional breadth)
       ≈  365 x 1.98
       ≈  723   →  call it 720
```

**30,000 zone events ≈ 720 effectively independent observations. Shrinkage factor 41.7x. Naive t-statistics are inflated by sqrt(41.7) = 6.5x.**

Sensitivity to `ρ̄`, which must be measured:

| `ρ̄` (avg pairwise correlation of signed zone outcomes) | `n_eff,cluster` | `n_eff` |
|---|---|---|
| 0.20 | 4.79 | 1,748 |
| 0.30 | 3.24 | 1,183 |
| **0.50 [A2]** | **1.98** | **723** |
| 0.70 | 1.42 | 519 |
| 0.90 | 1.11 | 405 |
| 1.00 | 1.00 | 365 |

The plausible range is **365–1,750**, and the regime dependence documented by Jahanshahloo et al. (ρ̄ rising from 0.49 to 0.78 in a downturn) means the pessimistic end applies exactly when performance matters. Use `n_eff = 720` for planning and report results at 365 and 1,183 as bounds.

### 4.4 Two consequences that change the research plan

**(a) More zone events buy nothing.** If the detector is loosened and event count doubles to 60,000, `Σū` is unchanged at 365 (it is bounded by `calendar / H_max`), concurrency rises to 164, and `n_eff,cluster` becomes `164/(1 + 163 x 0.5) = 1.988`. Total `n_eff` moves from 723 to 726 — a **0.4% gain**. Tuning the detector for more zones is statistically worthless. Only two things buy power:

| Lever | Effect on `n_eff` |
|---|---|
| More calendar history (2 y → 4 y) | 723 → ~1,450 (linear, but slow to acquire) |
| **Shorter outcome horizon (48 h → 8 h)** | 723 → `(17,520/8) x 1.99 = 4,360`, a **6x gain**, available immediately |
| Lower `ρ̄` (market-neutralise against BTC before labelling) | ρ̄ 0.5 → 0.2 raises `n_eff` to ~1,750, a 2.4x gain |
| More symbols (80 → 200) | `n_eff,cluster` 1.98 → 1.99. Nil. |

**The single highest-value methodological change available is to shorten the maximum outcome horizon and to residualise zone outcomes against contemporaneous BTC returns before labelling.** Those two together move `n_eff` from ~720 to ~10,000, which converts an underpowered study into a powered one. This should be decided before any strategy testing begins.

**(b) The existing live evidence is essentially empty.** 1,065 zones over 10 days:

```
concurrency  = 1,065 x 48 h / 240 h        = 213
Σū           = 1,065 / 213                 = 5.0
n_eff        = 5.0 x (213 / (1 + 212x0.5)) = 5.0 x 1.99 = ~10
```

**Ten effectively independent observations.** The 95% interval on the observed ~49% hit rate is `0.49 ± 1.96 x 0.5/sqrt(10) = [0.18, 0.80]`. That interval contains 50%, contains the 65% break-even (§5.1), and contains almost everything else. The 10-day live sample neither supports nor refutes anything and must not be cited as evidence in either direction. Similarly, the out-of-sample AUC of ~0.50 has a standard error of roughly `sqrt(11/(12 x 5 x 5)) = 0.19` at this effective sample size — it is compatible with AUC anywhere from 0.13 to 0.87.

---

## 5. Power analysis

Throughout: two-sided `α = 0.05`, power `1 - β = 0.80`, so `z_{α/2} + z_β = 1.9600 + 0.8416 = 2.8016`.

### 5.1 First, the break-even hit rate — because it reframes everything

For a symmetric ±100 bps barrier with round-trip cost `c` bps, net expectancy is `200p - 100 - c`, so the break-even hit rate is `p* = (100 + c)/200`.

| Round-trip cost | Symmetric ±1% → `p*` | +1% / −0.5% → `p*` | +1.5% / −0.5% → `p*` |
|---|---|---|---|
| 26 bps (optimistic) | **63.0%** | 50.7% | 38.0% |
| **30 bps [A7]** | **65.0%** | **53.3%** | **40.0%** |
| 40 bps | 70.0% | 60.0% | 45.0% |
| 52 bps (pessimistic) | **76.0%** | 68.0% | 51.0% |

Two things follow immediately.

1. **The relevant null hypothesis is not `p = 0.50`.** It is `p = 0.65`. Testing against 0.50 answers a question nobody is paying for.
2. **The payoff ratio is a far more powerful lever than the hit rate.** Moving from ±1% symmetric to +1%/−0.5% drops the break-even from 65% to 53.3% at the same cost. Reducing the required edge from 15 percentage points above coin-flip to 3.3 is worth more than any amount of feature engineering. **Barrier asymmetry should be treated as a first-class design variable, not a fixed setting.**

### 5.2 Barrier-hit-rate power

`n = (z_{α/2}·sqrt(p₀(1-p₀)) + z_β·sqrt(p₁(1-p₁)))² / (p₁ - p₀)²` with `p₀ = 0.50`. "Years" converts at **360 effectively independent observations per year** (§4.3).

| True hit rate | Required `n_eff` | Equivalent calendar years | Available (2 y ⇒ `n_eff` 720)? |
|---|---|---|---|
| 51% | **19,620** | 54.5 | ✗ (27x short) |
| 52% | **4,903** | 13.6 | ✗ (6.8x short) |
| 53% | **2,178** | 6.1 | ✗ (3.0x short) |
| **55.2%** | **~720** | **2.0** | **= exactly the detection limit** |
| 55% | 783 | 2.2 | borderline ✗ |
| 60% | 194 | 0.54 | ✓ (3.7x surplus) |
| 65% (break-even at 30 bps) | **85** | 0.24 | ✓ (8.5x surplus) |

**Minimum detectable hit rate at `n_eff = 720` and 80% power: 55.2%.** At the pessimistic `ρ̄ = 0.7` (`n_eff = 519`) it is 56.2%; at the optimistic `ρ̄ = 0.3` (`n_eff = 1,183`) it is 54.1%. Under no plausible correlation assumption does 52% become detectable.

### 5.3 Sharpe-ratio power

Because `t = SR_annualised × sqrt(years)`, required calendar time is `(2.8016 / SR)²` — and note that **trade count does not enter**. Sharpe is a per-unit-time quantity; adding correlated symbols does not shorten the required track record.

| Annualised Sharpe (net of cost) | Required years of independent history | Required independent daily obs (365 d/y) | Available (2.0 y)? |
|---|---|---|---|
| 0.3 | **87.2** | 31,828 | ✗ (44x short) |
| 0.5 | **31.4** | 11,461 | ✗ (16x short) |
| 0.75 | 13.9 | 5,074 | ✗ (7x short) |
| 1.0 | **7.85** | 2,865 | ✗ (3.9x short) |
| 1.5 | 3.49 | 1,274 | ✗ (1.7x short) |
| **1.98** | **2.00** | **730** | **= exactly the detection limit** |
| 2.5 | 1.26 | 460 | ✓ |

Crypto trades continuously, so annualisation throughout this document uses **365 days/year**, not 252. Required *years* are invariant to that choice; required observation counts are not.

**Minimum detectable annualised Sharpe on the full 2 years: 1.98.** On the 18-month development window alone: `2.8016/sqrt(1.5) = 2.29`. On the 5-month holdout alone: `2.8016/sqrt(0.42) = 4.32`. Independently, Lo (2002) gives the same answer from the other direction: the standard error of an annualised Sharpe is ≈ `1/sqrt(N_years)`, so 2 years yields `SE ≈ 0.71` and a Sharpe of 1.0 has a 95% interval of roughly `[-0.4, 2.4]`.

**MinTRL cross-check** (Bailey & López de Prado 2012, `SR* = 0`, `p = 0.95`, `γ₃ = -0.5`, `γ₄ = 6` **[A8]**):

| Claimed annualised Sharpe | MinTRL (daily obs) | MinTRL (years) | Does the 153-day holdout suffice? |
|---|---|---|---|
| 1.0 | 1,018 | 2.79 | ✗ |
| 1.5 | 461 | 1.26 | ✗ |
| 2.0 | 265 | 0.73 | ✗ |
| **2.67** | **153** | **0.42** | **= exactly the holdout's limit** |
| 3.0 | 123 | 0.34 | ✓ |

So the holdout window on its own can only certify an annualised Sharpe of about **2.67 or better**. Anything below that must be certified on the stitched walk-forward / CPCV evidence, with the holdout serving as a consistency check rather than as independent proof.

### 5.4 Reconciling the two tests via the fundamental law

Grinold (1989) with the Clarke–de Silva–Thorley (2002) correction, taking breadth `BR = 360` effectively independent bets per year (§4.3), transfer coefficient `TC = 0.6` **[A6]** (equal sizing, hard concurrency cap, no conviction weighting — squarely inside the 0.3–0.8 practical range), and `IC ≈ 2p - 1` for a signed binary forecast:

```
IR = TC · IC · sqrt(BR) = 0.6 · (2p - 1) · sqrt(360) = 11.38 · (2p - 1)
```

| True hit rate | Implied `IC` | Implied annualised `IR` | Years to detect the `IR` | Years to detect the hit rate directly |
|---|---|---|---|---|
| 51% | 0.02 | 0.23 | 151 | 54.5 |
| 52% | 0.04 | **0.46** | **37.9** | 13.6 |
| 53% | 0.06 | 0.68 | 16.8 | 6.1 |
| 55% | 0.10 | 1.14 | 6.1 | 2.2 |
| 60% | 0.20 | 2.28 | 1.5 | 0.54 |

The last two columns differ by exactly `1/TC² = 1/0.36 = 2.78x`, which is the whole content of the transfer coefficient: **testing realised P&L is 2.8x less statistically efficient than testing the raw directional signal, purely because of how the signal is turned into positions.** Practical implications:

- Do the *scientific* test on the barrier-hit rate and the AUC, where power is highest.
- Do the *economic* test on the portfolio P&L, where the number that matters lives.
- Raising `TC` (volatility-scaled sizing, conviction weighting, letting the queue prefer higher-scoring zones) recovers statistical power for free, before any new alpha is found.

And the sting in the table: a **true, real, persistent 52% edge corresponds to an annualised IR of 0.46** — a genuinely respectable strategy — and would require **38 years** of this data-generating process to certify from P&L. This is not a fixable defect of the protocol. It is what an 80-symbol, ρ̄ = 0.5, 48-hour-horizon design costs.

### 5.5 Verdict: is the planned study adequately powered?

**No — for the small-edge question. Yes — for the tradeability question.**

| Question | Powered? | What the study can actually resolve |
|---|---|---|
| "Is the barrier-hit rate above 50%?" at 51–53% | **No.** Needs 6–54 years. | Nothing |
| "Is the barrier-hit rate above 50%?" at ≥ 55.2% | **Yes** | Detectable at 80% power |
| "Is the annualised Sharpe above 0 ?" at 0.3–1.0 | **No.** Needs 8–87 years. | Nothing |
| "Is the annualised Sharpe above 0?" at ≥ 2.0 | **Yes** | Detectable, and coincidentally equal to the multiple-testing hurdle (§6.4) |
| **"Does this clear the 65% break-even and make money after costs?"** | **Yes, decisively.** Needs only 85 obs; 720 available. | **This is the question to run.** |
| "Is the ~49% live hit rate consistent with a tradeable edge?" | Already answered on 720 obs: 95% CI `[0.454, 0.526]`, which **excludes 63–76%** | The symmetric ±1% concept is already falsifiable today |

**What this implies for the plan:**

1. **Reframe the primary hypothesis** from "p > 0.50" to "p > `p*(c)`", the cost-adjusted break-even. The study is powered for that and only that.
2. **The symmetric ±1% design can be adjudicated now** and, on current evidence, should be declared falsified — the observed hit rate's 95% upper bound is ~0.53 against a required 0.63–0.76. Do not spend six weeks of walk-forward infrastructure confirming this.
3. **Any "we found a 52% edge" result from this study is uninterpretable by construction** and must be reported as inconclusive, not as a finding, no matter how the p-value comes out.
4. **Fix the design before testing it.** Shorten `H_max` to 8 h, residualise outcomes against BTC, and make the barrier ratio asymmetric. Those three changes raise `n_eff` from ~720 to ~10,000 and drop the break-even hit rate from 65% to ~53%. That is the difference between a study that can find something and a study that cannot.
5. **Accept the Type II cost explicitly** (Arnott–Harvey–Markowitz; Harvey & Liu 2020). With this design, small true edges *will* be missed. The correct response is to improve the design, not to loosen the thresholds.

---

## 6. The multiple-testing budget

### 6.1 The budget, from MinBTL

Bailey, Borwein, López de Prado & Zhu (2014): with 2 years of daily data, **no more than ~7 independent strategy configurations** may be tried before an in-sample Sharpe of 1.0 is the *expected* outcome under zero skill. Verifying their arithmetic:

```
bracket(N=7) = (1-γ)·Φ⁻¹(1 - 1/7) + γ·Φ⁻¹(1 - 1/(7e))
             = 0.4228 · 1.068 + 0.5772 · 1.621  = 1.387
MinBTL(years) = (bracket / E[max SR])²  = (1.387 / 1.0)²  = 1.92 years   ✓
```

The plan — 5 concepts x ~4 variants = 20 backtests, plus a 30–40 feature screen ≈ **55–60 raw tests** — is roughly **8x over budget** if those tests were independent. They are not independent, which is the only thing that saves it.

### 6.2 Split the hypotheses into two families with different error controls

| | **Family A — Confirmatory** | **Family B — Exploratory** |
|---|---|---|
| Contents | The 5 strategy concepts and their parameter variants | The 30–40 univariate feature screens |
| Raw test cap | **20** | 40 |
| Effective trial cap | **`N_eff` ≤ 7** | no cap, but see below |
| Error control | **FWER at 5%**, via Romano–Wolf StepM on the 20 OOS equity curves, with Hansen's SPA as the omnibus pre-test | **BH-FDR at `q = 0.10`** |
| Bootstrap | Stationary panel bootstrap, `b = 10 d`, `B = 5,000`, shared index sequence across all trials so cross-trial correlation is preserved | same |
| Status of a "pass" | A claim: "there is evidence of an edge" | **A lead only.** Never a finding. |
| Promotion path | — | A Family-B hit may be promoted into a later, freshly pre-registered Family-A round, **which consumes Family-A budget** |

FWER for Family A and FDR for Family B follows directly from the standard criterion (Benjamini–Hochberg 1995; and the FWER-vs-FDR guidance in the clinical-trials literature): use FWER where a single false positive will be **acted on** — here, allocating real capital — and FDR where flagged results will be **validated by follow-up**. It is a protocol violation to report a Family-B result as evidence of an edge. Note also that BH assumes independence or positive regression dependence; HAWK's features are strongly and not necessarily positively dependent, so use **Benjamini–Yekutieli** if any screened feature pair has correlation below −0.3.

### 6.3 Computing `N_eff` — mandatory, not optional

Both DSR and the t-hurdle need the *effective* number of trials. Grid-searching 20 parameter cells of one concept is not 20 experiments.

```python
def n_eff_trials(trial_returns):
    """trial_returns: (T, M) matrix of block-aggregated returns, one column per trial.
    Li & Ji (2005) effective number of independent tests."""
    lam = np.abs(np.linalg.eigvalsh(np.corrcoef(trial_returns, rowvar=False)))
    return float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))
```

Rules:
- Compute `N_eff` via Li–Ji on the **M x M correlation matrix of the trials' own return series**, using daily (not 15-minute) returns.
- Report Nyholt (2004) as a cross-check; if the two differ by more than 30%, use the **larger**.
- **`N_eff` must be reported in the pre-registration as a forecast and again after the fact as a measurement.** If the realised `N_eff` exceeds 7, the DSR hurdle rises accordingly (§6.4) — the budget is enforced by a harsher threshold, not by pretending the extra trials never happened.
- Count *every* configuration ever run against the development window, including abandoned ones, failed pipelines re-run after a bug fix, and changes to the labelling or cost model. This is the single most commonly violated rule in the literature and the one Bailey et al. identify as making most published backtests unassessable.

### 6.4 The numeric thresholds

**(a) Deflated Sharpe Ratio.** `SR₀ = sqrt(V[SR_trials]) · bracket(N_eff)`. Assuming a cross-trial annualised Sharpe dispersion of `sqrt(V[SR]) = 0.50` **[A9]** (must be measured from the actual trial set):

| `N_eff` | `bracket(N_eff)` | `SR₀` (annualised) | Annualised net SR needed for **DSR ≥ 0.95** |
|---|---|---|---|
| 5 | 1.193 | 0.60 | ~1.80 |
| **7 (the budget)** | **1.387** | **0.69** | **~1.90** |
| 10 | 1.574 | 0.79 | ~2.00 |
| 20 | 1.899 | 0.95 | **~2.16** |
| 40 | 2.189 | 1.09 | ~2.29 |
| 60 (the plan as stated) | 2.344 | 1.17 | ~2.37 |

(Computed with `T = 730` daily observations, `γ₃ = -0.5`, `γ₄ = 6` **[A8]**; recomputing on `T = 365` independent 48-hour blocks gives ~1.91 at `N_eff = 7`, so the figure is robust to that choice.)

**Required: DSR ≥ 0.95.** Note the coincidence worth internalising: the DSR hurdle at the trial budget (**~1.9**) and the study's own 80%-power detection limit (**~1.98**) are the same number. The protocol is self-consistent — a strategy this study can *detect* is roughly a strategy this study can *certify*, and nothing weaker will do either.

**(b) Probability of Backtest Overfitting**, via CSCV with S = 12 (924 partitions) over the 11 CPCV paths:

| PBO | Verdict |
|---|---|
| ≤ 0.25 | Pass |
| 0.25 – 0.50 | Inconclusive |
| > 0.50 | **Falsified** — the in-sample-optimal configuration is more likely than not below the out-of-sample median, i.e. the selection procedure is worse than random |

**(c) t-statistic hurdles** (Šidák, two-sided, FWER = 0.05), computed on the portfolio equity curve with a Ledoit–Wolf studentised bootstrap standard error and **`n_eff`, never `n_raw`**:

| `N_eff` | Šidák per-test α | Required &#124;t&#124; |
|---|---|---|
| 5 | 0.0102 | **2.57** |
| **7 (the budget)** | 0.0073 | **2.71** |
| 10 | 0.0051 | **2.80** |
| 20 | 0.0026 | **3.02** |
| 40 | 0.0013 | 3.22 |
| 60 | 0.00085 | 3.33 |

Harvey, Liu & Zhu's recommended **t > 3.0** corresponds to `N_eff ≈ 20` — a useful reality check on how modest a "20 effective trials" budget really is. **Prescription: demand |t| ≥ 2.71 at `N_eff` ≤ 7, and |t| ≥ 3.0 whenever `N_eff` > 15.** Never accept t ≈ 2.0.

**(d) Haircut Sharpe.** Run Harvey & Liu's (2015) `Haircut_SR.m` with the declared `N_eff` and the measured cross-trial correlation. Report the raw SR, the haircut SR under all three adjustments (Bonferroni, Holm, BHY), and the implied `Profit_Hurdle`. **The economic decision must be made on the haircut SR, not the raw SR.** A flat 50% haircut is not acceptable — the correction is non-linear.

**(e) Reality Check / SPA.** Before StepM, run Hansen's SPA (`arch.bootstrap.SPA`, `studentize=True`, `bootstrap="stationary"`, `block_size=10`, `reps=5000`) on all 20 Family-A curves against a **zero-net-return benchmark and a passive equal-weight-altcoin benchmark**. If SPA's consistent p-value exceeds 0.05, **stop**: no individual claim may be made, and StepM must not be reported. Prefer SPA over White's Reality Check precisely because the candidate set will contain many near-duplicate and several weak variants, which is the situation where unstudentised RC loses power.

### 6.5 Pre-registration rule

**A hypothesis that was not written down before the test was run is not a hypothesis. It is a description of the data.**

Before any Family-A test touches the development window, commit a file `docs/research/prereg/<YYYY-MM-DD>_<family>.md` to git and record its SHA. It must contain, and after the fact may not be edited:

1. The **exact** entry rule, exit rule, barrier levels, horizon, and universe filter — specified precisely enough that an independent implementer would produce identical trades.
2. The **complete** list of parameter values to be tried, with `M_raw` and a forecast of `N_eff`. Values not on this list may not be tried later against D.
3. The cost model, `K_max`, sizing rule, and queue-priority rule.
4. The primary metric, its confidence-interval method, and the block length.
5. The **numeric** pass / fail / inconclusive thresholds from §8, filled in with actual numbers, not formulas.
6. The direction of the predicted effect and an economic reason to expect it (Arnott–Harvey–Markowitz: a strategy with no ex-ante rationale is a data-mining result even if it passes).
7. The names of any Family-B leads being promoted, and the trial budget they consume.

After the run, publish a results file that references the pre-registration SHA and reports **every** pre-registered variant, including the losers. Reporting only the winner reintroduces exactly the selection bias the whole protocol exists to remove.

---

## 7. The metric suite

**Universal rules.** Every metric is reported with (i) the point estimate, (ii) a 95% interval from the stationary panel bootstrap of §3.4, (iii) `n_raw`, and (iv) `n_eff` / `Σū`. A metric reported without `n_eff` is not reportable. Portfolio-level metrics are computed on daily returns; trade-level metrics are descriptive only and may never be used in a pass/fail gate.

### 7.1 Definitions and computation

| Metric | Definition | CI method | Small-sample failure mode |
|---|---|---|---|
| **Expectancy** | Mean **net** P&L per trade in bps: `E = mean(r_i^gross) - c_i`, with `c_i` the per-trade round-trip cost from the historical spread proxy, not a constant | Stationary panel bootstrap, percentile-t, `b = 10 d`, `B = 10,000` | Dominated by a handful of fat-tail outcomes. The mean is a high-variance estimator on heavy-tailed data. **Report the trimmed mean (5%) and median alongside**; if the sign flips between mean and 5%-trimmed mean, the result rests on ≤ 5% of trades and is not a finding. |
| **Profit factor** | `Σ(gross wins) / |Σ(gross losses)|`, computed **net of costs** | Bootstrap percentile only. No closed form exists — it is a ratio of sums of heavy-tailed variables. | Severely right-skewed and **upward-biased** at small n; undefined when there are no losses; a single large win can double it. **Do not quote unless `n_eff` ≥ 100 losing trades.** Not comparable across strategies with different trade counts. |
| **Win rate** | `wins / (wins + losses)`, excluding time-stop exits which are reported as a third category | **Wilson score interval on `n_eff`, not `n_raw`.** Clopper–Pearson if a conservative bound is wanted. | Meaningless without the payoff ratio — trivially raised to 90% by moving the target closer. Using `n_raw` here is the single most common error in this study's setting and would understate the interval by ~6.5x. |
| **Average win / average loss** | Conditional means over the winning and losing subsamples | Bootstrap; report as a *pair* with win rate | These are conditional means over a **self-selected** subsample, so they move mechanically and inversely with the win rate. Reporting either in isolation is uninterpretable. Always present `(win rate, avg win, avg loss)` as one triple, and prefer expectancy, which combines them without the artefact. |
| **Trade frequency** | Zones/day and fills/day, per month and per volatility tercile | Poisson or bootstrap interval on the daily count | Strongly regime-dependent: zone rate spikes with volatility, so a whole-sample average hides that most trades came from two stressed weeks. **Always disaggregate by month and by volatility tercile.** |
| **MFE / MAE** | Maximum favourable / adverse excursion over the outcome window, reported **both** in bps and normalised by contemporaneous ATR; full deciles, not just the mean | Bootstrap quantile intervals | **Censored by the exit rule.** Once a barrier fires the path stops, so MFE above the target and MAE beyond the stop are structurally unobservable and the distribution is truncated. Measure MFE/MAE on the **un-barriered** forward path of fixed length `H_max`, then separately report the barriered version. Failing to normalise by volatility makes "high-vol zones have larger MFE" a tautology. |
| **Maximum drawdown** | Largest peak-to-trough decline of the daily portfolio equity curve | **Report the bootstrap distribution — median, 5th and 95th percentile — never the single realised number.** Also report time-under-water. | Three independent pathologies. (1) It is a **single order statistic of one path**, so its sampling variance is enormous. (2) It grows monotonically with sample length, so it cannot be compared across windows of different length — normalise, or compare only equal-length windows. (3) Bailey & López de Prado (2015): ignoring serial correlation **understates downside by up to 70%**, so any IID-based drawdown estimate is not merely noisy but biased low. |
| **Sharpe** | `mean(r_daily)/sd(r_daily) × sqrt(365)`, computed on the **net** portfolio equity curve | Lo (2002) SE with Newey–West HAC (lag = `⌊4(T/100)^{2/9}⌋`) **and** Ledoit–Wolf (2008) studentised circular-block-bootstrap interval. Report both; if they disagree, use the wider. | Non-normality, autocorrelation (sqrt-scaling invalid — Lo reports up to **65%** overstatement), and above all **selection bias**. A naive Sharpe from a search is not an estimate of anything. **Always report DSR next to it, in the same table cell if possible.** |
| **Sortino** | `mean(r) / sqrt(mean(min(r,0)²)) × sqrt(365)`, target 0 | Same bootstrap as Sharpe | Uses only the downside half of the sample, so the **effective n is roughly halved** and the interval is ~sqrt(2) wider than Sharpe's. Unstable and occasionally explosive when there are few losing days. |
| **Calmar** | Annualised return / |MaxDD| | Bootstrap percentile only | Inherits **every** MaxDD pathology, then compounds them by dividing one noisy quantity by another. **Report for continuity with industry convention, but never gate on it.** |
| **Turnover** | `Σ|notional traded| / average equity`, annualised; plus average holding period | Bootstrap | Not a performance metric but a **cost-sensitivity multiplier**: it is the derivative of net performance with respect to the cost assumption. A high-turnover strategy whose edge is 35 bps/trade against a 26–52 bps cost range is an untested strategy. Always report net performance at all three cost levels. |

### 7.2 Metrics that are gates versus metrics that are description

| Gate-eligible (§8) | Description only |
|---|---|
| Net expectancy with bootstrap lower bound | Profit factor |
| Portfolio Sharpe with DSR | Win rate in isolation |
| PBO | Average win / average loss |
| Fold-level sign consistency | Calmar |
| Bootstrap 95th-percentile MaxDD | Realised MaxDD |
| Barrier-hit rate against `p*(c)`, on `n_eff` | Any trade-level statistic |

---

## 8. Explicit pass / fail decision rules

Apply in order. Numeric, mechanical, no judgement calls. Every threshold must be copied into the pre-registration **before** the run.

### Gate 0 — Pre-registration
- [ ] A pre-registration file exists, is committed, and its SHA predates the first read of D for this family.
- [ ] It contains all seven items of §6.5, with numeric thresholds filled in.

**Fail ⇒ the entire result is EXPLORATORY.** It may generate leads; it may not be described as evidence.

### Gate 1 — Leakage (§9)
- [ ] All 12 leakage assertions pass.
- [ ] Same-bar barrier ambiguity ≤ **10%** of resolved outcomes.
- [ ] The `+1`-bar-shift ablation destroys the edge (net expectancy falls to within the bootstrap CI of zero).

**Fail ⇒ VOID.** Not "falsified" — void. Fix the pipeline and re-run; the re-run costs one trial.

### Gate 2 — Economic viability
- [ ] Net expectancy > 0 at the central cost of **30 bps [A7]**, with the bootstrap **95% lower bound > 0**.
- [ ] Net expectancy > 0 at the pessimistic cost of **52 bps** (point estimate; the lower bound may straddle zero).
- [ ] Barrier-hit rate `p̂`'s bootstrap 95% lower bound exceeds `p*(30 bps)` for the configured barrier ratio (65.0% symmetric ±1%; 53.3% at +1%/−0.5%).

**Fail with the 95% *upper* bound ≤ `p*` ⇒ FALSIFIED.** Fail otherwise ⇒ INCONCLUSIVE.

### Gate 3 — Walk-forward consistency
- [ ] Net expectancy positive in **≥ 5 of 6** folds.
- [ ] Stitched 360-day OOS net Sharpe > 0.
- [ ] **No single fold contributes > 50%** of total OOS P&L.
- [ ] **No single symbol contributes > 25%** of total OOS P&L, and no single calendar month > 35%.
- [ ] Rolling and anchored variants agree in sign.
- [ ] Conclusion unchanged across `K_max ∈ {2, 5, 10, 20}` and `b ∈ {4, 10, 20, 40}`.

**≤ 2 of 6 folds positive ⇒ FALSIFIED.** 3–4 of 6 ⇒ INCONCLUSIVE.

### Gate 4 — Multiple testing
- [ ] Hansen SPA consistent p-value ≤ **0.05** across the whole Family-A candidate set.
- [ ] Romano–Wolf StepM adjusted p-value ≤ **0.05** for this specific configuration.
- [ ] &#124;t&#124; on the OOS portfolio curve ≥ **2.71** (`N_eff` ≤ 7) or ≥ **3.00** (`N_eff` > 15), computed with a Ledoit–Wolf studentised bootstrap SE **on `n_eff`**.
- [ ] Realised `N_eff` (Li–Ji) ≤ **7** for Family A, and every configuration ever run against D is counted.
- [ ] Haircut Sharpe (Harvey–Liu, all three adjustments) remains above the economically required level.

**Fail ⇒ INCONCLUSIVE** (the strategy is not shown to be absent, only not shown to be present).

### Gate 5 — Overfitting
- [ ] **DSR ≥ 0.95** at the declared `N_eff`.
- [ ] **PBO ≤ 0.25** (CSCV, S = 12, 924 partitions).
- [ ] CPCV median OOS Sharpe ≥ **0.50 x** IS Sharpe (degradation ≤ 50%).
- [ ] **≥ 8 of the 11** CPCV paths have positive net expectancy.
- [ ] Observed track length ≥ **MinTRL** for the claimed Sharpe.

**PBO > 0.50, or CPCV median OOS Sharpe ≤ 0 ⇒ FALSIFIED.** Otherwise fail ⇒ INCONCLUSIVE.

### Gate 6 — Holdout (one shot, frozen parameters)
- [ ] Net expectancy > 0 on H with the pre-registered configuration, zero re-fitting.
- [ ] Annualised net Sharpe on H ≥ **1.0**.
- [ ] H's point estimate lies inside the development-period bootstrap 95% interval (a *higher* holdout result is as suspicious as a lower one and triggers a leakage re-audit).
- [ ] Sign agreement with D on expectancy, hit rate, and Sharpe.

**Net expectancy < 0 on H with the 95% interval excluding D's point estimate ⇒ FALSIFIED.**

### Gate 7 — Sample adequacy
- [ ] `n_eff` ≥ **300** in the evaluated window.
- [ ] ≥ **100 losing trades** before any profit factor or average-loss figure is quoted.
- [ ] Bootstrap **95th-percentile** MaxDD within the stated risk budget.
- [ ] `Σū` and `n_eff` reported in every table.

**Fail ⇒ INCONCLUSIVE regardless of every other gate.** An underpowered pass is not a pass.

### 8.1 Classification

| Verdict | Condition |
|---|---|
| **EVIDENCE OF AN EDGE** | Gates 0–7 **all** pass. |
| **FALSIFIED** | **Any** of: Gate 2 upper bound ≤ `p*`; ≤ 2 of 6 folds positive; PBO > 0.50; CPCV median OOS Sharpe ≤ 0; Gate 6 negative with the CI excluding D. |
| **INCONCLUSIVE** | Gates 0–3 pass but Gate 4, 5, 6 or 7 fails short of a falsification trigger. **This is the expected outcome for any true edge below ~55% / Sharpe ~2 (§5.5), and must be reported as such rather than escalated to a positive.** |
| **VOID** | Gate 1 fails. |
| **EXPLORATORY** | Gate 0 fails. |

### 8.2 The pre-emptive ruling on the current symmetric ±1% design

Applying Gate 2 to what is already known: the barrier-hit rate is ~47–51%, and on `n_eff = 720` the 95% interval is `0.49 ± 1.96 × 0.5/sqrt(720) = [0.454, 0.526]`. The upper bound of **0.526** is below `p*` at every cost level in the range (**0.630–0.760**).

**Verdict: FALSIFIED, on existing evidence, before any new infrastructure is built.** A symmetric ±1% barrier on mid-cap altcoins at these cost levels cannot be profitable at any hit rate this system plausibly produces. The correct next action is not more validation of this design but a change to the design: asymmetric barriers, a shorter horizon, BTC-residualised outcomes, and maker-side entry at the zone. Re-enter the protocol at Gate 0 with that new configuration.

---

## 9. Leakage checklist

Each item is an executable assertion. A failure aborts the run and voids the result (Gate 1). "Ablation" means: deliberately break the thing, confirm the edge disappears; if it does not, the edge was never coming from where you thought.

### 9.1 Point-in-time universe and survivorship

1. **Symbol listing calendar.** Build `(symbol, listed_at, delisted_at)` from exchange history, not from today's `exchangeInfo`. Assert every event's symbol satisfies `listed_at + 30 d ≤ t0 < delisted_at`. Report the count of events excluded — if using today's universe changed expectancy by more than **20%**, survivorship bias was material.
2. **Delisted and dead pairs must be present.** Assert the universe contains at least one symbol that was delisted or went to near-zero volume during the window. A universe with zero deaths over 2 crypto years is prima facie evidence of survivorship filtering. Pairs that collapsed are exactly where an order-block "retest" strategy loses catastrophically.
3. **Liquidity ranking must be trailing.** If the ~80 symbols were chosen by quote volume, assert the ranking used a **trailing** 30-day window as of `t0`, not full-sample volume. Full-sample liquidity selection is a look-ahead: it preferentially selects coins that *later* became popular, which is a momentum signal in disguise.
4. **Universe churn.** Report the number of symbols entering and leaving per month. A static 80-symbol list over 2 years is almost certainly not point-in-time.

### 9.2 Feature computation crossing the decision bar

5. **No future bars, asserted structurally.** For every feature, assert its computation touches only bars with `close_time ≤ t_decision`. Enforce with a truncated-frame test: for 20 randomly sampled events, recompute all features from a bar series hard-truncated at `t_decision` and assert **exact** equality with the pipeline's values.
6. **Rolling-window hygiene.** Grep-level assertions: no `center=True`; no `bfill`/`fillna(method='bfill')`; no `shift(-k)`; every rolling/EWM call has explicit `min_periods`; no `dropna()` applied after a forward-looking operation.
7. **No global fitting.** Assert every scaler, quantile binning, z-score, PCA, target encoding and feature-selection step is fitted **inside** the training fold only. Full-sample normalisation is the most common silent leak in ML pipelines and leaks the test-period distribution into training.
8. **Timestamp convention.** Assert whether bar timestamps are open-time or close-time, and that the convention is consistent across the detector, the feature builder, and the labeller. **A one-bar offset is the single most frequent cause of a spurious edge in 15-minute crypto research.** Ablation: shift features `+1` bar (later) — the edge must weaken to insignificance; shift `−1` bar (earlier, i.e. deliberately leaking) — the edge must *improve markedly*. If the `−1` result looks like the original, the original is leaking.
9. **Detector replay integrity.** The trailing swing high/low must be computed from a strictly backward window. Assert the replay never reads bars beyond `t_decision`, and hand-reconstruct 20 sampled zones from a truncated series to confirm the detector fires identically.

### 9.3 Same-bar barrier-order ambiguity

10. **Count it, resolve it pessimistically, bound it.** On a 15-minute bar where `low ≤ entry×0.99` **and** `high ≥ entry×1.01`, OHLC cannot say which barrier was touched first.
    - Assert the labeller resolves such bars as a **loss** (pessimistic).
    - Report the **fraction of outcomes resolved on an ambiguous bar** and the fraction resolved on the entry bar itself.
    - **If ambiguity > 10% of outcomes, the label is not computable at 15-minute resolution** and must be resolved on 1-minute (or trade-tick) data. This is a hard stop, not a caveat.
    - Also report the optimistic resolution (ambiguous ⇒ win) as an upper bound. **The conclusion must hold under the pessimistic resolution**; if it holds only under the optimistic one, the result is INCONCLUSIVE.

### 9.4 Outcome windows overlapping training data

11. **Purge and embargo, asserted per fold.** Run the `assert_no_leak` check of §2.3 for every fold of every walk-forward and every one of the 66 CPCV splits. Additionally assert `Σū` is recomputed *within* each training fold (uniqueness is fold-relative, not global) and that sample weights are passed to the estimator — a silent `sample_weight=None` reintroduces the full overlap bias.
12. **Holdout audit log.** Assert the H access log contains exactly one entry for this strategy family, and that its recorded config hash matches the pre-registration.

### 9.5 Cost and data-quality leaks

13. **Costs must be historical, not current.** Assert the spread component comes from a per-symbol, per-period OHLC proxy (Corwin–Schultz / Abdi–Ranaldo, per `evidence_context_costs.md`), not a constant and not today's spread. Applying 2026 liquidity to 2024 microcaps is a look-ahead that flatters the result precisely where it should not.
14. **Gaps and halts.** Assert no event's `[t0, t1]` window contains a bar gap of more than 3 consecutive missing bars. Events spanning a trading halt or a data outage must be dropped and counted, not interpolated.
15. **No outcome information in features.** Assert no feature is a function of any bar in `(t0, t1]`. In particular, the zone-defining volume spike is at `t0` and is legitimate; any "confirmation" feature computed after entry is not.
16. **Fill realism.** If entries assume a maker fill at the zone, assert the bar actually traded through the limit price with enough volume, and report the assumed fill rate. An unfilled-limit assumption of 100% is itself a look-ahead.

### 9.6 The two ablations that catch most of what the assertions miss

- **Label shuffle.** Randomly permute the outcome labels within each fold and re-run the entire pipeline end to end. Expected result: AUC 0.50, expectancy indistinguishable from `−c`. **Any residual edge on shuffled labels proves a structural leak** and voids the run.
- **Random-signal placebo.** Replace the decision layer with a fair coin at the same trade frequency, on the same zones, with the same costs. This is the correct benchmark for Gate 2, and its bootstrap distribution is the null the real strategy must beat. Run it with the same 10,000 bootstrap index sequences so the comparison is paired.

---

## 10. Reporting template

Every result — pass, fail, or inconclusive — is published as one file with these sections, in this order. No exceptions, and losers are published alongside winners.

1. Pre-registration SHA, run date, git commit SHA of the code.
2. `M_raw`, forecast `N_eff`, realised `N_eff` (Li–Ji and Nyholt), and a cumulative running total of trials spent against D.
3. Event counts: `n_raw`, `Σū`, `n_eff`, average concurrency, measured `ρ̄`.
4. Gates 0–7 as a literal checklist with pass/fail marks and the realised numbers.
5. Metric table (§7) with bootstrap intervals at `b ∈ {4, 10, 20, 40}`.
6. Per-fold and per-CPCV-path results; per-symbol and per-month P&L attribution.
7. DSR, PBO, MinTRL, haircut Sharpe, SPA p-value, StepM adjusted p-value.
8. Leakage checklist with the ablation outcomes.
9. Cost sensitivity at 26 / 30 / 52 bps.
10. Verdict per §8.1, stated in the exact prescribed words.

---

## 11. Assumption register

Nothing below is a measurement. Each must be replaced with a measured value before the protocol produces a defensible verdict, and each replacement must be logged.

| Tag | Assumed value | How to replace it with a measurement |
|---|---|---|
| **[A1]** | History spans 2024-09-01 → 2026-08-31 (730 days) | Read `min(open_time)` and `max(open_time)` from the backfilled klines per symbol; use the date by which ≥ 90% of the universe has continuous coverage. |
| **[A2]** | Average pairwise correlation of signed zone outcomes across symbols, `ρ̄ = 0.50` | Build the matrix of per-symbol signed barrier outcomes bucketed into 48 h windows; take the mean off-diagonal Pearson correlation. Compute separately for BTC-up, BTC-flat and BTC-down regimes and **use the highest**. Cross-check with Li–Ji `M_eff` on the same matrix and with the first eigenvalue's variance share. Grounded on Two Sigma's ~0.48 and Jahanshahloo et al.'s 0.49→0.78 regime shift; both suggest 0.50 is optimistic. |
| **[A3]** | Longest feature lookback `L_feat = 96` bars (24 h) | Take the max lookback across all features actually in the pipeline, including any EWM half-life times 5. Set the embargo to `H_max + L_feat`, rounded up. |
| **[A4]** | 30,000 zone events over the window | Count them after the point-in-time replay completes. Note (§4.4) that `n_eff` is nearly invariant to this number, so it does **not** need to be precise. |
| **[A5]** | Maximum outcome horizon `H_max = 48 h` | A design choice, not a measurement — **and the highest-leverage one available.** Measure the median and 90th-percentile time-to-resolution; if 90% of outcomes resolve within 8 h, set `H_max = 8 h` and gain ~6x in `n_eff`. |
| **[A6]** | Transfer coefficient `TC = 0.60` | Measure ex post as the cross-sectional correlation between the risk-adjusted zone score and the realised active weight, per Clarke–de Silva–Thorley. Practical range 0.3–0.8; `TC` below 0.5 means the position-construction layer is destroying more signal than the model layer is finding. |
| **[A7]** | Round-trip cost 30 bps central, 26 / 52 bps bounds | Already estimated in `docs/research/evidence_context_costs.md` (26–52 bps). Replace with a per-symbol, per-period OHLC spread proxy plus the realised Binance/MEXC fee tier and measured slippage from live fills. Binance spot base taker is 0.10% per side (0.075% with BNB); mid-cap USDT spreads run 3–8 bps. |
| **[A8]** | Return skew `γ₃ = -0.5`, kurtosis `γ₄ = 6` | Measure on the daily net portfolio returns of the candidate strategy. Barrier strategies with tight stops are typically negatively skewed and leptokurtic, which *raises* every MinTRL and *lowers* every DSR — so do not leave these at the Gaussian defaults of (0, 3). |
| **[A9]** | Cross-trial annualised Sharpe dispersion `sqrt(V[SR]) = 0.50` | Compute the sample standard deviation of the annualised Sharpe ratios across all Family-A trials actually run. This is directly observable once the trials exist and should never remain an assumption in a final report. |

### Cross-references

- `docs/research/evidence_context_costs.md` — cost model, OHLC spread proxies, BTC-state conditioning, MFE/ATR normalisation.
- `docs/research/evidence_signals.md` — candidate feature evidence feeding the Family-B screen.
