# Evidence Review: Contextual Filters and Execution Costs for HAWK

**Scope:** volatility regimes, BTC beta, calendar seasonality, liquidity/stop-hunt microstructure, and transaction costs.
**Deliberately out of scope:** signal-pattern evidence (order blocks, wick patterns) and backtest validation methodology — covered by other reviews.
**Compiled:** 2026-09-17. All exchange fee figures retrieved 2026-09-17 unless otherwise noted.

---

## Executive Summary

**The honest headline: the cost hurdle is large relative to the target move, and only two of the five contextual factors have evidence strong enough to justify hard-coding into a decision layer.**

1. **Cost hurdle.** For a MEXC USDT-perp round trip on a mid-cap altcoin at ~$40 notional, the realistic all-in cost is **~34 bps** taker/taker (**19 bps** for the most liquid names, **70 bps** for the illiquid tail). Binance spot at VIP 0 is **~40 bps**. The dominant surprise is that **MEXC charges API futures trades at 4 bps maker / 6 bps taker, explicitly overriding the 0–1 bp headline and promotional rates** — so fees alone are 12 bps per round trip, not the 4 bps the public schedule implies. A strategy targeting a 1% (100 bps) move therefore surrenders **about a third of its gross move to costs** and needs a **break-even win rate of 67%** with symmetric targets. At the observed ~50% directional accuracy, base-case expectancy is **−34 bps per trade**, or about −1.7% of posted margin per trade at 5× leverage. The fix must come from a better entry price and asymmetric payoffs, not from filtering alone: **maker entry at the zone saves ~12 bps** (dropping break-even to 61%) and is the mechanically natural way to express a retest, while capping the universe at `universe_rank` ≤ 45 avoids up to 36 bps.
2. **Volatility state is real; volatility *timing* is fragile.** Volatility clustering in crypto is one of the best-established facts in the literature (long-memory parameters significant and stable across 15–120-minute scales). But the flagship result that conditioning exposure on volatility improves risk-adjusted returns (Moreira & Muir 2017) **largely fails out-of-sample** in the most careful replication (Cederburg et al. 2020: certainty-equivalent return *lower* in 72 of 103 strategies). Use volatility to **size and to set stop distances**, which is well supported, not as a binary go/no-go directional filter.
3. **BTC beta is the strongest single contextual variable** for short-horizon altcoin returns and is regime-dependent, with correlations spiking toward 1 in selloffs. This supports keeping a BTC-state field, but the specific `btc_stable` boolean as currently defined is almost certainly a lossy encoding of a continuous, asymmetric relationship — and asymmetric in a way that matters (BTC-down contagion is much stronger than BTC-up co-movement).
4. **Calendar and funding-hour effects exist but are small and unstable.** Published crypto hour-of-day and day-of-week effects are frequently reversed or vanish in later samples. The one mechanically reliable intraday pattern is elevated volatility and order-flow activity around perpetual funding settlement (00:00 / 08:00 / 16:00 UTC), which is worth modelling as a **cost/risk** adjustment rather than a directional signal.
5. **Stop-hunt and liquidation-cascade mechanics are the best-documented reason an order-block retest could work at all** — stop-loss clustering at round numbers and prior extremes (Osler 2003), and forced-liquidation feedback loops in perps. Crucially, HAWK stores only OHLCV, but the literature provides validated OHLC-only proxies (Corwin–Schultz, Abdi–Ranaldo, Roll) so the liquidity dimension is *recoverable* from existing data.

**Priority ranking for the decision layer:** (1) build the cost model correctly and re-evaluate whether any edge survives; (2) BTC-state conditioning, replacing the boolean with continuous, signed features; (3) OHLC-derived spread/illiquidity proxies as both a cost input and a filter; (4) volatility for sizing; (5) calendar effects last, and only as cost adjustments.

---

## 1. Volatility Regimes and Volatility Filters

### 1.1 Volatility clustering is well established, including intraday crypto

Volatility clustering and long memory in crypto volatility are robustly documented and, importantly for HAWK, **stable across intraday sampling frequencies**.

- **"Long range dependence in the Bitcoin market: A study based on high-frequency data"** (Physica A, 2019; [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378437118313190)). Using local Whittle, exact local Whittle, and ARMA–FIAPARCH estimators, the long-memory parameter *d* is significant and "quite stable" for both unconditional and conditional volatility **across different time scales** (intraday intervals in the 15–120-minute range) and for realized volatility measures. Quarterly rolling-window analysis shows high persistence throughout the sample.
- **Phillip, Chan & Peiris / Bariviera et al., "Some stylized facts of the Bitcoin market"** (Physica A 484, 2017). Establishes the standard stylized-fact set for BTC: heavy tails, volatility clustering, and volatility seasonality.
- **Khuntia & Pattanayak, "Adaptive long memory in volatility of intra-day bitcoin returns and the impact of trading volume"** (Finance Research Letters, 2018; [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612318305488)). Using MFDFA on hourly BTC, long memory in volatility is **time-varying, not an all-or-nothing property** — it persists across the whole sample but concentrates in five identifiable episodes, and trading volume has explanatory power for the degree of long memory. This is a direct caution against treating a volatility regime label as a stationary feature.
- **Bouri et al., "Modelling long memory volatility in the Bitcoin market"** (International Journal of Finance & Economics, 2019; [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.1670)) and **Mensi et al.** (Finance Research Letters 29, 2019, 222–230; [RePEc](https://ideas.repec.org/a/eee/finlet/v29y2019icp222-230.html)). Both emphasise that **structural breaks materially reduce measured persistence**: once breaks are modelled, apparent long memory falls and forecasts improve. Practical reading — persistence estimated over a long window will overstate how predictable the next regime is.

**What this licenses:** volatility at time *t* is genuinely forecastable at short horizons. **What it does not license:** that forecastable volatility translates into forecastable *returns* or Sharpe improvement.

### 1.2 Volatility-managed portfolios: the original claim and the replication failure

This is the single most important piece of skepticism in this section, because a "trade only when vol is low/calm" filter is exactly the Moreira–Muir trade in miniature.

- **Moreira & Muir, "Volatility-Managed Portfolios"** (*Journal of Finance* 72(4), 2017, 1611–1644; [DOI](https://doi.org/10.1111/jofi.12513); [author PDF](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf)). Scaling factor exposure by the inverse of previous-month realized variance produced, for the US market factor, an annualized **alpha of 4.9%**, an **appraisal ratio of 0.33**, and a **~25% increase in the buy-and-hold Sharpe ratio** (roughly 0.42 → 0.51). Effects were also found in value, momentum, profitability, ROE, investment, BAB, and FX carry. Sample: long US post-1926 factor history. The mechanism claimed is that changes in volatility are **not** offset by proportional changes in expected return.
- **Cederburg, O'Doherty, Wang & Yan, "On the performance of volatility-managed portfolios"** (*Journal of Financial Economics* 138(1), 2020, 95–117; [DOI](https://doi.org/10.1016/j.jfineco.2020.04.015); [PDF](https://www.lehigh.edu/~xuy219/research/COWY.pdf)). Across **103 equity strategies**, they reproduce the positive spanning-regression alphas but show those regressions are **not implementable in real time**. Reasonable out-of-sample versions **earn lower certainty-equivalent returns in 72 of 103 cases**, and lower Sharpe ratios; for the market combination, managed Sharpe 0.42 vs unmanaged 0.46. The stated cause is **structural instability in the underlying spanning regressions** — the in-sample alpha does not carry information about the out-of-sample benefit. Benefits that do survive are concentrated in **momentum, profitability, and betting-against-beta**, i.e. strategies with crash-prone, negatively skewed payoffs.

The surviving-cases pattern is the useful nuance: volatility management helps where it **truncates a fat left tail**, not where it is used to pick moments of higher expected return.

### 1.3 Crypto-specific volatility scaling

- **Grobys, Kolari, Sandretto, Shahzad & Äijö, "Cryptocurrency momentum has (not) its moments"** (*Financial Markets and Portfolio Management*, 2025; [Springer](https://link.springer.com/article/10.1007/s11408-025-00474-9); [open PDF](https://osuva.uwasa.fi/bitstream/handle/10024/20018/Osuva_Grobys_Kolari_Sandretto_Shahzad_%C3%84ij%C3%B6_2025.pdf?sequence=2)). Sample: weekly, top-30 coins by market cap (liquidity control), April 2016 – December 2023. Plain crypto momentum earned 0.71%/week with **kurtosis of 121.81** and a −255% crash episode. Volatility-managed versions earned **1.86%/week (8-week vol window, significant at 5%)** and **2.40%/week (4-week window, only 10% significance)**, with kurtosis falling to 68–106. The benefit is explicitly framed as **crash mitigation**, consistent with the equity findings.
- **Habeli, Barakchian & Motavasseli, "Adaptive Risk Allocation in Crypto Markets: Evaluating Volatility-Scaled Portfolios"** (SSRN working paper 5090097, Dec 2024; [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5090097)). Scales the crypto market portfolio and seven strategies by recent volatility. Sharpe improves and abnormal returns appear **"in certain strategies, particularly momentum-based"**, robust to leverage constraints, and **explicitly confirms that volatility scaling does not systematically benefit all strategies**.

Two caveats for HAWK: both are **weekly-rebalanced, cross-sectional portfolio** studies, and neither is a 15-minute single-zone entry decision. Neither reports the kind of turnover-adjusted net figures that would matter at HAWK's frequency. Note also the self-selection hazard: the well-cited crypto vol-scaling results are portfolio-level, and portfolio-level vol scaling is a *sizing* rule, which is exactly the use case the evidence supports.

### 1.4 Which realized-volatility / ATR estimator for 15-minute crypto bars

Theoretical efficiency relative to close-to-close (standard reference table, reproduced in e.g. [Portfolio Optimizer's survey](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/) and [Financial Econometrics using R §5.1](https://bookdown.org/jarneric/financial_econometrics/5.1-ohlc-measures.html)):

| Estimator | Inputs | Efficiency vs close-to-close | Drift-robust | Gap-robust |
|---|---|---|---|---|
| Close-to-close | C | 1.0× | No | n/a |
| Parkinson (1980, *J. Business* 53(1):61) | H, L | ~5.2× | No | No |
| Garman–Klass (1980, *J. Business* 53(1):67) | O,H,L,C | ~7.4× | No | Partial |
| Rogers–Satchell (1991) | O,H,L,C | ~6–8× | **Yes** | No |
| Yang–Zhang (2000) | O,H,L,C + prior C | ~14× | **Yes** | **Yes** |

Crypto-specific empirical evidence:

- **"Beyond GARCH in cryptocurrency volatility modelling: superiority of range-based estimators"** (*Applied Economics Letters*, 2024; [DOI](https://doi.org/10.1080/13504851.2024.2363295)). Five coins (BTC, ETH, BNB, XRP, DOGE), 1 Jul 2019 – 30 Sep 2022, with Binance 5-minute realized measures as the truth proxy. Six GARCH specifications × four distributions vs four range-based estimators on daily candles: **the Garman–Klass estimator clearly outperforms all GARCH-family models in every setting**, in-sample and out-of-sample; other range estimators remain competitive with GARCH.
- **Allen, "Realised volatility estimation shortcuts: An empirical analysis"** (MODSIM 2025; [DOI](https://doi.org/10.36334/modsim2025.d02.allen)). Hourly Binance BTC, 1 Jan 2018 – 29 May 2025, benchmarking Yang–Zhang against Barndorff-Nielsen & Shephard bipower variation: **slope 0.979, R² 0.956**. OHLC-based estimation captures ~95% of the jump-robust high-frequency measure.

The decisive structural point for HAWK: **Yang–Zhang's efficiency advantage comes almost entirely from its overnight-gap term, and Binance 15-minute bars have no gaps** — each bar's open essentially equals the prior close. On continuous 24/7 data, Yang–Zhang degrades toward Rogers–Satchell, and the extra machinery buys nothing. Conversely, range estimators **amplify bad highs and lows**, and 15-minute altcoin bars on the thinner end of an 80-symbol universe are exactly where wick prints are least trustworthy.

### Implication for HAWK

- Compute volatility as **Garman–Klass or Rogers–Satchell over 15-minute bars** (the crypto evidence favours Garman–Klass; Rogers–Satchell if a trending 15m window makes the zero-drift assumption uncomfortable). Do **not** spend effort on Yang–Zhang — its advantage is gap handling, and there are no gaps. Keep a plain close-to-close or ATR series as a cross-check, because it is the estimator least corrupted by a single bad wick print.
- Treat `btc_range_4h_pct` as what it is: a **crude Parkinson-style range proxy on a 4-hour window**. It is defensible as a volatility level indicator but it is a single-observation range, so its sampling error is large; prefer a multi-bar Garman–Klass estimate built from the stored 15m OHLCV.
- **Use volatility for sizing and stop placement, not as a directional gate.** The Cederburg et al. result is a direct warning that an in-sample finding of the form "zones detected in low-vol conditions performed better" is exactly the kind of spanning-regression alpha that does not survive real-time implementation. If a volatility filter is tested, it must be evaluated with rolling/expanding-window parameter selection, and the honest benchmark is the unfiltered strategy's certainty-equivalent return, not its alpha.
- The one volatility use with genuine cross-market support is **left-tail truncation**: the surviving Moreira–Muir-style benefits concentrate in crash-prone strategies. A leveraged 5x altcoin retest strategy is crash-prone, so scaling *position size* down when realized volatility is in its top decile is the best-supported volatility intervention available.
- Because realized volatility sets the achievable MFE, **normalise the MFE/MAE records by contemporaneous volatility (e.g. MFE/ATR) before drawing any conclusion about which zones are better**. Otherwise a "high-vol zones win" finding is a tautology, and the cost hurdle in bps is not comparable across volatility states.

---

## 2. BTC as a Market/Beta Factor for Altcoins

### 2.1 A crypto "market factor" prices the cross-section — but it is not literally BTC

- **Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency"** (*Journal of Finance* 77(2), 2022, 1133–1177; [NBER WP 25882](https://www.nber.org/papers/w25882)). Three factors — **cryptocurrency market (CMKT), size (CSMB), and momentum (CMOM)** — capture the cross-section of expected crypto returns. Nine long-short strategies generate significant excess returns, and all nine are explained by the three-factor model. Critically for HAWK's framing: a **one-factor model with only the coin market return cannot account for most of the nine strategies' excess returns**, and the market+size two-factor model explains five of nine but **no momentum-related strategy**.
- **Jiatao Liu, "Factor Structure in Cryptocurrency Returns and Volatility"** (SSRN 3389152, rev. 2021; [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3389152)). Tick data on the nine most liquid coins. A common factor structure exists in both returns and volatility, but **the common factor's explanatory power is much stronger for volatility than for returns**. Bitcoin as a proposed "crypto market factor" has **only weak explanatory power** for returns, and the BTC relationship is **regime-dependent**: model-free realized betas with Bitcoin rose during the 2017 bubble and the explained fraction of altcoin variance **stayed elevated after the bubble collapsed**.
- **MSCI, "Could Factors Have Explained Cryptocurrency Risk?"** ([MSCI research blog](https://www.msci.com/research-and-insights/blog-post/could-factors-have-explained-cryptocurrency-risk)). A seven-factor technical model over 3 Jul 2017 – 24 Mar 2021 on 100 coins achieved **average R² of 0.45**. The characteristics with a distinguishable premium were **size and low liquidity**; notably **low beta** ranked among the better information ratios, i.e. high BTC beta was not itself rewarded.

The honest synthesis: **BTC beta explains altcoin *variance* far better than it explains altcoin *expected return*.** This is precisely the right shape for HAWK's use case — a BTC-state field should be read as a risk/noise control, not as a directional edge.

### 2.2 Short-horizon BTC→altcoin transmission, and why liquidity rank matters

This is the most directly applicable body of evidence for a 15-minute system.

- **"Price Transmission from Bitcoin to Altcoins: High-Frequency Evidence and Implications for Trading Strategy"** (*Asia-Pacific Financial Markets*, 2026; [DOI](https://doi.org/10.1007/s10690-026-09589-z)). **One-minute** data across multiple market regimes. Findings: (a) Granger causality is **unidirectional from BTC to altcoins** — reverse causality is generally absent; (b) **small-cap / low-trade-count coins exhibit significantly delayed responses** to BTC moves, i.e. lower liquidity ⇒ slower reaction; (c) a lag strategy using BTC's preceding returns as a leading indicator, with ML-based decisions, outperformed buy-and-hold across regimes. Caveat: the reported outperformance is the authors' own simulation and the benchmark is buy-and-hold, which is a weak bar; treat the *causality and liquidity-gradient* findings as the durable content, not the strategy return.
- **"Regime-Specific interdependencies in cryptocurrency markets: A high-frequency GMM-VAR approach"** (*Data Science in Finance and Economics*, 2025; [DOI](https://doi.org/10.3934/dsfe.2025017)). **One-minute BTC/ETH/XMR data, January 2020 – April 2025**, with regimes identified by a Gaussian mixture model rather than an imposed Markov structure. The core result is directly on point for `btc_stable`: **in the calm regime, return dynamics are primarily self-driven with limited cross-asset response; in the volatile regime, interlinkages are stronger and more persistent, with BTC acting as the principal shock transmitter.** Impulse responses and FEVDs confirm asymmetric interdependence across regimes. Bai–Perron tests found no deterministic breaks, supporting the regime classification.
- **"Do Bitcoin Shocks Dominate Other Cryptocurrencies?"** (*Asia-Pacific Financial Markets* 32(4), 2025; [RePEc](https://ideas.repec.org/a/kap/apfinm/v32y2025i4d10.1007_s10690-024-09493-4.html)). Daily data June 2011 – June 2020, ARMA-GARCH/TGARCH/EGARCH plus DCC and **ADCC**. Strong return and volatility spillover from BTC to seven major coins, and **time-varying conditional correlation with significant asymmetric effects** — the ADCC asymmetry term is the formal statement that correlations respond differently to negative than to positive shocks.
- **Counter-evidence on BTC's primacy:** ["Cryptocurrency Influence Networks from High-Frequency Returns"](https://www.emergentmind.com/papers/2606.25466) (Kraken trade data aggregated to 1-minute VWAP, Jan 2020 – Mar 2025, 275 weekly Granger-causality networks, universe growing from ~30 to ~390 assets) finds **Ethereum consistently the most influential asset and Bitcoin's relative importance gradually declining**, with substantial temporal variability in the influence ranking. So a BTC-only market proxy is a defensible but decaying choice.

### 2.3 Correlation spikes and downside asymmetry

- **"Dependence in cryptocurrencies: A partial correlation connectedness approach"** (*Finance Research Letters*, 2026; [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612326007233)). Quantile-specific connectedness controlling for indirect linkages. Total crypto connectedness **substantially exceeds** that of traditional financial and commodity markets; **lower-tail connectedness consistently exceeds upper-tail connectedness**, i.e. co-movement is tighter in selloffs than in rallies; and total connectedness rises markedly around major disruptions.
- **"Mapping Systemic Tail Risk in Crypto Markets: DeFi, Stablecoins, and Infrastructure Tokens"** (*Journal of Risk and Financial Management* 18(6):329, 2025; [MDPI](https://www.mdpi.com/1911-8074/18/6/329)). Sep 2021 – Mar 2025, eight tokens, partial-correlation network with quantile-specific connectedness. **Strong asymmetries**; systemic risk intensifies during adverse tail events (Terra/Luna, USDC depeg, post-halving 2024), and **ETH, LINK, UNI are the key transmitters of losses on the downside**.
- The conceptual anchor is **Forbes & Rigobon's** definition (*Journal of Finance* 57(5), 2002) that contagion is a *significant increase in correlation after a shock* — worth remembering because it also carries the warning that correlation estimates are **biased upward by heteroskedasticity**, so a naive "correlation spiked" reading during a volatile window is partly a measurement artifact.
- **Asymmetric volatility / leverage effect:** TGARCH news-impact curves show **negative news raises crypto volatility substantially more than positive news** for BTC, ETH and broad indices, with the amplification explicitly attributed to leverage and forced liquidations (["Towards Examining the Volatility of Top Market-Cap Cryptocurrencies…"](https://www.mdpi.com/2227-9091/13/3/57), *Risks* 13(3):57, 2025).

### Implication for HAWK

- **`btc_stable` as a boolean is the wrong shape, and probably the wrong sign convention.** The GMM-VAR evidence says a calm BTC regime is where altcoins are **most self-driven** — which is when an idiosyncratic order-block signal has the *best* chance of expressing itself rather than being overwhelmed by market beta. That is a coherent rationale for the filter, but it is a rationale about **signal-to-noise**, not about direction. Test it as such: the prediction is not "returns are higher when `btc_stable`" but "the zone-level signal's information ratio is higher when `btc_stable`", i.e. an interaction effect, which requires far more data to detect than a main effect.
- **Replace the boolean with three continuous, signed features** derived from data HAWK already has: (a) `btc_range_4h_pct` as a continuous percentile rather than a threshold; (b) **signed** BTC trend over the same window (the literature's downside asymmetry means BTC −2% and BTC +2% are not interchangeable, and lumping them into one "unstable" bucket destroys the most reliable part of the signal); (c) rolling BTC beta/correlation for the symbol, which HAWK can compute cheaply. Thresholding a continuous variable into a boolean discards information and, worse, creates a free parameter that invites overfitting at the cutoff.
- **Long and short zones should not share one BTC filter.** Lower-tail connectedness exceeding upper-tail connectedness implies BTC weakness is more informative for altcoin downside than BTC strength is for altcoin upside. Expect short-side zones to be more BTC-conditional than long-side zones, and allow the decision layer's BTC coefficient to differ by side.
- **Interact BTC state with `universe_rank`.** The one-minute evidence that low-liquidity coins respond to BTC with a *delay* means that for high-rank (illiquid) symbols, recent BTC returns carry residual predictive information at HAWK's 15-minute resolution, while for rank-1–20 symbols that information is already impounded. Concretely: `symbol_return_1h` minus (beta × BTC return over the same window) — a beta-adjusted idiosyncratic return — is a better-specified feature than the raw `symbol_return_1h`, and the delay structure should be allowed to vary with rank.
- **Do not use high BTC beta as a reason to prefer a symbol.** MSCI's cross-sectional work found low beta, not high beta, among the better-rewarded characteristics, and Liu (2021) found BTC explains altcoin variance far better than altcoin return. High beta buys you more variance per unit of signal — which, given a fixed cost hurdle, is not automatically good.
- **Correlation-spike caution for risk, not entry:** during a BTC selloff, an 80-symbol universe of simultaneously-triggered long zones is effectively **one position**. Portfolio-level exposure caps conditioned on BTC downside state are better supported by this literature than any per-zone directional rule.

---

## 3. Intraday and Calendar Seasonality in Crypto

This is the topic where the gap between "published effect" and "usable effect" is widest, and where the methodological literature is more valuable than the findings.

### 3.1 Return seasonality: found repeatedly, persists rarely

- **Baur, Cahill, Godfrey & Liu, "Bitcoin time-of-day, day-of-week and month-of-year effects in returns and trading volume"** (*Finance Research Letters* 31, 2019, 78–92; [RePEc](https://ideas.repec.org/a/eee/finlet/v31y2019icp78-92.html)). Over **15 million observations from seven continuously-traded BTC exchanges**, the authors find "time-specific anomalies in returns but **no persistent effects across time**." By contrast, they find **persistent differences in trading activity across all exchanges** — lower activity in local evening hours and on weekends. This is the single most important result in the section: **volume/liquidity seasonality is real and replicable; return seasonality is not.**
- **"A new perspective of the day-of-the-week effect on Bitcoin returns: evidence from an event study hourly approach"** (*Oeconomia Copernicana*, 2022; [DOI](https://doi.org/10.24136/oc.2022.022)). Kraken, 2016–2021. The authors do apply a genuine robustness protocol — re-estimating over one-, two-, and three-year **rolling windows** — and find that the **Friday 15:00 EST** effect survives, with 2-year-window significance percentages above 90% for cumulative returns 4–24h after 15:00. Mondays and Sundays do not hold up. They also cite Doyle & Chen (2009) on the day-of-week effect being "very sensitive to the choice of subperiods," which is the appropriate framing.
- **QuantPedia, "The Seasonality of Bitcoin"** ([link](https://quantpedia.com/the-seasonality-of-bitcoin/)). Gemini hourly, 9 Oct 2015 – extended through mid-2023, all UTC+0. Reports the largest returns concentrated in **21:00–23:00 UTC**, strongest on Fridays, and a two-hours-a-day strategy with ~37–41% annualized return and Calmar ~1.8–2.0. Treat with heavy discount: this is practitioner research with no multiple-testing correction across 24×7 = 168 hour-day cells, and the authors themselves note performance **weakened in 2022–23**. Searching 168 cells for the best one will produce a t-stat above 2 by construction.

### 3.2 The decisive skeptical result: UTC is the wrong clock

- **"Bitcoin on Wall Street Time: Natural Experiments on the Institutionalization of a 24/7 Market"** (*Journal of Risk and Financial Management* 19(9):707, 2026; [RePEc](https://ideas.repec.org/a/gam/jjrfmx/v19y2026i9p707-d2035752.html)). Ten years of hourly Kraken XBT/USD (2016–2025, **87,672 observations**). Findings, each with a clean identification strategy:
  - **When U.S. clocks change for daylight saving, the intraday volatility peak shifts by one hour in UTC**, tracking the displaced U.S. equity open. The shift appears **only in the institutionalized period** (post-2019).
  - On **weekday NYSE holidays**, the U.S.-hours share of realized variance **falls by 13.9 percentage points** versus matched weekdays, close to the uniform benchmark of 0.375. Also absent before 2019.
  - A circular index of intraday concentration **rises >40% over the decade**, with a structural break in **November 2021**.
  - **"Neither result is consistent with an explanation fixed in UTC."**
  - Methodological bombshell for HAWK: a **placebo simulation shows that naive whole-sample event contrasts on this trending series are significant for 100% of random pseudo-event dates.** Any hour-of-day finding computed on a long sample without detrending is uninterpretable.

  Practical consequence: a fixed UTC-hour dummy is a **mis-specified feature** that will silently drift by an hour twice a year and whose meaning has changed structurally since 2021.

### 3.3 Funding-settlement hours

- Mechanism and timing: on Binance and most major venues funding is exchanged **every 8 hours at 00:00, 08:00, 16:00 UTC**, and the rate approximately equals the average futures–spot spread over the preceding 8 hours (**"Fundamentals of Perpetual Futures"**, [arXiv:2212.06888](https://arxiv.org/html/2212.06888v6)). Confirmed in venue documentation, e.g. [Bitnomial funding-rate docs](https://bitnomial.com/exchange/docs/api/rest/funding-rates/): intervals fixed at 00:00–08:00, 08:00–16:00, 16:00–00:00 UTC, with `0.0001` = 1 bp per interval and annualization = rate × 3 × 365.
- **Critical caveat:** funding periods are **not uniform across venues or symbols**. Many venues and many high-funding altcoin perps use **1-hour or 4-hour** intervals, and some compute funding continuously (see [Coin Metrics' distinction between `period` and `interval`](https://gitbook-docs.coinmetrics.io/market-data/market-data-overview/market-funding-rates)). Hard-coding 00/08/16 across an 80-symbol universe will mislabel a meaningful subset.
- **Directional evidence around settlement is essentially null.** An [event study of Binance USD-M funding settlement timestamps](https://github.com/Kaleb0719/funding-timestamp-event-study/blob/main/README.md) (BTCUSDT, ETHUSDT, SOLUSDT; 1-minute OHLCV; ±30-minute windows; 1/5/15/30-minute post-settlement horizons) expanded an initial 30-day sample to 90 and 180 days: post-settlement mean returns went from **+0.0234%/+0.0288% (30d, 267 events)** to **−0.0044%/+0.0028% (90d, 810 events)** to **−0.0054%/−0.0084% (180d, 1,620 events)** at the 1- and 5-minute horizons. The authors' own conclusion: means "too small and unstable to support an actionable conclusion," with extreme-funding buckets showing larger magnitudes but **no consistently favorable direction**. This is unrefereed practitioner work, but it is exactly the right test design and its magnitudes — **under 3 bps even in the favourable subsample** — are an order of magnitude below the cost hurdle in Section 5.
- What *is* documented is concentrated **volume and realized-volatility activity** in the minutes after settlement, and that Binance's USDT perpetual is a major volatility transmitter to spot venues (**"The Role of Binance in Bitcoin Volatility Transmission"**, [arXiv:2107.00298](https://ar5iv.labs.arxiv.org/html/2107.00298)).

### 3.4 One intraday effect that did replicate — and why it should worry HAWK specifically

- **"Turn-of-the-candle effect in bitcoin returns"** (*Finance Research Letters* / PMC, 2023; [PMC10015199](https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/)). Positive returns of **0.58 bps per minute** are disproportionately concentrated **at minutes 0, 15, 30, 45 of each hour** — the turns of 15-minute candles — while average returns in other minutes are negative. **t-statistics above 9 for all seven sample exchanges in 2021**, robust in quantile regressions and TGARCH-M with heavy tails, and **persisting out-of-sample (Bitfinex, January–August 2022)**. The effect appears only from mid-to-late 2020 and is attributed to algorithmic trading reacting to the arrival of 15-minute candle information. Net-outperforming from ~$5,000 capital after fees and spreads.

This matters for HAWK beyond seasonality: it is direct evidence that **a large population of algorithms acts on 15-minute candle boundaries**, which is both why a 15-minute order-block detector might have something to react to, and why any edge at those exact timestamps is heavily contested. It also implies a **timestamp-alignment hazard**: entries mechanically placed at 15-minute boundaries sit in minutes with a documented, distinct return and liquidity profile.

### Implication for HAWK

- **Do not add a UTC hour-of-day directional dummy.** The Wall Street Time evidence shows the relevant clock is the U.S. equity calendar, not UTC, and that the mapping shifts with daylight saving and has broken structurally since late 2021. If an hour feature is used at all, it should be **hours relative to the U.S. equity open/close** (DST-aware), not raw UTC hour — and even then, the placebo result (100% of random pseudo-events significant on the untreated trending series) means any such test must be run on detrended data with rolling windows.
- **Do use time-of-day as a liquidity/cost input, which is the replicable part.** Volume is persistently lower in local evening hours and on weekends across every exchange studied. That means the **spread and slippage terms in the cost model of Section 5 should be time-of-day and weekend dependent** — plausibly 1.3–1.8× the median spread in the thinnest hours for high-`universe_rank` symbols. This is where UTC hour earns its place in HAWK: as a multiplier on cost, not as a term in expected return.
- **Day-of-week: treat as untested, allocate no parameters.** With ~80 symbols on 15m bars HAWK will accumulate zones quickly, but day-of-week × side × BTC-state cells fragment the sample fast, and the published effects (Friday afternoon) did not replicate across venues or into later samples.
- **Funding hours: handle as a cost/exit rule, not a signal.** Because HAWK holds up to 24h and measures MFE/MAE at 1/4/8/24h, a 24h hold crosses **~3 funding settlements** — that is a cost certainty (quantified in Section 5), whereas the directional effect is a documented near-zero. Record, per zone, the number of funding settlements crossed and the realized funding paid; that turns funding from an unmodelled leak into a measured line item. Before doing so, **verify the actual funding interval per symbol from the venue API** rather than assuming 8 hours.
- **Check for a 15-minute boundary artifact in HAWK's own records.** Given the turn-of-the-candle finding, compare fills/MFE for entries executed in the first minute of a 15-minute candle versus later minutes. If HAWK's retest detection systematically triggers on candle-close evaluation, part of any measured edge or cost may be this known microstructure effect rather than the order-block pattern.

---

## 4. Liquidity, Stop Hunts, and Liquidation Cascades

### 4.1 The Osler evidence — and why it cuts against the naive "sweep and reverse" story

This is the closest thing to a rigorous mechanism for why a level-sweep could reverse, and it is more specific than usually claimed.

- **Osler, "Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis"** (*Journal of Finance* 58(5), 2003, 1791–1819; [DOI](https://doi.org/10.1111/1540-6261.00588)). First data on individual currency stop-loss and take-profit orders. Two distinct clustering facts with **opposite** implications:
  - **Take-profit orders cluster particularly strongly at round numbers** → explains why trends **reverse** at support/resistance. These generate **negative-feedback** trading.
  - **Stop-loss orders cluster strongly just *beyond* round numbers** → explains why trends are **unusually rapid after** rates cross such levels. These generate **positive-feedback** trading.
  Quantitatively, the order-book data show heavy round-number concentration: e.g. a large share of orders request execution at rates ending in [01,10] versus only 6.9% in [90,99].
- **Osler, "Stop-Loss Orders and Price Cascades in Currency Markets"** (*Journal of International Money and Finance* 24(2), 2005, 219–241; [NY Fed Staff Report 150 PDF](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr150.pdf)). Three results: (1) trends are unusually rapid when rates reach documented stop-loss clusters; (2) **the response to stop-loss orders is larger than the response to take-profit orders**; (3) the stop-loss response **lasts longer**. Importantly: "Most results are statistically significant for **hours, although not for days**." Cascades plausibly explain fat tails in FX returns, and prices respond to **non-informative order flow**.

**The uncomfortable implication for HAWK.** The mechanism that is *stronger and longer-lasting* in Osler's data is the **continuation** mechanism (stop-loss cascades), not the reversal mechanism (take-profit provision). HAWK's detection rule requires the candle to sit **≥0.5% away from the trailing swing high/low** — that is, it explicitly selects candles that have **not** just broken a level. That specification pushes the setup toward the take-profit/liquidity-provision side (reversal) rather than the stop-cascade side (continuation), which is arguably the right choice given the strategy intent, but it means the stronger documented effect is being avoided rather than exploited. It also means the direction of the expected edge is genuinely ambiguous a priori — consistent with the coin-flip live results — and that **the distance-to-swing variable should be treated as a primary conditioning feature, not a fixed 0.5% gate**, since it is exactly the variable that determines which of the two opposing mechanisms dominates. Note also that Osler's significance horizon is **hours**, which brackets HAWK's 1h/4h MFE windows and argues that the 24h horizon is beyond where this mechanism has documented support.

### 4.2 Liquidation cascades in crypto perps: real, but less "critical" than folklore suggests

- **Lehar & Parlour / BIS, "Systemic fragility in decentralised markets"** (BIS Working Paper 1062; [PDF](https://www.bis.org/publ/work1062.pdf); also [SSRN 4164833](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4164833)). Collateral liquidations on Compound and Aave with blockchain transaction data. Documents **both temporary and permanent price impact** of liquidations, **cross-exchange contagion** (impact appears first on the transacting venue, then on others including off-chain), and **negative feedback loops** — liquidations push collateral prices down, triggering more liquidations. **Liquidated collateral exhibits heavier tails than unliquidated collateral**, plus evidence of strategic liquidator behaviour.
- **"Same Shock, Same Assets, Different Microstructure: … the October 10, 2025 Cryptocurrency Cascade"** ([Research Square preprint](https://www.researchsquare.com/article/rs-9459584/v1.pdf)). The 10 Oct 2025 event: **$6.93bn liquidated within 40 minutes**, **$9.89bn on CEXs by 03:00 UTC**, affecting **over 1.6 million accounts** — the largest single-day deleveraging on record. Key microstructure findings: spreads widened dramatically, **order-book depth collapsed**, identical assets showed substantial and persistent price divergence across venues despite arbitrage links, and cross-venue dynamics were **regime-dependent** (threshold cointegration preferred over linear models). Also documents that **CEX reporting is systematically incomplete during peak activity** — a data-integrity warning for anyone using exchange-reported liquidation feeds.
- **Skeptical counterweight: "Measuring the engine of a liquidation cascade: subcritical branching inside a first-order transition"** ([arXiv:2608.03616](https://arxiv.org/html/2608.03616)). Seven crash events 2022–2025, with the October 2025 crash measured directly from a transparent venue's on-chain fill log. The **branching ratio ran deeply subcritical (λ̂ ≈ 0.1–0.2 throughout)** — i.e. the cascade was **not self-sustaining**; severity came from the exogenous shock, leverage positioning, and liquidity withdrawal. **88% of post-onset forced selling landed within 30 minutes** and **63% was absorbed off-book** by the venue's backstop. Across all seven events, mean inter-asset coupling jumped **1.6–4.4 baseline standard deviations** into a near-fully-ordered phase, **price impact spiked by a factor of 1.2–3.5 (regressed) or 3.2–9.1 (quoted)**, and **open interest cleared by 25–70%**. The self-reinforcing Galton–Watson cascade account was **rejected at simulated power ≥0.96**.

The practical reading: cascades are **fast (≈30 minutes), accompanied by a 3–9× blowout in quoted price impact, and a near-total collapse of cross-sectional diversification** — but they are shock-driven events, not endogenous critical phenomena you can anticipate from price history alone.

### 4.3 Order-flow imbalance: strong, but out of reach without order-book data

- **Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events"** (*Journal of Financial Econometrics* 12(1), 2013, 47–88; [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)). NYSE TAQ, 50 US stocks. Mid-price changes are linear in **order-flow imbalance** with **average R² ≈ 65%**, slope **inversely proportional to market depth**, robust to intraday seasonality and stable across stocks and timescales. Excluding price-changing events from OFI to rule out tautology, R² **declines to 35–60%**. Crucially, the relation between price changes and **trade volume** is "noisy and less robust" than the OFI relation.
- **This is contemporaneous explanatory power, not prediction.** Cross-asset work ([Quantitative Finance, 2023](https://doi.org/10.1080/14697688.2023.2236159)) finds lagged cross-asset OFI does improve short-horizon forecasts and produces economic profit, but predictability "decays quickly" and manifests only "up to several minutes." A crypto-specific replication pipeline on one-second BTCUSDT ([AlphaLab](https://github.laiyagushi.com/ilibili-AI/AlphaLab)) found OFI sign associated with subsequent direction and absolute OFI monotonically ordering movement probability (AUC ~0.66–0.69), but **direct one-second return regression gave validation R² = −0.06**, underperforming a zero-return baseline.
- **HAWK cannot compute OFI.** It stores OHLCV only. The Cont et al. finding that **volume-based impact is markedly noisier than OFI-based impact** is a direct warning about HAWK's volume-spike criterion: volume magnitude is the weak version of the order-flow signal, and a **volume spike carries no directional sign**. This is a structural reason to expect ~50/50 directional outcomes from a volume-and-wick detector, and an argument for capturing at least aggregate taker buy/sell volume (available in Binance klines as `taker_buy_base_volume`) which would give a crude signed flow measure at near-zero cost.

### 4.4 OHLC-only liquidity proxies: what is and is not recoverable

- **Brauneis, Mestel, Riordan & Theissen, "How to measure the liquidity of cryptocurrency markets?"** (*Journal of Banking & Finance* 124, 2021, 106041; [DOI](https://doi.org/10.1016/j.jbankfin.2020.106041)). The definitive crypto study. Findings:
  - **Corwin & Schultz (2012)** and **Abdi & Ranaldo (2017)** outperform other measures at describing **time-series variation** in liquidity, "irrespective of the observation frequency, trading venue, high-frequency liquidity benchmark, and cryptocurrency," and perform well in high- and low-volatility/volume periods. Daily time-series correlations with quoted spread, effective spread, and round-trip cost range **0.54–0.75**; correlation with **price impact is 0.86** for both.
  - **Kyle & Obizhaeva (2016)** and the **Amihud (2002)** illiquidity ratio outperform at estimating liquidity **levels**, and reliably identify liquidity differences **between venues**. CS and AR capture the *level* of percentage effective spread well at the highest data frequency but **poorly at lower frequencies**.
  - **Roll (1984) performs worst.**
  - Conclusion: "there is not yet a universally best measure but there are reasonably good low-frequency measures."
- **Corwin & Schultz (2012)**, *Journal of Finance* 67(2) ([Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2012.01729.x)): the estimator exploits that daily highs are almost always buy trades and lows almost always sell trades, so the high–low ratio contains both variance and spread; the spread is separated using 1-day vs 2-day high–low ratios.
- **Critical confound for HAWK: "Do Liquidity Proxies Based on Daily Prices and Quotes Really Measure Liquidity?"** (*Entropy* 22(7):783, 2020; [DOI](https://doi.org/10.3390/e22070783)). Measures based on high and low prices — **Corwin–Schultz, Abdi–Ranaldo, and the high–low range** — are shown to be **more closely related to volatility estimates than to any liquidity benchmark**, confirmed by correlation, partial determination coefficients, and mutual information, in both cross-section and portfolio time series. Quote-based measures (Chung–Zhang closing quoted spread) track liquidity better. The paper explicitly notes that highs and lows are the same inputs used by Garman–Klass volatility estimators.

**Why this confound is severe in HAWK's case specifically:** HAWK's detection criterion is *a long wick relative to body* — i.e. an unusually large high–low range relative to open–close. Corwin–Schultz is computed *from* the high–low ratio. Therefore **a HAWK-flagged candle will mechanically produce an elevated Corwin–Schultz spread estimate by construction**, regardless of true liquidity. If CS-estimated spread is computed on the detection candle and used either as a filter or as the cost input, it will be circular and will bias cost estimates upward in a signal-correlated way. The proxy must be computed on a **window strictly preceding the detection candle** (e.g. the trailing 24–96 bars, excluding the signal bar), and cross-checked against Amihud and `universe_quote_volume`.

### Implication for HAWK

- **The mechanism evidence supports conditioning on *which side of a level* the zone sits, and it is already partly encoded.** Make the distance-to-swing measure (currently a 0.5% binary gate) a continuous feature and test it for a sign flip: near-level zones should behave like take-profit/reversal setups, and zones associated with a *break* of the level should behave like stop-cascade/continuation setups. This is the single best-motivated candidate for resolving the long/short/skip decision, because it maps to two documented mechanisms with **opposite** signs — which is exactly what would produce a 50/50 aggregate if both populations are currently pooled.
- **Restrict MFE horizons for this mechanism to 1h and 4h.** Osler's cascade and reversal results were significant "for hours, although not for days." The 8h and 24h MFE/MAE columns should be treated as risk diagnostics, not as the target the decision layer optimises.
- **Compute Amihud illiquidity and Corwin–Schultz/Abdi–Ranaldo spread from trailing 15m OHLCV — with strict exclusion of the signal bar.** Per Brauneis et al., use **Amihud** (and `universe_quote_volume`) for the **level** of illiquidity that drives the cost model, and **Corwin–Schultz or Abdi–Ranaldo** for **time-variation** in spread conditions. Do not use Roll. Do not use CS/AR for absolute level at 15m-aggregated-to-daily frequencies, where the JBF paper shows they perform poorly on levels.
- **Treat any "high estimated spread predicts better outcomes" finding as presumptively a volatility finding**, per the *Entropy* result. Include a Garman–Klass volatility control in the same specification; if the spread proxy loses significance once volatility is controlled, it was measuring volatility.
- **Add signed flow at near-zero cost.** Binance klines already include taker buy volume; deriving taker-buy share gives a crude OFI analogue. Given that Cont et al. show signed flow carries the explanatory power and unsigned volume is "noisy and less robust," this is the highest-value-per-unit-effort data addition available to HAWK, and it directly addresses why a volume-spike detector is directionally uninformative.
- **Hard risk rule from the cascade literature.** Price impact blows out **3–9×** and depth collapses during cascades, while 88% of forced selling completes within 30 minutes and inter-asset coupling jumps 1.6–4.4σ. Any zone whose retest occurs during a detected market-wide dislocation (e.g. BTC 15m range beyond its 99th percentile) should be **skipped or sized down**, not because direction is unpredictable but because **the assumed cost model is invalid in those minutes** and because 80 correlated positions become one. This is also a backtest-integrity issue: fills assumed at normal spreads during the October 2025-type windows will be badly optimistic.

---

## 5. Transaction Costs and Realistic Execution

### 5.1 Binance spot fees (retrieved 2026-09-17)

Official schedule: [binance.com/en/fee/schedule](https://www.binance.com/en/fee/schedule). The page is JavaScript-rendered and could not be captured directly; the figures below were cross-verified against a [secondary summary explicitly citing Binance's fee pages as displayed 7 September 2026](https://www.mexc.co/learn/article/binance-fees-explained-what-a-1-000-trade-really-costs-on-spot-futures-and-withdrawal/1). **The team must re-verify against its own account's fee tier and actual trade history before finalising the backtest.**

| Item | Standard (VIP 0) | With BNB deduction (25% off spot) |
|---|---|---|
| Spot maker | 0.100% (10.0 bps) | 0.075% (7.5 bps) |
| Spot taker | 0.100% (10.0 bps) | 0.075% (7.5 bps) |
| USDC-pair taker | 0.095% (9.5 bps) | 0.07125% (7.1 bps) |
| USDT-margined futures maker | 0.020% (2.0 bps) | 0.018% (1.8 bps) — 10% off futures |
| USDT-margined futures taker | 0.050% (5.0 bps) | 0.045% (4.5 bps) |

VIP 0 = under $1,000,000 of 30-day spot volume (or under $5,000,000 futures volume), no BNB holding required. **VIP 1 requires ≥$1,000,000 30-day spot volume plus 5 BNB.** At ~$40 notional per trade, HAWK would need ~25,000 round trips to reach VIP 1 in a 30-day window — so **VIP tiers are irrelevant and the backtest must assume VIP 0 permanently**. Note that Binance.US publishes a different schedule (0% maker / 0.01% taker on Tier 0 pairs, [binance.us/fees](https://www.binance.us/fees)) which does **not** apply to the global exchange HAWK sources data from.

### 5.2 MEXC fees — and the API surcharge that dominates everything

Official page: [mexc.com/fee](https://www.mexc.com/fee) (retrieved 2026-09-17; also JS-rendered, no rate tables served to a plain fetch).

| Item | Standard rate | With MX deduction (20%) |
|---|---|---|
| Spot maker | 0.00% | 0.00% |
| Spot taker | 0.05% (5.0 bps) | 0.04% (4.0 bps) |
| Futures maker | 0.01% (1.0 bp) | 0.008% (0.8 bps) |
| Futures taker | 0.04% (4.0 bps) | 0.032% (3.2 bps) |

Source: [MEXC, "How to Use MX for Trading Fee Discounts"](https://www.mexc.com/learn/article/how-to-use-mx-for-trading-fee-discounts/1). MEXC explicitly states that rates vary by pair, country and region — e.g. an [official announcement](https://www.mexc.com/announcements/article/adjustments-to-standard-futures-trading-fees-in-cis-region-17827791536704) sets CIS/Ukraine standard futures at maker 0% / taker 0.02% effective 7 Jul 2026, while [another](https://www.mexc.co/announcements/article/adjustments-to-standard-futures-trading-fees-in-georgia-17827791538245) sets Georgia ETH/GOLD futures to 0.01%/0.04% effective 6 Sep 2026.

**The critical finding — MEXC charges API futures trades at a materially higher, non-discountable rate.** Per [MEXC's official announcement "Updates to API Futures Trading Fees (May 1, 2026)"](https://www.mexc.com/announcements/article/updates-to-api-futures-trading-fees-may-1-2026-17827791535194), effective 1 May 2026 08:00 UTC:

| API Futures | Before | **After (current)** |
|---|---|---|
| Maker | 0.01% | **0.04% (4.0 bps)** |
| Taker | 0.05% | **0.06% (6.0 bps)** |

MEXC's own wording: *"API trading is subject to a separate fee structure, which takes precedence over any rates or promotional offers (such as 0-fee events, MX discounts, etc.)… Despite the web or app trading fees for BTCUSDT Futures being maker 0% and taker 0.01%, trades executed via API will be charged at maker 0.04% and taker 0.06%."* Applies to all futures pairs except Innovation Zone.

**Implication: HAWK is an automated system and will trade via API, so the correct fee assumption is 4 bps maker / 6 bps taker — not the 0–1 bp headline rates.** A taker-in/taker-out round trip costs **12 bps in fees alone**, three times the 4 bps a naive reading of the public schedule would suggest. Any existing backtest using 0.01%/0.04% understates fees by roughly 4 bps per round trip, and one using "0-fee promotional" rates understates by 12 bps.

### 5.3 Spreads for mid-cap altcoins

- **"Order Book Liquidity on Crypto Exchanges"** (*Journal of Risk and Financial Management* 18(3):124, 2025; [DOI](https://doi.org/10.3390/jrfm18030124)). Order-book data across Binance, Huobi, Kraken, OKEx. For **non-stablecoin cryptocurrencies, the average relative spread at the best level is ≈0.55% (55 bps) on Binance**, Huobi and Kraken, versus 1.9% on OKEx. Binance's 514 listed pairs give it breadth but not depth: despite a *wider* best-level spread than Kraken for some assets, Kraken and Huobi show **better liquidity at deeper levels (level 5+)**. Caution: this average spans Binance's entire long tail of pairs, so it is an **upper bound** for HAWK's top-80-by-quote-volume universe, not a central estimate.
- Practitioner benchmarks, consistent across sources: **large-cap (BTC/ETH) under 5 bps** and often under 1 bp in active sessions; **mid-cap 20–50 bps as the normal range**; **micro-cap frequently above 100 bps**; and spreads **widen in low-volume hours and under stress** (e.g. BTC moving from ~1 bp to 4–5 bps during the FTX week). See [this spread primer](https://academy.darkex.com/education/understanding-bid-ask-spread-in-cryptocurrency-trading/). These are unrefereed but align with the JRFM study's ordering.
- **Free calibration data exists and HAWK should use it.** Ardia, Guidotti & Kroencke publish **monthly effective bid-ask spread estimates for every crypto pair listed on Binance** ([Harvard Dataverse](https://doi.org/10.7910/dvn/9ava2b), with code at [github.com/eguidotti/bidask](https://github.com/eguidotti/bidask)). The estimator is OHLC-based (the Abdi–Ranaldo family / "efficient estimator") and returns the **root-mean-square effective spread over the sample window**. This is the single best available way to put a per-symbol, per-month spread number into HAWK's cost model without buying order-book data.

### 5.4 Funding costs across a 24-hour hold

- Binance's funding rate = `clamp(premium index, −0.05%, +0.05%) + interest rate`, with interest ≈ 0.01% (1 bp) per 8-hour period, settled at 00:00 / 08:00 / 16:00 UTC. **Standard cap is ±0.05% (5 bps) per window**, varying by pair.
- Empirical distribution from a **panel of 107,839 8-hour observations across 19 USDT-margined perpetuals, 2020-01 to 2025-12** ([premium-index analysis script](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/02_financial_data_universe/11_crypto_premium_analysis.py)): **all 19 symbols had a negative mean premium, between −0.03 and −2.6 bps.** Premium volatility: BTC std **5.6 bps**; ETH/ADA/DOT **6–7 bps**; XRP/UNI/COMP **10–11 bps**; **SOL std 36.7 bps with a −1,915 bps minimum during the FTX collapse.** The clamp is hit in **82.9% of 8-hour periods**, driven by the interest-rate baseline.
- A concrete single-symbol check: BNBUSDT 8h funding, Oct 2021 – Feb 2026, **4,836 settlements**: mean **−0.58 bps** per 8h (−6.36% annualized), median 0.00, min −20.9 bps, max +10.0 bps, std 2.36 bps ([dataset](https://huggingface.co/datasets/Torch-Trade/bnbusdt_perp_funding_8h_10_2021_to_02_2026)). Year-by-year averages for BNB range from **+21.7% annualized (2021) to −12.6% (2022)** ([funding history](https://cryptocurrencyhistory.com/coins/bnb/funding-rate-history/)).
- **Practical magnitude: 1–3 bps per settlement in absolute terms, with a hard cap at 5 bps.** A full 24-hour hold crosses **3 settlements ⇒ ~3–9 bps typical, up to 15 bps at the cap**, sign depending on side. This is small relative to spread, but it is **not negligible relative to a 1% target** and it is systematically adverse for one side at a time.
- **Interval heterogeneity is a real modelling risk.** Hyperliquid funds **hourly** with a ±1.50% cap; many venues and pairs use 1h or 4h periods; some compute funding continuously. Do not assume 8 hours across 80 symbols — pull `funding_interval` per symbol from the venue API (see the [`period` vs `interval` distinction](https://gitbook-docs.coinmetrics.io/market-data/market-data-overview/market-funding-rates)).

### 5.5 Published evidence on cost erosion at short horizons

This literature is unambiguous and directly on point for a 15-minute signal.

- **"Microstructure alpha: hierarchical learning and cross-asset transfer in cryptocurrency markets"** (*Frontiers in Blockchain*, 2026; [link](https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/full)). 5-minute rebalancing drives two-way turnover to **124–204× notional per day**. The authors find leakage-free forecasting information at minute frequency — range-based spread proxies, realized volatility and short-horizon momentum pass formal stability selection — yet **all net Sharpe ratios are deeply negative**. Their stated assumption: **20 bps per round trip on spot at Binance VIP-0**, and at 288 round trips/day "cumulative costs overwhelm any statistical edge by orders of magnitude." They also report that a gradient-boosted model **overfit severely and performed worse than chance** under strict temporal separation, while a simple linear model gave a small improvement over a random walk. Their three suggested remedies are exactly HAWK's available levers: **lower fee tier, longer holding horizon to amortise the round trip, and netting redundant trades across symbols.**
- **"Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting"** ([arXiv:2606.00060](https://arxiv.org/abs/2606.00060)). ~70,000 hourly BTC-USDT observations, 2018–2026, 27-fold walk-forward. All three models (XGBoost, LSTM, iTransformer) produce positive **gross** performance, but **naive sign-based strategies fail once 10 bps of transaction costs are imposed**. A **cost-aware execution filter** — trade only when forecast magnitude exceeds a cost-based threshold — **reduces turnover by more than an order of magnitude** and restores profitability in selected configurations (best long-only XGBoost: >65% annualized, Sharpe >1). Honest caveat from the authors: formal Sharpe comparisons against buy-and-hold **do not reject after bootstrap adjustment**, and performance is uneven across regimes. Their conclusion is the key transferable lesson: *"the main obstacle in hourly cryptocurrency trading is not only weak predictability, but also the way forecasts are converted into trades."*
- **"Crypto factor zoo"** (*International Review of Financial Analysis*, 2026; [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1057521926000645)). 565 coins, 36 factors, Jan 2018 – Jul 2024. Net performance declines with costs for **all** strategies; sensitivity varies sharply; "trading frictions meaningfully narrow the set of practically exploitable crypto factors."
- **"Machine learning and the cross-section of cryptocurrency returns"** (*International Review of Financial Analysis*, 2024; [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1057521924001765)). More optimistic — most ML strategies survive costs — but with two caveats that bite for HAWK: **trading costs consume on average more than half of long-short payoffs**, and **alphas are concentrated in hard-to-trade assets and depend on harvesting extreme returns on small, illiquid, volatile coins.** That is a warning that any HAWK backtest showing the edge concentrated in high-`universe_rank` symbols is probably measuring costs it will actually pay.

### 5.6 Cost Model Recommendation (concrete numbers for the backtest)

**Notional context.** HAWK paper config: ~$30 capital, 5× leverage, ~$8 margin per trade ⇒ **~$40 notional**. At $40, **market impact is effectively zero** — the order is negligible against top-of-book depth for any top-80 pair. Therefore **do not model a square-root impact term**; it would be spurious precision. The entire cost is fees + spread + execution noise + funding. Conversely, $40 notional means **no cost can be amortised**: fixed frictions (tick rounding, minimum notional rounding) hit at full relative weight.

#### Spread ladder by liquidity rank

Use `universe_rank` (by quote volume) as the key, and calibrate/override per symbol using Abdi–Ranaldo estimates from trailing 15m OHLCV (excluding signal bars) plus the Harvard Dataverse Binance spread dataset.

| `universe_rank` | Assumed quoted spread `S` | Taker spread cost, round trip (= `S`) | Slippage allowance (0.25 × `S`) |
|---|---|---|---|
| 1–10 | 4 bps | 4 bps | 1 bps |
| 11–25 | 8 bps | 8 bps | 2 bps |
| 26–45 | 16 bps | 16 bps | 4 bps |
| 46–65 | 28 bps | 28 bps | 7 bps |
| 66–80 | 45 bps | 45 bps | 11 bps |

Rationale: anchored to the practitioner mid-cap band of 20–50 bps and large-cap <5 bps, kept well below the JRFM all-pairs Binance average of 55 bps because HAWK's universe is the top ~80 by quote volume, not the full 514-pair tail.

**Slippage scaling rule.** Because impact is negligible at $40, slippage is not a function of size — it is a function of **spread and quote staleness**. Model it as a fraction of the quoted spread rather than a separate bps constant:

```
slippage_bps = 0.25 * S(rank) * time_of_day_mult * stress_mult
```

- `time_of_day_mult` = **1.0** in the US/EU overlap, **1.3** in the thin Asian-evening/early-UTC window, **1.5** on weekends. Justified by Baur et al. (2019): persistently lower trading activity in local evening hours and weekends across every exchange studied — the one replicable calendar effect.
- `stress_mult` = **3.0** when BTC's 15m realized range exceeds its trailing 99th percentile. Justified by the October 2025 cascade measurements: quoted price impact spiked **3.2–9.1×** and depth collapsed. Use 3.0 as a conservative lower bound of that range; better still, skip these zones entirely (see §4).
- Add a **tick-rounding term of 1 bp** for symbols whose tick size exceeds 1 bp of price, which is common among low-priced altcoins.

#### Venue A — MEXC USDT perps via API (the live-trading venue)

Fees: **maker 4.0 bps, taker 6.0 bps** (API schedule, non-discountable).

| Execution style | Fees | Spread | Slippage | Funding (≈1.5 settlements avg hold) | **Total round trip** |
|---|---|---|---|---|---|
| Taker in / taker out, rank 1–10 | 12.0 | 4 | 1 | 2 | **19 bps** |
| Taker in / taker out, rank 26–45 (**base case**) | 12.0 | 16 | 4 | 2 | **34 bps** |
| Taker in / taker out, rank 66–80 | 12.0 | 45 | 11 | 2 | **70 bps** |
| **Maker in / taker out, rank 26–45** | 10.0 | 8 (half) | 2 | 2 | **22 bps** |
| Maker in / maker out, rank 26–45 | 8.0 | 0 | 0 | 2 | **10 bps** |

Funding: charge **1.0 bp per settlement crossed** as the baseline (the interest-rate component), signed against longs; allow **5.0 bps per settlement** in a stress scenario; **count actual settlements crossed per zone** from the symbol's true funding interval rather than assuming 8h. A full 24h hold ⇒ 3 bps baseline, 15 bps stressed.

#### Venue B — Binance spot (long-only; also the data source)

Fees: **10.0 bps maker and taker at VIP 0**, or **7.5 bps with BNB deduction**. No funding. No leverage (so the 5× paper config is not representable here).

| Execution style | Fees | Spread | Slippage | **Total round trip** |
|---|---|---|---|---|
| Taker/taker, rank 26–45, no BNB | 20.0 | 16 | 4 | **40 bps** |
| Taker/taker, rank 26–45, with BNB | 15.0 | 16 | 4 | **35 bps** |
| Maker in / taker out, rank 26–45, with BNB | 15.0 | 8 | 2 | **25 bps** |

This sits right on top of the 20 bps-per-round-trip figure the *Frontiers* microstructure study used to conclude that no 5-minute Binance strategy survived — HAWK's cost structure is in the same regime, and its only structural advantage is a **longer holding period over which to amortise it**.

#### Break-even edge

Let `c` = round-trip cost in bps, `T` = gross target, `L` = gross stop, and `p` = win rate. Break-even requires `p·(T − c) = (1 − p)·(L + c)`, i.e.

```
p_breakeven = (L + c) / (T + L)
```

Equivalently, the strategy needs a **gross expected return per trade of at least `c` bps**.

| Scenario | Target / Stop | Cost `c` | Break-even win rate | Excess over coin flip |
|---|---|---|---|---|
| MEXC perp, base case, taker/taker | 100 / 100 | 34 bps | **67.0%** | +17.0 pp |
| MEXC perp, base case, **maker entry** | 100 / 100 | 22 bps | **61.0%** | +11.0 pp |
| MEXC perp, base case, maker entry, 2:1 R:R | 100 / 50 | 22 bps | **48.0%** | −2.0 pp |
| MEXC perp, base case, taker/taker, 2:1 R:R | 100 / 50 | 34 bps | **56.0%** | +6.0 pp |
| MEXC perp, illiquid (rank 66–80), taker/taker | 100 / 100 | 70 bps | **85.0%** | +35.0 pp |
| Binance spot, base case, taker/taker, no BNB | 100 / 100 | 40 bps | **70.0%** | +20.0 pp |

**Headline numbers to carry forward:**
- **Minimum gross edge to break even: ~34 bps per round trip in the base case (MEXC perp, mid-cap, taker/taker); ~22 bps with maker entry; ~70 bps for the illiquid tail.**
- **At the observed ~50% directional hit rate with symmetric 1% targets, base-case expectancy is −34 bps per trade.** On $40 notional that is −$0.14 per trade, which against $8 of posted margin is **−1.7% of margin per trade** — with 5× leverage, the account is consumed by frictions long before any directional edge could be measured.
- **Three structural levers, in order of magnitude:**
  1. **Maker entry at the zone: −12 bps** (34 → 22). This is free money and it is natural to HAWK's design, because the retest *is* a limit-order event — price must come back into the candle's high-low range, so a resting limit order is the mechanically correct way to express the entry. Cost: non-fill risk and adverse selection (you are filled preferentially when price keeps going), which must be modelled as a fill-probability assumption, not ignored.
  2. **Restrict the universe to `universe_rank` ≤ 45: −36 bps** versus the illiquid tail. The top of the universe costs roughly a quarter of what the bottom costs. Given Papers §5.5's warning that ML crypto alphas concentrate in illiquid coins, the illiquid tail is exactly where a backtest will look best and trade worst.
  3. **Asymmetric R:R (1% target / 0.5% stop): −11 pp of required win rate.** With a 50% hit rate this is the only configuration in the table that is not automatically negative — but it requires that the 0.5% stop is not hit by noise first, which at 15m altcoin volatility is a strong assumption that HAWK's own MAE records can test directly.

**Backtest hygiene requirements implied by the above:**
- Charge costs on **every** zone taken, including zones that are stopped out immediately.
- Never assume a fill better than the worse side of the spread for taker orders; for maker orders, require that price traded **through** the limit, not merely touched it.
- Apply `stress_mult` from realized BTC 15m range, not from hindsight labels.
- Re-verify both fee schedules against live account trade history before trusting any net result; the MEXC API surcharge alone moves the base case by 4–12 bps.

### Implication for HAWK

- **The cost model is not a detail to be appended after signal research — at these numbers it is the dominant term.** A 1% target with a ~34 bps base-case round trip means **costs consume about a third of the gross move**, and at the observed coin-flip hit rate the strategy's expectancy is negative by roughly the full cost. Re-running existing paper results with the ladder above should be the **first** action, before any filter work, because it determines whether there is anything to filter.
- **Switch the entry from market to resting limit at the zone.** This is the single highest-value change available, worth ~12 bps of the ~34 bps base case, and it aligns the execution mechanism with the strategy's own logic. Record fill/no-fill outcomes so the adverse-selection cost becomes measurable rather than assumed.
- **Use `universe_rank` and `universe_quote_volume` as cost inputs first and signal inputs second.** Their best-established role is setting `S(rank)`; their role as predictors of zone quality is unestablished and confounded with cost.
- **Log realized funding per zone.** With holds up to 24h crossing ~3 settlements, funding is a 3–15 bps line item with a systematic sign by side — enough to explain a persistent long/short asymmetry that would otherwise be misread as a directional finding.
- **Adopt the cost-aware execution filter from the walk-forward BTC study as the decision layer's actual form.** Rather than asking "long, short, or skip?" as a classification problem, the better-supported formulation is: **take the trade only when the model's expected move exceeds the symbol-specific round-trip cost by a margin.** That paper's central result is that this reframing — not better forecasts — is what turned a losing strategy into a profitable one, and it naturally produces "skip" for exactly the high-`universe_rank`, wide-spread symbols where the edge would be eaten.
- **Beware the illiquid-coin trap.** The cross-sectional ML literature finds crypto alphas concentrated in small, illiquid, volatile coins. HAWK's 80-symbol universe includes such names, and they will show the largest MFE in backtest while carrying 45–70 bps round trips. The cost ladder must be applied per-symbol, never as a universe-wide average, or the backtest will systematically select the symbols it can least afford to trade.

---

## 6. Prior Probability Table

My honest prior that **conditioning on each topic materially improves HAWK's risk-adjusted results** — "materially" meaning a net-of-cost improvement large enough to survive walk-forward validation on HAWK's own data, not merely an in-sample coefficient.

| # | Topic | Prior | Reasoning |
|---|---|---|---|
| 1 | **Volatility regimes / vol filters** (as a directional gate) | **15%** | Volatility clustering is near-certain, but Cederburg et al. (2020) showed the return-timing version fails out-of-sample in 72 of 103 equity strategies, and crypto's supporting evidence is weekly portfolio-level crash mitigation, not 15m entry timing. |
| 1b | **Volatility for position sizing and stop placement** (same field, different use) | **70%** | Left-tail truncation is the one volatility benefit that survived replication in both equities and crypto, and a 5×-leveraged altcoin strategy is exactly the crash-prone payoff where it helps. Listed separately because it changes the risk-adjusted metric without needing any return predictability. |
| 2 | **BTC market/beta conditioning** | **45%** | Strongest evidence base of the five: unidirectional 1-minute BTC→altcoin Granger causality, regime-dependent coupling (calm = self-driven, volatile = BTC-dominated), and documented downside asymmetry. Discounted from higher because BTC explains altcoin *variance* far better than *return*, the effect it predicts is an interaction (needs large samples), and the current boolean encoding discards the signed information that carries most of the value. |
| 3 | **Intraday / calendar seasonality** | **10%** | The published return effects did not persist across time or venues (Baur et al., 15m observations, seven exchanges), the relevant clock is the U.S. equity calendar rather than UTC and shifts with daylight saving, funding-hour directional effects measured under 3 bps and flipped sign as the sample grew, and a placebo test showed 100% of random pseudo-events look significant on this trending series. The low prior is for *directional* use; using hour-of-day as a **cost multiplier** is well supported and is folded into topic 5. |
| 4 | **Liquidity / stop-hunt / cascade microstructure** | **50%** | Highest prior among the directional factors because Osler supplies two documented mechanisms with **opposite signs** (take-profit clustering ⇒ reversal, stop-loss clustering ⇒ continuation), which is the most plausible explanation for HAWK's pooled 50/50 outcome, and the separating variable (distance to the swing level) is already recorded. Discounted by: HAWK cannot compute order-flow imbalance, its unsigned volume spike is the documented-noisy version of the signal, and the OHLC spread proxies are confounded with volatility in a way that is mechanically correlated with HAWK's own wick criterion. |
| 5 | **Transaction costs and execution** | **90%** | Not really a "filter" — it is an accounting correction with a near-certain effect. The MEXC API surcharge alone (4 bps maker / 6 bps taker, overriding promotional rates) is a 4–12 bps swing per round trip, switching to maker entry saves ~12 bps of a ~34 bps base case, and restricting to `universe_rank` ≤ 45 saves up to 36 bps. Two independent studies show short-horizon crypto strategies dying at 10–20 bps and one shows a **cost-aware trade filter converting a loser into a winner** without improving the forecast at all. The 10% reserved for failure is the possibility that no configuration clears the hurdle — in which case the correct conclusion is to stop, which is still a material improvement over trading. |

**Reading the table.** The two highest priors (5 at 90%, 1b at 70%) are not about predicting direction at all — they are about paying less and sizing better. The two most commonly assumed edges (volatility gating at 15%, calendar effects at 10%) are the weakest. The genuinely promising directional lead is topic 4's mechanism split, with topic 2 as a conditioning variable on top of it. **If the cost work in §5.6 shows no configuration clears break-even, none of the other four priors matter**, because a filter can only redistribute a negative expectancy.

---

## Appendix: Full Reference List

**Volatility**
- Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of Finance* 72(4), 1611–1644. https://doi.org/10.1111/jofi.12513
- Cederburg, S., O'Doherty, M.S., Wang, F. & Yan, X. (2020). "On the performance of volatility-managed portfolios." *Journal of Financial Economics* 138(1), 95–117. https://doi.org/10.1016/j.jfineco.2020.04.015
- Grobys, K., Kolari, J., Sandretto, D., Shahzad, S.J.H. & Äijö, J. (2025). "Cryptocurrency momentum has (not) its moments." *Financial Markets and Portfolio Management*. https://link.springer.com/article/10.1007/s11408-025-00474-9
- Habeli, M., Barakchian, S.M. & Motavasseli, A. (2024). "Adaptive Risk Allocation in Crypto Markets: Evaluating Volatility-Scaled Portfolios." SSRN 5090097.
- Parkinson, M. (1980). *Journal of Business* 53(1), 61. Garman, M. & Klass, M. (1980). *Journal of Business* 53(1), 67. Rogers & Satchell (1991). Yang & Zhang (2000).
- "Beyond GARCH in cryptocurrency volatility modelling: superiority of range-based estimators" (2024). *Applied Economics Letters*. https://doi.org/10.1080/13504851.2024.2363295
- Allen, D. (2025). "Realised volatility estimation shortcuts: An empirical analysis." MODSIM 2025. https://doi.org/10.36334/modsim2025.d02.allen
- "Long range dependence in the Bitcoin market" (2019). *Physica A*. Khuntia & Pattanayak (2018). *Finance Research Letters*. Bouri et al. (2019). *IJFE* 24. Mensi et al. (2019). *Finance Research Letters* 29, 222–230.

**BTC beta**
- Liu, Y., Tsyvinski, A. & Wu, X. (2022). "Common Risk Factors in Cryptocurrency." *Journal of Finance* 77(2), 1133–1177. https://www.nber.org/papers/w25882
- Liu, J. (2021). "Factor Structure in Cryptocurrency Returns and Volatility." SSRN 3389152.
- "Price Transmission from Bitcoin to Altcoins: High-Frequency Evidence" (2026). *Asia-Pacific Financial Markets*. https://doi.org/10.1007/s10690-026-09589-z
- "Regime-Specific interdependencies in cryptocurrency markets: A high-frequency GMM-VAR approach" (2025). *Data Science in Finance and Economics*. https://doi.org/10.3934/dsfe.2025017
- "Do Bitcoin Shocks Dominate Other Cryptocurrencies?" (2025). *Asia-Pacific Financial Markets* 32(4).
- "Dependence in cryptocurrencies: A partial correlation connectedness approach" (2026). *Finance Research Letters*.
- "Mapping Systemic Tail Risk in Crypto Markets" (2025). *JRFM* 18(6):329. https://www.mdpi.com/1911-8074/18/6/329
- Forbes, K. & Rigobon, R. (2002). *Journal of Finance* 57(5).

**Seasonality**
- Baur, D.G., Cahill, D., Godfrey, K. & Liu, Z. (2019). "Bitcoin time-of-day, day-of-week and month-of-year effects in returns and trading volume." *Finance Research Letters* 31, 78–92.
- "A new perspective of the day-of-the-week effect on Bitcoin returns" (2022). *Oeconomia Copernicana*. https://doi.org/10.24136/oc.2022.022
- "Bitcoin on Wall Street Time: Natural Experiments on the Institutionalization of a 24/7 Market" (2026). *JRFM* 19(9):707.
- "Turn-of-the-candle effect in bitcoin returns" (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/
- "Fundamentals of Perpetual Futures." arXiv:2212.06888. "The Role of Binance in Bitcoin Volatility Transmission." arXiv:2107.00298.

**Microstructure**
- Osler, C.L. (2003). "Currency Orders and Exchange Rate Dynamics." *Journal of Finance* 58(5), 1791–1819. https://doi.org/10.1111/1540-6261.00588
- Osler, C.L. (2005). "Stop-loss orders and price cascades in currency markets." *JIMF* 24(2), 219–241. https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr150.pdf
- "Systemic fragility in decentralised markets." BIS Working Paper 1062. https://www.bis.org/publ/work1062.pdf
- "Measuring the engine of a liquidation cascade." arXiv:2608.03616
- "Same Shock, Same Assets, Different Microstructure: … October 10, 2025 Cryptocurrency Cascade." Research Square rs-9459584.
- Cont, R., Kukanov, A. & Stoikov, S. (2013). "The Price Impact of Order Book Events." *Journal of Financial Econometrics* 12(1), 47–88. arXiv:1011.6402
- "Cross-impact of order flow imbalance in equity markets" (2023). *Quantitative Finance*. https://doi.org/10.1080/14697688.2023.2236159
- Brauneis, A., Mestel, R., Riordan, R. & Theissen, E. (2021). "How to measure the liquidity of cryptocurrency markets?" *Journal of Banking & Finance* 124, 106041. https://doi.org/10.1016/j.jbankfin.2020.106041
- Corwin, S. & Schultz, P. (2012). *Journal of Finance* 67(2). Abdi & Ranaldo (2017). Roll, R. (1984). *Journal of Finance* 39. Amihud, Y. (2002). Kyle & Obizhaeva (2016).
- "Do Liquidity Proxies Based on Daily Prices and Quotes Really Measure Liquidity?" (2020). *Entropy* 22(7):783. https://doi.org/10.3390/e22070783

**Costs**
- Binance fee schedule: https://www.binance.com/en/fee/schedule (retrieved 2026-09-17)
- MEXC fee overview: https://www.mexc.com/fee (retrieved 2026-09-17)
- MEXC, "Updates to API Futures Trading Fees (May 1, 2026)": https://www.mexc.com/announcements/article/updates-to-api-futures-trading-fees-may-1-2026-17827791535194
- "Order Book Liquidity on Crypto Exchanges" (2025). *JRFM* 18(3):124. https://doi.org/10.3390/jrfm18030124
- Ardia, D., Guidotti, E. & Kroencke, T.A. (2024). "Bid-Ask Spread Estimates for Crypto Pairs in Binance." Harvard Dataverse. https://doi.org/10.7910/dvn/9ava2b
- "Microstructure alpha: hierarchical learning and cross-asset transfer in cryptocurrency markets" (2026). *Frontiers in Blockchain*.
- "Machine Learning-Based Bitcoin Trading Under Transaction Costs" (2026). arXiv:2606.00060
- "Crypto factor zoo" (2026). *IRFA*. "Machine learning and the cross-section of cryptocurrency returns" (2024). *IRFA*.
