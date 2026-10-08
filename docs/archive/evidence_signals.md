# Evidence Review: Direction-Selection Signals for HAWK Order Blocks

**Scope.** Five signal families that could be layered on top of HAWK's order-block detector to
produce a *directional* forecast. Execution-cost modelling and validation methodology are covered
by separate reviews and are only referenced here where a cited result hinges on them.

**What HAWK actually detects.** Reading `src/hawk_eye/logic/order_blocks.py`, a 15m bar is flagged
when, against the prior 60 bars only: `volume >= 1.35 * mean(prior volume)`,
`max(upper_wick, lower_wick) / |close - open| >= 1.2`, and
`min(|high - prior_high|, |low - prior_low|) / close * 100 >= 0.5`. Four properties of this
definition matter for mapping the literature onto it, and they are flagged again in the relevant
sections:

1. `wick_ratio` uses `max(upper, lower)` — **the side of the wick is discarded.** The literature on
   reversal candles is entirely about signed wick direction (hammer vs. shooting star), so HAWK's
   feature is by construction orthogonal to the one dimension the candlestick literature says
   might carry information (Topic 3).
2. `distance_to_swing` is the distance to the **nearer** of the two prior extremes, and the filter
   requires it to be *large*. So HAWK deliberately excludes bars at the swing high/low — i.e. it
   excludes exactly the locations where the support/resistance and stop-clustering literature
   (Topic 1) predicts unusual price behaviour. This is close to an inversion of the documented
   effect.
3. `distance_to_price = |midpoint - close| / close` is computed **from the signal bar's own close**,
   not from a later market price. It is an algebraic restatement of where the close sits inside the
   bar's range — i.e. a body/range geometry term that is mechanically correlated with `wick_ratio`,
   not a measure of distance to current price. It carries no independent information about market
   proximity and should not be treated as one.
4. `direction` is `close > open`, i.e. the sign of the signal bar's own body. It is a label, not a
   forecast, and the ~47–51% hit rate on ±1% barriers already reported from ~10 days of live data
   is what an uninformative label should produce.

---

## Executive Summary

| Family | Verdict on published evidence |
|---|---|
| 1. Order blocks / SMC / ICT | **Unsupported folklore** as a named method. Zero peer-reviewed validation. The *adjacent* claim — that horizontal levels and round numbers attract order clustering and alter price paths — is well documented (Osler 2003, JF), but it concerns levels HAWK's `distance_to_swing >= 0.5%` filter explicitly discards, and the trading-rule version of it (Brock/Lakonishok/LeBaron 1992) collapsed out-of-sample under Sullivan/Timmermann/White (1999). |
| 2. Abnormal volume | **Real but weak, wrong horizon, and probably wrong sign for HAWK.** Gervais/Kaniel/Mingelgrin (2001, JF) is a genuine, replicated cross-sectional effect, but it is a *daily-volume, weekly-to-monthly-horizon, small-cap equity* premium, not an intraday continuation signal. At intraday crypto horizons the better-documented relation is volume → *volatility*, not volume → signed return. |
| 3. Candlestick / long-wick reversal bars | **Negative evidence.** Marshall/Young/Rose (2006, JBF) found candlestick single- and multi-bar patterns unprofitable on DJIA 1992–2002; Horton (2009) likewise. Lo/Mamaysky/Wang (2000, JF) found *statistical* information in some geometric patterns but did not demonstrate tradable profit. Critically, all of it is about signed wick direction, which HAWK discards. |
| 4. Short-horizon crypto reversal vs. continuation | **The strongest of the five, and it points the opposite way to HAWK's label.** Intraday/hourly crypto returns show robust short-horizon *reversal*, especially after volume and volatility shocks, while momentum appears at weekly-to-monthly horizons. If anything predicts HAWK's outcome, it is a reversal, not continuation, of the signal bar's own sign. |
| 5. Cross-sectional crypto factors | **Real at weekly-to-monthly horizons, undemonstrated at 15m–48h.** Liu/Tsyvinski/Wu (2022, JF) establish market, size and momentum factors in crypto, but at weekly and longer horizons on a large universe. HAWK's ~80-symbol universe and `universe_rank` / `universe_quote_volume` are a plausible but *unvalidated* short-horizon application. |

**One-line conclusion.** Nothing in the literature supports the *order-block construct itself* as a
directional signal. The two families with a defensible reason to expect signal in HAWK's own data
are (4) short-horizon crypto reversal and (5) cross-sectional crypto ranking, in that order — and
family (4) predicts that HAWK's current `direction` label has the *wrong sign*.

---

## Topic 1 — Order blocks, supply/demand zones, SMC/ICT, and horizontal support/resistance

### 1a. The named method: no credible evidence

There is **no peer-reviewed study validating "order blocks," "fair value gaps," "break of
structure," or the SMC/ICT framework** as predictive of returns. This is not a case of thin
evidence; it is a case of absent evidence. The construct originates in retail trading education
(Michael Huddleston / "Inner Circle Trader"), and is itself a rebranding of older discretionary
material — Richard Wyckoff's "composite operator" accumulation/distribution narrative and Sam
Seiden's supply-and-demand zones. Surveys of the citation landscape find that none of the sources
invoked in SMC pedagogy validate SMC as a method
([IndicatorEdge, *Smart Money Concepts* evidence survey](https://indicatoredge.io/smart-money-research);
[LuxAlgo Library, *Smart Money Concepts*](https://www.luxalgo.com/library/concept/smart-money-concepts/)).
Both explicitly state that the "order block as something distinct from classic support/resistance"
claim and the accumulation→manipulation→distribution narrative have no rigorous support, and that
widely circulated retest/fill rates of "80–90%" are unsourced.

**Label: unsupported folklore.** Specifically unsupported are (i) that a high-volume wick bar marks
institutional order accumulation, (ii) that the bar's own body sign predicts the direction of the
subsequent move, and (iii) any specific retest-fill or win-rate figure quoted for these patterns.
Note that HAWK's own ~83% retest rate is *not* evidence for the construct: a zone defined as a
recent bar's high–low range will be retested at a high rate under a random walk with realistic
15m volatility, because the zone is wide (`zone_size_pct`) and near the current price. Retest rate
is a property of the geometry, not of the signal.

The only quantitative tests that exist are grey literature (non-peer-reviewed practitioner
backtests), and they are *not* encouraging. Treat these as weak evidence — they are unrefereed and
their own authors' methodology is unaudited — but they are the only mechanical tests available:

- A study of 648 backtests codifying four ICT entries (order blocks, FVGs, liquidity sweeps,
  "optimal trade entry") across SPY/QQQ/DIA/IWM found **0 of 648 variants beat buy-and-hold on net
  profit**, and on SPY the best forward-return effect was the order block at +0.121% over 5 days
  across 736 events, **t = 1.22** (below significance)
  ([StatOasis, *I Backtested ICT / Smart Money Concepts — What Survives*](https://statoasis.com/overfit/research/ict-backtest-what-survives)).
  Notably, ICT entries were *not worse* than textbook entries — they were merely not better than
  doing nothing.
- A mechanical FVG-after-sweep rule run on 2.55M one-minute EURUSD bars (2019-01-01 to 2025-12-31)
  across 54 rule variations found the best win rate at 56.3% and **0 of 54 variants profitable
  after a 0.5-pip cost** ([IndicatorEdge](https://indicatoredge.io/smart-money-research)).

The honest summary: the *phenomena* SMC gestures at (stop clustering, session volatility patterns,
order-imbalance price impact) are real and documented; the *specific mechanical rules* have never
been shown to produce a cost-surviving edge by anyone who published their methodology.

### 1b. The adjacent, better-studied question: horizontal levels and round numbers

Here there is real evidence, but it supports a narrower claim than traders usually take from it.

**Osler (2003), "Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive
Success of Technical Analysis," *Journal of Finance* 58(5), 1791–1819**
([DOI](https://doi.org/10.1111/1540-6261.00588);
[NY Fed Staff Report 125 working version](https://www.newyorkfed.org/research/staff_reports/sr125.html)).
Using the first dataset of individual currency stop-loss and take-profit orders from a large FX
dealing bank, Osler documents two distinct clustering patterns: **take-profit orders cluster
strongly at round numbers** (which mechanically produces reversals at those levels, since TP orders
are counter-trend limit orders), while **stop-loss orders cluster just beyond round numbers** (which
mechanically produces acceleration once a level is crossed, since SL orders are trend-following
market orders). This is a microstructural *mechanism*, established on order-book data, for why
prices behave unusually at obvious levels. It is the single strongest citation in this whole
family. Asset class: FX spot. It is a statement about conditional price *paths*, not about a
tradable net-of-cost profit, and Osler does not claim the latter.

**Osler (2000), "Support for Resistance: Technical Analysis and Intraday Exchange Rates," *FRBNY
Economic Policy Review* 6(2)**
([link](https://www.newyorkfed.org/research/epr/00v06n2/0007osle.html)). Using support/resistance
levels actually published daily by six market-making firms, Jan 1996 – Mar 1998, tested against
1-minute DEM/USD, JPY/USD and GBP/USD quotes: levels **stopped intraday trends 56.2% of the time on
average**, statistically significant and better than arbitrary levels, for all six firms — but with
substantial variation across firms and currencies. Note the effect size: 56.2% against a ~50%
null is real but small, and this is a *trend-interruption* rate, not a directional return.

**Curcio & Goodhart (1992), "When Support/Resistance Levels are Broken, Can Profits Be Made?
Evidence From the Foreign Exchange Market," LSE Financial Markets Group DP 142**
([abstract](https://www.fmg.ac.uk/publications/discussion-papers/when-supportresistance-levels-are-broken-can-profits-be-made))
initially reported that chartist-supplied trading-range breaks generated profitable signals net of
costs. **Curcio, Goodhart, Guillaume & Payne (1997), "Do technical trading rules generate profits?
Conclusions from the intra-day foreign exchange market," *International Journal of Finance &
Economics***
([PDF](http://technicalanalysis.org.uk/support-and-resistance/Curcio-etal1997.pdf)) then walked this
back on better intraday data: profits existed **only during strongly trending sub-periods, were not
profitable on average, and essentially vanished once bid-ask spreads and transaction costs were
included.** They also re-ran the Brock et al. rules on their intraday data and found them
unprofitable there. This is the canonical "looked good, didn't survive costs" arc for level-based
rules.

**Brock, Lakonishok & LeBaron (1992), "Simple Technical Trading Rules and the Stochastic Properties
of Stock Returns," *Journal of Finance* 47(5), 1731–1764**
([EconPapers](https://econpapers.repec.org/RePEc:bla:jfinan:v:47:y:1992:i:5:p:1731-64);
[SFI working paper PDF](https://sfi-edu.s3.amazonaws.com/sfi-edu/production/uploads/sfi-com/dev/uploads/filer/17/91/1791d085-e000-427d-a5d2-94d20b072bcb/91-01-006.pdf)).
DJIA, 1897–1986, 25,806 daily observations, testing moving-average and **trading-range-break
(i.e. support/resistance)** rules with bootstrap inference against random-walk, AR(1), GARCH-M and
EGARCH nulls. Variable-length MA buy signals earned **+0.042%/day (~+12% annualized)** vs. sell
signals **−0.025%/day (~−7% annualized)**; daily volatility was 0.89% in buy periods vs. 1.34% in
sell periods; average buy−sell spread across the 10 VMA rules was 0.00067. The fixed-length rule
gave a 10-day buy−sell spread of 0.77% (no band) / 1.09% (1% band). The authors' own caveat:
*"transactions costs should be carefully considered before such strategies can be implemented."*

**Sullivan, Timmermann & White (1999), "Data-Snooping, Technical Trading Rule Performance, and the
Bootstrap," *Journal of Finance* 54(5)**
([PDF](https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Sullivan_Timmermann_White.pdf)).
This is the decisive re-examination. Applying White's Reality Check bootstrap to an expanded
universe of **7,846 rule parameterizations** (vs. BLL's 26) on 100 years of DJIA data, they found
BLL's in-sample result *does* survive the data-snooping adjustment — in all four BLL sub-periods —
and that some unreported rules performed even better. **But the best in-sample rule produced no
superior performance in the 1987–1996 out-of-sample period; the results "are completely
reversed."** They interpret this as increasing market efficiency. The lesson for HAWK is the
methodological one: a rule selected as best-of-N in-sample tells you nothing until it is tested
forward, and the number of rule variants you searched over determines how large an in-sample
t-statistic has to be before it means anything.

### Implication for HAWK

Do **not** spend effort trying to validate the order-block construct as such; there is no prior to
build on and the grey-literature tests are null. Two concrete, cheap tests follow instead.

First, `distance_to_swing` is currently used with the **opposite sign to what the level literature
implies.** Osler's mechanism (TP clustering at round numbers, SL clustering just beyond) says
unusual conditional behaviour happens *at* obvious levels; HAWK's filter requires the bar to be
≥0.5% *away* from the prior-60-bar extreme, discarding those bars. Test `distance_to_swing` as a
continuous conditioning variable with the filter relaxed to 0 — i.e. does forward ±1% asymmetry
vary monotonically with proximity to the swing extreme, and specifically is the *near-extreme*
bucket (the one currently excluded) where any signal lives? This is the highest-information single
test available from existing features, and it is free.

Second, HAWK has the raw ingredients for a genuine round-number test that the level literature
directly motivates: from 15m OHLCV, compute for each zone the distance from `zone_midpoint` (and
from the ±1% barriers) to the nearest round price level at the coin's natural scale, and test
whether outcome asymmetry depends on whether the barrier sits just before or just past a round
number. This tests Osler's mechanism rather than the SMC narrative. Be aware the crypto analogue of
"round number" is scale-dependent across an 80-coin universe (BTC at 10^4 vs. a coin at 10^-4), so
the level grid must be defined per-symbol by order of magnitude, and that this is a
multiple-comparison trap — pre-register the grid definition.

Third, and non-negotiable given Sullivan/Timmermann/White: whatever variant count you search over
for a direction rule, record it, and apply a multiplicity-aware criterion. HAWK's current position
(AUC ≈ 0.50 out-of-sample on stored features) is exactly the BLL→STW arc's endpoint arrived at
early, which is the cheap way to learn it.

Do **not** interpret the ~83% retest rate as validation of anything. Before it can mean anything it
needs a matched control: the same-direction, same-width band placed at the same distance from price
on a randomly chosen non-signal bar. If a random band is also retested ~83% of the time — which is
the null expectation — the statistic is measuring 15m crypto volatility, not order flow.

---

## Topic 2 — Abnormal volume as a return predictor

### 2a. The volume–volatility relation is the robust finding; volume–signed-return is not

**Karpoff (1987), "The Relation Between Price Changes and Trading Volume: A Survey," *Journal of
Financial and Quantitative Analysis* 22(1), 109–126**
([RePEc](https://ideas.repec.org/a/cup/jfinqa/v22y1987i01p109-126_01.html)). The canonical survey
establishes two relations: volume is positively related to the **magnitude** of the price change
(|Δp|) in all markets, and — in equity markets only — to the signed price change. The first is
among the most replicated findings in empirical finance; the second is weaker and market-specific.
The clean reading is: *a volume spike tells you the next move will be large, not which way.*

**Jones, Kaul & Lipson (1994), "Transactions, Volume, and Volatility," *Review of Financial
Studies* 7(4), 631–651** ([DOI](https://doi.org/10.1093/rfs/7.4.631)) sharpens this materially: the
volatility–volume relation is really a volatility–**number-of-transactions** relation. Trade *size*
has no information content beyond trade *frequency*. This is directly actionable for HAWK, because
`volume_ratio` is a size-of-volume measure and the more informative variable per JKL is the trade
count, which Binance klines expose alongside OHLCV.

### 2b. The high-volume return premium (equities): real, replicated, but wrong horizon

**Gervais, Kaniel & Mingelgrin (2001), "The High-Volume Return Premium," *Journal of Finance*
56(3), 877–919**
([DOI](https://doi.org/10.1111/0022-1082.00349);
[Wharton WP PDF](https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9901.pdf)).
NYSE stocks, CRSP daily data, **1963-08-15 to 1996-12-31**, 161 non-overlapping 50-trading-day
intervals. Stocks with unusually high (low) volume over one day to one week subsequently appreciate
(depreciate) over the following month. Effect sizes for the zero-investment high-minus-low-volume
portfolio, **net of a trading-cost adjustment**: +0.35% (small caps) to +0.12% (large caps) after
1 day, and +0.94% to +0.50% after 20 trading days; t-statistics generally ≥ 3. Weekly-formation
20-day returns reach +1.36% (small) / +0.90% (large). The effect is *stronger* when the volume
shock is **not** accompanied by an extreme return, is not explained by return autocorrelation,
announcements, market risk or liquidity, and the authors attribute it to a visibility/investor-
attention channel.

Three caveats that matter for HAWK: (i) this is a **cross-sectional** ranking effect — high-volume
stocks vs. low-volume stocks — not a statement that any single high-volume bar predicts its own
continuation; (ii) the formation window is 1 day to 1 week and the holding period is up to a month,
i.e. **two to three orders of magnitude longer than HAWK's 15m bar**; (iii) the magnitudes are
tenths of a percent, comparable to or below realistic crypto round-trip costs at HAWK's frequency.

**Lee & Swaminathan (2000), "Price Momentum and Trading Volume," *Journal of Finance* 55(5),
2017–2069** ([DOI](https://doi.org/10.1111/0022-1082.00280);
[PDF](https://www.lsvasset.com/pdf/research-papers/Price-Momentum-Trad-Vol-2000.pdf)) gives the
interaction: past volume predicts the magnitude *and persistence* of momentum. High-volume winners
and losers show stronger 12-month momentum (a high-volume momentum strategy beats plain momentum
by **2–7%/year**), but **high-volume winners reverse faster**; momentum reverses over years 2–5.
The interpretation — volume proxies investor interest and therefore the *imminence of reversal* —
is the single most relevant piece of intuition for HAWK: a volume spike is evidence of crowding,
and crowding precedes reversal. Note this result has been substantially challenged:
**Chen, Kim, Ma & Zhang (2021), "Momentum life cycle, revisited," *Journal of Banking & Finance*
127** ([RePEc](https://ideas.repec.org/a/eee/jbfina/v127y2021ics0378426621000777.html)) argues the
momentum life cycle is largely a mechanical product of two separate effects, and that what remains
disappears under risk adjustment or with NASDAQ included. Cite Lee–Swaminathan for the *hypothesis*,
not as settled fact.

### 2c. Crypto-specific evidence: the cross-sectional sign is *reversed*

This is the most decision-relevant finding in this section.

**Liu, Tsyvinski & Wu (2022), "Common Risk Factors in Cryptocurrency," *Journal of Finance* 77(2),
1133–1177** ([working paper PDF](https://economics.yale.edu/sites/default/files/2022-10/LiuTsyvinskiWu2019%20COMMON%20RISK%20FACTORS.pdf))
sorts coins weekly into quintiles on volume measures. The dollar-volume long-short spread is
**−3.2% per week, significant at 5%** — that is, **low**-dollar-volume coins *outperform* high-
dollar-volume coins, the **opposite sign to GKM in equities**. The volume and scaled-volume
variants are not statistically significant. Crucially, the authors show the dollar-volume-volatility
premium is **subsumed by the crypto size factor** — dollar volume is mostly a proxy for coin size
and price level. So the apparent crypto "volume premium" is a small-cap/illiquidity effect wearing
a volume costume, and it runs against, not with, the equity result.

**Balcilar, Bouri, Gupta & Roubaud (2017), "Can volume predict Bitcoin returns and volatility? A
quantiles-based approach," *Economic Modelling*
([RePEc](https://ideas.repec.org/p/hal/journl/hal-02008551.html))**: nonparametric
causality-in-quantiles on BTC, 2011-12-19 to 2016-04-25. Volume predicts returns **except in bear
and bull regimes** (i.e. only in the middle of the conditional distribution) and **cannot predict
volatility at any quantile**. Note this last result contradicts the equity volume–volatility
consensus and is itself contested.

**Bouri, Lau, Lucey & Roubaud (2019), "Trading volume and the predictability of return and
volatility in the cryptocurrency market," *Finance Research Letters* 29, 340–346**
([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612318303519)): seven
major coins, daily data, copula-quantile causality. Volume Granger-causes **extreme negative and
extreme positive returns** for all seven — i.e. it predicts the *tails*, both sides, which is again
a magnitude result, not a direction result. Volume Granger-causes volatility for only 3 of 7 coins
and only under a squared-return proxy, not GARCH.

**Fousekis & Grigoriadis (2021), "Directional predictability between returns and volume in
cryptocurrencies markets," *Studies in Economics and Finance* 38(4), 693–711**
([RePEc](https://ideas.repec.org/a/eme/sefpps/sef-08-2020-0318.html)): cross-quantilogram on BTC,
ETH, XRP, LTC daily data. **Low volume carries no information about future returns; high volume
tends to precede extreme positive returns.** The dominant direction of causality, though, runs
*returns → volume*: extreme returns (either sign) lead high volume. That is the ordering you would
expect if volume is a consequence of price moves rather than a leading indicator.

None of these crypto papers work at intraday frequency, and none report transaction-cost-adjusted
strategy returns at anything like a 15-minute bar.

### Implication for HAWK

`volume_ratio` at 1.35× over a 60-bar trailing mean is, on this evidence, a **volatility
forecaster, not a direction forecaster.** That is not useless — it is exactly what you want for
sizing, barrier placement, and for deciding *whether* a ±1% barrier is even a meaningful distance
for this symbol at this moment — but do not expect it to signal sign. Concretely:

1. **Reframe `volume_ratio` as a volatility feature.** Test whether it predicts realized
   absolute return over the next 1/4/24h and the *time* to first barrier touch, not the sign. If it
   does (very likely, per Karpoff), use it to normalize the ±1% barriers into volatility units
   (e.g. barriers at ±k × forward-expected-σ), which will make the 47–51% hit rate interpretable
   for the first time. Right now a fixed ±1% barrier mixes low-vol and high-vol symbols and states,
   which alone can wash out a real effect.
2. **Do not port GKM to HAWK as a continuation signal.** Its horizon (20 days), unit of analysis
   (cross-sectional stock ranking), and effect size (0.12–0.94%) make a 15m intrabar translation
   unjustified. The one legitimate GKM-flavoured test HAWK *can* run is cross-sectional and belongs
   to Topic 5: rank the ~80 symbols by `universe_quote_volume` shock and check forward relative
   returns at 4h/24h. Given Liu–Tsyvinski–Wu, **expect the negative sign** (high-volume coins
   underperform) and test that direction explicitly rather than assuming the equity sign.
3. **Add trade count if it is cheap.** Per Jones/Kaul/Lipson, the informative activity variable is
   number of trades, not traded quantity. Binance klines already return trade count and taker-buy
   base/quote volume. Taker-buy share is a *signed* order-imbalance proxy and is, on the
   microstructure evidence Osler and the order-flow literature rest on, far more plausibly
   directional than anything currently in the feature set. This is the single highest-expected-value
   data addition: HAWK is currently trying to infer order flow from candle shape when the exchange
   publishes a direct measurement of it.
4. **Test the volume-as-crowding (reversal) hypothesis, not the volume-as-conviction
   (continuation) hypothesis.** Lee–Swaminathan and the crypto reversal literature (Topic 4) agree
   that high volume marks late-stage moves. Interact `volume_ratio` with `symbol_return_1h/4h` and
   ask whether high `volume_ratio` *strengthens* subsequent reversal of the recent move. That is a
   theory-motivated, single-hypothesis test rather than a fishing expedition.

---

## Topic 3 — Candlestick patterns and long-wick / rejection bars

### 3a. The equity evidence on candlesticks is predominantly negative

**Marshall, Young & Rose (2006), "Candlestick technical trading strategies: Can they create value
for investors?," *Journal of Banking & Finance* 30(8), 2303–2323**
([RePEc](https://ideas.repec.org/a/eee/jbfina/v30y2006i8p2303-2323.html);
[PDF](https://scispace.com/pdf/candlestick-technical-trading-strategies-can-they-create-2whgphf6vm.pdf)).
The methodologically strongest test in this family. 28 candlestick rules (single lines and
multi-bar patterns, bullish and bearish), 35 DJIA component stocks, **1992–2002**, with an
extension of the bootstrap that generates random *open, high, low and close* series jointly — which
is exactly the right null for a rule defined on intrabar geometry, because it preserves the
statistical structure of OHLC while destroying any pattern information. Results:

- **22 of 28 rules produced no statistically significant p-value on any of the 35 stocks.**
- The maximum count for any rule was **3 significant stocks out of 35** (White Shooting Star) —
  entirely consistent with multiple-testing noise at 28 × 35 = 980 tests.
- **Mean profit following all bullish single lines was *higher* on the random bootstrap series than
  on the actual series** — i.e. the point estimate goes the wrong way.

The authors note the sample is a genuine out-of-sample test of the technique: candlesticks were
developed on 18th-century Japanese rice data, and by 1992–2002 Western practitioners had both the
knowledge and the software to trade them, so this is a direct efficiency test, not a data-mined
in-sample exercise.

**Horton (2009), "Stars, crows, and doji: The use of candlesticks in stock selection," *Quarterly
Review of Economics and Finance* 49(2), 283–294**
([RePEc](https://ideas.repec.org/a/eee/quaeco/v49y2009i2p283-294.html)). Nine candlestick patterns,
349 S&P 500 stocks. Explicitly contradicts an earlier positive paper, finds "little value" in
candlesticks, and concludes that using Stars, Crows or Doji for individual stock selection is not
recommended. Supports weak-form efficiency.

**Lo, Mamaysky & Wang (2000), "Foundations of Technical Analysis: Computational Algorithms,
Statistical Inference, and Empirical Implementation," *Journal of Finance* 55(4), 1705–1765**
([PDF](https://web.mit.edu/Alo/www/Papers/1705-1765.pdf);
[NBER w7613](https://www.nber.org/papers/w7613)). Often mis-cited as validating technical analysis.
What it actually shows: using nonparametric kernel-regression pattern recognition on US stocks
**1962–1996**, the conditional return distribution following **7 of 10** geometric patterns differs
significantly from the unconditional distribution on NYSE/AMEX (exceptions: broadening bottom,
triple top, double bottom), and **all 10** on Nasdaq. This is a *distributional-information* result,
not a profit result. The authors themselves note that "patterns that are optimal for detecting
statistical anomalies need not be optimal for indicating trading profits, and vice versa." In the
published discussion attached to the paper, Jegadeesh concludes that the evidence "does not support
the use of technical analysis as a tool to independently identify profitable trading opportunities."
Also relevant: the patterns are **multi-bar geometric shapes** (head-and-shoulders, double bottoms),
not single-bar wick bars.

Counter-evidence exists but is weaker and market-specific: e.g. **Lu, Shiu & Liu (2012),
"Profitable candlestick trading strategies — The evidence from a new perspective," *Review of
Financial Economics* 21(2)** ([DOI](https://doi.org/10.1016/j.rfe.2012.02.001)) finds some
profitability on Taiwanese data. The pattern across the literature is that positive candlestick
results cluster in smaller, less liquid, emerging markets and do not replicate on liquid
large-cap markets.

### 3b. Crypto: the one genuinely positive, data-snooping-controlled result

**Moser & Brauneis (2026), "Intraday price forecasts using candlestick patterns in cryptocurrency
markets," *International Review of Economics & Finance* 108**
([DOI](https://doi.org/10.1016/j.iref.2026.105158);
[RePEc](https://ideas.repec.org/a/eee/reveco/v108y2026ics1059056026002716.html)). This is the most
directly relevant paper in the entire review and should be read in full by the team. **Hourly** OHLC,
**07/2018–01/2022**, ~400 cryptocurrencies, 2,000 trading pairs, 36 exchanges, **200 million
observations**, 55 TA-Lib reversal patterns, with profitability assessed via the **stepwise Superior
Predictive Ability test** to control data-snooping across the 55-pattern family. Findings:

- Bullish Harami and Hikkake precede statistically significant positive returns; **Bearish Harami
  and Hanging Man** precede significant declines.
- Results are stable across time, market regimes, coins and exchanges, and are **"unaffected by
  volume or volatility."** That last clause is important and unfavourable for HAWK: it means
  conditioning a candlestick signal on a volume spike did **not** improve it in the largest crypto
  candlestick study available.
- The surviving patterns are mostly **multi-bar** (Harami, Hikkake are two/three-bar constructs).
  The exception, Hanging Man, is exactly a single-bar **long-lower-wick, small-body** bar — the
  closest published analogue to HAWK's signal bar.

### 3c. Does wick *direction* matter? Yes — and it is the whole content of the signal

This is the crux for HAWK. In every candlestick taxonomy the geometry HAWK measures maps to
patterns whose predicted sign depends entirely on two things it currently throws away:

| Pattern | Geometry | Prior trend | Predicted sign |
|---|---|---|---|
| Hammer | long **lower** wick, small body | after downtrend | bullish |
| Hanging Man | long **lower** wick, small body | after uptrend | **bearish** |
| Inverted Hammer | long **upper** wick, small body | after downtrend | bullish |
| Shooting Star | long **upper** wick, small body | after uptrend | **bearish** |

Hammer and Hanging Man are the *same bar* with *opposite* predictions; so are Inverted Hammer and
Shooting Star. The taxonomy carries information only through (a) which side the wick is on and
(b) the preceding trend. HAWK's `wick_ratio = max(upper, lower) / |close - open|` **collapses the
first dimension**, and the detector conditions on nothing about the preceding trend, so it also
drops the second. Meanwhile HAWK's `direction` label is `close > open` — the body sign — which in
the candlestick literature is the *least* informative component of a small-body bar and is nearly
arbitrary when the body is small by construction (`wick_ratio >= 1.2` guarantees a small body). A
`wick_ratio` of 1.2+ with a tiny body means the close/open ordering is close to a coin flip on
microstructure noise. **This is a sufficient mechanistic explanation for the observed 47–51% barrier
hit rate and the AUC ≈ 0.50 — the direction label is very close to independent of anything.**

Honest verdict on single-bar reversal candles: the equity evidence says no tradable edge
(Marshall/Young/Rose, Horton); the best crypto evidence (Moser & Brauneis) says a *small number* of
signed, trend-conditioned reversal patterns survive data-snooping control at hourly frequency, with
Hanging Man the only single-bar survivor, and that volume conditioning adds nothing. Anyone claiming
a specific win rate for "rejection candles" in general is stating **unsupported folklore**.

### Implication for HAWK

This is the cheapest high-value change in the whole review and it requires no new data.

1. **Split `wick_ratio` into two signed features immediately:** `upper_wick_ratio = (high -
   max(open, close)) / range` and `lower_wick_ratio = (min(open, close) - low) / range`. Normalizing
   by `range` rather than by body avoids the pathological blow-up as `|close - open| → 0` that the
   current definition has (a near-doji produces an unbounded `wick_ratio`, which is almost certainly
   putting heavy-tailed garbage into the ML feature matrix and is a plausible contributor to the
   AUC ≈ 0.50). Keep the old feature for continuity but do not expect it to carry sign.
2. **Replace the `direction = close > open` label with a trend-conditioned, wick-side-conditioned
   hypothesis.** HAWK already stores `symbol_return_1h/4h/24h`, which is precisely the prior-trend
   context the Hammer/Hanging-Man distinction needs. The pre-registered hypothesis to test is the
   four-cell table above: does forward ±1% asymmetry differ across
   {upper-wick, lower-wick} × {prior 4h return positive, negative}? That is 4 cells, one test, and
   it directly tests the only claim in the candlestick literature with any support. Predicted signs
   from the taxonomy: upper wick after an up-move → bearish; lower wick after a down-move → bullish.
3. **Do not expect the volume filter to help the candlestick signal.** Moser & Brauneis explicitly
   report their patterns' predictive power is unaffected by volume. If HAWK's `volume_ratio >= 1.35`
   filter is justified, it will be as a *volatility/liquidity* gate (Topic 2), not as a confirmation
   of the wick pattern.
4. **Adopt the Marshall/Young/Rose null, not a random-walk-on-close null.** When testing any
   wick-geometry rule, the correct benchmark is an OHLC bootstrap that jointly resamples
   open/high/low/close so the null preserves intrabar geometry statistics. Testing against a
   close-only random walk will make a geometry rule look significant when it is not. This is the
   single most important methodological borrow available.
5. Recognize that `distance_to_price` is not an independent feature: with `midpoint = (h + l) / 2`,
   `|midpoint - close| / close` is a deterministic function of the same OHLC geometry that produces
   `wick_ratio`. Including both in a model adds collinearity, not information. Drop it or replace it
   with `(close - low) / (high - low)` (close location value), which is at least a standard,
   bounded, signed measure of where the bar closed in its range — and is the signed quantity the
   candlestick literature actually uses.

---

## Topic 4 — Short-horizon reversal vs. continuation in crypto (15m to 48h)

This is the best-evidenced family for HAWK's horizon, and the evidence is fairly consistent:
**at intraday-to-daily horizons crypto reverses; momentum appears only from ~1 week out, and
reversal vs. momentum flips sign with liquidity.**

### 4a. Intraday (the horizon HAWK actually trades)

**Wen, Bouri, Xu & Zhao (2022), "Intraday return predictability in the cryptocurrency markets:
Momentum, reversal, or both," *North American Journal of Economics and Finance* 62, 101733**
([DOI](https://doi.org/10.1016/j.najef.2022.101733);
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080253)). 5-minute Bitcoin data,
**2013-03-03 to 2020-05-31**, aggregated to hourly (and half-hourly in robustness). Both intraday
momentum and intraday reversal are present. In the half-hour specification, of the pairs with
significant in-sample predictability **and positive out-of-sample R²** (their "efficient" pairs),
**33 survived: 12 momentum, 21 reversal — reversal dominates roughly 2:1.** Predictability patterns
change materially in the presence of **large intraday price jumps**, around FOMC releases, with
liquidity level, and during COVID. A timing strategy on the intraday predictors beat always-long and
buy-and-hold on economic value. The authors attribute the momentum component to late-informed
investors and — critically for HAWK — attribute the reversal component, which they describe as
**"unique to the cryptocurrency market," to investors' overreaction to *non-fundamental*
information and overconfidence.** A high-volume, long-wick 15m bar is close to a textbook example
of a non-fundamental overreaction event.

Two important qualifications: this is a *time-of-day* predictability structure (the *i*-th
half-hour predicting the *j*-th), not a bar-shape signal, and it is single-asset BTC, not a
cross-section. Transaction costs are addressed through economic-value comparison rather than a
full cost model.

**Borgards & Czudaj (2020), "The prevalence of price overreactions in the cryptocurrency market,"
*Journal of International Financial Markets, Institutions & Money* 65, 101194**
([RePEc](https://ideas.repec.org/a/eee/intfin/v65y2020ics1042443120300780.html)). Twelve
cryptocurrencies vs. the US stock market, dynamic threshold definition of overreaction, tested at
**interday and multiple intraday frequencies including hourly**. Price overreactions are "highly
prevalent ... for all frequencies," **negative overreactions slightly more prevalent than positive**,
and the returns to an overreaction (i.e. contrarian) trading strategy are **significantly higher in
crypto than in US stocks, with overreaction magnitude the dominant driver of profitability.** This
is the most direct published support for a contrarian-after-shock rule at HAWK's frequency.

**Counter-evidence, and it should be taken seriously: Caporale & Plastun (2020), "Momentum effects
in the cryptocurrency market after one-day abnormal returns," *Financial Markets and Portfolio
Management* 34(3), 251–266** ([working paper](https://ideas.repec.org/p/ces/ceswps/_7917.html)).
BTC, ETH, LTC vs. USD, **2017-01-01 to 2019-09-01**, hourly returns. They find that on an
overreaction day, **prices tend to continue in the direction of the overreaction until the end of
the day, and the effect persists into the following day** — i.e. *momentum*, not reversal, at the
intraday-to-1-day horizon, with exploitable profits in a trading simulation. Contrarian behaviour
appears in only two of six cases (BTCUSD positive overreactions, ETHUSD negative overreactions).
So the sign at HAWK's exact horizon is **genuinely contested**: Borgards/Czudaj and Wen et al. lean
reversal, Caporale/Plastun lean continuation within the same day. The disagreement plausibly turns
on how the shock is defined (daily abnormal return vs. dynamic intraday threshold) and on sample
period. **Do not assume either sign; test both, and expect the answer to be conditional.**

### 4b. Daily to weekly cross-section: reversal, but only in illiquid coins

**Kozlowski, Puleo & Zhou (2021), "Cryptocurrency return reversals," *Applied Economics Letters*
28(11), 887–893** ([DOI](https://doi.org/10.1080/13504851.2020.1784831);
[open PDF](https://digitalcommons.fairfield.edu/cgi/viewcontent.cgi?article=1249&context=business-facultypubs)).
200 cryptocurrencies, **2015–2019**. Loser-minus-winner long-short spreads: **7.37% (t = 24.23)
daily, 10.03% (t = 7.03) weekly, 28.22% (t = 3.28) monthly.** Robust to size, turnover and
illiquidity controls, present in both sample halves and in both high- and low-implied-volatility
regimes. Reversals are **most pronounced among smaller-cap and less liquid coins**, consistent with
a mix of inefficiency and liquidity-provision compensation. Note: these are gross spreads on a
universe of 200 coins including very illiquid ones; the paper does not present a full
transaction-cost-net implementation, and at these frequencies in illiquid altcoins costs are
first-order.

**Zaremba, Bilgin, Long, Mercik & Szczygielski (2021), "Up or down? Short-term reversal, momentum,
and liquidity effects in cryptocurrency markets," *International Review of Financial Analysis* 78,
101908** ([DOI](https://doi.org/10.1016/j.irfa.2021.101908);
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1057521921002349)).
Daily prices, **>3,600 coins, 2015–2021**. The **last day's return is a powerful negative predictor** of the
next day's cross-sectional return: yesterday's losers outperform yesterday's winners. But the
effect is **liquidity-conditional** — the authors argue it results from illiquidity, and **"the
handful of largest and most tradeable coins exhibit daily momentum rather than a reversal."**

This liquidity-conditional sign flip is confirmed independently: **"Impact of size and volume on
cryptocurrency momentum and reversal"**
([working paper PDF](https://quantitative.cz/wp-content/uploads/2023/09/impact_of_size_and_volume_on_cryptocurrency_momentum_and_reversal.pdf))
reports the weekly reversal is **significant only for small, illiquid, low-volume coins
(Newey-West t = −7.31)**, while **large, liquid coins show weekly *momentum* (t = 2.33 at 1 week,
2.68 at 2 weeks)**. Beyond 4 weeks reversal reappears even for large coins but insignificantly
(t = −1.91 at 26 weeks). They also find distance from the 1-week high predicts **negatively** for
small/illiquid coins (t = −9.03) and **positively** for large/liquid coins (t = 4.93).

**Dobrynskaya, "Cryptocurrency Momentum and Reversal"**
([conference PDF](https://conference.hse.ru/files/download_file_ex?hash=FAE0AB2DC7A67656E89A0B1CB27D8C7D&id=3B5EE9A5-0B18-458A-9458-B4ED0F6C6664)),
2,000 largest coins: **positive momentum at horizons up to 2–4 weeks, significant and economically
large reversal beyond 4–6 weeks**, driven mainly by past losers, not explained by standard crypto
risk factors.

### 4c. How this differs from equities

In equities the canonical short-horizon reversal (Lehmann 1990; Jegadeesh & Titman 1995 contrarian
profits) is weekly-to-monthly and is largely a bid-ask bounce / liquidity-provision phenomenon,
while momentum lives at 3–12 months. Crypto compresses the whole term structure: **reversal at
hours to days, momentum at ~1–4 weeks, reversal again beyond a month.** The compression is
consistent with a retail-dominated, 24/7, high-volatility, thin-order-book market with no earnings
cycle. Practically, this means equity-calibrated intuitions about horizon are misleading by roughly
an order of magnitude, and the *liquidity interaction is stronger in crypto than in equities* —
liquidity is not a nuisance control here, it is the variable that determines the sign.

### Implication for HAWK

This is where to spend the testing budget. Four concrete tests, in priority order:

1. **Test reversal of the signal bar's own move as the direction rule, against HAWK's current
   body-sign rule.** Define `bar_return = (close - open) / open` on the signal bar and take the
   *opposite* side. The overreaction literature (Borgards/Czudaj; Wen et al.) predicts this beats
   the current label. Simultaneously test the continuation side (Caporale/Plastun), because the sign
   is contested. This is a single binary hypothesis with a clear pre-specified alternative, so it
   is statistically cheap.
2. **Interact direction with liquidity — this is the highest-information interaction available.**
   HAWK already has `universe_quote_volume` and `universe_rank`. Every paper in 4b finds the
   reversal/momentum sign **flips** between the liquid and illiquid halves of the universe. The
   pre-registered hypothesis: among HAWK's ~80 symbols, the low-`universe_quote_volume` tercile
   reverses and the high-`universe_quote_volume` tercile continues. If HAWK has been pooling all 80
   symbols, it has been averaging two opposite-signed effects toward zero, **which is an entirely
   sufficient explanation for AUC ≈ 0.50 even if both effects are real.** Test this before
   concluding there is no signal.
3. **Use `symbol_return_1h/4h/24h` as the overreaction magnitude, not as a momentum feature.**
   Borgards/Czudaj find overreaction *magnitude* is the dominant driver of contrarian profitability.
   So test forward outcome against the *size* of the prior 1h/4h move, with the direction rule being
   "fade it," and check monotonicity across magnitude deciles. Monotonicity is much harder to fake
   by chance than a single-threshold result and is the right robustness check here.
4. **Add time-of-day.** Wen et al.'s entire result is a time-of-day predictability structure, and
   crypto session effects (Asia/Europe/US overlap) are among the most replicated intraday
   regularities. HAWK has 15m timestamps for free; bucket by UTC hour before concluding anything
   about bar shape. Be disciplined: 24 buckets is 24 tests, so treat this as a control variable to
   condition on rather than a signal to mine.

What **not** to do: do not test a continuation rule on the largest, most liquid symbols and a
reversal rule on the illiquid tail and then report the better one. That is the exact
best-of-N-in-sample failure Sullivan/Timmermann/White documented. Pre-commit to the liquidity split
and the predicted sign in each half *before* looking.

---

## Topic 5 — Cross-sectional crypto factors

### 5a. The canonical references

**Liu, Tsyvinski & Wu (2022), "Common Risk Factors in Cryptocurrency," *Journal of Finance* 77(2),
1133–1177** ([DOI](https://doi.org/10.1111/jofi.13119);
[NBER w25882 PDF](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). Universe:
all coins with market cap > $1M (≈1,064 coins by 2019), weekly rebalancing. They construct crypto
analogues of ~24 equity price/market factors; **10 form significant long-short strategies**, and all
10 are captured by a **three-factor model: crypto market (CMKT), crypto size (CSMB), crypto momentum
(CMOM)**. Momentum effect sizes (top-minus-bottom quintile, per week, monotonic across quintiles):
**2.7% (1-week), 3.3% (2-week), 4.1% (3-week, significant at 1%), 2.5% (4-week)** — i.e. ~3%/week,
significant at 5%. Direction: **long recent winners, short recent losers** (continuation).
Two findings are directly load-bearing for HAWK:

- **Crypto momentum is much stronger among LARGE coins** (from their double-sort on size and
  3-week momentum). This reconciles with Topic 4's liquidity-conditional sign flip.
- The **dollar-volume long-short spread is −3.2%/week** (low volume outperforms high volume) and
  **CSMB accounts for the market-cap, price, max-day-price, dollar-volume and dollar-volume-
  volatility strategies.** In other words: volume-based sorts in crypto are size in disguise.

**Liu & Tsyvinski (2021), "Risks and Returns of Cryptocurrency," *Review of Financial Studies*
34(6), 2689–2727** ([RePEc](https://ideas.repec.org/a/oup/rfinst/v34y2021i6p2689-2727..html);
[NBER w24877 PDF](https://www.nber.org/system/files/working_papers/w24877/w24877.pdf)). BTC, ETH,
XRP. Crypto returns have **no exposure** to standard stock, macro, currency or commodity factors;
what does predict them is crypto-specific: **strong time-series momentum at daily and weekly
frequency** (a 1-standard-deviation-scale prior BTC return predicts a **+0.33%** next-day return in
their specification) and **investor-attention proxies** (a one-SD increase in Google searches for
"Bitcoin" → **+2.3%** two-week-ahead return; the top-minus-bottom attention quintile spread is
1.07% at 1 week). This is a *time-series* result on three large coins, and the horizon is daily to
weekly. Note the attention channel is conceptually the same mechanism GKM invoke for the equity
high-volume premium (Topic 2) — visibility driving subsequent demand.

### 5b. Does cross-sectional ranking work at HAWK's horizon? Partially — and one paper is directly on point

**Guo, Sang, Tu & Wang (2024), "Cross-cryptocurrency return predictability," *Journal of Economic
Dynamics and Control* 163, 104863** ([DOI](https://doi.org/10.1016/j.jedc.2024.104863);
[open working version](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7900&context=lkcsb_research)).
**This is the closest published analogue to HAWK's setting: minute-frequency Binance data, 30
coins.** Findings: lagged returns of *other* coins significantly predict a focal coin's return
**for up to ten minutes**, consistent with slow information diffusion / limited attention, and
**without reversal** over that window. Magnitudes: a one-SD increase in the pooled lagged return of
other coins raises an individual coin's return by **0.40–4.82 bps per minute (≈24–289 bps hourly)**.
Adaptive LASSO selects at least six other coins' lagged returns for every one of the 30 coins;
**BTC is a positive, significant predictor for every non-stablecoin**, with BNB and TRX next. A
long-short portfolio on this signal earned a reported **2.16% daily out-of-sample return after
accounting for Binance's highest transaction costs.**

Skeptical reading required: a 2.16%/day net-of-cost figure is implausibly large as a deployable
return and almost certainly reflects negligible capacity, no slippage/market-impact model, and a
short, favourable sample. Treat the **existence and sign of the minute-scale cross-coin lead-lag as
credible** (it is mechanically plausible, has an established equity analogue in Hou 2007 on
industry information diffusion, and survives LASSO selection) and treat the **magnitude as an upper
bound that will not survive real execution.** Related: **Jia, Wu, Yan & Liu (2023), "A seesaw
effect in the cryptocurrency market: Understanding the return cross predictability of
cryptocurrencies," *Journal of Empirical Finance* 74, 101428.**

### 5c. A cautionary note: crypto factor results frequently fail to replicate

The crypto MAX/lottery literature is the cleanest illustration and should calibrate how much any
single crypto factor paper is worth:

- **Grobys & Junttila (2021), "Speculation and lottery-like demand in cryptocurrency markets,"
  *Journal of International Financial Markets, Institutions and Money* 71, 101289**
  ([RePEc](https://ideas.repec.org/a/eee/intfin/v71y2021ics1042443121000081.html)): 20 coins
  excluding BTC, Jan 2016 – Dec 2019. High-MAX coins **underperform**; low-minus-high MAX quintile
  spread **exceeds 1.50%/week** (the equity-consistent negative MAX effect).
- **Ozdamar, Akdeniz & Sensoy (2021), "Lottery-like preferences and the MAX effect in the
  cryptocurrency market," *Financial Innovation* 7(1)**
  ([DOI](https://doi.org/10.1186/s40854-021-00291-9)): a much larger universe (17 growing to 523
  coins > $5M cap). High-MAX coins **outperform**; top-minus-bottom MAX decile spread **+3.03% raw
  / +1.99% risk-adjusted per week**, robust to size, price, momentum, short-term reversal,
  liquidity, volatility, skewness and sentiment controls. They explicitly note their results
  "contradict" Grobys & Junttila, Grobys & Sapkota (2019) and Jia et al. (2020), and attribute the
  difference to sample size.
- **"MAX momentum in the cryptocurrency market"**
  ([PDF](https://centaur.reading.ac.uk/99101/1/MAX%20Momentum.R1.pdf)) independently finds the
  *positive* sign and names it "MAX momentum."

So the same characteristic, in the same asset class, at the same weekly horizon, yields
**+3.03%/week in one paper and −1.50%/week in another**, with the disagreement traced to universe
construction and sample period. This is not an argument for ignoring the literature; it is an
argument for weighting HAWK's own carefully-constructed out-of-sample test above any single
published crypto factor result, and for expecting published crypto effect sizes to be biased
upward. The same caution applies to Kozlowski et al.'s 7.37%/day reversal spread: gross, on a
200-coin universe, with no execution model.

### Implication for HAWK

Cross-sectional ranking is HAWK's **most under-exploited** asset, because HAWK already computes a
universe rank across ~80 symbols on a common clock — which is precisely the data structure these
papers use, and is much harder to fake than a bar-shape signal.

1. **The single highest-prior test in this document: does BTC's (and ETH's) contemporaneous 15m
   return predict altcoin 15m returns, and does the signal-bar outcome depend on it?** Guo et al.
   find BTC positively predicts every non-stablecoin at minute scale. HAWK has raw 15m OHLCV for
   ~80 symbols, so `btc_return_15m`, `btc_return_1h` and the symbol's *residual* return
   (`symbol_return_1h` minus its beta-scaled BTC return) are computable today at zero data cost.
   Two hypotheses worth separating: (a) an order block is directionally informative only when it
   agrees with concurrent BTC direction (beta/spillover), and (b) the *idiosyncratic* part of the
   signal bar's move — the part not explained by BTC — is what reverses. Hypothesis (b) is the
   sharper one, and it is the natural union of Topic 4's overreaction-to-non-fundamental-information
   mechanism with Topic 5's spillover mechanism. **Right now HAWK's `symbol_return_1h/4h/24h` mixes
   market beta and idiosyncratic move, which will blur any reversal signal.** Decomposing them is a
   few lines of work and is the most likely source of a genuine improvement.
2. **Use `universe_rank` / `universe_quote_volume` as a size/liquidity conditioner with the sign
   pre-registered from the literature, not as a raw feature.** Liu/Tsyvinski/Wu: momentum is
   stronger in large coins, dollar-volume sorts are size in disguise, and low-dollar-volume coins
   earn more unconditionally at weekly horizon. Zaremba et al. and the size/volume paper: reversal
   in the illiquid tail, momentum in the liquid head. The pre-registered prediction is therefore a
   **sign flip across the liquidity split**, and a model that includes `universe_quote_volume` as a
   main effect without the interaction will miss it entirely. If HAWK's ML model included these as
   flat features, that is a second sufficient explanation for AUC ≈ 0.50.
3. **Do not expect a direct payoff from importing CSMB/CMOM.** HAWK's universe is ~80 Binance
   perp/spot symbols — the *liquid head* of the market, not the 1,000-coin long tail where size and
   reversal effects live. Weekly-horizon factor premia of 2–4% also do not translate to a 15m signal
   bar. What does translate is the *conditioning structure*: rank the 80 symbols cross-sectionally
   at the moment of the signal, and use rank rather than raw level.
4. **Test a cross-sectional relative form of the outcome.** Instead of asking "does this symbol hit
   +1% before −1%," ask "does this symbol outperform the median of the other 79 over the next
   1/4/24h." A market-relative target strips out the dominant common factor (which is BTC-driven
   and roughly 60–80% of altcoin 15m variance), massively improves the signal-to-noise ratio of the
   label, and matches what every paper in this section actually estimates. Given ~10 days of live
   data and a directionless ±1% barrier, HAWK's current label is mostly measuring market beta.
   **Changing the label is probably worth more than changing the features.**

---

## Closing: prior probabilities

Priors that each family shows a **real, cost-surviving directional edge in HAWK's own data** —
i.e. survives out-of-sample validation and realistic 15m round-trip costs on Binance, with the
multiplicity discipline Sullivan/Timmermann/White demands. These are priors *before* testing, not
conclusions.

| # | Signal family | Prior | Reasoning |
|---|---|---|---|
| 1 | Order blocks / SMC / supply-demand zones, as currently defined | **4%** | Zero peer-reviewed support, two independent grey-literature mechanical tests came back null (0/648 beat buy-and-hold; 0/54 profitable after 0.5 pip), and HAWK's `distance_to_swing >= 0.5%` filter actively excludes the levels where the one well-documented mechanism (Osler's stop/take-profit clustering) operates. The construct's own direction label is the sign of a deliberately small body. |
| 2 | Abnormal volume as a **direction** predictor | **10%** | The robust finding is volume → absolute return (Karpoff; and trade *count* rather than size per Jones/Kaul/Lipson), not volume → sign. GKM's high-volume premium is a 20-day cross-sectional equity effect of 0.12–0.94%, and the crypto cross-sectional analogue (Liu/Tsyvinski/Wu) has the **opposite sign** and is subsumed by size. As a **volatility** feature for barrier scaling my prior is ~85%, which is why `volume_ratio` should be kept — just not as a direction signal. |
| 3 | Candlestick / long-wick rejection bars | **14%** | Strongly negative equity evidence under the correct OHLC bootstrap null (Marshall/Young/Rose: 22/28 rules significant on zero stocks; bullish patterns did *better* on random data), and Horton concurs. The one credible positive is Moser & Brauneis's hourly crypto study with stepwise-SPA control, where a small number of patterns including the single-bar Hanging Man survive — but it also reports the patterns are *unaffected by volume*, undercutting HAWK's combination. Conditional on HAWK splitting the wick into signed upper/lower and conditioning on prior trend, I would raise this to ~22%; with the current `max(upper, lower)` definition, ~5%. |
| 4 | Short-horizon crypto reversal vs. continuation | **32%** | Best-evidenced family at HAWK's horizon, from multiple independent peer-reviewed studies (Wen/Bouri/Xu/Zhao: 21 reversal vs. 12 momentum among OOS-positive half-hour pairs; Borgards & Czudaj: overreactions prevalent at hourly frequency with contrarian profits larger than in equities; Kozlowski et al.: 7.37%/day gross reversal spread). Marked down from higher because the sign is contested at exactly the 1-day horizon (Caporale & Plastun find continuation), the published spreads are gross, and HAWK's liquid-head universe is the part of the market where the literature says reversal *weakens* or flips to momentum. |
| 5 | Cross-sectional crypto factors / universe ranking | **26%** | Liu/Tsyvinski/Wu is a *Journal of Finance* result with monotonic quintile sorts, and Guo et al. (2024) demonstrate minute-scale, cost-adjusted cross-coin predictability on Binance specifically — the right exchange, frequency and universe structure. Marked down because the documented horizons are weekly (factors) or ≤10 minutes (lead-lag) with little in between, the reported magnitudes are implausible as deployable returns, and the MAX-effect replication failure (+3.03% vs. −1.50% per week for the same characteristic) shows how fragile crypto factor findings are. Most of this 26% is the BTC-lead-lag / market-relative-label channel, not the size/momentum factors themselves. |

**Joint prior.** That *some* combination of these — most plausibly a market-relative label plus an
idiosyncratic-reversal direction rule, conditioned on liquidity and volatility — clears realistic
15m Binance round-trip costs out-of-sample: **~20%.** That HAWK's *current* detector-plus-body-sign
configuration does: **<5%.** The gap between those two numbers is the value of the work, and almost
all of it comes from three changes that require no new data source: sign the wick, decompose the
return into market and idiosyncratic components, and make the label market-relative.

**Non-negotiable methodological carry-overs from this literature.** (i) Use the
Marshall/Young/Rose joint-OHLC bootstrap as the null for any bar-geometry rule, not a close-only
random walk. (ii) Count and report every rule variant searched, and apply a Reality-Check-style or
stepwise-SPA multiplicity adjustment (Sullivan/Timmermann/White; Moser & Brauneis). (iii) Every
statistic about HAWK's zones — retest rate above all — needs a matched control drawn from
non-signal bars with the same geometry, or it is measuring volatility rather than signal.
(iv) Ten days of live data is far too short for any of this; the tests above should be run on
backfilled 15m history, and the detector is already written point-in-time, which makes that
feasible.
