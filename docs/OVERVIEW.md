# Overview

## The question

"Order blocks" are a popular idea in discretionary crypto trading: price zones where large
players supposedly left unfilled orders, which price later revisits and reacts to. This project
asks whether they predict anything that can be traded: on 76 Binance altcoins, 15-minute bars,
September 2024 to September 2026, after realistic costs, without look-ahead, and compared with
what random candles of the same kind would have done.

## How it was tested

1. **Replay without look-ahead.** Each detector is run bar by bar so a zone exists only from the
   bar at which it could have been known. A truncation test recomputes every feature with the
   future removed and finds no difference.
2. **Compare against matched random candles.** Every zone is paired with ordinary bars on the
   same symbol, within a day, with similar volume and volatility. A zone must beat its matched
   controls, not just zero, because busy bars behave differently from quiet ones.
3. **Trade it with costs.** Fixed strategy grids (enter at the close or on a retest; exit after a
   time or at a take-profit/stop), priced with exchange fees and slippage, at most five positions
   at a time.
4. **Guard against luck.** Monthly walk-forward folds with purging, a 20% holdout, intervals from
   a block bootstrap that respects the fact that one market move fires dozens of zones at once,
   and a deflated Sharpe ratio against every configuration recorded in the trial ledger.
5. **Test the test.** The pipeline must find a planted +25 bps edge (it measures +21.6 bps) and
   must flag a random-direction strategy as significant only about 5% of the time (4 of 60).

Details: [`methodology.md`](methodology.md).

## Results

**Round 1: the original detector.** The detector came from a live alerting system. It flags
high-volume, long-wick candles near a recent swing and calls them bullish or bearish by the
candle's colour. Over 152,266 zones:

- Direction does not beat matched candles: zone minus control, signed 4h return, −0.31 bps
  [−3.38, +2.69].
- Taken literally, the direction is slightly *wrong* before costs: −3.42 bps at 4h, interval
  excluding zero. Fading it is gross-positive by the same tiny amount, far below the ~14 bps it
  costs to trade.
- Every one of 24 configurations is significantly net-negative under MEXC futures costs, and the
  direction rule is net-negative in every BTC regime. 0 of 21 walk-forward months are
  net-positive. Deflated Sharpe 0.00 against 72 trials.
- The best leg found, bearish zones only with a retest entry and a 2:1 barrier, makes +10.4 bps
  gross and loses −8.86 bps per trade net. Same-bar barrier ambiguity barely matters here: 0.42%
  of its trades touch both barriers in one bar, and counting all of them as losses moves it to
  +9.1 bps gross and −10.1 bps net.

**Round 2: textbook order blocks.** Round 1's detector is not what traders mean by an order
block, so round 2 implemented the textbook definition (the last opposite-coloured candle before a
break of market structure; 3-bar fractal swings; fixed rules, nothing tuned) and ran it through
the same pipeline. Only 4.2% of its 218,217 zones are candles the round-1 detector also flagged.

- Direction does not beat matched candles: +0.50 bps [−2.17, +3.63] at 4h.
- Every configuration is net-negative; the best, with a fair-value gap and a retest entry, loses
  −7.6 bps per trade [−13.3, −2.0]. 3 of 21 months net-positive. Deflated Sharpe 0.00.

**What zones do carry:** a little more volatility. In the 4h after a zone, realised volatility is
higher than after matched candles (+2.35 bps for round 1, +0.77 bps for round 2, both intervals
excluding zero). That is information about *how much* price moves, not *which way*, and nothing
here turns it into a profitable trade.

**Family A: the pre-registered follow-up.** Two more careful systems specified in advance (a
maker limit at the far zone edge; zones against tightly matched controls) were both FALSIFIED:
S1 loses 33.5 bps per trade net. In S5 zones lose about 7 bps less than their controls, but both
lose more than 30 bps per trade.
[`archive/s1_s5_results.md`](archive/s1_s5_results.md).

**Side studies,** all negative: straddling the zone's range (five variants), fading the day's
biggest movers (−0.95% per trade out of sample), and market-neutral pairs and funding/basis
(6,408 cells, none passing). One breakout variant with a tight trailing stop shows small profits
after fees (+0.17% per trade; about +0.01% after the spread), but random controls do better, so it
is not a zone effect, and it rests on an assumed intrabar price path. Each study has a README under
[`experiments/`](../experiments/README.md).

## What this does and does not show

It shows that for this universe, timeframe, period and cost structure, neither detector's
direction carries information beyond a matched random candle, and no tested way of trading it
makes money. It does not show that order blocks fail on every timeframe or asset, with
higher-timeframe structure, with confluence filters or with maker execution in rounds 1 and 2;
those are untested. Rounds 1 and 2 were not pre-registered, so by the protocol's rules they are exploratory;
an exploratory search that finds nothing is still a search that found nothing.

## How much to trust it

The parts most worth checking are the cost model (modelled, not measured), the same-bar barrier
rule (pessimistic for longs, optimistic for shorts), the round-1 thresholds (hand-picked
for the live bot, not fitted to this data) and the deviations from the protocol in [`deviations.md`](deviations.md). Where the code
and the original documents disagree, [`PROVENANCE.md`](PROVENANCE.md) and the archive index say
so.
