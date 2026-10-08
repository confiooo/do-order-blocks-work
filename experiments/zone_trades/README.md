# Zone trades (side study)

Stages: `python -m experiments run zone_trades` (`zt_harvest`, `zt_width`, `zt_symsl`, `zt_gainbuf`,
`zt_oco`, `zt_breakout`). Not pre-registered; exploratory.

**Question.** Instead of predicting direction, can a trader profit from the zone's *range*? Each
script trades the round-1 alerts (the zones the live system would have sent, 73,047 of them) with a
different mechanical rule, against matched random bars where a control is defined. Costs: MEXC API
taker fee 6 bps per fill (`config/costs.json`), with the spread shown separately where reported.

| Script | Rule | Result |
|---|---|---|
| `two_way_harvest` | Long and short at the close, take-profits at both zone edges, stops just beyond | −23.2 bps per alert after fees; zones minus random bars −0.08 bps [−0.49, +0.29] |
| `two_way_width_sweep` | Same straddle around the alert price with synthetic widths | No width and horizon profitable |
| `two_way_symmetric_sl` | Straddle with a symmetric remaining stop | No configuration profitable |
| `two_way_gain_buffer` | Straddle; after the first take-profit, lock part of the gain | No configuration profitable |
| `two_way_oco_fast` | Straddle; after the first take-profit, a resting limit plus an emergency stop | No configuration profitable |
| `breakout_trail` | Enter on the first break of a zone edge, trail a stop | Positive after fees for zones **and for random controls**; see below |

**The breakout result is not a zone edge.** With a 0.25% trailing stop the zone arm makes +0.17% per
trade after fees, but random controls make +0.20%. After the study's own 0.16% spread estimate the
zone arm is about +0.01%. The simulation fills trailing stops on 15-minute bars with an assumed
intrabar path (up bars open-low-high-close, down bars open-high-low-close); with a stop this tight
the result depends on that assumption and has not been checked on 1-minute data. Treat it as an
untested lead about trailing stops in general, not about order blocks.

Same-bar ambiguity in these scripts is resolved pessimistically for the trade (see each script's
docstring), and `ambiguous_pct` is reported where it applies.
