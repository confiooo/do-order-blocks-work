# Daily fade (side study)

Stages: `python -m experiments run daily_fade` (`df_study`, `df_zone_trade`, `df_tp_stops`). Not
pre-registered; exploratory. Unrelated to order blocks: it tests whether the day's biggest movers
pull back.

**Rule.** At each UTC day's close, short the largest gainer and buy the largest loser among 84
symbols, entering at the next day's first 15-minute open. `df_study` compares a fixed list of exits
over 1,478 events (Sep 2024 - Sep 2026). `df_zone_trade` fixes a "rollback zone" when the candle
closes: the target is a retrace of that candle sized from the same kind of day **before 2026-01-01
only**. A rule is chosen on that fitting sample (capture at least 60%, target hit at least 45%,
positive mean net, then the highest capture) and applied once to 2026 data.
Costs: 24 bps round trip (Binance spot taker 10 bps per side plus a 4 bps spread).

**Result.** The chosen rule (`market_open|p50|time_24h|all_days`) on the 2026 test sample: 516
trades over 258 days, target hit 53.5% of the time and 61% of trades net-positive, yet mean net
−0.95% per trade, compounding to a 98.8% loss: the wins are small and the losses large. `df_tp_stops` scores 26
take-profit and stop variants on the fitting and test samples; every one has a negative mean
net return.

Two files in the original project's data folder, `rollback_features.parquet` and
`y2026_trades.parquet`, were produced by interactive analysis whose code was not kept. Both are derived from published
outputs and add nothing: `y2026_trades.parquet` holds the same 516 2026 test trades as
`zone_bot_test.parquet`, and `rollback_features.parquet` is a per-event feature table for the
1,478 events in `events.parquet`. They are not reproduced here.
