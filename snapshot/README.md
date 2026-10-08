# Live zones extract

`live_zones.parquet` is the `zones` table of the original live alerting system's database,
exported unchanged (1,543 zones detected live between 2026-09-07 and 2026-09-17, 35 columns:
symbol, times, zone prices, detector measurements, confidence, the BTC gate, and the outcome
the live tracker recorded). It contains market data only, no users, chats, orders or trades.

It lets anyone run the checks that tie the replay to the live system:

- `s3`: the replay reproduces 1,510 of the 1,543 live zones with every compared field identical;
- `s3b`: each of the 33 misses is attributed to a cause (history shorter than the detector's
  400-bar window);
- `s5` / `s5b`: the replay's retest, entry price and MFE/MAE match the live tracker.

The stages read the owner's full snapshot database instead when it is present
(`data/snapshot/db.sqlite`); the results are identical.
