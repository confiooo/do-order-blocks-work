# Experiments: the side studies

Three studies that came out of the same project but are not part of the order-block study:

| Study | Question | Result |
|---|---|---|
| [`zone_trades/`](zone_trades/README.md) | Can a trader profit from a zone's *range* (straddles, breakouts) instead of its direction? | No configuration beats random bars after costs |
| [`daily_fade/`](daily_fade/README.md) | Do the day's biggest movers pull back? (unrelated to order blocks) | −0.95% per trade out of sample |
| [`market_neutral/`](market_neutral/README.md) | Is there a market-neutral edge: relative-value pairs, funding/basis? (unrelated to order blocks) | 6,408 cells, none passing |

No headline number of the order-block study (rounds 1 and 2, Family A) depends on this code, and
`hawk-research run all` does not run it. The results above are the published ones; the code that
produced them is kept here unchanged so they stay reproducible, and every cell they evaluate is still
logged to the same trial ledger (`data/ledger.sqlite`).

Run from the repository root (the package is not installed with `hawk_research`):

```bash
hawk-research --data-dir data run s5                 # the zone trades read the round-1 zone dataset
python -m experiments --data-dir data run mn_ingest  # network: USD-M funding and 1h perpetual bars
python -m experiments --data-dir data run all        # every offline side-study stage
python -m experiments --data-dir data verify         # their 11 headline numbers
python -m experiments stages                         # the stage table
```

Tests: `pytest experiments/tests`.
