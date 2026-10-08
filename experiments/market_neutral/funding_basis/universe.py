"""The twelve names from relative-value phase 1. Family B does not re-rank."""

PHASE1_NAMES = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "PEPEUSDT",
    "SUIUSDT",
    "TRXUSDT",
    "ADAUSDT",
    "ENAUSDT",
    "LINKUSDT",
)

# Phase-1 backtest book. PEPEUSDT has no USD-M contract under that ticker.
FAMILY_B_NAMES = tuple(name for name in PHASE1_NAMES if name != "PEPEUSDT")

PEPE_EXCLUSION_NOTE = (
    "PEPEUSDT is excluded from Family B phase 1. Binance USD-M has no contract "
    "under that name. 1000PEPEUSDT was not remapped onto it."
)

# A name needs this much tier-D overlap, on both funding and perpetual prices,
# before it can enter a Family B study. Shorter histories stay on disk and are
# marked excluded in the ingestion log.
MIN_DEVELOPMENT_DAYS = 60
