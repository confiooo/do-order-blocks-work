# Common tasks. Every target is a thin wrapper around a documented command.
PY ?= python
DATA ?= data

.PHONY: install test test-all lint format smoke download reproduce verify coverage clean-smoke experiments

install:          ## editable install with dev tools
	$(PY) -m pip install -e ".[dev]"

test:             ## unit tests (no data, no network), under a minute
	$(PY) -m pytest -q -m "not integration"

test-all:         ## unit + integration tests (integration runs the smoke pipeline, ~2-5 min)
	$(PY) -m pytest -q

lint:
	ruff check src tests experiments
	ruff format --check src tests experiments
	mypy

format:
	ruff check --fix src tests experiments
	ruff format src tests experiments

coverage:
	$(PY) -m pytest -q -m "not integration" --cov=hawk_research --cov-report=term

smoke:            ## end-to-end run on a synthetic market, under 5 minutes
	hawk-research smoke

download:         ## OHLCV (~200 MB) and the point-in-time universe, all public Binance data
	hawk-research --data-dir $(DATA) download --pit-universe
	hawk-research --data-dir $(DATA) verify-data

reproduce:        ## every offline stage, in dependency order, then compare headline numbers
	hawk-research --data-dir $(DATA) run all
	hawk-research --data-dir $(DATA) verify

verify:
	hawk-research --data-dir $(DATA) verify

experiments:      ## the side studies (experiments/): market-neutral inputs, every stage, their headline numbers
	$(PY) -m experiments --data-dir $(DATA) run mn_ingest
	$(PY) -m experiments --data-dir $(DATA) run all
	$(PY) -m experiments --data-dir $(DATA) verify
