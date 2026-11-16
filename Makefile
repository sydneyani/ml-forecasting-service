PYTHON ?= python

.PHONY: install lint test train train-ci download register serve

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

download:
	$(PYTHON) -m forecasting.data.download

train:
	$(PYTHON) -m forecasting.training --config configs/train.yaml

train-ci:
	$(PYTHON) -m forecasting.training --config configs/ci.yaml

register:
	$(PYTHON) -m forecasting.tracking.register

serve:
	$(PYTHON) -m forecasting.serving

drift:
	$(PYTHON) -m forecasting.monitoring.drift
