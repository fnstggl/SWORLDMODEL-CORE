.PHONY: install test lint typecheck banxico banxico-eval synthetic clean all check

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

typecheck:
	$(PY) -m mypy

# Run the Banxico pastcast and write the SEALED pre-outcome artifacts.
# Uses the deterministic offline backends unless SWORLDMODEL_MODEL_API_KEY is set.
banxico:
	$(PY) -m sworldmodel banxico run

# Compare the sealed pre-outcome forecast against the known result.
# Refuses to run until the pre-outcome artifact exists and is hashed.
banxico-eval:
	$(PY) -m sworldmodel banxico evaluate

# Generalization proof: synthetic committees through the same runtime.
synthetic:
	$(PY) -m sworldmodel synthetic run

check: lint typecheck test

all: check banxico

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
