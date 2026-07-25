.PHONY: install test lint typecheck clean check

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

check: lint typecheck test

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
