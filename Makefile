.PHONY: help setup setup-ml setup-all test test-fast lint format typecheck clean smoke reproduce datasets-check gpu-check paper-skeleton dashboard docker-check

PYTHON ?= python
PIP    ?= pip

help:
	@echo "make setup       - install core + dev deps (no torch/PyG)"
	@echo "make setup-ml    - core + ML stack (torch, sb3, codecarbon)"
	@echo "make setup-all   - everything (torch, PyG, causal-learn, avalanche, dashboards)"
	@echo "make test        - full test suite (skips needs_dataset by default)"
	@echo "make test-fast   - unit tests only, no coverage"
	@echo "make lint        - ruff check"
	@echo "make format      - ruff format"
	@echo "make typecheck   - mypy"
	@echo "make smoke       - end-to-end smoke via CLI"
	@echo "make datasets-check - verify Kaggle CSVs are on disk"
	@echo "make reproduce   - re-run every headline experiment (Phases 1-6)"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

setup-ml:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[ml,drift,dev]

setup-all:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[all]

test:
	pytest -m "not needs_dataset and not gpu and not slow" --cov=cadence --cov-report=term-missing

test-fast:
	pytest tests/unit -q -x

lint:
	ruff check cadence tests

format:
	ruff format cadence tests

typecheck:
	mypy cadence

smoke:
	cadence config-show -c configs/ci.yaml
	cadence smoke -c configs/ci.yaml

datasets-check:
	cadence datasets-check -c configs/default.yaml

reproduce:
	@echo "Phase-by-phase reproduce pipeline lands as phases complete. See docs/results.md."

gpu-check:
	cadence gpu-check

paper-skeleton:
	$(PYTHON) -m scripts.build_paper_skeleton
	@echo "wrote docs/paper/skeleton.md — commit if you want the appendix in sync"

dashboard:
	streamlit run dashboard/app.py

docker-check:
	docker-compose config --quiet && echo "docker-compose config OK"

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
