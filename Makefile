# curate: recommender platform. Every target runs through uv so the same interpreter is used everywhere.
UV ?= uv
PY := $(UV) run python
SEED ?= 42
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: setup lint format test bench check train serve docker-build docker-run clean help

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Install the environment and fetch or build the datasets (cached under data/)
	$(UV) sync
	$(PY) -m curate.cli data --seed $(SEED)

lint: ## Ruff lint and format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Apply ruff formatting and safe fixes
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

test: ## Run the test suite
	$(UV) run pytest -q

bench: ## Train, evaluate, write benchmarks/results.json and results.md, then copy the tables into README.md
	$(PY) -m curate.bench --seed $(SEED)
	$(PY) -m curate.bench --update-readme

check: ## Compare benchmarks/results.json against benchmarks/thresholds.json (CI quality gate)
	$(PY) -m curate.bench --check

train: ## Fit the serving bundle into artifacts/
	$(PY) -m curate.cli train --seed $(SEED)

serve: ## Serve the API (trains the bundle first when artifacts/ is empty)
	$(PY) -m curate.cli serve --host $(HOST) --port $(PORT)

docker-build: ## Build the container image
	docker build -t curate:local .

docker-run: ## Run the container image on port 8000
	docker run --rm -p 8000:8000 curate:local

clean: ## Remove caches, artifacts and benchmark outputs (keeps downloaded data)
	rm -rf .pytest_cache .ruff_cache .hypothesis artifacts/* benchmarks/results.json benchmarks/results.md
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
