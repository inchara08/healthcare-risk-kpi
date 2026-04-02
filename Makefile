.PHONY: help setup up down pipeline download validate load features train score kpis report test lint fmt clean

PYTHON := python3
PIPELINE := $(PYTHON) scripts/run_pipeline.py

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	pip install -e ".[dev]"

up:  ## Start PostgreSQL + pgAdmin via Docker
	docker compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@docker compose exec postgres pg_isready -U pguser -d healthcare_risk || sleep 5

down:  ## Stop Docker services
	docker compose down

# ─── Pipeline stages ──────────────────────────────────────────────────────────

pipeline: up  ## Run full 8-stage pipeline end-to-end
	$(PIPELINE) all

download:  ## Stage 1: Download CMS SynPUF files
	$(PIPELINE) download

validate:  ## Stage 2: Validate raw data with pandera
	$(PIPELINE) validate

load: up  ## Stage 3: Load data into PostgreSQL
	$(PIPELINE) load

features:  ## Stage 4: Build feature parquet files
	$(PIPELINE) features

train:  ## Stage 5: Train readmission + LOS + high-cost models
	$(PIPELINE) train

score: up  ## Stage 6: Batch score and write to analytics.risk_scores
	$(PIPELINE) score

kpis: up  ## Stage 7: Compute KPIs and write to analytics.kpi_snapshots
	$(PIPELINE) kpis

report:  ## Stage 8: Generate weekly HTML report
	$(PYTHON) scripts/generate_report.py

# ─── Quality ──────────────────────────────────────────────────────────────────

test:  ## Run pytest with coverage
	pytest tests/ -v

lint:  ## Run ruff linter
	ruff check src/ tests/ scripts/

fmt:  ## Format with black
	black src/ tests/ scripts/

fmt-check:  ## Check formatting without modifying
	black --check src/ tests/ scripts/

# ─── Utilities ────────────────────────────────────────────────────────────────

clean:  ## Remove generated artifacts (keeps raw data)
	rm -rf data/processed/*.parquet
	rm -rf reports/*.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
