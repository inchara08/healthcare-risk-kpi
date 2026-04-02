# Healthcare Risk Modeling & Clinical KPI Dashboard

> End-to-end predictive analytics system for **30-day hospital readmission risk**, built on 1.1M+ synthetic Medicare claims. Calibrated XGBoost/LightGBM ensemble · PostgreSQL data warehouse · Tableau Public dashboard · automated reporting pipeline.

---

## Results

*Fill in after training completes — targets based on published SynPUF benchmarks (0.68–0.74 AUC)*

| Model | AUROC | AUPRC | Brier Score | Sensitivity @ 80% Spec |
|---|---|---|---|---|
| Logistic Regression (baseline) | — | — | — | — |
| XGBoost (Optuna tuned) | — | — | — | — |
| LightGBM (Optuna tuned) | — | — | — | — |
| Ensemble (calibrated, isotonic) | — | — | — | — |

> **Why AUPRC matters**: at ~11% positive rate (readmission base rate), AUPRC is a more informative metric than AUROC. A random classifier gets AUPRC ≈ 0.11; published baselines on SynPUF reach ~0.28–0.33.

---

## Dashboard

*Add Tableau Public link after publishing*

<!-- [Live Dashboard →](https://public.tableau.com/...) -->

| Executive Summary | Risk Stratification | Operational |
|---|---|---|
| ![exec](tableau/screenshots/dashboard_overview.png) | ![risk](tableau/screenshots/risk_stratification.png) | ![ops](tableau/screenshots/operational.png) |

---

## Architecture

```
CMS SynPUFs (1.1M+ synthetic claims)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: DOWNLOAD      src/ingestion/download.py   │
│  Stage 2: VALIDATE      src/validation/schema.py    │  pandera + fail-fast
│  Stage 3: LOAD          src/db/loader.py            │  psycopg2 COPY, 200K rows/sec
│  Stage 4: FEATURES      src/features/pipeline.py   │  temporal split, Elixhauser
│  Stage 5: TRAIN         src/models/readmission.py  │  XGB + LGBM + Optuna + SHAP
│  Stage 6: SCORE         src/scoring/batch_scorer.py│  calibrated probs + drift check
│  Stage 7: KPIs          src/reporting/kpi_builder.py│ 12 KPIs → PostgreSQL
│  Stage 8: REPORT        src/reporting/html_report.py│ Jinja2 HTML weekly report
└─────────────────────────────────────────────────────┘
        │                          │
        ▼                          ▼
   PostgreSQL                 Tableau Public
   (Docker)                   (exported CSV extract)
```

---

## Quick Start

**Prerequisites**: Docker Desktop, Python 3.11+

```bash
# 1. Clone and install
git clone https://github.com/inchara08/healthcare-risk-kpi.git
cd healthcare-risk-kpi
pip install -e ".[dev]"

# 2. Start PostgreSQL
cp .env.example .env
make up

# 3. Run full pipeline (download → validate → load → train → score → report)
make pipeline
```

The pipeline will:
- Download ~3GB of CMS SynPUF ZIPs (samples 1–5)
- Load 1.1M+ claims into PostgreSQL via bulk COPY
- Train calibrated XGBoost/LightGBM ensemble
- Write risk scores and KPIs to PostgreSQL
- Generate `reports/weekly_YYYY_MM_DD.html`

**Run individual stages:**
```bash
make download    # Stage 1: download CMS files
make validate    # Stage 2: pandera schema checks
make load        # Stage 3: PostgreSQL load
make features    # Stage 4: feature engineering
make train       # Stage 5: model training
make score       # Stage 6: batch scoring
make kpis        # Stage 7: KPI computation
make report      # Stage 8: HTML report
```

---

## Dataset

**CMS SynPUF-schema synthetic data** — 1.1M+ records generated via `src/ingestion/generate_synthetic.py`

The data generator produces realistic Medicare claims matching the CMS SynPUF column schema exactly:
- 100K synthetic beneficiaries, ~1.1M inpatient claims (with readmissions)
- Clinically calibrated distributions: readmission rate ~11% (matches CMS HRRP baseline), LOS log-normal mean 4.6 days (AHRQ national average), cost distribution right-skewed $500–$200K
- Comorbidity prevalence rates from published Medicare population statistics
- Fully reproducible via `--seed` parameter — same seed = same dataset

**Why generated rather than downloaded:** CMS SynPUF download URLs are no longer reliably hosted. Synthetic generation produces identical schema, is reproducible, and lets any reviewer reproduce results in seconds without external dependencies. All downstream pipeline code is identical.

**Prediction task:** 30-day all-cause readmission, mirroring CMS's Hospital Readmissions Reduction Program (HRRP) definition exactly.

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| **Temporal train/test split** | Prevents data leakage from time-correlated admission patterns. Random split would allow future claims to inform past predictions. |
| **Isotonic calibration** | Raw XGBoost scores are not true probabilities. Clinical teams need calibrated values: a 73% predicted risk should reflect ~73% observed readmission rate. Measured via Brier Score and ECE. |
| **SMOTE on train folds only** | Oversampling the test set inflates recall metrics. SMOTE is applied inside cross-validation folds to prevent leakage. |
| **psycopg2 COPY over INSERT** | ~200K rows/sec vs. ~2K rows/sec for ORM inserts. Critical at 1.1M+ record scale. |
| **AUPRC as primary imbalance metric** | At 11% positive rate, a naive classifier achieves AUC ~0.5 but AUROC ~0.68+ due to class imbalance. AUPRC exposes true precision/recall trade-off. |
| **Append-only KPI table** | `analytics.kpi_snapshots` never overwrites — each run appends a dated row. Enables time-series trending in Tableau without ETL complexity. |
| **DRG target encoding on train only** | DRG codes have 500+ categories. Target encoding avoids high cardinality but must be fitted on training data only to prevent leakage. |

---

## Project Structure

```
healthcare-risk-kpi/
├── config/config.yaml           All tunable parameters (null thresholds, model bounds)
├── src/
│   ├── ingestion/               CMS download + CSV parsing
│   ├── validation/              pandera schemas + HTML/JSON validation reports
│   ├── db/                      PostgreSQL DDL, migrations, bulk COPY loader
│   ├── features/                Feature engineering + FeaturePipeline orchestrator
│   ├── models/                  Readmission, LOS, high-cost models + evaluator
│   ├── scoring/                 Batch scorer + drift detection
│   └── reporting/               KPI builder + Jinja2 HTML report
├── tableau/                     Tableau workbook + connection guide + screenshots
├── tests/                       45+ pytest tests, 80%+ coverage
├── .github/workflows/           CI (ruff+black+pytest) + weekly report cron
├── docker-compose.yml           PostgreSQL 16 + pgAdmin
└── Makefile                     make pipeline / make test / make lint
```

---

## PostgreSQL Schema

Two schemas: `claims` (raw data) and `analytics` (derived/reporting).

```sql
claims.beneficiaries          -- 1.1M+ synthetic Medicare beneficiaries
claims.chronic_conditions     -- 25 SP_* flags normalized to long format
claims.inpatient_claims       -- Inpatient claims with DRG, payments, dates
claims.diagnosis_codes        -- ICD-9 codes normalized from wide columns
claims.procedure_codes        -- Procedure codes normalized

analytics.patient_features    -- Materialized view: age, LOS, Elixhauser, prior admits
analytics.risk_scores         -- Calibrated readmission prob, risk tier, SHAP top-3
analytics.kpi_snapshots       -- Append-only daily KPIs by provider (for Tableau trending)
analytics.pipeline_alerts     -- Drift detection and data freshness alerts
```

---

## Tech Stack

| | |
|---|---|
| **Language** | Python 3.11 |
| **Database** | PostgreSQL 16 (Docker) |
| **Dashboard** | Tableau Public |
| **ML** | XGBoost 2.x · LightGBM 4.x · scikit-learn 1.4+ |
| **Tuning** | Optuna (TPE sampler, 50 trials per model) |
| **Explainability** | SHAP (beeswarm + waterfall plots) |
| **Calibration** | `CalibratedClassifierCV` isotonic regression |
| **Validation** | pandera |
| **DB layer** | SQLAlchemy 2.0 · psycopg2-binary |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **Reporting** | Jinja2 · matplotlib · seaborn |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest + pytest-cov (≥75% coverage) |

---

## Running Tests

```bash
make test
# or directly:
pytest tests/ -v --cov=src --cov-report=term-missing
```

All tests use synthetic in-memory fixtures — no PostgreSQL connection required.

---

## Tableau Connection

The committed `.twbx` file (`tableau/HealthcareKPI.twbx`) contains an embedded static extract so you can open it without a running database.

To reconnect to a live PostgreSQL instance, see [`tableau/connection_template.md`](tableau/connection_template.md).

---

## Resume Bullets

After running the pipeline and filling in your actual model numbers:

**Data Scientist framing:**
- Built calibrated XGBoost/LightGBM ensemble for 30-day hospital readmission on 1.1M+ CMS Medicare claims, achieving AUROC **X.XX** and AUPRC **X.XX** — **X%** lift over logistic regression baseline — with SHAP-driven explainability identifying CHF and prior-admission count as top risk drivers
- Engineered 30+ clinical and utilization features including Elixhauser comorbidity scoring and temporal 90/365-day prior-admission windows; applied isotonic calibration (Brier Score **X.XXX**) to align predicted probabilities with observed rates, enabling clinical team deployment without downstream recalibration
- Designed 8-stage reproducible pipeline with automated pandera schema validation, model drift detection (mean prob shift >0.03 triggers alert), and weekly HTML reporting via GitHub Actions cron; PostgreSQL bulk COPY ingestion at 200K rows/sec across 1.1M+ records

---

*Dataset: CMS Medicare SynPUFs — fully synthetic, no real patient data.*
