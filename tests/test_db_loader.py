"""Integration tests for src/db/loader.py — requires a live PostgreSQL instance.

These tests are skipped automatically when PostgreSQL is not reachable (e.g. local
dev without Docker running). In CI a postgres service container is spun up so they
always execute there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.db.loader import (
    load_beneficiaries,
    load_chronic_conditions,
    load_inpatient_claims,
    refresh_materialized_views,
)

SCHEMA_SQL = Path(__file__).parent.parent / "src" / "db" / "schema.sql"


def _make_engine() -> Engine:
    return create_engine(
        "postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}".format(
            user=os.environ.get("PG_USER", "pguser"),
            pw=os.environ.get("PG_PASS", "pgpassword"),
            host=os.environ.get("PG_HOST", "localhost"),
            port=os.environ.get("PG_PORT", "5432"),
            db=os.environ.get("PG_DB", "healthcare_risk"),
        )
    )


@pytest.fixture(scope="session")
def db_engine():
    """Session-scoped engine. Skips all tests if PostgreSQL is not reachable."""
    engine = _make_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL not available — skipping integration tests")

    # Apply schema (all statements are CREATE IF NOT EXISTS / idempotent)
    with engine.connect() as conn:
        conn.execute(text(SCHEMA_SQL.read_text()))
        conn.commit()

    yield engine
    engine.dispose()


@pytest.fixture()
def clean_db(db_engine):
    """Truncate all base tables before each test so tests are fully isolated."""
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE claims.diagnosis_codes, claims.procedure_codes, "
                "claims.inpatient_claims, claims.chronic_conditions, "
                "claims.beneficiaries RESTART IDENTITY CASCADE"
            )
        )
    yield


# ─── load_beneficiaries ──────────────────────────────────────────────────────


def test_load_beneficiaries_returns_count(db_engine, clean_db, sample_beneficiaries):
    count = load_beneficiaries(sample_beneficiaries, db_engine)
    assert count == 500


def test_load_beneficiaries_rows_persisted(db_engine, clean_db, sample_beneficiaries):
    load_beneficiaries(sample_beneficiaries, db_engine)
    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM claims.beneficiaries")).scalar()
    assert n == 500


def test_load_beneficiaries_birth_dt_not_null(db_engine, clean_db, sample_beneficiaries):
    load_beneficiaries(sample_beneficiaries, db_engine)
    with db_engine.connect() as conn:
        nulls = conn.execute(
            text("SELECT COUNT(*) FROM claims.beneficiaries WHERE birth_dt IS NULL")
        ).scalar()
    assert nulls == 0


def test_load_beneficiaries_idempotent(db_engine, clean_db, sample_beneficiaries):
    """Loading the same data twice must not raise (upsert fallback handles duplicates)."""
    load_beneficiaries(sample_beneficiaries, db_engine)
    count2 = load_beneficiaries(sample_beneficiaries, db_engine)
    assert count2 >= 0


# ─── load_chronic_conditions ─────────────────────────────────────────────────


def test_load_chronic_conditions_returns_count(db_engine, clean_db, sample_beneficiaries):
    load_beneficiaries(sample_beneficiaries, db_engine)
    count = load_chronic_conditions(sample_beneficiaries, db_engine)
    assert count > 0


def test_load_chronic_conditions_valid_indicators_only(db_engine, clean_db, sample_beneficiaries):
    load_beneficiaries(sample_beneficiaries, db_engine)
    load_chronic_conditions(sample_beneficiaries, db_engine)
    with db_engine.connect() as conn:
        bad = conn.execute(
            text("SELECT COUNT(*) FROM claims.chronic_conditions " "WHERE indicator NOT IN (1, 2)")
        ).scalar()
    assert bad == 0


def test_load_chronic_conditions_no_sp_columns_returns_zero(db_engine, sample_beneficiaries):
    """DataFrame without SP_* columns returns 0 without touching the database."""
    import pandas as pd

    df_no_sp = pd.DataFrame({"DESYNPUF_ID": sample_beneficiaries["DESYNPUF_ID"]})
    count = load_chronic_conditions(df_no_sp, db_engine)
    assert count == 0


# ─── load_inpatient_claims ───────────────────────────────────────────────────


def test_load_inpatient_claims_returns_count(
    db_engine, clean_db, sample_beneficiaries, sample_inpatient
):
    load_beneficiaries(sample_beneficiaries, db_engine)
    count = load_inpatient_claims(sample_inpatient, db_engine)
    assert count == 1500


def test_load_inpatient_claims_rows_persisted(
    db_engine, clean_db, sample_beneficiaries, sample_inpatient
):
    load_beneficiaries(sample_beneficiaries, db_engine)
    load_inpatient_claims(sample_inpatient, db_engine)
    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM claims.inpatient_claims")).scalar()
    assert n == 1500


def test_load_inpatient_claims_icd9_normalized(
    db_engine, clean_db, sample_beneficiaries, sample_inpatient
):
    """ICD9 diagnosis codes must be unpivoted into claims.diagnosis_codes."""
    load_beneficiaries(sample_beneficiaries, db_engine)
    load_inpatient_claims(sample_inpatient, db_engine)
    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM claims.diagnosis_codes")).scalar()
    assert n > 0


def test_load_inpatient_claims_admit_dt_not_null(
    db_engine, clean_db, sample_beneficiaries, sample_inpatient
):
    load_beneficiaries(sample_beneficiaries, db_engine)
    load_inpatient_claims(sample_inpatient, db_engine)
    with db_engine.connect() as conn:
        nulls = conn.execute(
            text("SELECT COUNT(*) FROM claims.inpatient_claims WHERE admit_dt IS NULL")
        ).scalar()
    assert nulls == 0


# ─── refresh_materialized_views ──────────────────────────────────────────────


def test_refresh_materialized_view_populates(
    db_engine, clean_db, sample_beneficiaries, sample_inpatient
):
    load_beneficiaries(sample_beneficiaries, db_engine)
    load_inpatient_claims(sample_inpatient, db_engine)
    refresh_materialized_views(db_engine)
    with db_engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM analytics.patient_features")).scalar()
    assert n == 1500


def test_refresh_materialized_view_derives_age(
    db_engine, clean_db, sample_beneficiaries, sample_inpatient
):
    load_beneficiaries(sample_beneficiaries, db_engine)
    load_inpatient_claims(sample_inpatient, db_engine)
    refresh_materialized_views(db_engine)
    with db_engine.connect() as conn:
        nulls = conn.execute(
            text("SELECT COUNT(*) FROM analytics.patient_features WHERE age_at_admit IS NULL")
        ).scalar()
    assert nulls == 0
