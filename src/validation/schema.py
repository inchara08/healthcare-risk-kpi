"""
Stage 2: pandera schema definitions for CMS SynPUF beneficiary and inpatient claims DataFrames.

Validation runs pre-load (before writing to PostgreSQL) to fail fast on bad data.
Schema contracts enforce dtype, value ranges, and null rate thresholds from config.yaml.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

# ─── Beneficiary schema ──────────────────────────────────────────────────────

BENEFICIARY_SCHEMA = DataFrameSchema(
    columns={
        "DESYNPUF_ID": Column(str, nullable=False, unique=True),
        "BENE_BIRTH_DT": Column("datetime64[ns]", nullable=False),
        "BENE_DEATH_DT": Column("datetime64[ns]", nullable=True),
        "BENE_SEX_IDENT_CD": Column(
            pd.Int16Dtype(),
            Check.isin([1, 2]),
            nullable=True,
        ),
        "BENE_RACE_CD": Column(
            pd.Int16Dtype(),
            Check.isin([1, 2, 3, 4, 5, 6]),
            nullable=True,
        ),
        # Chronic condition flags: 0 or 1, nullable
        "SP_ALZHDMTA": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_CHF": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_CHRNKIDN": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_CNCR": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_COPD": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_DEPRESSN": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_DIABETES": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_ISCHMCHT": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_OSTEOPRS": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_RA_OA": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
        "SP_STRKETIA": Column(pd.Int16Dtype(), Check.isin([0, 1, 2]), nullable=True),
    },
    coerce=False,
    strict=False,  # allow extra columns not listed here
)

# ─── Inpatient claims schema ──────────────────────────────────────────────────

INPATIENT_SCHEMA = DataFrameSchema(
    columns={
        "CLM_ID": Column(str, nullable=False, unique=True),
        "DESYNPUF_ID": Column(str, nullable=False),
        "CLM_ADMSN_DT": Column("datetime64[ns]", nullable=False),
        "NCH_BENE_DSCHRG_DT": Column("datetime64[ns]", nullable=True),
        "CLM_PMT_AMT": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=True,
        ),
        "CLM_DRG_CD": Column(pd.Int16Dtype(), nullable=True),
        "PTNT_DSCHRG_STUS_CD": Column(pd.Int16Dtype(), nullable=True),
    },
    coerce=False,
    strict=False,
)


# ─── Null-rate checker ────────────────────────────────────────────────────────

def check_null_rates(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    table_name: str,
) -> dict[str, float]:
    """
    Verify that null rate per column does not exceed the configured threshold.
    Returns a dict of {col: actual_null_rate} for columns that violated thresholds.
    """
    violations: dict[str, float] = {}
    for col, max_rate in thresholds.items():
        if col not in df.columns:
            continue
        actual = df[col].isna().mean()
        if actual > max_rate:
            logger.warning(
                "[%s] Column '%s': null rate %.1f%% exceeds threshold %.1f%%",
                table_name,
                col,
                actual * 100,
                max_rate * 100,
            )
            violations[col] = actual
    return violations


# ─── Critical non-null check ─────────────────────────────────────────────────

def check_critical_non_null(
    df: pd.DataFrame,
    critical_cols: list[str],
    table_name: str,
) -> None:
    """Raise ValueError if any critical column has nulls."""
    for col in critical_cols:
        if col not in df.columns:
            raise ValueError(f"[{table_name}] Critical column '{col}' not found in DataFrame")
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"[{table_name}] Critical column '{col}' has {null_count} null values — aborting"
            )


# ─── Full validation runner ───────────────────────────────────────────────────

def validate_beneficiaries(
    df: pd.DataFrame,
    critical_cols: list[str],
    null_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Run all beneficiary validations. Returns a results dict."""
    results: dict[str, Any] = {"table": "beneficiaries", "rows": len(df), "violations": {}}

    check_critical_non_null(df, critical_cols, "beneficiaries")

    try:
        BENEFICIARY_SCHEMA.validate(df, lazy=True)
        logger.info("Beneficiary schema validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.warning("Beneficiary schema errors:\n%s", e.failure_cases.to_string())
        results["schema_errors"] = e.failure_cases.to_dict("records")

    results["violations"] = check_null_rates(df, null_thresholds, "beneficiaries")
    return results


def validate_inpatient(
    df: pd.DataFrame,
    critical_cols: list[str],
    null_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Run all inpatient claims validations. Returns a results dict."""
    results: dict[str, Any] = {"table": "inpatient", "rows": len(df), "violations": {}}

    check_critical_non_null(df, critical_cols, "inpatient")

    try:
        INPATIENT_SCHEMA.validate(df, lazy=True)
        logger.info("Inpatient schema validation passed (%d rows)", len(df))
    except pa.errors.SchemaErrors as e:
        logger.warning("Inpatient schema errors:\n%s", e.failure_cases.to_string())
        results["schema_errors"] = e.failure_cases.to_dict("records")

    results["violations"] = check_null_rates(df, null_thresholds, "inpatient")
    return results
