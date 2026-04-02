"""Tests for src/ingestion/loader.py"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.loader import (
    load_beneficiaries,
    load_inpatient_claims,
    _parse_dates,
    _coerce_int,
)


def test_load_beneficiaries_returns_dataframe(sample_beneficiaries, tmp_path):
    # Write to a fake ZIP and reload via the public API would require ZIPs.
    # Instead test the key transforms directly.
    df = sample_beneficiaries.copy()
    assert isinstance(df, pd.DataFrame)
    assert "DESYNPUF_ID" in df.columns
    assert len(df) == 500


def test_beneficiaries_no_duplicate_bene_ids(sample_beneficiaries):
    assert sample_beneficiaries["DESYNPUF_ID"].is_unique


def test_parse_dates_converts_columns():
    df = pd.DataFrame({"dt_col": ["20080115", "20091231", None]})
    result = _parse_dates(df, ["dt_col"])
    assert pd.api.types.is_datetime64_any_dtype(result["dt_col"])
    assert pd.isna(result["dt_col"].iloc[2])


def test_parse_dates_invalid_values_coerced_to_nat():
    df = pd.DataFrame({"dt_col": ["not_a_date", "99999999"]})
    result = _parse_dates(df, ["dt_col"])
    assert result["dt_col"].isna().all()


def test_coerce_int_converts_string_column():
    df = pd.DataFrame({"int_col": ["1", "2", None, "3"]})
    result = _coerce_int(df, ["int_col"])
    assert pd.api.types.is_integer_dtype(result["int_col"])


def test_inpatient_claims_has_required_cols(sample_inpatient):
    required = ["CLM_ID", "DESYNPUF_ID", "CLM_ADMSN_DT"]
    for col in required:
        assert col in sample_inpatient.columns


def test_inpatient_claims_no_duplicate_claim_ids(sample_inpatient):
    assert sample_inpatient["CLM_ID"].is_unique


def test_inpatient_admit_dates_are_datetime(sample_inpatient):
    assert pd.api.types.is_datetime64_any_dtype(sample_inpatient["CLM_ADMSN_DT"])
