"""Tests for src/validation/schema.py and report.py"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.validation.report import write_validation_report
from src.validation.schema import (
    check_critical_non_null,
    check_null_rates,
    validate_beneficiaries,
    validate_inpatient,
)


def test_check_critical_non_null_passes_clean_data(sample_beneficiaries):
    check_critical_non_null(sample_beneficiaries, ["DESYNPUF_ID", "BENE_BIRTH_DT"], "bene")


def test_check_critical_non_null_raises_on_nulls():
    df = pd.DataFrame({"DESYNPUF_ID": ["A", None, "C"], "BENE_BIRTH_DT": pd.to_datetime(["2000-01-01"] * 3)})
    with pytest.raises(ValueError, match="Critical column 'DESYNPUF_ID'"):
        check_critical_non_null(df, ["DESYNPUF_ID"], "test")


def test_check_null_rates_detects_violation():
    df = pd.DataFrame({"col_a": [None] * 50 + [1] * 50})
    violations = check_null_rates(df, {"col_a": 0.10}, "test")
    assert "col_a" in violations
    assert abs(violations["col_a"] - 0.50) < 0.01


def test_check_null_rates_passes_under_threshold():
    df = pd.DataFrame({"col_a": [None] + [1] * 99})
    violations = check_null_rates(df, {"col_a": 0.05}, "test")
    assert "col_a" not in violations


def test_validate_beneficiaries_passes_clean_data(sample_beneficiaries):
    result = validate_beneficiaries(
        sample_beneficiaries,
        critical_cols=["DESYNPUF_ID", "BENE_BIRTH_DT"],
        null_thresholds={},
    )
    assert result["rows"] == 500
    assert not result["violations"]


def test_validate_inpatient_passes_clean_data(sample_inpatient):
    result = validate_inpatient(
        sample_inpatient,
        critical_cols=["CLM_ID", "DESYNPUF_ID", "CLM_ADMSN_DT"],
        null_thresholds={},
    )
    assert result["rows"] == 1500
    assert not result["violations"]


def test_write_validation_report_creates_files(tmp_path, sample_beneficiaries, sample_inpatient):
    bene_result = {"table": "beneficiaries", "rows": 500, "violations": {}}
    json_path, html_path = write_validation_report([bene_result], tmp_path)
    assert json_path.exists()
    assert html_path.exists()
    data = json.loads(json_path.read_text())
    assert data["overall_pass"] is True
    assert "<html" in html_path.read_text().lower()


def test_write_validation_report_marks_fail_on_violations(tmp_path):
    result = {"table": "inpatient", "rows": 1000, "violations": {"CLM_PMT_AMT": 0.15}}
    json_path, html_path = write_validation_report([result], tmp_path)
    data = json.loads(json_path.read_text())
    assert data["overall_pass"] is False
