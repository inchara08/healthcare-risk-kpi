"""Tests for src/features/"""

from __future__ import annotations

import pandas as pd

from src.features.claims import (
    add_admit_date_features,
    add_financial_ratios,
    add_utilization_flags,
    build_readmission_label,
)
from src.features.comorbidities import add_comorbidity_flags
from src.features.demographics import add_age_features, add_sex_flag

# ─── Demographics ────────────────────────────────────────────────────────────


def test_add_age_features_creates_age_group(sample_patient_features):
    result = add_age_features(sample_patient_features)
    assert "age_group" in result.columns
    assert "age_group_ord" in result.columns
    assert result["age_group_ord"].notna().all()


def test_age_group_ord_is_non_negative(sample_patient_features):
    result = add_age_features(sample_patient_features)
    assert (result["age_group_ord"] >= 0).all()


def test_add_sex_flag_binary(sample_patient_features):
    result = add_sex_flag(sample_patient_features, col="sex_cd")
    assert set(result["is_female"].dropna().unique()).issubset({0, 1})


# ─── Claims date features ────────────────────────────────────────────────────


def test_add_admit_date_features_columns(sample_patient_features):
    result = add_admit_date_features(sample_patient_features, date_col="admit_dt")
    for col in ["admit_month", "admit_quarter", "admit_dow", "admit_month_sin", "admit_month_cos"]:
        assert col in result.columns


def test_admit_month_range(sample_patient_features):
    result = add_admit_date_features(sample_patient_features)
    assert result["admit_month"].between(1, 12).all()


def test_cyclical_encoding_bounded(sample_patient_features):
    result = add_admit_date_features(sample_patient_features)
    for col in ["admit_month_sin", "admit_month_cos", "admit_dow_sin", "admit_dow_cos"]:
        assert result[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


# ─── Financial ratios ────────────────────────────────────────────────────────


def test_add_financial_ratios_log_pmt_non_negative(sample_patient_features):
    result = add_financial_ratios(sample_patient_features)
    assert (result["log_claim_pmt"] >= 0).all()


def test_pass_thru_ratio_bounded(sample_patient_features):
    result = add_financial_ratios(sample_patient_features)
    # ratio should be 0..1 after fillna
    assert (result["pass_thru_ratio"] >= 0).all()


# ─── Utilization flags ────────────────────────────────────────────────────────


def test_utilization_flags_binary(sample_patient_features):
    result = add_utilization_flags(sample_patient_features)
    for col in ["had_prior_admit_90d", "had_prior_admit_365d", "is_frequent_flyer"]:
        assert set(result[col].dropna().unique()).issubset({0, 1}), col


def test_capped_cols_have_upper_limit(sample_patient_features):
    result = add_utilization_flags(sample_patient_features)
    assert result["prior_admits_90d_capped"].max() <= 10
    assert result["prior_admits_365d_capped"].max() <= 20
    assert result["los_days_capped"].max() <= 60


# ─── Readmission label ────────────────────────────────────────────────────────


def test_readmission_label_is_binary(sample_patient_features):
    labels = build_readmission_label(sample_patient_features, window_days=30)
    assert set(labels.dropna().unique()).issubset({0, 1})


def test_readmission_label_length_matches(sample_patient_features):
    labels = build_readmission_label(sample_patient_features)
    assert len(labels) == len(sample_patient_features)


def test_readmission_label_rate_realistic():
    """Synthetic data with a known readmission should label first admission correctly."""
    # Patient P001: admitted Jan 1, discharged Jan 5, readmitted Jan 20 (15 days later — within 30d window)
    # Patient P002: single admission — no readmission
    df = pd.DataFrame(
        {
            "bene_id": ["P001", "P001", "P002"],
            "admit_dt": pd.to_datetime(["2008-01-01", "2008-01-20", "2008-02-01"]),
            "discharge_dt": pd.to_datetime(["2008-01-05", "2008-01-25", "2008-02-05"]),
        }
    )
    labels = build_readmission_label(df, window_days=30)
    # After sort by (bene_id, admit_dt): P001-Jan1, P001-Jan20, P002-Feb1
    # P001's first row: next_admit=Jan20, discharge=Jan5, diff=15d → readmitted=1
    assert labels.iloc[0] == 1  # P001 first admission: readmitted
    assert labels.iloc[1] == 0  # P001 second admission: no further admission
    assert labels.iloc[2] == 0  # P002: no readmission


# ─── Comorbidities ────────────────────────────────────────────────────────────


def test_add_comorbidity_flags_adds_cond_columns(sample_patient_features, sample_conditions):
    result = add_comorbidity_flags(sample_patient_features, sample_conditions)
    assert "cond_chf" in result.columns
    assert "cond_diabetes" in result.columns
    assert "has_high_weight_comorbidity" in result.columns


def test_comorbidity_flags_binary(sample_patient_features, sample_conditions):
    result = add_comorbidity_flags(sample_patient_features, sample_conditions)
    assert set(result["cond_chf"].dropna().unique()).issubset({0, 1})
