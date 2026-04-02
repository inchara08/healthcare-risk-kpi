"""
Claim-level and utilization history features.

All window-based features (prior_admits, LOS) are pre-computed in
analytics.patient_features materialized view. This module handles
additional Python-side transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_admit_date_features(df: pd.DataFrame, date_col: str = "admit_dt") -> pd.DataFrame:
    """Add cyclical and ordinal encodings for admit date."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["admit_month"] = dt.dt.month.astype("Int8")
    df["admit_quarter"] = dt.dt.quarter.astype("Int8")
    df["admit_dow"] = dt.dt.dayofweek.astype("Int8")  # 0=Monday

    # Cyclical encoding — preserves periodicity for tree models
    df["admit_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12).astype("float32")
    df["admit_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12).astype("float32")
    df["admit_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7).astype("float32")
    df["admit_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7).astype("float32")

    return df


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Derive financial ratio features."""
    df = df.copy()
    # Pass-through ratio: what fraction of payment is pass-through
    df["pass_thru_ratio"] = (
        (df["pass_thru_amt"] / df["claim_pmt_amt"].replace(0, np.nan)).fillna(0).astype("float32")
    )

    # Log-transformed payment (handles skew)
    df["log_claim_pmt"] = np.log1p(df["claim_pmt_amt"].fillna(0)).astype("float32")

    return df


def add_utilization_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags derived from utilization history."""
    df = df.copy()
    df["had_prior_admit_90d"] = (df["prior_admits_90d"] > 0).astype("Int8")
    df["had_prior_admit_365d"] = (df["prior_admits_365d"] > 0).astype("Int8")
    df["is_frequent_flyer"] = (df["prior_admits_365d"] >= 3).astype("Int8")

    # Cap outliers for LOS and prior admits
    df["prior_admits_90d_capped"] = df["prior_admits_90d"].clip(upper=10).astype("Int8")
    df["prior_admits_365d_capped"] = df["prior_admits_365d"].clip(upper=20).astype("Int8")
    df["los_days_capped"] = df["los_days"].clip(upper=60).fillna(0).astype("Int16")

    return df


def add_discharge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode discharge status as risk-stratified features.

    Raw discharge_status_cd codes are not ordinal — code 20 (expired) is
    highest risk but numerically largest. Explicit binary and ordinal
    encodings let tree models exploit this signal without discovering the
    non-linear split on their own.
    """
    df = df.copy()
    # Binary: any non-routine discharge = higher readmission risk
    df["is_not_routine_discharge"] = (df["discharge_status_cd"].fillna(1) != 1).astype("Int8")

    # Ordinal risk score mapped from CMS discharge status codes
    # 1=home(low), 6=home+health(med-low), 2=transfer(med), 3=SNF(med-high),
    # 30=still-pt(med-high), 20=expired(high)
    risk_map = {1: 0, 6: 1, 2: 2, 3: 3, 30: 3, 20: 4}
    df["discharge_risk_score"] = (
        df["discharge_status_cd"].fillna(1).map(risk_map).fillna(2)
    ).astype("Int8")

    return df


def build_readmission_label(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """
    Construct 30-day readmission label.

    Mirrors CMS HRRP definition: same patient admitted to any inpatient
    facility within window_days of the prior discharge.

    df must have columns: bene_id, admit_dt, discharge_dt (all rows in dataset).
    Returns a boolean Series aligned to df.index.
    """
    df = df.copy()
    df["discharge_dt"] = pd.to_datetime(df["discharge_dt"])
    df["admit_dt"] = pd.to_datetime(df["admit_dt"])
    df = df.sort_values(["bene_id", "admit_dt"])

    # For each claim, check if there's a subsequent admission within window_days of discharge
    df["next_admit_dt"] = df.groupby("bene_id")["admit_dt"].shift(-1)
    df["readmitted_30d"] = (
        (df["next_admit_dt"] - df["discharge_dt"]).dt.days.between(
            0, window_days, inclusive="right"
        )
    ).astype("Int8")

    # Last admission per patient cannot have a readmission in dataset
    df.loc[df["next_admit_dt"].isna(), "readmitted_30d"] = 0

    return df["readmitted_30d"]
