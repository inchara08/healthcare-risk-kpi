"""
Elixhauser comorbidity score and individual condition flags.

Maps CMS chronic_conditions table to binary feature columns for model training.
"""

from __future__ import annotations

import pandas as pd

# All condition codes loaded from claims.chronic_conditions
ALL_CONDITIONS = [
    "alzheimers",
    "chf",
    "chronic_kidney",
    "cancer",
    "copd",
    "depression",
    "diabetes",
    "ischemic_heart",
    "osteoporosis",
    "ra_oa",
    "stroke_tia",
]

# Conditions with highest Elixhauser weight for readmission prediction
HIGH_WEIGHT_CONDITIONS = {"chf", "chronic_kidney", "cancer", "copd", "stroke_tia"}


def add_comorbidity_flags(
    features_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join per-patient condition flags onto features_df.

    conditions_df must have columns: bene_id, condition_code, indicator
    """
    df = features_df.copy()

    # Pivot: one row per bene_id, one column per condition
    pivot = (
        conditions_df[conditions_df["indicator"].isin([1, 2])]
        .assign(has_condition=1)
        .pivot_table(
            index="bene_id",
            columns="condition_code",
            values="has_condition",
            aggfunc="max",
            fill_value=0,
        )
    )
    pivot.columns = [f"cond_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    df = df.merge(pivot, on="bene_id", how="left")

    # Fill any missing condition columns with 0
    for cond in ALL_CONDITIONS:
        col = f"cond_{cond}"
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype("Int8")

    # High-weight comorbidity flag (any of the high-weight conditions present)
    high_weight_cols = [f"cond_{c}" for c in HIGH_WEIGHT_CONDITIONS if f"cond_{c}" in df.columns]
    if high_weight_cols:
        df["has_high_weight_comorbidity"] = (
            df[high_weight_cols].max(axis=1).astype("Int8")
        )

    return df
