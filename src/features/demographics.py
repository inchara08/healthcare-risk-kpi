"""Demographic feature transforms: age bins, sex encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_age_features(
    df: pd.DataFrame,
    age_col: str = "age_at_admit",
    bins: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if bins is None:
        bins = [0, 18, 35, 50, 65, 75, 85, 120]
    if labels is None:
        labels = ["<18", "18-34", "35-49", "50-64", "65-74", "75-84", "85+"]

    df = df.copy()
    df["age_group"] = pd.cut(
        df[age_col], bins=bins, labels=labels, right=False, include_lowest=True
    )
    # Ordinal encode age_group for tree models
    df["age_group_ord"] = df["age_group"].cat.codes.astype("Int8")
    return df


def add_sex_flag(df: pd.DataFrame, col: str = "sex_cd") -> pd.DataFrame:
    df = df.copy()
    df["is_female"] = (df[col] == 2).astype("Int8")
    return df
