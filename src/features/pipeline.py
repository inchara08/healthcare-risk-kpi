"""
Stage 4: FeaturePipeline — orchestrates all feature transforms and produces
train/test parquet files ready for model training.

Design decisions:
- Temporal split (not random): train on admits before split_date, test on after.
  This prevents data leakage from time-correlated admission patterns.
- DRG target encoding fitted on training set only, then applied to test set.
- SMOTE is NOT applied here — it's applied inside model training folds only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sqlalchemy.engine import Engine

from src.db.connection import get_engine
from src.features.claims import (
    add_admit_date_features,
    add_discharge_features,
    add_financial_ratios,
    add_utilization_flags,
    build_readmission_label,
)
from src.features.comorbidities import add_comorbidity_flags
from src.features.demographics import add_age_features, add_sex_flag

logger = logging.getLogger(__name__)

# Columns selected as model features (subset of all engineered columns)
FEATURE_COLS = [
    # Demographics
    "age_at_admit", "age_group_ord", "is_female", "race_cd",
    # Admission
    "drg_cd_encoded", "admit_source_cd", "admit_type_cd", "discharge_status_cd",
    "is_not_routine_discharge", "discharge_risk_score",
    "admit_month", "admit_quarter", "admit_dow",
    "admit_month_sin", "admit_month_cos", "admit_dow_sin", "admit_dow_cos",
    # Clinical / comorbidities
    "elixhauser_count",
    "cond_chf", "cond_diabetes", "cond_copd", "cond_chronic_kidney",
    "cond_cancer", "cond_ischemic_heart", "cond_stroke_tia", "cond_depression",
    "has_high_weight_comorbidity",
    # Utilization history
    "prior_admits_90d_capped", "prior_admits_365d_capped",
    "had_prior_admit_90d", "had_prior_admit_365d", "is_frequent_flyer",
    "los_days_capped", "days_since_last_admit",
    # Financial
    "log_claim_pmt", "pass_thru_ratio", "hmo_coverage_months",
]

TARGET_COL = "readmitted_30d"
LOS_TARGET_COL = "los_days_log"
HIGH_COST_TARGET_COL = "is_high_cost"


class FeaturePipeline:
    def __init__(self, config: dict, engine: Engine | None = None):
        self.config = config
        self.engine = engine or get_engine()
        self.drg_encoder: TargetEncoder | None = None
        self.split_date = pd.Timestamp(config["features"]["temporal_split_date"])

    def load_from_db(self) -> pd.DataFrame:
        """Load analytics.patient_features materialized view from PostgreSQL."""
        query = """
            SELECT *
            FROM analytics.patient_features
            ORDER BY admit_dt
        """
        logger.info("Loading patient features from PostgreSQL...")
        df = pd.read_sql(query, self.engine, parse_dates=["admit_dt", "discharge_dt", "death_dt"])
        logger.info("Loaded %d rows from analytics.patient_features", len(df))
        return df

    def load_conditions_from_db(self) -> pd.DataFrame:
        query = "SELECT bene_id, condition_code, indicator FROM claims.chronic_conditions"
        return pd.read_sql(query, self.engine)

    def build_features(self, df: pd.DataFrame, conditions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms in sequence."""
        df = add_age_features(df, age_col="age_at_admit",
                              bins=self.config["features"]["age_bins"],
                              labels=self.config["features"]["age_labels"])
        df = add_sex_flag(df, col="sex_cd")
        df = add_admit_date_features(df, date_col="admit_dt")
        df = add_financial_ratios(df)
        df = add_utilization_flags(df)
        df = add_discharge_features(df)
        df = add_comorbidity_flags(df, conditions_df)
        return df

    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all target columns."""
        cfg = self.config
        df = df.copy()

        # Readmission label
        df[TARGET_COL] = build_readmission_label(
            df, window_days=cfg["features"]["readmission_window_days"]
        )

        # LOS label (log-transformed for regression)
        df[LOS_TARGET_COL] = np.log1p(df["los_days"].clip(lower=0).fillna(0)).astype("float32")

        # High-cost label
        pct = cfg["features"]["high_cost_percentile"] / 100
        threshold = df["claim_pmt_amt"].quantile(pct)
        df[HIGH_COST_TARGET_COL] = (df["claim_pmt_amt"] >= threshold).astype("Int8")

        return df

    def encode_drg(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Target-encode DRG code — fitted on train only to prevent leakage."""
        self.drg_encoder = TargetEncoder(cols=["drg_cd"], smoothing=10)
        train = train.copy()
        test = test.copy()
        train["drg_cd_encoded"] = self.drg_encoder.fit_transform(
            train[["drg_cd"]], train[TARGET_COL]
        )["drg_cd"]
        test["drg_cd_encoded"] = self.drg_encoder.transform(test[["drg_cd"]])["drg_cd"]
        return train, test

    def temporal_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["admit_dt"] < self.split_date].copy()
        test = df[df["admit_dt"] >= self.split_date].copy()
        logger.info(
            "Temporal split: train=%d rows (before %s), test=%d rows",
            len(train), self.split_date.date(), len(test)
        )
        pos_rate = train[TARGET_COL].mean()
        logger.info("Train readmission rate: %.1f%%", pos_rate * 100)
        return train, test

    def run(self, processed_dir: Path) -> tuple[Path, Path]:
        """
        Execute the full feature pipeline.

        Returns (train_path, test_path) for the saved parquet files.
        """
        processed_dir.mkdir(parents=True, exist_ok=True)

        df = self.load_from_db()
        conditions_df = self.load_conditions_from_db()

        df = self.build_features(df, conditions_df)
        df = self.build_labels(df)

        train, test = self.temporal_split(df)
        train, test = self.encode_drg(train, test)

        # Select only model-relevant columns + targets + identifiers
        keep_cols = (
            ["claim_id", "bene_id", "admit_dt", "provider_id"]
            + FEATURE_COLS
            + [TARGET_COL, LOS_TARGET_COL, HIGH_COST_TARGET_COL]
        )
        train = train[[c for c in keep_cols if c in train.columns]]
        test = test[[c for c in keep_cols if c in test.columns]]

        train_path = processed_dir / "train_features.parquet"
        test_path = processed_dir / "test_features.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        logger.info("Saved train features: %s (%d rows)", train_path, len(train))
        logger.info("Saved test features:  %s (%d rows)", test_path, len(test))
        return train_path, test_path
