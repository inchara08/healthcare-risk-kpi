"""
Stage 5 (cont.): High-cost claim binary classifier.

Label: claim payment in top-20th percentile for the dataset.
Useful for cost-reduction narrative on resume and dashboard.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features.pipeline import FEATURE_COLS, HIGH_COST_TARGET_COL
from src.models.evaluator import evaluate

logger = logging.getLogger(__name__)


def train_high_cost_model(
    train_path: Path,
    test_path: Path,
    model_dir: Path,
    cfg: dict,
    version: str,
) -> dict:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[feat_cols].fillna(-1).astype("float32")
    y_train = train[HIGH_COST_TARGET_COL].fillna(0).astype(int).values
    X_test = test[feat_cols].fillna(-1).astype("float32")
    y_test = test[HIGH_COST_TARGET_COL].fillna(0).astype(int).values

    pos_rate = y_train.mean()
    logger.info("High-cost label positive rate: %.1f%%", pos_rate * 100)

    model = lgb.LGBMClassifier(
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=300,
        is_unbalance=True,
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, prob, "high_cost_lgbm")

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / f"high_cost_v{version}.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(model_dir / f"high_cost_card_v{version}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
