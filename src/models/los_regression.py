"""
Stage 5 (cont.): Length of Stay (LOS) regression model.

Predicts log1p(los_days) using LightGBM, then back-transforms predictions.
Used for operational capacity planning KPI.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.pipeline import FEATURE_COLS, LOS_TARGET_COL

logger = logging.getLogger(__name__)


def train_los_model(
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
    y_train = train[LOS_TARGET_COL].fillna(0).values
    X_test = test[feat_cols].fillna(-1).astype("float32")
    y_test = test[LOS_TARGET_COL].fillna(0).values

    model = lgb.LGBMRegressor(
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds_log = model.predict(X_test)
    preds_days = np.expm1(preds_log)  # back-transform
    actual_days = np.expm1(y_test)

    mae = mean_absolute_error(actual_days, preds_days)
    rmse = np.sqrt(mean_squared_error(actual_days, preds_days))
    r2 = r2_score(actual_days, preds_days)
    mape = np.mean(np.abs((actual_days - preds_days) / (actual_days + 1e-6))) * 100

    metrics = {
        "model": "los_lgbm_regression",
        "mae_days": round(mae, 3),
        "rmse_days": round(rmse, 3),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 2),
    }
    logger.info(
        "LOS model: MAE=%.2f days | RMSE=%.2f | R²=%.4f | MAPE=%.1f%%",
        mae,
        rmse,
        r2,
        mape,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / f"los_lgbm_v{version}.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(model_dir / f"los_model_card_v{version}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
