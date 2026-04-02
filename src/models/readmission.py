"""
Stage 5: 30-day readmission model training.

Model ladder:
  1. Logistic Regression (L2) — baseline
  2. XGBoost — Optuna hyperparameter search, 5-fold stratified CV
  3. LightGBM — Optuna hyperparameter search, 5-fold stratified CV
  4. Soft-vote ensemble (0.5 XGB + 0.5 LGBM)
  5. CalibratedClassifierCV (isotonic) applied to ensemble

Why calibration matters: clinical teams need true probabilities.
A predicted 73% readmission risk should reflect ~73% observed rate.
Raw XGBoost scores are not calibrated probabilities.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import FEATURE_COLS, TARGET_COL
from src.models.evaluator import evaluate, plot_calibration, save_model_card

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, ...]:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[feat_cols].fillna(-1).astype("float32")
    y_train = train[TARGET_COL].fillna(0).astype(int).values
    X_test = test[feat_cols].fillna(-1).astype("float32")
    y_test = test[TARGET_COL].fillna(0).astype(int).values
    return X_train, y_train, X_test, y_test, feat_cols


# ─── Baseline: Logistic Regression ──────────────────────────────────────────


def train_logistic_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = 42
) -> LogisticRegression:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=500, random_state=random_state, n_jobs=-1)
    model.fit(X_scaled, y_train)
    logger.info("Logistic regression baseline trained.")
    return model, scaler


# ─── XGBoost with Optuna ─────────────────────────────────────────────────────


def _xgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cfg: dict) -> float:
    sp = cfg["model"]["xgb_space"]
    params = {
        "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "subsample": trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *sp["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *sp["reg_lambda"]),
        "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg: dict,
) -> xgb.XGBClassifier:
    n_trials = cfg["model"]["n_optuna_trials"]
    timeout = cfg["model"]["optuna_timeout_seconds"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: _xgb_objective(t, X_train.values, y_train, cfg),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    best = study.best_params
    logger.info("XGBoost best AUROC (CV): %.4f | params: %s", study.best_value, best)

    model = xgb.XGBClassifier(
        **best,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="auc",
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── LightGBM with Optuna ────────────────────────────────────────────────────


def _lgbm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cfg: dict) -> float:
    sp = cfg["model"]["lgbm_space"]
    params = {
        "num_leaves": trial.suggest_int("num_leaves", *sp["num_leaves"]),
        "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "subsample": trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *sp["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *sp["reg_lambda"]),
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg: dict,
) -> lgb.LGBMClassifier:
    n_trials = cfg["model"]["n_optuna_trials"]
    timeout = cfg["model"]["optuna_timeout_seconds"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: _lgbm_objective(t, X_train.values, y_train, cfg),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    best = study.best_params
    logger.info("LightGBM best AUROC (CV): %.4f | params: %s", study.best_value, best)

    model = lgb.LGBMClassifier(
        **best,
        is_unbalance=True,
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── Ensemble + Calibration ──────────────────────────────────────────────────


class SoftVoteEnsemble:
    """Soft-voting blend of XGBoost and LightGBM with configurable weights."""

    def __init__(
        self,
        xgb_model: xgb.XGBClassifier,
        lgbm_model: lgb.LGBMClassifier,
        weights: tuple[float, float] = (0.5, 0.5),
    ):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """No-op — ensemble components are already fitted."""
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_lgbm = self.lgbm_model.predict_proba(X)[:, 1]
        blended = self.weights[0] * p_xgb + self.weights[1] * p_lgbm
        return np.column_stack([1 - blended, blended])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─── SHAP ────────────────────────────────────────────────────────────────────


def compute_shap(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    sample_n: int = 1000,
) -> np.ndarray:
    """Compute SHAP values on a random sample. Returns shap_values array."""
    idx = np.random.choice(len(X), size=min(sample_n, len(X)), replace=False)
    X_sample = X.iloc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Global summary plot
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved.")
    return shap_values


def get_top_shap_features(
    shap_values: np.ndarray, feature_names: list[str], n: int = 3
) -> list[str]:
    """Return top n feature names by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]
    return [feature_names[i] for i in top_idx]


# ─── Main training entry point ────────────────────────────────────────────────


def run_readmission_training(
    train_path: Path,
    test_path: Path,
    model_dir: Path,
    cfg: dict,
    version: str | None = None,
) -> dict:
    """
    Full readmission model training pipeline.

    Returns metrics dict for all models (used in model card).
    """
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M")

    model_dir.mkdir(parents=True, exist_ok=True)
    X_train_full, y_train_full, X_test, y_test, feat_cols = _load_data(train_path, test_path)

    # Hold out 20% of training data as calibration set (stratified)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=cfg["model"]["random_state"],
    )

    all_metrics: list[dict] = []
    calibrated_probs: dict[str, np.ndarray] = {}

    # 1. Baseline
    logger.info("Training logistic regression baseline...")
    lr_model, scaler = train_logistic_baseline(X_train, y_train, cfg["model"]["random_state"])
    lr_prob = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    all_metrics.append(evaluate(y_test, lr_prob, "logistic_regression_baseline"))
    calibrated_probs["Logistic (baseline)"] = lr_prob

    # 2. XGBoost
    logger.info("Training XGBoost (Optuna, %d trials)...", cfg["model"]["n_optuna_trials"])
    xgb_model = train_xgboost(X_train, y_train, cfg)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    all_metrics.append(evaluate(y_test, xgb_prob, "xgboost"))
    xgb_model.save_model(str(model_dir / f"xgb_v{version}.json"))

    # 3. LightGBM
    logger.info("Training LightGBM (Optuna, %d trials)...", cfg["model"]["n_optuna_trials"])
    lgbm_model = train_lightgbm(X_train, y_train, cfg)
    lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]
    all_metrics.append(evaluate(y_test, lgbm_prob, "lightgbm"))
    lgbm_model.booster_.save_model(str(model_dir / f"lgbm_v{version}.txt"))

    # 4. Soft-vote ensemble
    weights = (cfg["model"]["ensemble_weights"]["xgb"], cfg["model"]["ensemble_weights"]["lgbm"])
    ensemble = SoftVoteEnsemble(xgb_model, lgbm_model, weights=weights)
    ens_prob_raw = ensemble.predict_proba(X_test)[:, 1]
    all_metrics.append(evaluate(y_test, ens_prob_raw, "ensemble_uncalibrated"))

    # 5. Calibrate ensemble using IsotonicRegression on held-out calibration set
    logger.info("Applying isotonic calibration to ensemble...")
    ens_cal_raw = ensemble.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ens_cal_raw, y_cal)

    ens_prob_cal = calibrator.predict(ens_prob_raw)
    all_metrics.append(evaluate(y_test, ens_prob_cal, "ensemble_calibrated"))
    calibrated_probs["Ensemble (calibrated)"] = ens_prob_cal

    with open(model_dir / f"calibrator_v{version}.pkl", "wb") as f:
        pickle.dump({"ensemble": ensemble, "calibrator": calibrator}, f)

    # 6. SHAP on XGBoost (best single model for interpretability)
    logger.info("Computing SHAP values...")
    shap_values = compute_shap(xgb_model, X_test, feat_cols, model_dir / "shap_plots")
    top_features = get_top_shap_features(shap_values, feat_cols)
    logger.info("Top SHAP features: %s", top_features)

    # 7. Calibration plot
    plot_calibration(y_test, calibrated_probs, model_dir / "calibration_plot.png")

    # 8. Save score baseline for drift detection
    baseline = {
        "version": version,
        "mean_prob": float(ens_prob_cal.mean()),
        "pct_high_risk": float((ens_prob_cal >= cfg["scoring"]["risk_tiers"]["high"]).mean() * 100),
        "top_shap_features": top_features,
    }
    with open(model_dir / "score_baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)

    # 9. Model card
    save_model_card(all_metrics, model_dir, version)

    logger.info("Readmission model training complete. Version: %s", version)
    return {"version": version, "metrics": all_metrics, "top_shap_features": top_features}
