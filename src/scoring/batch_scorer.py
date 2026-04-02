"""
Stage 6: Batch scoring — load latest calibrated model, score all claims,
write results to analytics.risk_scores, and check for model drift.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_engine
from src.features.pipeline import FEATURE_COLS, TARGET_COL
from src.models.readmission import SoftVoteEnsemble, get_top_shap_features

logger = logging.getLogger(__name__)


def _latest_file(model_dir: Path, pattern: str) -> Path:
    files = sorted(model_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model file matching '{pattern}' in {model_dir}")
    return files[-1]


def load_calibrated_model(model_dir: Path) -> tuple[object, object, str]:
    cal_path = _latest_file(model_dir, "calibrator_v*.pkl")
    with open(cal_path, "rb") as f:
        saved = pickle.load(f)
    version = cal_path.stem.replace("calibrator_v", "")
    ensemble = saved["ensemble"]
    calibrator = saved["calibrator"]
    logger.info("Loaded calibrated model: %s (version %s)", cal_path.name, version)
    return ensemble, calibrator, version


def load_score_baseline(model_dir: Path) -> dict:
    path = model_dir / "score_baseline.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def detect_drift(
    current_probs: np.ndarray,
    baseline: dict,
    cfg: dict,
    engine: Engine,
) -> list[str]:
    """
    Compare current score distribution against stored baseline.
    Writes to analytics.pipeline_alerts if thresholds breached.
    Returns list of alert messages.
    """
    alerts: list[str] = []
    if not baseline:
        logger.info("No baseline found — skipping drift detection")
        return alerts

    thresh_mean = cfg["scoring"]["drift_mean_prob_threshold"]
    thresh_pct = cfg["scoring"]["drift_high_risk_pct_threshold"]
    high_cutoff = cfg["scoring"]["risk_tiers"]["high"]

    current_mean = float(current_probs.mean())
    current_high_pct = float((current_probs >= high_cutoff).mean() * 100)
    baseline_mean = baseline.get("mean_prob", current_mean)
    baseline_high_pct = baseline.get("pct_high_risk", current_high_pct)

    mean_shift = abs(current_mean - baseline_mean)
    pct_shift = abs(current_high_pct - baseline_high_pct)

    if mean_shift > thresh_mean:
        msg = (
            f"Score drift detected: mean prob shifted {mean_shift:.4f} "
            f"(baseline={baseline_mean:.4f}, current={current_mean:.4f})"
        )
        logger.warning(msg)
        alerts.append(msg)
        _write_alert(engine, "score_drift", msg)

    if pct_shift > thresh_pct:
        msg = (
            f"High-risk % drift: {pct_shift:.1f}pp shift "
            f"(baseline={baseline_high_pct:.1f}%, current={current_high_pct:.1f}%)"
        )
        logger.warning(msg)
        alerts.append(msg)
        _write_alert(engine, "score_drift", msg)

    if not alerts:
        logger.info(
            "Drift check passed (mean=%.4f, high_risk=%.1f%%)", current_mean, current_high_pct
        )
    return alerts


def _write_alert(engine: Engine, alert_type: str, message: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO analytics.pipeline_alerts(alert_type, message) VALUES (:t, :m)"
            ),
            {"t": alert_type, "m": message},
        )


def _assign_risk_tier(prob: float, tiers: dict) -> str:
    if prob >= tiers["high"]:
        return "high"
    if prob >= tiers["medium"]:
        return "medium"
    return "low"


def score_claims(
    test_path: Path,
    model_dir: Path,
    cfg: dict,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """
    Score all claims in test_path, write to analytics.risk_scores.
    Returns the scored DataFrame.
    """
    engine = engine or get_engine()
    ensemble, calibrator, version = load_calibrated_model(model_dir)
    baseline = load_score_baseline(model_dir)

    test = pd.read_parquet(test_path)
    feat_cols = [c for c in FEATURE_COLS if c in test.columns]
    X = test[feat_cols].fillna(-1).astype("float32")

    logger.info("Scoring %d claims...", len(X))
    probs = calibrator.predict(ensemble.predict_proba(X)[:, 1])

    # Drift detection
    detect_drift(probs, baseline, cfg, engine)

    # SHAP top-3 per prediction (use XGBoost component of ensemble)
    try:
        xgb_model = ensemble.xgb_model
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X)
        top_features_per_row = []
        for row_shap in shap_vals:
            top_idx = np.argsort(np.abs(row_shap))[::-1][:3]
            top_features_per_row.append([feat_cols[i] for i in top_idx])
    except Exception:
        logger.warning("SHAP per-row computation failed — using global top features")
        global_top = get_top_shap_features(
            np.zeros((1, len(feat_cols))), feat_cols
        )
        top_features_per_row = [global_top] * len(X)

    tiers = cfg["scoring"]["risk_tiers"]
    scores_df = pd.DataFrame({
        "claim_id": test["claim_id"].values,
        "bene_id": test["bene_id"].values,
        "model_version": version,
        "readmission_prob": probs.astype(float),
        "risk_tier": [_assign_risk_tier(p, tiers) for p in probs],
        "shap_top_feature_1": [f[0] if len(f) > 0 else None for f in top_features_per_row],
        "shap_top_feature_2": [f[1] if len(f) > 1 else None for f in top_features_per_row],
        "shap_top_feature_3": [f[2] if len(f) > 2 else None for f in top_features_per_row],
        "readmission_label": test[TARGET_COL].fillna(0).astype(int).values
        if TARGET_COL in test.columns else None,
    })

    # Write to analytics.risk_scores (upsert on claim_id)
    _upsert_risk_scores(scores_df, engine)
    logger.info(
        "Scored %d claims — high=%.1f%%, medium=%.1f%%, low=%.1f%%",
        len(scores_df),
        (scores_df["risk_tier"] == "high").mean() * 100,
        (scores_df["risk_tier"] == "medium").mean() * 100,
        (scores_df["risk_tier"] == "low").mean() * 100,
    )
    return scores_df


def _upsert_risk_scores(df: pd.DataFrame, engine: Engine) -> None:
    cols = [c for c in df.columns if c != "readmission_label" or "readmission_label" in df.columns]
    chunk_size = 5000
    with engine.begin() as conn:
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            for row in chunk.itertuples(index=False):
                conn.execute(
                    text("""
                        INSERT INTO analytics.risk_scores
                            (claim_id, bene_id, model_version, readmission_prob, risk_tier,
                             shap_top_feature_1, shap_top_feature_2, shap_top_feature_3,
                             readmission_label)
                        VALUES
                            (:claim_id, :bene_id, :model_version, :readmission_prob, :risk_tier,
                             :s1, :s2, :s3, :label)
                        ON CONFLICT (claim_id) DO UPDATE SET
                            readmission_prob = EXCLUDED.readmission_prob,
                            risk_tier = EXCLUDED.risk_tier,
                            model_version = EXCLUDED.model_version,
                            scored_at = now()
                    """),
                    {
                        "claim_id": row.claim_id,
                        "bene_id": row.bene_id,
                        "model_version": row.model_version,
                        "readmission_prob": float(row.readmission_prob),
                        "risk_tier": row.risk_tier,
                        "s1": row.shap_top_feature_1,
                        "s2": row.shap_top_feature_2,
                        "s3": row.shap_top_feature_3,
                        "label": int(row.readmission_label) if row.readmission_label is not None else None,
                    },
                )
    logger.info("Upserted %d rows into analytics.risk_scores", len(df))
