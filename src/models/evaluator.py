"""
Shared model evaluation utilities.

Reports AUROC, AUPRC, Brier Score, Sensitivity@80%Spec, and ECE.
Produces calibration reliability diagram and saves model card JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.80,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    # Find the point closest to target_specificity
    idx = np.argmin(np.abs(specificity - target_specificity))
    return float(tpr[idx])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """Compute and log all evaluation metrics. Returns metrics dict."""
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    sens_80spec = sensitivity_at_specificity(y_true, y_prob, 0.80)
    ece = expected_calibration_error(y_true, y_prob)

    metrics = {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "sensitivity_at_80pct_specificity": round(sens_80spec, 4),
        "expected_calibration_error": round(ece, 4),
    }

    logger.info(
        "[%s] AUROC=%.4f | AUPRC=%.4f | Brier=%.4f | Sens@80%%Spec=%.1f%% | ECE=%.4f",
        model_name,
        auroc,
        auprc,
        brier,
        sens_80spec * 100,
        ece,
    )
    return metrics


def plot_calibration(
    y_true: np.ndarray,
    y_prob_dict: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot reliability diagram for one or more models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, y_prob in y_prob_dict.items():
        fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_predicted, fraction_of_positives, "s-", label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Reliability Diagram — 30-day Readmission")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration plot saved: %s", output_path)


def save_model_card(metrics_list: list[dict], model_dir: Path, version: str) -> Path:
    """Write a model_card JSON that serves as the record of model performance."""
    card = {
        "version": version,
        "task": "30-day all-cause hospital readmission (binary classification)",
        "dataset": "CMS Medicare SynPUFs (synthetic), samples 1–5",
        "label_definition": "Same patient readmitted within 30 days of discharge (mirrors CMS HRRP)",
        "split": "Temporal — train: admits before 2009-07-01, test: admits on/after",
        "class_imbalance_handling": "SMOTE applied to training folds only",
        "calibration": "CalibratedClassifierCV (isotonic regression)",
        "results": metrics_list,
    }
    path = model_dir / f"model_card_v{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    logger.info("Model card saved: %s", path)
    return path
