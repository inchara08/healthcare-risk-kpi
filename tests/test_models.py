"""Tests for src/models/ — evaluator, readmission training helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluator import (
    evaluate,
    expected_calibration_error,
    sensitivity_at_specificity,
)
from src.models.readmission import SoftVoteEnsemble

# ─── Evaluator ───────────────────────────────────────────────────────────────

@pytest.fixture
def binary_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, size=1000), 0, 1)
    return y_true, y_prob


def test_evaluate_returns_all_metrics(binary_predictions):
    y_true, y_prob = binary_predictions
    metrics = evaluate(y_true, y_prob, "test_model")
    for key in ["auroc", "auprc", "brier_score", "sensitivity_at_80pct_specificity", "expected_calibration_error"]:
        assert key in metrics, f"Missing metric: {key}"


def test_auroc_above_random(binary_predictions):
    y_true, y_prob = binary_predictions
    metrics = evaluate(y_true, y_prob, "test_model")
    assert metrics["auroc"] > 0.5, "AUROC should be above random (0.5)"


def test_brier_score_bounded(binary_predictions):
    y_true, y_prob = binary_predictions
    metrics = evaluate(y_true, y_prob, "test_model")
    assert 0 <= metrics["brier_score"] <= 1


def test_perfect_predictions_auroc():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    metrics = evaluate(y_true, y_prob, "perfect")
    assert metrics["auroc"] == 1.0


def test_ece_bounded(binary_predictions):
    y_true, y_prob = binary_predictions
    ece = expected_calibration_error(y_true, y_prob)
    assert 0 <= ece <= 1


def test_sensitivity_at_specificity_returns_float(binary_predictions):
    y_true, y_prob = binary_predictions
    sens = sensitivity_at_specificity(y_true, y_prob, 0.80)
    assert 0 <= sens <= 1


def test_evaluate_positive_rate_field(binary_predictions):
    y_true, y_prob = binary_predictions
    metrics = evaluate(y_true, y_prob, "test")
    assert abs(metrics["positive_rate"] - y_true.mean()) < 1e-6


# ─── SoftVoteEnsemble ────────────────────────────────────────────────────────

class _MockClassifier:
    """Minimal classifier that returns fixed probabilities."""
    def __init__(self, prob: float):
        self._prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self._prob), np.full(n, self._prob)])


def test_soft_vote_ensemble_blends_probabilities():
    m1 = _MockClassifier(0.8)
    m2 = _MockClassifier(0.4)
    ensemble = SoftVoteEnsemble(m1, m2, weights=(0.5, 0.5))
    X = np.zeros((10, 5))
    proba = ensemble.predict_proba(X)
    assert proba.shape == (10, 2)
    # Expected blend: 0.5 * 0.8 + 0.5 * 0.4 = 0.6
    assert abs(proba[0, 1] - 0.6) < 1e-6


def test_soft_vote_ensemble_predict_binary():
    m1 = _MockClassifier(0.8)
    m2 = _MockClassifier(0.8)
    ensemble = SoftVoteEnsemble(m1, m2)
    X = np.zeros((5, 3))
    preds = ensemble.predict(X)
    assert set(preds).issubset({0, 1})


def test_soft_vote_ensemble_custom_weights():
    m1 = _MockClassifier(1.0)
    m2 = _MockClassifier(0.0)
    ensemble = SoftVoteEnsemble(m1, m2, weights=(0.3, 0.7))
    X = np.zeros((1, 2))
    # 0.3 * 1.0 + 0.7 * 0.0 = 0.3
    assert abs(ensemble.predict_proba(X)[0, 1] - 0.3) < 1e-6
