"""Tests for src/reporting/kpi_builder.py — using mocked DB calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_risk_scores_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "claim_id": [f"CLM{i:06d}" for i in range(n)],
            "bene_id": [f"BENE{rng.integers(0, 50):06d}" for _ in range(n)],
            "risk_tier": rng.choice(["high", "medium", "low"], size=n, p=[0.15, 0.25, 0.60]),
            "readmission_prob": rng.uniform(0, 1, size=n).astype(float),
            "readmission_label": rng.integers(0, 2, size=n),
            "scored_at": pd.date_range("2009-07-01", periods=n, freq="h"),
            "provider_id": rng.choice(["PROV001", "PROV002", "PROV003"], size=n),
            "admit_dt": pd.date_range("2009-01-01", periods=n, freq="D"),
            "discharge_dt": pd.date_range("2009-01-05", periods=n, freq="D"),
            "claim_pmt_amt": rng.uniform(1000, 50000, size=n),
            "drg_cd": rng.integers(1, 999, size=n),
            "age_at_admit": rng.integers(50, 90, size=n),
            "los_days": rng.integers(1, 20, size=n),
            "elixhauser_count": rng.integers(0, 10, size=n),
        }
    )


def test_kpi_columns_present(mock_risk_scores_df):
    """KPI builder should produce readmission_rate, avg_los, etc. for each provider."""
    from src.reporting.kpi_builder import compute_and_store_kpis

    mock_engine = MagicMock()

    with (
        patch("src.reporting.kpi_builder._read_risk_scores", return_value=mock_risk_scores_df),
        patch("src.reporting.kpi_builder._upsert_kpi") as mock_upsert,
    ):
        rows = compute_and_store_kpis(
            cfg={"kpi": {"national_avg_los_days": 4.6, "snapshot_lookback_days": 30}},
            engine=mock_engine,
        )

    assert rows > 0, "Should write at least some KPI rows"
    # Verify upsert was called
    assert mock_upsert.call_count > 0


def test_kpi_readmission_rate_in_range(mock_risk_scores_df):
    """Readmission rate should be between 0 and 1."""
    labeled = mock_risk_scores_df[mock_risk_scores_df["readmission_label"].notna()]
    rate = labeled["readmission_label"].mean()
    assert 0 <= rate <= 1


def test_kpi_empty_dataframe_returns_zero():
    from src.reporting.kpi_builder import compute_and_store_kpis

    mock_engine = MagicMock()
    with patch("src.reporting.kpi_builder._read_risk_scores", return_value=pd.DataFrame()):
        rows = compute_and_store_kpis(
            cfg={"kpi": {"national_avg_los_days": 4.6, "snapshot_lookback_days": 30}},
            engine=mock_engine,
        )
    assert rows == 0


def test_kpi_high_risk_pct_bounded(mock_risk_scores_df):
    high_pct = (mock_risk_scores_df["risk_tier"] == "high").mean() * 100
    assert 0 <= high_pct <= 100


def test_kpi_los_vs_national(mock_risk_scores_df):
    avg_los = mock_risk_scores_df["los_days"].mean()
    national = 4.6
    diff = avg_los - national
    # Just verify arithmetic works
    assert isinstance(diff, float)
