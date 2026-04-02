"""
Export analytics tables to CSV for Tableau Public.

Usage:
    python scripts/export_tableau.py

Writes three CSVs to tableau/data/:
  - risk_scores.csv       — 334K scored claims (risk tier, prob, SHAP features)
  - kpi_snapshots.csv     — daily KPI aggregates by provider
  - patient_features.csv  — claim-level features + derived fields (admit, LOS, age, etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.db.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("tableau/data")

EXPORTS = {
    "risk_scores.csv": """
        SELECT
            rs.claim_id,
            rs.bene_id,
            rs.scored_at::DATE            AS score_date,
            rs.model_version,
            rs.readmission_prob,
            rs.risk_tier,
            rs.shap_top_feature_1,
            rs.shap_top_feature_2,
            rs.shap_top_feature_3,
            rs.readmission_label,
            pf.provider_id,
            pf.admit_dt,
            pf.discharge_dt,
            pf.los_days,
            pf.age_at_admit,
            pf.sex_cd,
            pf.race_cd,
            pf.elixhauser_count,
            pf.prior_admits_90d,
            pf.claim_pmt_amt
        FROM analytics.risk_scores rs
        LEFT JOIN analytics.patient_features pf USING (claim_id)
        ORDER BY rs.readmission_prob DESC
    """,
    "kpi_snapshots.csv": """
        SELECT
            snapshot_date,
            provider_id,
            metric_name,
            metric_value
        FROM analytics.kpi_snapshots
        ORDER BY snapshot_date, provider_id, metric_name
    """,
    "patient_features.csv": """
        SELECT
            claim_id,
            bene_id,
            provider_id,
            admit_dt,
            discharge_dt,
            los_days,
            age_at_admit,
            sex_cd,
            race_cd,
            drg_cd,
            admit_type_cd,
            discharge_status_cd,
            elixhauser_count,
            prior_admits_90d,
            prior_admits_365d,
            days_since_last_admit,
            claim_pmt_amt,
            died_within_30d,
            hmo_coverage_months
        FROM analytics.patient_features
        ORDER BY admit_dt
    """,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    for filename, query in EXPORTS.items():
        logger.info("Exporting %s ...", filename)
        df = pd.read_sql(query.strip(), engine)
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        logger.info("  → %s  (%d rows, %.1f MB)", path, len(df), path.stat().st_size / 1e6)

    logger.info("Done. Import the CSVs in tableau/data/ into Tableau Public.")


if __name__ == "__main__":
    main()
