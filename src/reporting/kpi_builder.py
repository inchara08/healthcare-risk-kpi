"""
Stage 7: Compute 12 clinical KPIs and append to analytics.kpi_snapshots.

The kpi_snapshots table is append-only — each run inserts new rows for today's date.
This design enables time-series trending in Tableau without overwriting history.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_engine

logger = logging.getLogger(__name__)


def _read_risk_scores(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT rs.claim_id, rs.bene_id, rs.risk_tier, rs.readmission_prob,
               rs.readmission_label, rs.scored_at,
               ic.provider_id, ic.admit_dt, ic.discharge_dt,
               ic.claim_pmt_amt, ic.drg_cd,
               pf.age_at_admit, pf.los_days, pf.elixhauser_count
        FROM analytics.risk_scores rs
        LEFT JOIN claims.inpatient_claims ic ON ic.claim_id = rs.claim_id
        LEFT JOIN analytics.patient_features pf ON pf.claim_id = rs.claim_id
        """,
        engine,
        parse_dates=["admit_dt", "discharge_dt", "scored_at"],
    )


def _upsert_kpi(
    engine: Engine,
    snapshot_date: date,
    provider_id: str,
    metric_name: str,
    metric_value: float,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO analytics.kpi_snapshots
                    (snapshot_date, provider_id, metric_name, metric_value)
                VALUES (:d, :p, :m, :v)
                ON CONFLICT (snapshot_date, provider_id, metric_name)
                DO UPDATE SET metric_value = EXCLUDED.metric_value
            """),
            {"d": snapshot_date, "p": provider_id, "m": metric_name, "v": metric_value},
        )


def compute_and_store_kpis(cfg: dict, engine: Engine | None = None) -> int:
    """
    Compute all KPIs by provider and store in analytics.kpi_snapshots.
    Returns total rows written.
    """
    engine = engine or get_engine()
    df = _read_risk_scores(engine)
    if df.empty:
        logger.warning("No risk scores found — skipping KPI computation")
        return 0

    today = date.today()
    national_avg_los = cfg["kpi"]["national_avg_los_days"]
    rows_written = 0

    providers = df["provider_id"].dropna().unique().tolist()
    providers.append("ALL")  # aggregate across all providers

    for prov in providers:
        sub = df if prov == "ALL" else df[df["provider_id"] == prov]
        if len(sub) == 0:
            continue

        kpis: dict[str, float] = {}

        # ── Readmission KPIs ────────────────────────────────────────────────
        if "readmission_label" in sub.columns and sub["readmission_label"].notna().any():
            labeled = sub[sub["readmission_label"].notna()]
            kpis["readmission_rate_30d"] = labeled["readmission_label"].mean()
            kpis["high_risk_count"] = (sub["risk_tier"] == "high").sum()
            kpis["high_risk_pct"] = (sub["risk_tier"] == "high").mean() * 100
            kpis["avg_readmit_prob"] = sub["readmission_prob"].mean()

        # ── LOS KPIs ────────────────────────────────────────────────────────
        if "los_days" in sub.columns and sub["los_days"].notna().any():
            kpis["avg_los_days"] = sub["los_days"].mean()
            kpis["median_los_days"] = sub["los_days"].median()
            kpis["los_vs_national_avg"] = sub["los_days"].mean() - national_avg_los

        # ── Cost KPIs ───────────────────────────────────────────────────────
        if "claim_pmt_amt" in sub.columns and sub["claim_pmt_amt"].notna().any():
            kpis["avg_cost_per_admission"] = sub["claim_pmt_amt"].mean()
            kpis["total_cost"] = sub["claim_pmt_amt"].sum()

        # ── Volume KPIs ─────────────────────────────────────────────────────
        kpis["admission_count"] = len(sub)

        # ── Comorbidity KPI ─────────────────────────────────────────────────
        if "elixhauser_count" in sub.columns:
            kpis["avg_elixhauser_score"] = sub["elixhauser_count"].mean()

        # ── Age KPI ─────────────────────────────────────────────────────────
        if "age_at_admit" in sub.columns:
            kpis["avg_age_at_admit"] = sub["age_at_admit"].mean()

        for metric_name, metric_value in kpis.items():
            if pd.isna(metric_value):
                continue
            _upsert_kpi(engine, today, prov, metric_name, float(metric_value))
            rows_written += 1

    logger.info(
        "KPI computation complete: %d rows written for %d providers", rows_written, len(providers)
    )
    return rows_written
