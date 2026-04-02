-- Migration 003: Add convenience views over kpi_snapshots for Tableau connections

CREATE OR REPLACE VIEW analytics.v_kpi_readmission AS
SELECT
    snapshot_date,
    provider_id,
    metric_value AS readmission_rate
FROM analytics.kpi_snapshots
WHERE metric_name = 'readmission_rate_30d';

CREATE OR REPLACE VIEW analytics.v_kpi_avg_los AS
SELECT
    snapshot_date,
    provider_id,
    metric_value AS avg_los_days
FROM analytics.kpi_snapshots
WHERE metric_name = 'avg_los_days';

CREATE OR REPLACE VIEW analytics.v_risk_summary AS
SELECT
    DATE_TRUNC('month', scored_at)::DATE  AS month,
    risk_tier,
    COUNT(*)                               AS patient_count,
    AVG(readmission_prob)                  AS avg_readmit_prob
FROM analytics.risk_scores
GROUP BY 1, 2;
