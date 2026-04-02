-- ============================================================
-- Healthcare Risk KPI — PostgreSQL Schema
-- CMS Medicare SynPUF data warehouse
-- ============================================================
-- Run order: this file is auto-mounted in docker-compose.yml
-- and executed on first container start.
-- For subsequent changes, use the migrations/ directory.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS claims;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ─── claims.beneficiaries ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims.beneficiaries (
    bene_id                     VARCHAR(20)  PRIMARY KEY,
    birth_dt                    DATE         NOT NULL,
    death_dt                    DATE,
    sex_cd                      SMALLINT,    -- 1=Male, 2=Female
    race_cd                     SMALLINT,    -- 1=White, 2=Black, 3=Other, 4=Asian, 5=Hispanic, 6=North American Native
    state_cd                    SMALLINT,
    county_cd                   SMALLINT,
    esrd_ind                    CHAR(1),
    hi_coverage_months          SMALLINT,
    smi_coverage_months         SMALLINT,
    hmo_coverage_months         SMALLINT,
    plan_coverage_months        SMALLINT,
    medreimb_ip                 NUMERIC(12,2),
    benres_ip                   NUMERIC(12,2),
    pppymt_ip                   NUMERIC(12,2),
    medreimb_op                 NUMERIC(12,2),
    benres_op                   NUMERIC(12,2),
    pppymt_op                   NUMERIC(12,2),
    medreimb_car                NUMERIC(12,2),
    benres_car                  NUMERIC(12,2),
    pppymt_car                  NUMERIC(12,2),
    created_at                  TIMESTAMPTZ  DEFAULT now()
);

-- ─── claims.chronic_conditions ───────────────────────────────────────────────
-- 25 SP_* flags normalized from wide → long for query efficiency
CREATE TABLE IF NOT EXISTS claims.chronic_conditions (
    bene_id                     VARCHAR(20)  NOT NULL REFERENCES claims.beneficiaries(bene_id) ON DELETE CASCADE,
    condition_code              VARCHAR(30)  NOT NULL,  -- 'alzheimers', 'chf', 'diabetes', etc.
    indicator                   SMALLINT     NOT NULL,  -- 1=diagnosed before year, 2=diagnosed during year
    PRIMARY KEY (bene_id, condition_code)
);

-- ─── claims.inpatient_claims ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims.inpatient_claims (
    claim_id                    VARCHAR(30)  PRIMARY KEY,
    bene_id                     VARCHAR(20)  NOT NULL REFERENCES claims.beneficiaries(bene_id),
    admit_dt                    DATE         NOT NULL,
    discharge_dt                DATE,
    provider_id                 VARCHAR(10),
    drg_cd                      SMALLINT,
    admit_source_cd             SMALLINT,
    admit_type_cd               SMALLINT,
    discharge_status_cd         SMALLINT,
    claim_pmt_amt               NUMERIC(12,2),
    pass_thru_amt               NUMERIC(12,2),
    nch_prmry_pyr_clm_pd_amt    NUMERIC(12,2),
    ip_ddctbl_amt               NUMERIC(12,2),
    pta_coinsrnc_amt            NUMERIC(12,2),
    blood_pnts_qty              SMALLINT,
    utilization_day_cnt         SMALLINT,
    admitting_icd9              VARCHAR(8),
    created_at                  TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_inpatient_bene_admit
    ON claims.inpatient_claims(bene_id, admit_dt);
CREATE INDEX IF NOT EXISTS idx_inpatient_provider
    ON claims.inpatient_claims(provider_id);
CREATE INDEX IF NOT EXISTS idx_inpatient_admit_dt
    ON claims.inpatient_claims(admit_dt);

-- ─── claims.diagnosis_codes ──────────────────────────────────────────────────
-- Normalized from wide ICD9_DGNS_CD_1..10 columns
CREATE TABLE IF NOT EXISTS claims.diagnosis_codes (
    claim_id                    VARCHAR(30)  NOT NULL REFERENCES claims.inpatient_claims(claim_id) ON DELETE CASCADE,
    seq_num                     SMALLINT     NOT NULL,
    icd9_code                   VARCHAR(8)   NOT NULL,
    PRIMARY KEY (claim_id, seq_num)
);

CREATE INDEX IF NOT EXISTS idx_diag_icd9
    ON claims.diagnosis_codes(icd9_code);

-- ─── claims.procedure_codes ──────────────────────────────────────────────────
-- Normalized from wide ICD9_PRCDR_CD_1..6 columns
CREATE TABLE IF NOT EXISTS claims.procedure_codes (
    claim_id                    VARCHAR(30)  NOT NULL REFERENCES claims.inpatient_claims(claim_id) ON DELETE CASCADE,
    seq_num                     SMALLINT     NOT NULL,
    icd9_prcdr_code             VARCHAR(8)   NOT NULL,
    PRIMARY KEY (claim_id, seq_num)
);

-- ─── analytics.risk_scores ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analytics.risk_scores (
    claim_id                    VARCHAR(30)  PRIMARY KEY,
    bene_id                     VARCHAR(20)  NOT NULL,
    scored_at                   TIMESTAMPTZ  DEFAULT now(),
    model_version               VARCHAR(20),
    readmission_prob            FLOAT,       -- calibrated probability 0–1
    risk_tier                   VARCHAR(10)  CHECK (risk_tier IN ('low', 'medium', 'high')),
    los_predicted_days          FLOAT,
    high_cost_prob              FLOAT,
    shap_top_feature_1          VARCHAR(60),
    shap_top_feature_2          VARCHAR(60),
    shap_top_feature_3          VARCHAR(60),
    readmission_label           SMALLINT     -- 0/1 ground truth (populated post-scoring)
);

CREATE INDEX IF NOT EXISTS idx_risk_scores_bene
    ON analytics.risk_scores(bene_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_scored_at
    ON analytics.risk_scores(scored_at);
CREATE INDEX IF NOT EXISTS idx_risk_scores_tier
    ON analytics.risk_scores(risk_tier);

-- ─── analytics.kpi_snapshots ─────────────────────────────────────────────────
-- Append-only daily KPI aggregates — enables time-series trending in Tableau
CREATE TABLE IF NOT EXISTS analytics.kpi_snapshots (
    snapshot_date               DATE         NOT NULL,
    provider_id                 VARCHAR(10)  NOT NULL,
    metric_name                 VARCHAR(60)  NOT NULL,
    metric_value                FLOAT,
    PRIMARY KEY (snapshot_date, provider_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_kpi_date_provider
    ON analytics.kpi_snapshots(snapshot_date, provider_id);

-- ─── analytics.pipeline_alerts ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analytics.pipeline_alerts (
    alert_id                    SERIAL       PRIMARY KEY,
    alert_type                  VARCHAR(40)  NOT NULL,  -- 'stale_kpi', 'score_drift', 'null_spike'
    message                     TEXT,
    severity                    VARCHAR(10)  DEFAULT 'WARNING',
    resolved                    BOOLEAN      DEFAULT FALSE,
    created_at                  TIMESTAMPTZ  DEFAULT now()
);

-- ─── analytics.patient_features (materialized view) ─────────────────────────
-- Refreshed after each load cycle. Used as the feature source for ML training.
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.patient_features AS
SELECT
    ic.claim_id,
    ic.bene_id,
    ic.admit_dt,
    ic.discharge_dt,
    ic.provider_id,
    ic.drg_cd,
    ic.admit_source_cd,
    ic.admit_type_cd,
    ic.discharge_status_cd,
    ic.claim_pmt_amt,
    ic.pass_thru_amt,
    ic.utilization_day_cnt,
    ic.admitting_icd9,
    -- Derived: age at admission
    EXTRACT(YEAR FROM AGE(ic.admit_dt, b.birth_dt))::INT          AS age_at_admit,
    -- Derived: length of stay in days
    (ic.discharge_dt - ic.admit_dt)::INT                          AS los_days,
    -- Derived: death within 30 days of discharge (proxy for mortality outcome)
    CASE
        WHEN b.death_dt IS NOT NULL
         AND b.death_dt <= ic.discharge_dt + INTERVAL '30 days' THEN 1
        ELSE 0
    END                                                            AS died_within_30d,
    -- Derived: Elixhauser comorbidity count
    (
        SELECT COUNT(*)
        FROM claims.chronic_conditions cc
        WHERE cc.bene_id = ic.bene_id AND cc.indicator IN (1, 2)
    )::INT                                                         AS elixhauser_count,
    -- Derived: prior admissions in past 90 days
    (
        SELECT COUNT(*)
        FROM claims.inpatient_claims prior
        WHERE prior.bene_id = ic.bene_id
          AND prior.admit_dt >= ic.admit_dt - INTERVAL '90 days'
          AND prior.admit_dt < ic.admit_dt
    )::INT                                                         AS prior_admits_90d,
    -- Derived: prior admissions in past 365 days
    (
        SELECT COUNT(*)
        FROM claims.inpatient_claims prior
        WHERE prior.bene_id = ic.bene_id
          AND prior.admit_dt >= ic.admit_dt - INTERVAL '365 days'
          AND prior.admit_dt < ic.admit_dt
    )::INT                                                         AS prior_admits_365d,
    -- Derived: days since last admission
    (
        SELECT (ic.admit_dt - MAX(prior.admit_dt))::INT
        FROM claims.inpatient_claims prior
        WHERE prior.bene_id = ic.bene_id
          AND prior.admit_dt < ic.admit_dt
    )                                                              AS days_since_last_admit,
    -- Beneficiary attributes
    b.sex_cd,
    b.race_cd,
    b.hmo_coverage_months,
    b.hi_coverage_months,
    b.medreimb_ip,
    b.death_dt
FROM claims.inpatient_claims ic
JOIN claims.beneficiaries b ON b.bene_id = ic.bene_id
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_features_claim
    ON analytics.patient_features(claim_id);
CREATE INDEX IF NOT EXISTS idx_patient_features_admit_dt
    ON analytics.patient_features(admit_dt);

-- ─── Data freshness trigger ───────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION analytics.check_kpi_freshness()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF (SELECT MAX(snapshot_date) FROM analytics.kpi_snapshots) < CURRENT_DATE - INTERVAL '2 days' THEN
        INSERT INTO analytics.pipeline_alerts(alert_type, message, severity)
        VALUES (
            'stale_kpi',
            'KPI snapshots not updated in 48+ hours. Last snapshot: '
                || (SELECT MAX(snapshot_date)::TEXT FROM analytics.kpi_snapshots),
            'WARNING'
        );
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER trg_kpi_freshness
AFTER INSERT ON analytics.kpi_snapshots
FOR EACH STATEMENT
EXECUTE FUNCTION analytics.check_kpi_freshness();
