-- Migration 002: Add model_run_id and confidence_interval columns to risk_scores
-- Safe to run multiple times (uses IF NOT EXISTS / DO blocks).

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'analytics'
          AND table_name = 'risk_scores'
          AND column_name = 'model_run_id'
    ) THEN
        ALTER TABLE analytics.risk_scores ADD COLUMN model_run_id VARCHAR(40);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'analytics'
          AND table_name = 'risk_scores'
          AND column_name = 'readmit_prob_ci_lower'
    ) THEN
        ALTER TABLE analytics.risk_scores ADD COLUMN readmit_prob_ci_lower FLOAT;
        ALTER TABLE analytics.risk_scores ADD COLUMN readmit_prob_ci_upper FLOAT;
    END IF;
END;
$$;
