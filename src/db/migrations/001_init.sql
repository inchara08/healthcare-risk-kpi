-- Migration 001: Initial schema (same as schema.sql, idempotent)
-- Applied automatically by db/loader.py on first run.
-- All statements use IF NOT EXISTS for safe reruns.

-- Schemas
CREATE SCHEMA IF NOT EXISTS claims;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Verify both schemas exist (used as a health check)
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('claims', 'analytics');
