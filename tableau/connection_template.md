# Tableau Public — PostgreSQL Connection Guide

The committed `HealthcareKPI.twbx` workbook contains a static data extract so you can open and interact with it without a database connection. This guide explains how to reconnect it to a live PostgreSQL instance.

## Prerequisites

- Tableau Desktop (free 14-day trial) or Tableau Public Desktop
- PostgreSQL ODBC driver or Tableau's native PostgreSQL connector (built-in since 2021.3+)
- Docker running (`make up`)

## Steps

1. Open `HealthcareKPI.twbx` in Tableau Desktop
2. Go to **Data → Edit Connection**
3. Set:
   - **Server**: `localhost`
   - **Port**: `5432`
   - **Database**: `healthcare_risk`
   - **Username**: `pguser` (or whatever you set in `.env`)
   - **Password**: `pgpassword`
4. Click **Sign In**
5. Go to **Data → Extract → Refresh All Extracts** to pull live data

## Tables Used

| Tableau Datasource | PostgreSQL Table |
|---|---|
| Risk Scores | `analytics.risk_scores` |
| KPI Trend | `analytics.kpi_snapshots` |
| Patient Features | `analytics.patient_features` |
| Provider Summary | `analytics.v_risk_summary` (view) |

## Publishing to Tableau Public

1. In Tableau Desktop: **Server → Tableau Public → Save to Tableau Public**
2. Sign in with your free Tableau Public account
3. The workbook is published with an embedded data extract (no live DB needed by viewers)
4. Copy the public URL and add it to your README and resume

## Dashboard Pages

| Page | Key Visuals |
|---|---|
| Executive Summary | Readmission rate trend, high-risk volume KPI card, avg LOS vs. national avg, cost per admission by DRG |
| Risk Stratification | Risk tier by provider (stacked bar), SHAP feature importance, readmission by age group |
| Operational | LOS predicted vs. actual (scatter), admission volume heatmap by day-of-week |
