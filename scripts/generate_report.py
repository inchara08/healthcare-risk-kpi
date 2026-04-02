"""Standalone script for GitHub Actions weekly report cron job."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

if __name__ == "__main__":
    cfg_path = Path("config/config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from src.reporting.html_report import generate_html_report
    from src.reporting.kpi_builder import compute_and_store_kpis

    # Refresh KPIs before generating report
    rows = compute_and_store_kpis(cfg)
    logging.info("KPIs refreshed: %d rows", rows)

    reports_dir = Path(cfg["reporting"]["reports_dir"])
    model_dir = Path(cfg["scoring"]["model_dir"])
    path = generate_html_report(cfg, reports_dir, model_dir)
    print(f"Report generated: {path}")
