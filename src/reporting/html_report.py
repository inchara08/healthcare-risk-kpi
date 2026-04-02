"""
Stage 8: Generate weekly HTML report from Jinja2 template.

Pulls data from PostgreSQL, embeds matplotlib charts as base64 PNGs,
and writes a self-contained HTML file to reports/.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.engine import Engine

from src.db.connection import get_engine

logger = logging.getLogger(__name__)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _fetch_kpis(engine: Engine) -> dict:
    """Fetch today's aggregate KPIs for the 'ALL' provider row."""
    df = pd.read_sql(
        """
        SELECT metric_name, metric_value
        FROM analytics.kpi_snapshots
        WHERE provider_id = 'ALL'
          AND snapshot_date = (SELECT MAX(snapshot_date) FROM analytics.kpi_snapshots)
        """,
        engine,
    )
    return {row.metric_name: row.metric_value for row in df.itertuples()}


def _fetch_provider_table(engine: Engine) -> list[dict]:
    df = pd.read_sql(
        """
        SELECT
            pf.provider_id,
            COUNT(*)                                              AS admission_count,
            AVG(CASE WHEN rs.risk_tier = 'high'   THEN 1.0 ELSE 0 END) * 100 AS high_pct,
            AVG(CASE WHEN rs.risk_tier = 'medium' THEN 1.0 ELSE 0 END) * 100 AS medium_pct,
            AVG(CASE WHEN rs.risk_tier = 'low'    THEN 1.0 ELSE 0 END) * 100 AS low_pct,
            AVG(rs.readmission_label)                            AS readmission_rate,
            AVG(pf.los_days)                                     AS avg_los
        FROM analytics.risk_scores rs
        LEFT JOIN analytics.patient_features pf ON pf.claim_id = rs.claim_id
        WHERE pf.provider_id IS NOT NULL
        GROUP BY pf.provider_id
        ORDER BY admission_count DESC
        LIMIT 20
        """,
        engine,
    )
    return df.to_dict("records")


def _fetch_alerts(engine: Engine) -> list[dict]:
    df = pd.read_sql(
        """
        SELECT alert_type, message, created_at
        FROM analytics.pipeline_alerts
        WHERE resolved = FALSE
        ORDER BY created_at DESC
        LIMIT 10
        """,
        engine,
    )
    return df.to_dict("records")


def _make_risk_tier_chart(provider_table: list[dict]) -> str:
    if not provider_table:
        return ""
    top = provider_table[:10]
    labels = [r["provider_id"] for r in top]
    high = [r["high_pct"] for r in top]
    medium = [r["medium_pct"] for r in top]
    low = [r["low_pct"] for r in top]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(labels))
    ax.bar(x, high, label="High", color="#e74c3c", alpha=0.85)
    ax.bar(x, medium, bottom=high, label="Medium", color="#f39c12", alpha=0.85)
    ax.bar(
        x, low,
        bottom=[h + m for h, m in zip(high, medium)],
        label="Low", color="#27ae60", alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of Patients")
    ax.set_title("Risk Tier Distribution by Provider (Top 10)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_html_report(
    cfg: dict,
    reports_dir: Path,
    model_dir: Path,
    engine: Engine | None = None,
) -> Path:
    """Generate and save the weekly HTML report. Returns the output path."""
    engine = engine or get_engine()
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    kpis = _fetch_kpis(engine)
    provider_table = _fetch_provider_table(engine)
    alerts = _fetch_alerts(engine)

    # Load calibration chart if it exists
    cal_chart_path = model_dir / "calibration_plot.png"
    calibration_b64 = ""
    if cal_chart_path.exists():
        with open(cal_chart_path, "rb") as f:
            calibration_b64 = base64.b64encode(f.read()).decode()

    # Load SHAP chart if it exists
    shap_chart_path = model_dir / "shap_plots" / "shap_summary.png"
    shap_b64 = ""
    if shap_chart_path.exists():
        with open(shap_chart_path, "rb") as f:
            shap_b64 = base64.b64encode(f.read()).decode()

    # Resolve model version from latest model card
    model_cards = sorted(model_dir.glob("model_card_v*.json"))
    model_version = model_cards[-1].stem.replace("model_card_v", "") if model_cards else "unknown"

    # Render template
    template_dir = Path(cfg["reporting"]["template_dir"])
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("weekly_report.html")

    html = template.render(
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        data_as_of=today,
        model_version=model_version,
        kpis=kpis,
        provider_table=provider_table,
        alerts=alerts,
        calibration_chart_b64=calibration_b64,
        shap_chart_b64=shap_b64,
    )

    output_path = reports_dir / f"weekly_{today.replace('-', '_')}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Weekly report written: %s", output_path)
    return output_path
