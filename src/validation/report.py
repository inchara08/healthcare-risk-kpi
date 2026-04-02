"""
Stage 2 (cont.): Write pandera validation results to JSON and HTML reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Validation Report — {timestamp}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #1a5276; }}
  h2 {{ color: #2e86c1; border-bottom: 2px solid #aed6f1; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  th {{ background: #2e86c1; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #d5d8dc; }}
  tr:nth-child(even) {{ background: #f2f3f4; }}
  .pass {{ color: #1e8449; font-weight: 600; }}
  .fail {{ color: #c0392b; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; }}
  .badge-pass {{ background: #d5f5e3; color: #1e8449; }}
  .badge-fail {{ background: #fadbd8; color: #c0392b; }}
</style>
</head>
<body>
<h1>Healthcare Risk KPI — Data Validation Report</h1>
<p><strong>Generated:</strong> {timestamp}</p>
<p><strong>Overall status:</strong>
  <span class="badge {badge_class}">{status_label}</span>
</p>
{tables_html}
</body>
</html>
"""

TABLE_HTML = """
<h2>Table: {table}</h2>
<p>Rows loaded: <strong>{rows:,}</strong></p>
<h3>Null-rate violations</h3>
{violations_html}
{schema_html}
"""


def _violations_table(violations: dict[str, float]) -> str:
    if not violations:
        return '<p class="pass">No null-rate violations.</p>'
    rows = "".join(
        f"<tr><td>{col}</td><td class='fail'>{rate:.2%}</td></tr>"
        for col, rate in violations.items()
    )
    return (
        "<table><thead><tr><th>Column</th><th>Actual Null Rate</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _schema_errors_table(errors: list[dict]) -> str:
    if not errors:
        return '<p class="pass">No schema errors.</p>'
    rows = "".join(
        f"<tr><td>{e.get('column','')}</td><td>{e.get('check','')}</td>"
        f"<td>{e.get('failure_case','')}</td></tr>"
        for e in errors[:50]  # cap at 50 rows for readability
    )
    note = f"<p><em>(showing first 50 of {len(errors)} errors)</em></p>" if len(errors) > 50 else ""
    return (
        "<table><thead><tr><th>Column</th><th>Check</th><th>Failure Case</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{note}"
    )


def write_validation_report(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Write validation results as JSON + HTML to output_dir.

    Returns (json_path, html_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    overall_pass = all(not r.get("violations") and not r.get("schema_errors") for r in results)

    # ── JSON ──
    json_payload = {
        "timestamp": ts,
        "overall_pass": overall_pass,
        "tables": results,
    }
    json_path = output_dir / f"validation_report_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2, default=str)
    logger.info("Validation JSON report written: %s", json_path)

    # ── HTML ──
    tables_html = ""
    for r in results:
        tables_html += TABLE_HTML.format(
            table=r["table"],
            rows=r["rows"],
            violations_html=_violations_table(r.get("violations", {})),
            schema_html=_schema_errors_table(r.get("schema_errors", [])),
        )

    html_content = HTML_TEMPLATE.format(
        timestamp=ts,
        badge_class="badge-pass" if overall_pass else "badge-fail",
        status_label="PASS" if overall_pass else "FAIL",
        tables_html=tables_html,
    )

    html_path = output_dir / f"validation_report_{ts}.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info("Validation HTML report written: %s", html_path)

    if not overall_pass:
        logger.error("Validation FAILED — review %s before proceeding", html_path)

    return json_path, html_path
