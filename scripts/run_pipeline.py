"""
Healthcare Risk KPI — Pipeline CLI

Usage:
    python scripts/run_pipeline.py all          # full 8-stage pipeline
    python scripts/run_pipeline.py download     # stage 1
    python scripts/run_pipeline.py validate     # stage 2
    python scripts/run_pipeline.py load         # stage 3
    python scripts/run_pipeline.py features     # stage 4
    python scripts/run_pipeline.py train        # stage 5
    python scripts/run_pipeline.py score        # stage 6
    python scripts/run_pipeline.py kpis         # stage 7
    python scripts/run_pipeline.py report       # stage 8
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import typer
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
app = typer.Typer(help="Healthcare Risk KPI pipeline CLI")

CONFIG_PATH = Path("config/config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@app.command()
def download(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
    n_beneficiaries: int = typer.Option(100_000, help="Number of synthetic beneficiaries"),
    n_claims: int = typer.Option(500_000, help="Number of base inpatient claims"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Stage 1: Generate synthetic SynPUF-schema data (replaces CMS download)."""
    cfg = yaml.safe_load(config.read_text())
    from src.ingestion.generate_synthetic import generate_and_save
    raw_dir = Path(cfg["data"]["raw_dir"])
    bene_path, inpatient_path = generate_and_save(
        raw_dir,
        n_beneficiaries=n_beneficiaries,
        n_claims=n_claims,
        random_seed=seed,
    )
    typer.echo(f"✓ Synthetic data generated: {bene_path.name}, {inpatient_path.name}")


@app.command()
def validate(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Stage 2: Validate raw data with pandera schemas."""
    cfg = yaml.safe_load(config.read_text())
    from src.ingestion.loader import load_raw_data
    from src.validation.schema import validate_beneficiaries, validate_inpatient
    from src.validation.report import write_validation_report

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    dataset = load_raw_data(raw_dir)
    val_cfg = cfg["validation"]

    bene_results = validate_beneficiaries(
        dataset.beneficiaries,
        critical_cols=val_cfg["critical_non_null"]["beneficiaries"],
        null_thresholds=val_cfg["null_thresholds"],
    )
    inp_results = validate_inpatient(
        dataset.inpatient,
        critical_cols=val_cfg["critical_non_null"]["inpatient"],
        null_thresholds=val_cfg["null_thresholds"],
    )

    json_path, html_path = write_validation_report(
        [bene_results, inp_results], processed_dir
    )

    if bene_results.get("violations") or inp_results.get("violations"):
        typer.echo(f"⚠ Validation warnings — review {html_path}", err=True)
    else:
        typer.echo("✓ Validation passed")

    # Save parquet checkpoints for downstream stages
    dataset.beneficiaries.to_parquet(processed_dir / "beneficiaries.parquet", index=False)
    dataset.inpatient.to_parquet(processed_dir / "inpatient_claims.parquet", index=False)


@app.command()
def load(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Stage 3: Load data into PostgreSQL."""
    cfg = yaml.safe_load(config.read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    migrations_dir = Path("src/db/migrations")

    import pandas as pd
    from src.db.connection import get_engine, ping
    from src.db.loader import (
        load_beneficiaries,
        load_chronic_conditions,
        load_inpatient_claims,
        refresh_materialized_views,
        run_migrations,
    )

    engine = get_engine()
    if not ping(engine):
        typer.echo("✗ Cannot reach PostgreSQL. Is Docker running? Run: make up", err=True)
        raise typer.Exit(1)

    run_migrations(engine, migrations_dir)

    bene = pd.read_parquet(processed_dir / "beneficiaries.parquet")
    inpatient = pd.read_parquet(processed_dir / "inpatient_claims.parquet")

    load_beneficiaries(bene, engine)
    load_chronic_conditions(bene, engine)
    load_inpatient_claims(inpatient, engine)
    refresh_materialized_views(engine)
    typer.echo("✓ Load complete")


@app.command()
def features(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Stage 4: Build feature parquet files."""
    cfg = yaml.safe_load(config.read_text())
    from src.features.pipeline import FeaturePipeline
    processed_dir = Path(cfg["data"]["processed_dir"])
    pipeline = FeaturePipeline(cfg)
    train_path, test_path = pipeline.run(processed_dir)
    typer.echo(f"✓ Features built: {train_path}, {test_path}")


@app.command()
def train(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
    version: str = typer.Option(None, help="Model version tag (default: YYYYMM_HHMM)"),
) -> None:
    """Stage 5: Train readmission + LOS + high-cost models."""
    cfg = yaml.safe_load(config.read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["scoring"]["model_dir"])
    v = version or datetime.now().strftime("%Y%m%d_%H%M")

    from src.models.readmission import run_readmission_training
    from src.models.los_regression import train_los_model
    from src.models.high_cost import train_high_cost_model

    train_path = processed_dir / "train_features.parquet"
    test_path = processed_dir / "test_features.parquet"

    result = run_readmission_training(train_path, test_path, model_dir, cfg, version=v)
    typer.echo(f"✓ Readmission model trained (v{v})")

    train_los_model(train_path, test_path, model_dir, cfg, version=v)
    typer.echo("✓ LOS model trained")

    train_high_cost_model(train_path, test_path, model_dir, cfg, version=v)
    typer.echo("✓ High-cost model trained")

    # Print results table to stdout
    typer.echo("\n── Model Results ───────────────────────────────────────────")
    for m in result["metrics"]:
        typer.echo(
            f"  {m['model']:<35} AUROC={m.get('auroc','N/A')}  "
            f"AUPRC={m.get('auprc','N/A')}  Brier={m.get('brier_score','N/A')}"
        )


@app.command()
def score(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Stage 6: Batch score claims and write to analytics.risk_scores."""
    cfg = yaml.safe_load(config.read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["scoring"]["model_dir"])

    from src.scoring.batch_scorer import score_claims
    score_claims(processed_dir / "test_features.parquet", model_dir, cfg)
    typer.echo("✓ Scoring complete")


@app.command()
def kpis(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Stage 7: Compute KPIs and write to analytics.kpi_snapshots."""
    cfg = yaml.safe_load(config.read_text())
    from src.reporting.kpi_builder import compute_and_store_kpis
    rows = compute_and_store_kpis(cfg)
    typer.echo(f"✓ KPIs computed: {rows} rows written")


@app.command()
def report(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Stage 8: Generate weekly HTML report."""
    cfg = yaml.safe_load(config.read_text())
    from src.reporting.html_report import generate_html_report
    reports_dir = Path(cfg["reporting"]["reports_dir"])
    model_dir = Path(cfg["scoring"]["model_dir"])
    path = generate_html_report(cfg, reports_dir, model_dir)
    typer.echo(f"✓ Report written: {path}")


@app.command()
def all(
    config: Path = typer.Option(CONFIG_PATH, help="Path to config.yaml"),
) -> None:
    """Run the full 8-stage pipeline end-to-end."""
    typer.echo("=" * 60)
    typer.echo("  Healthcare Risk KPI — Full Pipeline")
    typer.echo("=" * 60)
    for stage in [download, validate, load, features, train, score, kpis, report]:
        typer.echo(f"\n▶ Running: {stage.__name__}...")
        try:
            ctx = typer.Context(stage)
            stage.callback(config=config) if hasattr(stage, "callback") else stage(config=config)
        except Exception as exc:
            typer.echo(f"✗ Stage '{stage.__name__}' failed: {exc}", err=True)
            raise typer.Exit(1)
    typer.echo("\n✓ Pipeline complete!")


if __name__ == "__main__":
    app()
