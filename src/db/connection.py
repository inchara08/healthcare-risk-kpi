"""SQLAlchemy engine factory — reads credentials from .env or environment variables."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()
logger = logging.getLogger(__name__)


def _build_dsn() -> str:
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5432")
    db = os.environ.get("PG_DB", "healthcare_risk")
    user = os.environ.get("PG_USER", "pguser")
    password = os.environ.get("PG_PASS", "pgpassword")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine(pool_size: int = 5, echo: bool = False) -> Engine:
    dsn = _build_dsn()
    engine = create_engine(dsn, pool_size=pool_size, max_overflow=10, echo=echo)
    logger.info("SQLAlchemy engine created: %s", dsn.split("@")[-1])
    return engine


def ping(engine: Engine | None = None) -> bool:
    """Return True if the database is reachable."""
    eng = engine or get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database ping failed: %s", exc)
        return False
