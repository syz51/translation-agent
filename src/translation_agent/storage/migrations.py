from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def upgrade_database(dsn: str, *, revision: str = "head") -> None:
    """Apply Alembic migrations to the configured Postgres database."""

    command.upgrade(_build_alembic_config(dsn), revision)


def normalize_sqlalchemy_url(dsn: str) -> str:
    """Translate runtime DSNs to the SQLAlchemy psycopg driver URL when needed."""

    if dsn.startswith("postgresql+psycopg://"):
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+psycopg://", 1)
    return dsn


def _build_alembic_config(dsn: str) -> Config:
    project_root = Path(__file__).resolve().parents[3]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", normalize_sqlalchemy_url(dsn).replace("%", "%%"))
    return config
