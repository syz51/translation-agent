"""Runtime configuration and environment validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    workspace_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / ".translation-agent")
    blob_dir: Path = Field(default_factory=lambda: Path.cwd() / ".translation-agent" / "blobs")
    state_db_dsn: str | None = None
    trace_dir: Path = Field(default_factory=lambda: Path.cwd() / ".translation-agent" / "traces")
    log_level: str = "INFO"
    emit_console_logs: bool = True

    def model_post_init(self, __context: object) -> None:
        blob_dir_overridden = "blob_dir" in self.model_fields_set
        trace_dir_overridden = "trace_dir" in self.model_fields_set
        self.workspace_dir = self.workspace_dir.expanduser().resolve()
        self.data_dir = self.data_dir.expanduser().resolve()
        if blob_dir_overridden:
            self.blob_dir = self.blob_dir.expanduser().resolve()
        else:
            self.blob_dir = self.data_dir / "blobs"
        if trace_dir_overridden:
            self.trace_dir = self.trace_dir.expanduser().resolve()
        else:
            self.trace_dir = self.data_dir / "traces"


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    checked_paths: tuple[Path, ...]
    state_backend: str
    state_db_ok: bool
    state_db_target: str
    state_db_error: str | None = None


def load_settings() -> Settings:
    """Load settings from the process environment."""

    return Settings()


def validate_environment(settings: Settings, *, create_dirs: bool = True) -> ValidationResult:
    """Validate the configured local runtime paths and Postgres connectivity."""

    paths = _runtime_paths(settings)
    if create_dirs:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    state_db_target = sanitize_db_target(settings.state_db_dsn)
    state_db_ok = False
    state_db_error: str | None = None

    if not settings.state_db_dsn:
        state_db_error = "TA_STATE_DB_DSN is required"
    else:
        try:
            with psycopg.connect(settings.state_db_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            state_db_ok = True
        except Exception as exc:
            state_db_error = str(exc)

    return ValidationResult(
        ok=all(path.exists() for path in paths) and state_db_ok,
        checked_paths=paths,
        state_backend="postgres",
        state_db_ok=state_db_ok,
        state_db_target=state_db_target,
        state_db_error=state_db_error,
    )


def sanitize_db_target(state_db_dsn: str | None) -> str:
    """Return a credential-free database target string for logs and CLI output."""

    if not state_db_dsn:
        return "<missing>"

    try:
        conninfo = conninfo_to_dict(state_db_dsn)
    except Exception:
        return "<invalid>"

    host_value = conninfo.get("host")
    port_value = conninfo.get("port")
    dbname_value = conninfo.get("dbname")
    host = str(host_value) if host_value else ""
    port = str(port_value) if port_value else ""
    dbname = str(dbname_value) if dbname_value else ""

    target = "postgresql://"
    if host:
        target += host
    if port:
        if host:
            target += f":{port}"
        else:
            target += f"localhost:{port}"
    if dbname:
        target += f"/{dbname}"
    elif not host and not port:
        target += "/"
    return target


def _runtime_paths(settings: Settings) -> tuple[Path, Path, Path]:
    return (settings.data_dir, settings.blob_dir, settings.trace_dir)
