"""Runtime configuration and environment validation."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    adapter_mode: Literal["fake", "real"] = "fake"
    allow_langgraph_py314_warning: bool = False
    ffmpeg_binary: str = "ffmpeg"
    provider_timeout_seconds: float = 30.0
    adapter_retry_attempts: int = Field(default=3, ge=1, le=5)
    adapter_initial_backoff_seconds: float = Field(default=0.25, gt=0, le=5)
    adapter_max_backoff_seconds: float = Field(default=2.0, gt=0, le=30)
    adapter_poll_interval_seconds: float = Field(default=1.0, gt=0, le=30)
    adapter_poll_attempts: int = Field(default=120, ge=1, le=10_000)
    assemblyai_api_key: str | None = None
    assemblyai_base_url: str = "https://api.assemblyai.com"
    speechmatics_api_key: str | None = None
    speechmatics_base_url: str = "https://eu1.asr.api.speechmatics.com/v2"
    deepgram_api_key: str | None = None
    deepgram_base_url: str = "https://api.deepgram.com/v1/listen"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1/responses"
    translation_model_id: str = "gpt-5.4-mini"
    translation_prompt_version: str = "phase-3-v1"

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
    adapter_mode: str
    runtime_compatibility_ok: bool
    provider_config_ok: bool
    state_db_error: str | None = None
    runtime_compatibility_error: str | None = None
    provider_config_error: str | None = None


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

    provider_config_error = validate_provider_configuration(settings)
    runtime_compatibility_error = validate_runtime_compatibility(settings)
    effective_error = state_db_error
    if effective_error is None and provider_config_error is not None:
        effective_error = provider_config_error
    if effective_error is None and runtime_compatibility_error is not None:
        effective_error = runtime_compatibility_error

    return ValidationResult(
        ok=(
            all(path.exists() for path in paths)
            and state_db_ok
            and provider_config_error is None
            and runtime_compatibility_error is None
        ),
        checked_paths=paths,
        state_backend="postgres",
        state_db_ok=state_db_ok,
        state_db_target=state_db_target,
        adapter_mode=settings.adapter_mode,
        runtime_compatibility_ok=runtime_compatibility_error is None,
        provider_config_ok=provider_config_error is None,
        state_db_error=effective_error,
        runtime_compatibility_error=runtime_compatibility_error,
        provider_config_error=provider_config_error,
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


def validate_provider_configuration(settings: Settings) -> str | None:
    """Validate provider credentials required for the selected adapter mode."""

    if settings.adapter_mode != "real":
        return None

    missing = [
        name
        for name, value in (
            ("TA_ASSEMBLYAI_API_KEY", settings.assemblyai_api_key),
            ("TA_SPEECHMATICS_API_KEY", settings.speechmatics_api_key),
            ("TA_DEEPGRAM_API_KEY", settings.deepgram_api_key),
            ("TA_OPENAI_API_KEY", settings.openai_api_key),
        )
        if not value
    ]
    if missing:
        return f"real adapter mode requires {', '.join(missing)}"
    return None


def validate_runtime_compatibility(settings: Settings) -> str | None:
    """Gate incompatible LangGraph runtimes before real adapters are enabled."""

    if settings.adapter_mode != "real":
        return None
    compatibility_error = langgraph_real_mode_compatibility_error(
        allow_warning=settings.allow_langgraph_py314_warning,
    )
    if compatibility_error is not None:
        return compatibility_error
    return None


def langgraph_real_mode_compatibility_error(*, allow_warning: bool) -> str | None:
    """Gate real adapter mode if LangGraph still emits the Python 3.14 legacy warning."""

    if allow_warning or os.environ.get("TA_ALLOW_LANGGRAPH_PY314_WARNING") == "1":
        return None

    warning_message = _langgraph_py314_warning()
    if warning_message is None:
        return None
    return (
        "LangGraph real-adapter mode is gated on Python 3.14 because langchain-core still emits "
        f"the legacy compatibility warning: {warning_message}. "
        "Set TA_ALLOW_LANGGRAPH_PY314_WARNING=1 to opt in explicitly."
    )


def _langgraph_py314_warning() -> str | None:
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            from langgraph.graph import StateGraph  # noqa: F401
    except Exception:
        return None

    for warning in captured:
        message = str(warning.message)
        if "Core Pydantic V1 functionality isn't compatible with Python 3.14" in message:
            return message
    return None
