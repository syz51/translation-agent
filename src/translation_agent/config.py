"""Runtime configuration and environment validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    blob_dir: Path | None = None
    state_db_path: Path | None = None
    trace_dir: Path | None = None
    log_level: str = "INFO"
    emit_console_logs: bool = True

    def model_post_init(self, __context: object) -> None:
        self.workspace_dir = self.workspace_dir.expanduser().resolve()
        self.data_dir = self.data_dir.expanduser().resolve()
        if self.blob_dir is None:
            self.blob_dir = self.data_dir / "blobs"
        else:
            self.blob_dir = self.blob_dir.expanduser().resolve()
        if self.state_db_path is None:
            self.state_db_path = self.data_dir / "state" / "runs.sqlite3"
        else:
            self.state_db_path = self.state_db_path.expanduser().resolve()
        if self.trace_dir is None:
            self.trace_dir = self.data_dir / "traces"
        else:
            self.trace_dir = self.trace_dir.expanduser().resolve()


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    checked_paths: tuple[Path, ...]


def load_settings() -> Settings:
    """Load settings from the process environment."""

    return Settings()


def validate_environment(settings: Settings, *, create_dirs: bool = True) -> ValidationResult:
    """Validate the configured local runtime paths."""

    paths = (
        settings.data_dir,
        settings.blob_dir,
        settings.state_db_path.parent,
        settings.trace_dir,
    )
    if create_dirs:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    return ValidationResult(ok=all(path.exists() for path in paths), checked_paths=paths)
