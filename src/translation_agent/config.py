"""Runtime configuration and environment validation."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import psycopg
from psycopg.conninfo import conninfo_to_dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from translation_agent.storage import SQLiteOperationalStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
_USE_DEFAULT_ENV_FILE = object()
type TranscriptionProviderId = Literal["assemblyai", "speechmatics", "deepgram"]
type LlmProviderId = Literal["gemini", "openai"]
DEFAULT_TRANSCRIPTION_PROVIDERS: tuple[TranscriptionProviderId, ...] = (
    "assemblyai",
    "speechmatics",
    "deepgram",
)
_SUPPORTED_TRANSCRIPTION_PROVIDER_IDS = frozenset(DEFAULT_TRANSCRIPTION_PROVIDERS)
_TRANSCRIPTION_PROVIDER_ENV_VARS: dict[TranscriptionProviderId, str] = {
    "assemblyai": "TA_ASSEMBLYAI_API_KEY",
    "speechmatics": "TA_SPEECHMATICS_API_KEY",
    "deepgram": "TA_DEEPGRAM_API_KEY",
}
_SUPPORTED_LLM_PROVIDER_IDS = frozenset({"gemini", "openai"})
_LLM_PROVIDER_ENV_VARS: dict[LlmProviderId, str] = {
    "gemini": "TA_GEMINI_API_KEY",
    "openai": "TA_OPENAI_API_KEY",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TA_",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workspace_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / ".translation-agent")
    blob_dir: Path = Field(default_factory=lambda: Path.cwd() / ".translation-agent" / "blobs")
    state_db_dsn: str | None = None
    state_db_path: Path = Field(
        default_factory=lambda: Path.cwd() / ".translation-agent" / "state.sqlite3"
    )
    trace_dir: Path = Field(default_factory=lambda: Path.cwd() / ".translation-agent" / "traces")
    log_level: str = "INFO"
    emit_console_logs: bool = True
    adapter_mode: Literal["fake", "real"] = "fake"
    allow_langgraph_py314_warning: bool = False
    ffmpeg_binary: str = "ffmpeg"
    provider_timeout_seconds: float = 300.0
    translation_timeout_seconds: float = 90.0
    assemblyai_timeout_seconds: float = 300.0
    adapter_retry_attempts: int = Field(default=3, ge=1, le=5)
    adapter_initial_backoff_seconds: float = Field(default=0.25, gt=0, le=5)
    adapter_max_backoff_seconds: float = Field(default=2.0, gt=0, le=30)
    adapter_poll_interval_seconds: float = Field(default=1.0, gt=0, le=30)
    adapter_poll_attempts: int = Field(default=120, ge=1, le=10_000)
    transcription_providers: str | tuple[str, ...] | None = None
    assemblyai_api_key: str | None = None
    assemblyai_base_url: str = "https://api.assemblyai.com"
    speechmatics_api_key: str | None = None
    speechmatics_base_url: str = "https://eu1.asr.api.speechmatics.com/v2"
    deepgram_api_key: str | None = None
    deepgram_base_url: str = "https://api.deepgram.com/v1/listen"
    deepgram_utterance_split_seconds: float = Field(default=0.8, ge=0.2, le=3.0)
    translation_provider: LlmProviderId = "gemini"
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    default_source_language: str = "en"
    default_target_language: str = "zh"
    translation_model_id: str = "gemini-3-flash"
    reasoning_provider: LlmProviderId = "openai"
    reasoning_model_id: str = "gpt-5.4"
    translation_prompt_version: str = "phase-3-v1"
    transcription_max_workers: int | None = Field(default=None, ge=1, le=16)
    translation_candidate_max_workers: int = Field(default=2, ge=1, le=16)
    translation_chunk_max_workers: int = Field(default=4, ge=1, le=16)
    review_max_workers: int = Field(default=2, ge=1, le=16)
    reference_evaluation_max_workers: int = Field(default=4, ge=1, le=16)
    memory_drain_max_workers: int = Field(default=2, ge=1, le=16)
    translation_max_chunk_characters: int = Field(default=5_000, ge=250, le=50_000)
    translation_max_chunk_segments: int = Field(default=100, ge=1, le=1_000)
    translation_context_segment_window: int = Field(default=2, ge=0, le=16)

    def model_post_init(self, __context: object) -> None:
        blob_dir_overridden = "blob_dir" in self.model_fields_set
        trace_dir_overridden = "trace_dir" in self.model_fields_set
        state_db_path_overridden = "state_db_path" in self.model_fields_set
        self.workspace_dir = self.workspace_dir.expanduser().resolve()
        self.data_dir = self.data_dir.expanduser().resolve()
        if blob_dir_overridden:
            self.blob_dir = self.blob_dir.expanduser().resolve()
        else:
            self.blob_dir = self.data_dir / "blobs"
        if state_db_path_overridden:
            self.state_db_path = self.state_db_path.expanduser().resolve()
        else:
            self.state_db_path = self.data_dir / "state.sqlite3"
        if trace_dir_overridden:
            self.trace_dir = self.trace_dir.expanduser().resolve()
        else:
            self.trace_dir = self.data_dir / "traces"
        if self.transcription_max_workers is None:
            self.transcription_max_workers = min(
                _configured_transcription_provider_count(self.transcription_providers),
                4,
            )


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


def load_settings(*, env_file: Path | str | None | object = _USE_DEFAULT_ENV_FILE) -> Settings:
    """Load settings from the process environment and the repo-root dotenv file."""

    if env_file is _USE_DEFAULT_ENV_FILE:
        env_file = DEFAULT_ENV_FILE
    resolved_env_file: Path | str | None
    if isinstance(env_file, Path):
        resolved_env_file = env_file.expanduser().resolve()
    else:
        resolved_env_file = cast("Path | str | None", env_file)
    return Settings(_env_file=resolved_env_file)  # pyright: ignore[reportCallIssue]


def validate_environment(settings: Settings, *, create_dirs: bool = True) -> ValidationResult:
    """Validate the configured local runtime paths and operational store connectivity."""

    paths = _runtime_paths(settings)
    if create_dirs:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    state_backend = _state_backend(settings)
    state_db_target = sanitize_db_target(
        settings.state_db_dsn if state_backend == "postgres" else settings.state_db_path
    )
    state_db_ok = False
    state_db_error: str | None = None

    if state_backend == "postgres" and settings.state_db_dsn is not None:
        try:
            with psycopg.connect(settings.state_db_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            state_db_ok = True
        except Exception as exc:
            state_db_error = str(exc)
    else:
        try:
            with SQLiteOperationalStore(settings.state_db_path):
                pass
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
        state_backend=state_backend,
        state_db_ok=state_db_ok,
        state_db_target=state_db_target,
        adapter_mode=settings.adapter_mode,
        runtime_compatibility_ok=runtime_compatibility_error is None,
        provider_config_ok=provider_config_error is None,
        state_db_error=effective_error,
        runtime_compatibility_error=runtime_compatibility_error,
        provider_config_error=provider_config_error,
    )


def sanitize_db_target(state_db_target: str | Path | None) -> str:
    """Return a display-safe database target string for logs and CLI output."""

    if state_db_target is None:
        return "<missing>"
    if isinstance(state_db_target, Path):
        return str(state_db_target.expanduser().resolve())

    state_db_dsn = state_db_target
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


def _state_backend(settings: Settings) -> str:
    return "postgres" if settings.state_db_dsn else "sqlite"


def resolve_transcription_providers(settings: Settings) -> tuple[TranscriptionProviderId, ...]:
    """Return the effective real-mode transcription provider tuple."""

    configured = settings.transcription_providers
    if configured is None:
        return DEFAULT_TRANSCRIPTION_PROVIDERS

    normalized_tokens = tuple(_normalized_transcription_provider_tokens(configured))
    if not normalized_tokens:
        raise ValueError("TA_TRANSCRIPTION_PROVIDERS must select at least one provider when set")

    unsupported: list[str] = []
    unsupported_seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_seen: set[str] = set()
    resolved: list[TranscriptionProviderId] = []
    seen: set[str] = set()

    for provider_id in normalized_tokens:
        if provider_id in seen:
            if provider_id not in duplicate_seen:
                duplicates.append(provider_id)
                duplicate_seen.add(provider_id)
            continue
        seen.add(provider_id)
        if provider_id not in _SUPPORTED_TRANSCRIPTION_PROVIDER_IDS:
            if provider_id not in unsupported_seen:
                unsupported.append(provider_id)
                unsupported_seen.add(provider_id)
            continue
        resolved.append(cast("TranscriptionProviderId", provider_id))

    if unsupported:
        unsupported_values = ", ".join(unsupported)
        raise ValueError(
            f"TA_TRANSCRIPTION_PROVIDERS contains unsupported providers: {unsupported_values}"
        )
    if duplicates:
        duplicate_values = ", ".join(duplicates)
        raise ValueError(
            f"TA_TRANSCRIPTION_PROVIDERS contains duplicate providers: {duplicate_values}"
        )
    return tuple(resolved)


def validate_provider_configuration(settings: Settings) -> str | None:
    """Validate provider credentials required for the selected adapter mode."""

    if settings.adapter_mode != "real":
        return None

    try:
        transcription_providers = resolve_transcription_providers(settings)
    except ValueError as exc:
        return str(exc)

    provider_api_keys: dict[TranscriptionProviderId, str | None] = {
        "assemblyai": settings.assemblyai_api_key,
        "speechmatics": settings.speechmatics_api_key,
        "deepgram": settings.deepgram_api_key,
    }
    missing = [
        _TRANSCRIPTION_PROVIDER_ENV_VARS[provider_id]
        for provider_id in transcription_providers
        if not provider_api_keys[provider_id]
    ]
    translation_provider_error = _missing_llm_provider_env_var(
        settings,
        provider_id=settings.translation_provider,
    )
    if translation_provider_error is not None:
        missing.append(translation_provider_error)
    if _reasoning_provider_enabled(settings):
        reasoning_provider_error = _missing_llm_provider_env_var(
            settings,
            provider_id=settings.reasoning_provider,
        )
        if reasoning_provider_error is not None:
            missing.append(reasoning_provider_error)
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


def _normalized_transcription_provider_tokens(
    configured: str | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(configured, str):
        raw_tokens = configured.split(",")
    else:
        raw_tokens = configured
    return tuple(token.strip().lower() for token in raw_tokens if token.strip())


def _configured_transcription_provider_count(configured: str | tuple[str, ...] | None) -> int:
    if configured is None:
        return len(DEFAULT_TRANSCRIPTION_PROVIDERS)
    return max(1, len(_normalized_transcription_provider_tokens(configured)))


def llm_provider_api_key(settings: Settings, provider_id: LlmProviderId) -> str | None:
    """Return the configured API key for the selected LLM provider."""

    if provider_id == "gemini":
        return settings.gemini_api_key
    return settings.openai_api_key


def llm_provider_base_url(settings: Settings, provider_id: LlmProviderId) -> str | None:
    """Return the configured base URL for the selected LLM provider."""

    if provider_id == "gemini":
        return settings.gemini_base_url
    return settings.openai_base_url


def llm_provider_base_url_source(settings: Settings, provider_id: LlmProviderId) -> str:
    """Describe where the active provider base URL was sourced from."""

    if provider_id == "gemini":
        return "TA_GEMINI_BASE_URL" if "gemini_base_url" in settings.model_fields_set else "default"
    if settings.openai_base_url:
        return "TA_OPENAI_BASE_URL"
    return "openai-sdk-default"


def _missing_llm_provider_env_var(
    settings: Settings,
    *,
    provider_id: LlmProviderId,
) -> str | None:
    if provider_id not in _SUPPORTED_LLM_PROVIDER_IDS:  # pragma: no cover - defensive
        raise RuntimeError(f"unsupported LLM provider: {provider_id}")
    if llm_provider_api_key(settings, provider_id):
        return None
    return _LLM_PROVIDER_ENV_VARS[provider_id]


def _reasoning_provider_enabled(settings: Settings) -> bool:
    """Keep reasoning credential validation dormant until a live adapter exists."""

    del settings
    return False
