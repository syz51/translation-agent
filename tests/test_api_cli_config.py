from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from translation_agent.api import (
    RunJobRequest,
    RunJobResult,
    _failure_details,
    _final_status,
    list_runs,
    resolve_review,
    resume_transcription,
    resume_translation,
    review_job,
    run_job,
)
from translation_agent.cli import main
from translation_agent.config import (
    ValidationResult,
    load_settings,
    resolve_transcription_providers,
    sanitize_db_target,
    validate_environment,
    validate_runtime_compatibility,
)
from translation_agent.errors import TranscriptionProvidersFailedError
from translation_agent.graph import GraphState
from translation_agent.graph.state import RoutingFact
from translation_agent.models import (
    AssetContextInput,
    CandidatePreference,
    JobContext,
    ReviewBundle,
    Segment,
    StructuredEvidence,
    SynthesizedTranscriptArtifact,
    TranslationCandidate,
)
from translation_agent.observability import TraceEvent
from translation_agent.review_flow import _build_contradiction_summary
from translation_agent.run_status import PhaseCounters, RecentRunEvent, RunStatusSnapshot
from translation_agent.storage import (
    LocalBlobStore,
    PostgresOperationalStore,
    PostgresRunStore,
    SQLiteOperationalStore,
    job_path,
    operational_job_key,
)


def _job_context(job_id: str = "job-123") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-local",
        project_id="project-local",
        source_video_ref="input.mp4",
        target_language="zh-CN",
        source_language="en",
        requested_by="system@local",
        created_at=datetime(2026, 3, 31, 0, 0, tzinfo=UTC),
        profile_ref="profiles/default",
        media_key=f"source-ref:{job_id}",
    )


def _artifact_path(*parts: str) -> Path:
    return Path(job_path(_job_context(), *parts))


def _translation_candidate(job_id: str = "job-123") -> TranslationCandidate:
    return TranslationCandidate(
        candidate_id=f"candidate-{job_id}",
        job_id=job_id,
        source_transcript_candidate_id=f"transcript-{job_id}",
        model_id="gpt-5.4",
        prompt_variant_id="prompt-variant-a",
        prompt_version="translation-v1",
        language="fr",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=1_200,
                source_text="Hello.",
                target_text="Bonjour.",
            ),
            Segment(
                segment_id="seg-2",
                start_ms=1_200,
                end_ms=2_400,
                source_text="Skip me.",
                target_text="   ",
            ),
            Segment(
                segment_id="seg-3",
                start_ms=2_400,
                end_ms=3_600,
                source_text="Thank you.",
                target_text="Merci.",
            ),
        ),
        full_text="Bonjour. Merci.",
        normalization_version="translation-candidate/v1",
    )


def _write_translation_candidate(
    path: Path,
    candidate: TranslationCandidate | None = None,
) -> TranslationCandidate:
    persisted_candidate = candidate or _translation_candidate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(persisted_candidate.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return persisted_candidate


def _configure_real_mode_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    transcription_providers: str | None = None,
    translation_provider: str | None = None,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_ADAPTER_MODE", "real")
    monkeypatch.setenv("TA_ALLOW_LANGGRAPH_PY314_WARNING", "1")
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)
    monkeypatch.delenv("TA_ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("TA_SPEECHMATICS_API_KEY", raising=False)
    monkeypatch.delenv("TA_DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("TA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TA_TRANSLATION_PROVIDER", raising=False)
    monkeypatch.delenv("TA_REASONING_PROVIDER", raising=False)
    monkeypatch.delenv("TA_REASONING_MODEL_ID", raising=False)
    if transcription_providers is None:
        monkeypatch.delenv("TA_TRANSCRIPTION_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("TA_TRANSCRIPTION_PROVIDERS", transcription_providers)
    if translation_provider is not None:
        monkeypatch.setenv("TA_TRANSLATION_PROVIDER", translation_provider)


@pytest.mark.unit
def test_load_settings_reads_environment(monkeypatch, tmp_path: Path) -> None:
    postgres_dsn = (
        "postgresql://user:secret@db.example.com:5432/translation_agent"
        "?sslmode=require"
    )  # pragma: allowlist secret
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_DEFAULT_TARGET_LANGUAGE", "ja")
    monkeypatch.setenv("TA_ASSEMBLYAI_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("TA_STATE_DB_DSN", postgres_dsn)

    settings = load_settings()

    assert settings.data_dir == tmp_path / "runtime"
    assert settings.blob_dir == settings.data_dir / "blobs"
    assert settings.trace_dir == settings.data_dir / "traces"
    assert settings.default_target_language == "ja"
    assert settings.assemblyai_timeout_seconds == 300.0
    assert settings.state_db_dsn == (
        "postgresql://user:secret@db.example.com:5432/translation_agent?sslmode=require"
    )


@pytest.mark.unit
def test_load_settings_defaults_provider_timeout_to_five_minutes(monkeypatch) -> None:
    monkeypatch.delenv("TA_PROVIDER_TIMEOUT_SECONDS", raising=False)

    settings = load_settings(env_file=None)

    assert settings.provider_timeout_seconds == 300.0


@pytest.mark.unit
def test_load_settings_autoloads_repo_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"TA_DATA_DIR={tmp_path / 'runtime-from-dotenv'}",
                "TA_ADAPTER_MODE=fake",
                "TA_STATE_DB_DSN=postgresql://dotenv:secret@127.0.0.1:55432/translation_agent",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("translation_agent.config.DEFAULT_ENV_FILE", env_file)

    settings = load_settings()

    assert settings.data_dir == tmp_path / "runtime-from-dotenv"
    assert settings.state_db_dsn == ("postgresql://dotenv:secret@127.0.0.1:55432/translation_agent")


@pytest.mark.unit
def test_environment_variables_override_repo_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"TA_DATA_DIR={tmp_path / 'runtime-from-dotenv'}",
                "TA_STATE_DB_DSN=postgresql://dotenv:secret@127.0.0.1:55432/translation_agent",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("translation_agent.config.DEFAULT_ENV_FILE", env_file)
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime-from-env"))
    monkeypatch.setenv(
        "TA_STATE_DB_DSN", "postgresql://env:secret@127.0.0.1:55433/translation_agent"
    )

    settings = load_settings()

    assert settings.data_dir == tmp_path / "runtime-from-env"
    assert settings.state_db_dsn == "postgresql://env:secret@127.0.0.1:55433/translation_agent"


@pytest.mark.unit
def test_validate_environment_defaults_to_local_sqlite_without_state_db_dsn(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is True
    assert result.state_backend == "sqlite"
    assert result.adapter_mode == "fake"
    assert result.state_db_ok is True
    assert result.state_db_target == str((tmp_path / "runtime" / "state.sqlite3").resolve())
    assert result.state_db_error is None
    assert result.runtime_compatibility_ok is True
    assert result.provider_config_ok is True
    for path in result.checked_paths:
        assert path.exists()
    assert (tmp_path / "runtime" / "state.sqlite3").exists()


@pytest.mark.unit
def test_validate_environment_fails_cleanly_for_unreachable_dsn(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv(
        "TA_STATE_DB_DSN",
        "postgresql://user:secret@127.0.0.1:1/translation_agent?connect_timeout=1",
    )

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.state_db_ok is False
    assert result.state_db_target == "postgresql://127.0.0.1:1/translation_agent"
    assert "secret" not in result.state_db_target
    assert "connect_timeout" not in result.state_db_target
    assert result.state_db_error is not None


@pytest.mark.unit
def test_validate_environment_real_mode_requires_provider_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", "postgresql://user:secret@127.0.0.1:1/app")
    monkeypatch.setenv("TA_ADAPTER_MODE", "real")
    monkeypatch.delenv("TA_ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("TA_SPEECHMATICS_API_KEY", raising=False)
    monkeypatch.delenv("TA_DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("TA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TA_OPENAI_API_KEY", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error is not None
    assert "TA_ASSEMBLYAI_API_KEY" in result.provider_config_error
    assert "TA_GEMINI_API_KEY" in result.provider_config_error


@pytest.mark.unit
def test_validate_environment_real_mode_unset_selector_requires_all_stt_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(monkeypatch, tmp_path)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error == (
        "real adapter mode requires TA_ASSEMBLYAI_API_KEY, "
        "TA_SPEECHMATICS_API_KEY, TA_DEEPGRAM_API_KEY, TA_GEMINI_API_KEY"
    )


@pytest.mark.unit
def test_validate_environment_real_mode_assemblyai_selector_requires_only_selected_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(monkeypatch, tmp_path, transcription_providers="assemblyai")

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error == (
        "real adapter mode requires TA_ASSEMBLYAI_API_KEY, TA_GEMINI_API_KEY"
    )


@pytest.mark.unit
def test_validate_environment_real_mode_subset_selector_requires_only_subset_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(monkeypatch, tmp_path, transcription_providers="assemblyai,deepgram")

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error == (
        "real adapter mode requires TA_ASSEMBLYAI_API_KEY, TA_DEEPGRAM_API_KEY, TA_GEMINI_API_KEY"
    )


@pytest.mark.unit
def test_validate_environment_real_mode_rejects_unknown_transcription_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(monkeypatch, tmp_path, transcription_providers="assemblyai,foo")

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error == (
        "TA_TRANSCRIPTION_PROVIDERS contains unsupported providers: foo"
    )


@pytest.mark.unit
def test_validate_environment_real_mode_rejects_empty_transcription_provider_selector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(monkeypatch, tmp_path, transcription_providers=" , , ")

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error == (
        "TA_TRANSCRIPTION_PROVIDERS must select at least one provider when set"
    )


@pytest.mark.unit
def test_validate_environment_real_mode_rejects_duplicate_transcription_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(
        monkeypatch,
        tmp_path,
        transcription_providers="assemblyai, deepgram, AssemblyAI",
    )

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error == (
        "TA_TRANSCRIPTION_PROVIDERS contains duplicate providers: assemblyai"
    )


@pytest.mark.unit
def test_resolve_transcription_providers_normalizes_case_and_whitespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(monkeypatch, tmp_path, transcription_providers="ASSEMBLYAI, deepgram")

    settings = load_settings()

    assert resolve_transcription_providers(settings) == ("assemblyai", "deepgram")


@pytest.mark.unit
def test_validate_environment_real_mode_requires_langgraph_py314_opt_in(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", "postgresql://user:secret@127.0.0.1:1/app")
    monkeypatch.setenv("TA_ADAPTER_MODE", "real")
    monkeypatch.setenv("TA_ASSEMBLYAI_API_KEY", "assembly")
    monkeypatch.setenv("TA_SPEECHMATICS_API_KEY", "speech")
    monkeypatch.setenv("TA_DEEPGRAM_API_KEY", "deepgram")
    monkeypatch.setenv("TA_GEMINI_API_KEY", "gemini")
    monkeypatch.delenv("TA_ALLOW_LANGGRAPH_PY314_WARNING", raising=False)
    monkeypatch.setattr(
        "translation_agent.config._langgraph_py314_warning",
        lambda: "Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    )
    monkeypatch.setattr(
        "translation_agent.config._langgraph_py314_warning",
        lambda: "Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    )

    compatibility_error = validate_runtime_compatibility(load_settings())

    assert compatibility_error is not None
    assert "TA_ALLOW_LANGGRAPH_PY314_WARNING=1" in compatibility_error


@pytest.mark.unit
def test_cli_validate_config_json_missing_dsn(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["validate-config", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["adapter_mode"] == "fake"
    assert payload["runtime_compatibility_ok"] is True
    assert payload["provider_config_ok"] is True
    assert payload["state_backend"] == "sqlite"
    assert payload["state_db_ok"] is True
    assert payload["state_db_target"] == str((tmp_path / "runtime" / "state.sqlite3").resolve())


@pytest.mark.unit
def test_cli_validate_config_json_unreachable_dsn(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv(
        "TA_STATE_DB_DSN",
        "postgresql://user:secret@127.0.0.1:1/translation_agent?connect_timeout=1",
    )

    exit_code = main(["validate-config", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["adapter_mode"] == "fake"
    assert payload["state_db_ok"] is False
    assert payload["state_db_target"] == "postgresql://127.0.0.1:1/translation_agent"


@pytest.mark.unit
def test_validate_environment_real_mode_requires_provider_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_ADAPTER_MODE", "real")
    monkeypatch.setenv(
        "TA_STATE_DB_DSN",
        "postgresql://user:secret@127.0.0.1:1/translation_agent?connect_timeout=1",
    )
    monkeypatch.delenv("TA_ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("TA_SPEECHMATICS_API_KEY", raising=False)
    monkeypatch.delenv("TA_DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("TA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TA_OPENAI_API_KEY", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.adapter_mode == "real"
    assert result.provider_config_ok is False
    assert result.provider_config_error is not None
    assert "TA_ASSEMBLYAI_API_KEY" in result.provider_config_error
    assert "TA_GEMINI_API_KEY" in result.provider_config_error
    assert result.runtime_compatibility_ok is True


@pytest.mark.unit
def test_cli_validate_config_human_readable_output_includes_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv(
        "TA_STATE_DB_DSN",
        "postgresql://user:secret@127.0.0.1:1/translation_agent?connect_timeout=1",
    )

    exit_code = main(["validate-config"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "configuration invalid" in output
    assert "postgres: postgresql://127.0.0.1:1/translation_agent" in output
    assert "database connectivity failed" in output
    assert "runtime compatibility ok" in output
    assert "provider configuration ok" in output


@pytest.mark.unit
def test_cli_migrate_db_requires_postgres_dsn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["migrate-db"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "TA_STATE_DB_DSN is required for migrate-db" in output


@pytest.mark.unit
def test_cli_migrate_db_runs_upgrade_with_loaded_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str]] = []
    dsn = "postgresql://user:secret@127.0.0.1:55432/translation_agent"
    monkeypatch.setenv("TA_STATE_DB_DSN", dsn)
    monkeypatch.setattr(
        "translation_agent.cli.upgrade_database",
        lambda received_dsn, *, revision: calls.append((received_dsn, revision)),
    )

    exit_code = main(["migrate-db", "--revision", "head"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == [(dsn, "head")]
    assert "migrated postgresql://127.0.0.1:55432/translation_agent to head" in output


@pytest.mark.unit
def test_cli_convert_json_to_srt_json_output_includes_conversion_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "candidates" / "translations" / "candidate-job-123.json"
    candidate = _write_translation_candidate(source_path)

    exit_code = main(["convert-json-to-srt", str(source_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "source_path": str(source_path.resolve()),
        "output_path": str(source_path.with_suffix(".srt").resolve()),
        "job_id": candidate.job_id,
        "candidate_id": candidate.candidate_id,
        "language": candidate.language,
        "subtitle_count": 2,
    }


@pytest.mark.unit
def test_cli_convert_json_to_srt_plain_output_reports_output_path_and_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "published" / "translation.json"
    _write_translation_candidate(source_path)

    exit_code = main(["convert-json-to-srt", str(source_path)])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exit_code == 0
    assert lines == [
        str(source_path.with_suffix(".srt").resolve()),
        "subtitles: 2",
    ]


@pytest.mark.unit
def test_cli_run_job_plain_output_reports_run_status_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-plain"])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    default_output_path = (
        tmp_path
        / "runtime"
        / "blobs"
        / job_path(_job_context("job-plain"), "exports", "translation.srt")
    ).resolve()
    assert exit_code == 0
    assert len(lines) == 7
    assert lines[1] == "completed"
    assert lines[2] == "source_language: en"
    assert lines[3] == "target_language: zh-CN"
    assert lines[4] == f"sqlite: {(tmp_path / 'runtime' / 'state.sqlite3').resolve()}"
    assert Path(lines[5]).exists()
    assert lines[6] == f"default_output_path: {default_output_path}"


@pytest.mark.unit
def test_cli_run_job_json_includes_default_output_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-json", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "zh-CN"
    assert payload["default_output_path"] == str(
        (
            tmp_path
            / "runtime"
            / "blobs"
            / job_path(_job_context("job-json"), "exports", "translation.srt")
        ).resolve()
    )


@pytest.mark.unit
def test_cli_run_job_tty_live_panel_renders_recent_events_and_counters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = Path("/tmp/trace-live.jsonl")

    def fake_run_job(request, settings=None, live_trace_sink=None) -> RunJobResult:
        del settings
        if live_trace_sink is not None:
            live_trace_sink.record(
                TraceEvent(
                    run_id="run-live",
                    name="run.bootstrapped",
                    attributes={"job_id": request.job_id or "job-live"},
                )
            )
            live_trace_sink.record(
                TraceEvent(
                    run_id="run-live",
                    name="run.started",
                    attributes={"job_id": request.job_id or "job-live"},
                )
            )
            live_trace_sink.record(
                TraceEvent(
                    run_id="run-live",
                    name="node.started",
                    attributes={"node_name": "fanout_transcription", "execution_id": "exec-1"},
                )
            )
            live_trace_sink.record(
                TraceEvent(
                    run_id="run-live",
                    name="transcription.provider.completed",
                    attributes={
                        "provider_id": "deepgram",
                        "provider_total": 2,
                        "candidate_id": "tr-1",
                    },
                )
            )
            live_trace_sink.record(
                TraceEvent(
                    run_id="run-live",
                    name="transcription.provider.failed",
                    attributes={
                        "provider_id": "speechmatics",
                        "provider_total": 2,
                        "error": "timeout",
                    },
                )
            )
            live_trace_sink.record(
                TraceEvent(
                    run_id="run-live",
                    name="run.completed",
                    attributes={"status": "completed"},
                )
            )
        return RunJobResult(
            run_id="run-live",
            job_id=request.job_id or "job-live",
            status="completed",
            source=request.source,
            source_language="en",
            target_language="zh-CN",
            blob_root=Path("/tmp/blob-root"),
            trace_path=trace_path,
            state_backend="sqlite",
            state_db_target="/tmp/state.sqlite3",
        )

    monkeypatch.setattr("translation_agent.cli.run_job", fake_run_job)
    monkeypatch.setattr("translation_agent.cli._has_tty", lambda: True)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-live"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "providers: active=0 completed=1 failed=1 total=2" in output
    assert "recent events:" in output
    assert "Failed transcription provider speechmatics: timeout" in output
    assert "run-live" in output
    assert "completed" in output


@pytest.mark.unit
def test_cli_show_run_json_payload_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("TA_DATA_DIR", str(runtime_dir))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    with SQLiteOperationalStore(runtime_dir / "state.sqlite3") as store:
        store.create_run(
            run_id="run-show",
            status="completed",
            input_data={"job_id": "job-show", "source": "input.wav"},
            created_at="2026-04-03T00:00:00+00:00",
        )
        store.update_run(
            "run-show",
            status="completed",
            output_data={"final_stage": "finalize_outputs"},
            updated_at="2026-04-03T00:00:05+00:00",
        )
    trace_path = runtime_dir / "traces" / "run-show.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-show",
                        "name": "run.started",
                        "timestamp": "2026-04-03T00:00:00+00:00",
                        "attributes": {"job_id": "job-show"},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-show",
                        "name": "run.completed",
                        "timestamp": "2026-04-03T00:00:05+00:00",
                        "attributes": {"status": "completed"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(["show-run", "run-show", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["run_id"] == "run-show"
    assert payload["job_id"] == "job-show"
    assert payload["status"] == "completed"
    assert payload["current_stage"] == "finalize_outputs"
    assert payload["active_node"] is None
    assert payload["elapsed_seconds"] == 5.0
    assert payload["trace_path"] == str(trace_path)
    assert len(payload["recent_events"]) == 2


@pytest.mark.unit
def test_cli_watch_run_tty_exits_on_terminal_state_and_renders_recent_events(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshots = iter(
        [
            RunStatusSnapshot(
                run_id="run-watch",
                job_id="job-watch",
                status="running",
                current_stage="fanout_transcription",
                active_node="fanout_transcription",
                elapsed_seconds=2.0,
                trace_path=Path("/tmp/run-watch.jsonl"),
                transcription_providers=PhaseCounters(total=2, active=1, completed=0, failed=0),
                recent_events=(
                    RecentRunEvent(
                        timestamp="2026-04-03T00:00:02+00:00",
                        name="transcription.provider.started",
                        message="Started transcription provider deepgram",
                    ),
                ),
            ),
            RunStatusSnapshot(
                run_id="run-watch",
                job_id="job-watch",
                status="completed",
                current_stage="finalize_outputs",
                active_node=None,
                elapsed_seconds=5.0,
                trace_path=Path("/tmp/run-watch.jsonl"),
                recent_events=(
                    RecentRunEvent(
                        timestamp="2026-04-03T00:00:05+00:00",
                        name="run.completed",
                        message="Run completed with status completed",
                    ),
                ),
            ),
        ]
    )
    monkeypatch.setattr("translation_agent.cli._has_tty", lambda: True)
    monkeypatch.setattr("translation_agent.cli.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "translation_agent.cli.get_run_status",
        lambda run_id, settings=None: next(snapshots),
    )

    exit_code = main(["watch-run", "run-watch", "--interval", "0.01"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Started transcription provider deepgram" in output
    assert "Run completed with status completed" in output


@pytest.mark.unit
def test_cli_list_runs_plain_output_reports_summary_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        store.create_run(
            run_id="run-older",
            status="completed",
            input_data={"job_id": "job-older", "source": "older.wav"},
            created_at="2026-04-01T00:00:00+00:00",
        )
        store.create_run(
            run_id="run-newer",
            status="failed",
            input_data={"job_id": "job-newer", "source": "newer.wav"},
            created_at="2026-04-02T00:00:00+00:00",
        )

    exit_code = main(["list-runs"])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exit_code == 0
    assert lines == [
        (
            "run-newer status=failed job_id=job-newer created_at=2026-04-02T00:00:00+00:00 "
            "source=newer.wav"
        ),
        (
            "run-older status=completed job_id=job-older created_at=2026-04-01T00:00:00+00:00 "
            "source=older.wav"
        ),
    ]


@pytest.mark.unit
def test_cli_list_runs_json_returns_persisted_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        store.create_run(
            run_id="run-older",
            tenant_id="tenant-local",
            project_id="project-local",
            status="completed",
            input_data={"job_id": "job-older", "source": "older.wav"},
            metadata={"kind": "test"},
            created_at="2026-04-01T00:00:00+00:00",
        )
        store.create_run(
            run_id="run-newer",
            tenant_id="tenant-local",
            project_id="project-local",
            status="failed",
            input_data={"job_id": "job-newer", "source": "newer.wav"},
            metadata={"kind": "test"},
            created_at="2026-04-02T00:00:00+00:00",
        )

    exit_code = main(["list-runs", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == [
        {
            "run_id": "run-newer",
            "tenant_id": "tenant-local",
            "project_id": "project-local",
            "status": "failed",
            "created_at": "2026-04-02T00:00:00+00:00",
            "updated_at": "2026-04-02T00:00:00+00:00",
            "input_data": {"job_id": "job-newer", "source": "newer.wav"},
            "output_data": None,
            "metadata": {"kind": "test"},
            "error": None,
        },
        {
            "run_id": "run-older",
            "tenant_id": "tenant-local",
            "project_id": "project-local",
            "status": "completed",
            "created_at": "2026-04-01T00:00:00+00:00",
            "updated_at": "2026-04-01T00:00:00+00:00",
            "input_data": {"job_id": "job-older", "source": "older.wav"},
            "output_data": None,
            "metadata": {"kind": "test"},
            "error": None,
        },
    ]


@pytest.mark.unit
def test_cli_run_job_plain_output_reports_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "runtime" / "traces" / "run-failed.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "translation_agent.cli.run_job",
        lambda request, settings=None: RunJobResult(
            run_id="run-failed",
            job_id=request.job_id or "job-failed",
            status="translation_failed",
            source=request.source,
            source_language=request.source_language or "en",
            target_language=request.target_language or "zh-CN",
            blob_root=tmp_path / "runtime" / "blobs",
            trace_path=trace_path,
            state_backend="sqlite",
            state_db_target=str((tmp_path / "runtime" / "state.sqlite3").resolve()),
            failure_ref="jobs/job-failed/published/translation-failed.json",
            failure_summary="All translation variants failed; transcript preserved for recovery.",
            failure_reasons=(
                "variant-a: simulated translation failure for variant-a",
                "variant-b: simulated translation failure for variant-b",
            ),
        ),
    )

    exit_code = main(["run-job", "input.wav", "--job-id", "job-failed"])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exit_code == 0
    assert lines == [
        "run-failed",
        "translation_failed",
        "source_language: en",
        "target_language: zh-CN",
        f"sqlite: {(tmp_path / 'runtime' / 'state.sqlite3').resolve()}",
        str(trace_path),
        "All translation variants failed; transcript preserved for recovery.",
        "variant-a: simulated translation failure for variant-a",
        "variant-b: simulated translation failure for variant-b",
    ]


@pytest.mark.unit
def test_cli_unsupported_command_uses_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeParser:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def parse_args(self, argv: list[str] | None) -> Namespace:
            del argv
            return Namespace(command="mystery")

        def error(self, message: str) -> None:
            self.messages.append(message)
            raise SystemExit(2)

    parser = FakeParser()
    monkeypatch.setattr("translation_agent.cli.build_parser", lambda: parser)

    with pytest.raises(SystemExit) as exc_info:
        main(["mystery"])

    assert exc_info.value.code == 2
    assert parser.messages == ["unsupported command: mystery"]


@pytest.mark.integration
def test_run_job_bootstraps_local_artifacts_and_postgres_record(
    migrated_postgres_dsn: str, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", migrated_postgres_dsn)

    result = run_job(RunJobRequest(source="input.mp4", job_id="job-123"))

    assert result.status == "completed"
    assert result.source_language == "en"
    assert result.target_language == "zh-CN"
    assert result.blob_root.exists()
    assert result.trace_path.exists()
    assert (
        result.default_output_path
        == (result.blob_root / _artifact_path("exports", "translation.srt")).resolve()
    )
    assert result.default_output_path is not None
    assert (result.blob_root / "jobs" / f"{result.run_id}-request.json").exists()
    assert (result.blob_root / _artifact_path("published", "transcript.json")).exists()
    assert (result.blob_root / _artifact_path("published", "translation.json")).exists()
    assert result.default_output_path.exists()
    assert result.state_backend == "postgres"
    assert result.state_db_target == sanitize_db_target(migrated_postgres_dsn)
    assert result.failure_ref is None
    assert result.failure_summary is None
    assert result.failure_reasons == ()

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(result.run_id)
        node_executions = store.list_node_executions(result.run_id)

    assert record is not None
    assert record.status == "completed"
    assert record.input_data == {
        "artifact_ref": f"jobs/{result.run_id}-request.json",
        "asset_id": None,
        "job_id": "job-123",
        "media_fingerprint": None,
        "media_key": f"source-ref:{result.run_id}",
        "reference_mode": "none",
        "source": "input.mp4",
    }
    assert record.output_data is not None
    assert record.output_data["final_stage"] == "finalize_outputs"
    assert len(node_executions) == 16


@pytest.mark.unit
def test_run_job_uses_configured_default_target_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_DEFAULT_TARGET_LANGUAGE", "zh")
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(RunJobRequest(source="input.mp4", job_id="job-default-target"))
    request_payload = json.loads(
        (result.blob_root / "jobs" / f"{result.run_id}-request.json").read_text(encoding="utf-8")
    )

    assert request_payload["source_language"] == "en"
    assert request_payload["target_language"] == "zh-CN"
    assert result.source_language == "en"
    assert result.target_language == "zh-CN"
    assert (
        result.default_output_path
        == (
            result.blob_root
            / Path(job_path(_job_context("job-default-target"), "exports", "translation.srt"))
        ).resolve()
    )


@pytest.mark.integration
def test_run_job_persists_request_asset_context_in_postgres_runtime(
    migrated_postgres_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", migrated_postgres_dsn)

    request = RunJobRequest(
        source="input.mp4",
        job_id="job-asset-context",
        asset_context=AssetContextInput(
            canonical_title="Episode 1",
            content_type="episode",
            series_id="series-1",
            franchise_id="franchise-1",
            channel_id="channel-1",
            speaker_ids=("speaker-1",),
            topic_tags=("finance",),
            style_profile_id="style-1",
            metadata_confidence="high",
            metadata_sources=("request",),
        ),
    )

    result = run_job(request)
    request_payload = json.loads(
        (result.blob_root / "jobs" / f"{result.run_id}-request.json").read_text(encoding="utf-8")
    )

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(result.run_id)

    with PostgresOperationalStore(migrated_postgres_dsn) as store:
        asset_context = store.get_asset_context(record.input_data["media_key"] if record else "")

    assert request_payload["asset_context"]["series_id"] == "series-1"
    assert asset_context is not None
    assert asset_context.canonical_title == "Episode 1"


@pytest.mark.integration
def test_cli_convert_json_to_srt_matches_published_export_in_postgres_runtime(
    migrated_postgres_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", migrated_postgres_dsn)

    result = run_job(RunJobRequest(source="input.mp4", job_id="job-convert"))
    published_translation_path = result.blob_root / job_path(
        _job_context("job-convert"),
        "published",
        "translation.json",
    )
    existing_export_path = result.blob_root / job_path(
        _job_context("job-convert"),
        "exports",
        "translation.srt",
    )
    converted_output_path = tmp_path / "converted" / "translation-from-json.srt"

    exit_code = main(
        [
            "convert-json-to-srt",
            str(published_translation_path),
            "--output",
            str(converted_output_path),
        ]
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exit_code == 0
    assert lines == [
        str(converted_output_path.resolve()),
        "subtitles: 2",
    ]
    assert converted_output_path.read_text(encoding="utf-8") == existing_export_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_run_job_defaults_to_local_sqlite_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(RunJobRequest(source="input.mp4", job_id="job-local"))

    assert result.status == "completed"
    assert result.source_language == "en"
    assert result.target_language == "zh-CN"
    assert result.state_backend == "sqlite"
    assert result.state_db_target == str((tmp_path / "runtime" / "state.sqlite3").resolve())
    assert result.failure_ref is None
    assert result.failure_summary is None
    assert result.failure_reasons == ()
    assert (tmp_path / "runtime" / "state.sqlite3").exists()
    local_job = _job_context(job_id="job-local")
    assert (
        result.default_output_path
        == (result.blob_root / Path(job_path(local_job, "exports", "translation.srt"))).resolve()
    )
    assert (result.blob_root / Path(job_path(local_job, "published", "transcript.json"))).exists()
    assert (result.blob_root / Path(job_path(local_job, "published", "translation.json"))).exists()
    assert result.default_output_path is not None
    assert result.default_output_path.exists()


@pytest.mark.unit
def test_run_job_rejects_invalid_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = load_settings().model_copy(update={"data_dir": tmp_path / "runtime"})
    validation = ValidationResult(
        ok=False,
        checked_paths=(tmp_path / "runtime",),
        state_backend="sqlite",
        state_db_ok=True,
        state_db_target=str((tmp_path / "runtime" / "state.sqlite3").resolve()),
        adapter_mode="real",
        runtime_compatibility_ok=False,
        provider_config_ok=False,
        state_db_error=None,
        runtime_compatibility_error="runtime blocked",
        provider_config_error="missing provider keys",
    )
    monkeypatch.setattr("translation_agent.api.validate_environment", lambda _: validation)

    with pytest.raises(RuntimeError, match="invalid runtime configuration"):
        run_job(RunJobRequest(source="input.mp4"), settings=settings)


@pytest.mark.unit
def test_final_status_prefers_translation_failure_then_human_review_then_degraded() -> None:
    base = GraphState(
        run_id="run-status",
        job=_job_context(),
        current_stage="finalize_outputs",
        source_video_ref="input.mp4",
    )

    assert (
        _final_status(
            base.model_copy(update={"translation_failed": True, "human_review_required": True})
        )
        == "translation_failed"
    )
    assert _final_status(base.model_copy(update={"human_review_required": True})) == (
        "human_review_required"
    )
    assert (
        _final_status(
            base.model_copy(
                update={
                    "routing_facts": (
                        RoutingFact(
                            stage="fanout_transcription",
                            fact_type="transcription_provider_failed",
                            value="speechmatics",
                        ),
                    )
                }
            )
        )
        == "completed_with_degraded_transcription"
    )
    assert _final_status(base) == "completed"


@pytest.mark.unit
def test_run_job_persists_long_term_memory_across_separate_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    first = RunJobRequest(source="input.mp4", job_id="job-memory-a")
    second = RunJobRequest(source="input.mp4", job_id="job-memory-b")

    run_job(first)
    second_result = run_job(second)

    second_job = _job_context(job_id="job-memory-b")
    memory_bundle = json.loads(
        (
            second_result.blob_root
            / Path(job_path(second_job, "memory", "recall", "translation-review.json"))
        ).read_text(encoding="utf-8")
    )

    assert any(
        entry["content"].startswith("translation adjudication trusted tl-variant-a-job-memory-a-")
        for entry in memory_bundle["semantic_memory"]
    )


@pytest.mark.unit
def test_run_job_returns_translation_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-translation-failed",
            metadata={"scenario": "translation_failed"},
        )
    )

    assert result.status == "translation_failed"
    assert result.source_language == "en"
    assert result.target_language == "zh-CN"
    assert result.default_output_path is None
    assert result.failure_ref == str(
        job_path(_job_context("job-translation-failed"), "published", "translation-failed.json")
    )
    assert result.failure_summary == (
        "All translation variants failed; transcript preserved for recovery."
    )
    assert result.failure_reasons == (
        "variant-a: simulated translation failure for variant-a",
        "variant-b: simulated translation failure for variant-b",
    )


@pytest.mark.unit
def test_run_job_persists_transcription_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    settings = load_settings()

    def fake_build_runtime(**kwargs: object) -> object:
        return SimpleNamespace(run_store=kwargs["run_store"], scenario="happy")

    def fail_run_workflow(*_: object) -> object:
        raise TranscriptionProvidersFailedError(
            {
                "assemblyai": "network failure: The write operation timed out",
                "speechmatics": "http 401",
            }
        )

    monkeypatch.setattr("translation_agent.api.build_runtime", fake_build_runtime)
    monkeypatch.setattr("translation_agent.api.run_workflow", fail_run_workflow)

    with pytest.raises(RuntimeError, match="all transcription providers failed"):
        run_job(RunJobRequest(source=str(source_path)), settings=settings)

    with SQLiteOperationalStore(settings.state_db_path) as store:
        persisted_run = store.list_runs()[0]

    assert persisted_run.status == "failed"
    assert persisted_run.error == {
        "message": "all transcription providers failed",
        "category": "transcription_failed",
        "reason": "all_transcription_providers_failed",
        "provider_errors": [
            {
                "provider_id": "assemblyai",
                "message": "network failure: The write operation timed out",
            },
            {
                "provider_id": "speechmatics",
                "message": "http 401",
            },
        ],
    }


@pytest.mark.unit
def test_list_runs_returns_reverse_chronological_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    state_db_path = tmp_path / "runtime" / "state.sqlite3"
    with SQLiteOperationalStore(state_db_path) as store:
        store.create_run(
            run_id="run-older",
            status="completed",
            input_data={"job_id": "job-older", "source": "older.wav"},
            created_at="2026-04-01T00:00:00+00:00",
        )
        store.create_run(
            run_id="run-newer",
            status="failed",
            input_data={"job_id": "job-newer", "source": "newer.wav"},
            created_at="2026-04-02T00:00:00+00:00",
        )

    records = list_runs()

    assert [record.run_id for record in records] == ["run-newer", "run-older"]
    assert [record.status for record in records] == ["failed", "completed"]
    assert records[0].input_data == {"job_id": "job-newer", "source": "newer.wav"}


@pytest.mark.unit
def test_run_job_human_review_required_leaves_default_output_path_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-human-review",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )

    assert result.status == "human_review_required"
    assert result.review_required_stage == "translation"
    assert result.source_language == "en"
    assert result.target_language == "zh-CN"
    assert result.default_output_path is None


@pytest.mark.unit
def test_run_job_persists_structured_translation_failure_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-translation-failed-structured",
            metadata={"scenario": "translation_failed"},
        )
    )

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        record = store.get_run(result.run_id)

    assert record is not None
    assert record.status == "translation_failed"
    assert record.error == {
        "category": "translation_failed",
        "reason": "all_translation_variants_failed",
        "message": "All translation variants failed; transcript preserved for recovery.",
        "provider_id": "fake-translation",
        "failure_ref": str(
            job_path(
                _job_context("job-translation-failed-structured"),
                "published",
                "translation-failed.json",
            )
        ),
        "failure_summary": "All translation variants failed; transcript preserved for recovery.",
        "failure_reasons": [
            "variant-a: simulated translation failure for variant-a",
            "variant-b: simulated translation failure for variant-b",
        ],
        "retryable": False,
    }
    assert record.output_data["failure_ref"] == str(
        job_path(
            _job_context("job-translation-failed-structured"),
            "published",
            "translation-failed.json",
        )
    )
    assert record.output_data["failure_summary"] == (
        "All translation variants failed; transcript preserved for recovery."
    )
    assert record.output_data["failure_reasons"] == [
        "variant-a: simulated translation failure for variant-a",
        "variant-b: simulated translation failure for variant-b",
    ]


@pytest.mark.unit
def test_resume_translation_from_failed_run_skips_transcription(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    source = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-resume-failed",
            metadata={"scenario": "translation_failed"},
        )
    )

    resumed = resume_translation(source.run_id)

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        resumed_record = store.get_run(resumed.run_id)
        node_names = [record.node_name for record in store.list_node_executions(resumed.run_id)]
        resumed_transcripts = store.list_transcript_candidates(
            resumed.job_id,
            storage_job_id=operational_job_key(_job_context(resumed.job_id)),
        )

    assert resumed.status == "translation_failed"
    assert resumed.run_id != source.run_id
    assert resumed.job_id != source.job_id
    assert resumed_record is not None
    assert resumed_record.input_data["resumed_from_run_id"] == source.run_id
    assert node_names == [
        "generate_translation_candidates",
        "normalize_translations",
        "review_translations",
        "adjudicate_translation",
        "background_memory_pipeline",
        "finalize_outputs",
    ]
    assert len(resumed_transcripts) == 3
    assert all(candidate.job_id == resumed.job_id for candidate in resumed_transcripts)
    assert all(source.job_id in candidate.candidate_id for candidate in resumed_transcripts)
    assert all(resumed.job_id not in candidate.candidate_id for candidate in resumed_transcripts)
    assert resumed.failure_ref == str(
        job_path(_job_context(resumed.job_id), "published", "translation-failed.json")
    )


@pytest.mark.unit
def test_resume_translation_from_partial_variant_run_completes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    source = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-resume-partial",
            metadata={"scenario": "translation_single_variant"},
        )
    )

    resumed = resume_translation(source.run_id)

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        node_names = [record.node_name for record in store.list_node_executions(resumed.run_id)]

    assert resumed.status == "completed"
    assert resumed.failure_ref is None
    assert resumed.default_output_path is not None
    assert "fanout_transcription" not in node_names
    assert "extract_audio" not in node_names
    assert node_names[0] == "generate_translation_candidates"


def test_resume_transcription_retries_only_failed_providers_and_preserves_successful_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    source = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-resume-transcription",
            metadata={"scenario": "degraded_stt"},
        )
    )

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        source_record = store.get_run(source.run_id)
        assert source_record is not None
        source_artifact_ref = source_record.input_data["artifact_ref"]
        store.update_run(source.run_id, metadata={"scenario": "happy"})
    blob_store = LocalBlobStore(tmp_path / "runtime" / "blobs")
    source_request = json.loads(blob_store.read_bytes(source_artifact_ref).decode("utf-8"))
    source_request["metadata"] = {"scenario": "happy"}
    blob_store.put_bytes(
        source_artifact_ref,
        (json.dumps(source_request, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    resumed = resume_transcription(
        source.run_id,
        provider_ids=("speechmatics",),
        settings=load_settings(),
    )

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        resumed_record = store.get_run(resumed.run_id)
        node_names = [record.node_name for record in store.list_node_executions(resumed.run_id)]
        resumed_transcripts = store.list_transcript_candidates(
            resumed.job_id,
            storage_job_id=operational_job_key(_job_context(resumed.job_id)),
        )

    assert resumed.status == "completed"
    assert resumed_record is not None
    assert resumed_record.input_data["resumed_from_run_id"] == source.run_id
    assert "ingest" not in node_names
    assert "extract_audio" not in node_names
    assert node_names[0] == "fanout_transcription"
    assert {candidate.provider_id for candidate in resumed_transcripts} == {
        "assemblyai",
        "speechmatics",
        "deepgram",
    }
    assert any(
        candidate.provider_id == "assemblyai" and source.job_id in candidate.candidate_id
        for candidate in resumed_transcripts
    )
    assert any(
        candidate.provider_id == "deepgram" and source.job_id in candidate.candidate_id
        for candidate in resumed_transcripts
    )
    assert any(
        candidate.provider_id == "speechmatics" and candidate.job_id == resumed.job_id
        for candidate in resumed_transcripts
    )


@pytest.mark.unit
def test_failure_details_returns_ref_when_manifest_missing(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    final_state = GraphState(
        run_id="run-missing-failure-manifest",
        job=_job_context("job-missing-failure-manifest"),
        current_stage="finalize_outputs",
        source_video_ref="input.mp4",
        translation_failed=True,
    )

    failure_ref, failure_summary, failure_reasons = _failure_details(
        final_state=final_state,
        blob_store=blob_store,
    )

    assert failure_ref == str(
        job_path(
            _job_context("job-missing-failure-manifest"),
            "published",
            "translation-failed.json",
        )
    )
    assert failure_summary is None
    assert failure_reasons == ()


def test_cli_resume_transcription_repeated_provider_flags_and_conditional_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    source = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-resume-transcription-cli",
            metadata={"scenario": "degraded_stt"},
        )
    )

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        source_record = store.get_run(source.run_id)
        assert source_record is not None
        source_artifact_ref = source_record.input_data["artifact_ref"]
        store.update_run(source.run_id, metadata={"scenario": "happy"})
    blob_store = LocalBlobStore(tmp_path / "runtime" / "blobs")
    source_request = json.loads(blob_store.read_bytes(source_artifact_ref).decode("utf-8"))
    source_request["metadata"] = {"scenario": "happy"}
    blob_store.put_bytes(
        source_artifact_ref,
        (json.dumps(source_request, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    exit_code = main(
        [
            "resume-transcription",
            source.run_id,
            "--provider",
            "speechmatics",
            "--provider",
            "deepgram",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "ingest" not in output
    assert "extract_audio" not in output
    assert "review_required_stage" not in output
    assert "interactive review requires a real TTY" not in output


@pytest.mark.integration
def test_run_job_marks_bootstrap_failure_and_emits_failed_trace(
    migrated_postgres_dsn: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", migrated_postgres_dsn)

    def fail_build_runtime(**_: object) -> object:
        raise RuntimeError("bootstrap exploded")

    monkeypatch.setattr("translation_agent.api.build_runtime", fail_build_runtime)

    with pytest.raises(RuntimeError, match="bootstrap exploded"):
        run_job(RunJobRequest(source="input.mp4", job_id="job-bootstrap-fail"))

    with PostgresRunStore(migrated_postgres_dsn) as store:
        records = store.list_runs()

    assert len(records) == 1
    record = records[0]
    assert record.status == "failed"
    assert record.error == {"message": "bootstrap exploded"}
    assert record.output_data == {"final_stage": "bootstrap"}

    trace_files = list((tmp_path / "runtime" / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    trace_records = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        item["name"] == "run.failed" and item["attributes"]["phase"] == "bootstrap"
        for item in trace_records
    )


@pytest.mark.integration
def test_cli_validate_config_json_with_working_dsn(
    postgres_dsn: str, monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", postgres_dsn)

    exit_code = main(["validate-config", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["state_backend"] == "postgres"
    assert payload["state_db_ok"] is True
    assert payload["state_db_target"] == sanitize_db_target(postgres_dsn)


@pytest.mark.integration
def test_cli_run_job_json(migrated_postgres_dsn: str, monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", migrated_postgres_dsn)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-123", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["job_id"] == "job-123"
    assert payload["status"] == "completed"
    assert payload["source_language"] == "en"
    assert payload["state_backend"] == "postgres"
    assert payload["state_db_target"] == sanitize_db_target(migrated_postgres_dsn)
    assert payload["target_language"] == "zh-CN"
    assert payload["failure_ref"] is None
    assert payload["failure_summary"] is None
    assert payload["failure_reasons"] == []
    assert Path(payload["trace_path"]).exists()

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(payload["run_id"])
        node_executions = store.list_node_executions(payload["run_id"])

    assert record is not None
    assert len(node_executions) == 16


@pytest.mark.unit
def test_cli_run_job_review_auto_non_tty_prints_resume_instructions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "translation_agent.cli.run_job",
        lambda request, settings: RunJobResult(
            run_id="run-review-auto",
            job_id="job-review-auto",
            status="human_review_required",
            source=request.source,
            source_language="en",
            target_language="zh-CN",
            blob_root=Path("/tmp/blob-root"),
            trace_path=Path("/tmp/trace.jsonl"),
            state_backend="sqlite",
            state_db_target="/tmp/state.sqlite3",
            review_required_stage="translation",
            resume_commands=(
                "uv run translation-agent review-job run-review-auto",
                "uv run translation-agent approve-review run-review-auto "
                "--candidate-id <candidate-id>",
            ),
        ),
    )

    exit_code = main(["run-job", "input.wav", "--job-id", "job-review-auto", "--review", "auto"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "review_required_stage: translation" in output
    assert "uv run translation-agent review-job run-review-auto" in output
    assert "uv run translation-agent approve-review run-review-auto" in output


@pytest.mark.unit
def test_cli_resume_translation_review_auto_non_tty_prints_resume_instructions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "translation_agent.cli.resume_translation",
        lambda run_id, review_mode, settings: RunJobResult(
            run_id="run-resume-auto",
            job_id="job-resume-auto",
            status="human_review_required",
            source="input.wav",
            source_language="en",
            target_language="zh-CN",
            blob_root=Path("/tmp/blob-root"),
            trace_path=Path("/tmp/trace.jsonl"),
            state_backend="sqlite",
            state_db_target="/tmp/state.sqlite3",
            review_required_stage="translation",
            resume_commands=(
                "uv run translation-agent review-job run-resume-auto",
                "uv run translation-agent approve-review run-resume-auto "
                "--candidate-id <candidate-id>",
            ),
        ),
    )

    exit_code = main(["resume-translation", "run-source", "--review", "auto"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "review_required_stage: translation" in output
    assert "uv run translation-agent review-job run-resume-auto" in output
    assert "uv run translation-agent approve-review run-resume-auto" in output


@pytest.mark.unit
def test_review_job_json_exposes_exception_only_review_contract_and_legacy_review_diffs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-json",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )

    exit_code = main(["review-job", result.run_id, "--json"])

    payload = json.loads(capsys.readouterr().out)
    candidate = payload["candidates"][0]
    assert exit_code == 0
    assert payload["review_mode"] == "exception_only"
    assert payload["review_required_stage"] == "translation"
    assert payload["recommended_candidate_id"] is not None
    assert payload["candidate_count"] >= 1
    assert candidate["source_transcript_candidate_id"]
    assert candidate["source_transcript"]["provider_id"] in {
        "assemblyai",
        "speechmatics",
        "deepgram",
    }
    contradictory_candidate = next(
        candidate for candidate in payload["candidates"] if candidate["contradiction_count"] >= 1
    )
    assert contradictory_candidate["blocking_hard_contradiction_count"] >= 1
    contradiction = contradictory_candidate["contradictions"][0]
    assert contradiction["dimension"] in {"meaning", "entity", "number_date_unit", "coverage"}
    assert contradiction["evidence_text"]
    assert contradiction["time_range"]
    assert payload["human_review_summary"]["contradiction_count"] >= 1
    assert payload["flagged_spans"]
    flagged_span = payload["flagged_spans"][0]
    assert flagged_span["source_span_id"]
    assert flagged_span["recommended_variant_id"]
    assert flagged_span["selected_variant_id"]
    assert isinstance(flagged_span["acknowledged"], bool)
    assert payload["blocking_span_count"] >= 1
    assert payload["warning_span_count"] >= 0
    assert payload["auto_accepted_span_count"] >= 0
    assert payload["review_spans"]
    review_span = payload["review_spans"][0]
    assert review_span["source_span_id"]
    assert review_span["variants"]
    assert review_span["current_draft_decision"]["selected_base_variant_id"]
    assert "acknowledged" in review_span["current_draft_decision"]
    assert review_span["transcript_provenance_options"]
    assert payload["review_diffs"]
    review_diff = payload["review_diffs"][0]
    assert review_diff["diff_id"]
    assert review_diff["source_excerpt"]
    assert review_diff["left_candidate"]["candidate_id"]
    assert review_diff["right_candidate"]["candidate_id"]
    assert review_diff["left_candidate"]["target_excerpt"]
    assert review_diff["right_candidate"]["target_excerpt"]
    assert Path(candidate["translation_preview_json_path"]).exists()
    assert Path(candidate["translation_preview_srt_path"]).exists()
    assert payload["transcript_review_summary"]["decision_ref"].endswith(
        "/decisions/transcript.json"
    )


@pytest.mark.unit
def test_review_job_interactive_launches_textual_app_when_tty_is_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-interactive",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    payload = review_job(result.run_id)
    captured: dict[str, object] = {}

    class FakeReviewApp:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr("translation_agent.cli.review_job", lambda run_id, settings=None: payload)
    monkeypatch.setattr("translation_agent.cli.ReviewTerminalApp", FakeReviewApp)
    monkeypatch.setattr("translation_agent.cli._has_tty", lambda: True)

    exit_code = main(["review-job", result.run_id])

    assert exit_code == 0
    assert captured["ran"] is True
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs["run_id"] == result.run_id
    assert kwargs["payload"] == payload


@pytest.mark.unit
def test_review_job_interactive_requires_real_tty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-no-tty",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    monkeypatch.setattr("translation_agent.cli._has_tty", lambda: False)

    exit_code = main(["review-job", result.run_id])

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "interactive review requires a real TTY" in output


@pytest.mark.unit
def test_review_diff_builder_prefers_preferred_candidate_for_multi_candidate_group() -> None:
    candidates = [
        TranslationCandidate(
            candidate_id="candidate-a",
            job_id="job-review-diffs",
            source_transcript_candidate_id="transcript-a",
            model_id="gpt-5.4",
            prompt_variant_id="prompt-a",
            prompt_version="translation-v1",
            language="zh",
            segments=(
                Segment(
                    segment_id="seg-1",
                    start_ms=0,
                    end_ms=1_000,
                    source_text="Source line.",
                    target_text="Alpha rendering.",
                ),
            ),
            full_text="Alpha rendering.",
            normalization_version="translation-candidate/v1",
        ),
        TranslationCandidate(
            candidate_id="candidate-b",
            job_id="job-review-diffs",
            source_transcript_candidate_id="transcript-b",
            model_id="gpt-5.4",
            prompt_variant_id="prompt-b",
            prompt_version="translation-v1",
            language="zh",
            segments=(
                Segment(
                    segment_id="seg-1",
                    start_ms=0,
                    end_ms=1_000,
                    source_text="Source line.",
                    target_text="Beta rendering.",
                ),
            ),
            full_text="Beta rendering.",
            normalization_version="translation-candidate/v1",
        ),
        TranslationCandidate(
            candidate_id="candidate-c",
            job_id="job-review-diffs",
            source_transcript_candidate_id="transcript-c",
            model_id="gpt-5.4",
            prompt_variant_id="prompt-c",
            prompt_version="translation-v1",
            language="zh",
            segments=(
                Segment(
                    segment_id="seg-1",
                    start_ms=0,
                    end_ms=1_000,
                    source_text="Source line.",
                    target_text="Gamma rendering.",
                ),
            ),
            full_text="Gamma rendering.",
            normalization_version="translation-candidate/v1",
        ),
    ]
    reviews = (
        ReviewBundle(
            review_id="review-1",
            job_id="job-review-diffs",
            stage="translation",
            reviewer_role="faithfulness_reviewer",
            candidate_preferences=(
                CandidatePreference(candidate_id="candidate-b", rank=1),
                CandidatePreference(candidate_id="candidate-a", rank=2),
                CandidatePreference(candidate_id="candidate-c", rank=3),
            ),
            confidence=0.82,
            structured_evidence=(
                StructuredEvidence(
                    source_span_id="span:0:1000",
                    candidate_id="candidate-a",
                    dimension="meaning",
                    polarity="refutes",
                    normalized_value="conflict",
                    severity="major",
                    evidence_text="Competing meaning detected.",
                ),
                StructuredEvidence(
                    source_span_id="span:0:1000",
                    candidate_id="candidate-b",
                    dimension="meaning",
                    polarity="refutes",
                    normalized_value="conflict",
                    severity="major",
                    evidence_text="Competing meaning detected.",
                ),
                StructuredEvidence(
                    source_span_id="span:0:1000",
                    candidate_id="candidate-c",
                    dimension="meaning",
                    polarity="refutes",
                    normalized_value="conflict",
                    severity="critical",
                    evidence_text="Competing meaning detected.",
                ),
            ),
        ),
    )

    summary = _build_contradiction_summary(
        translation_candidates=candidates,
        reviews=reviews,
        preferred_candidate_id="candidate-b",
        candidate_rank_map={
            "candidate-a": 1,
            "candidate-b": 2,
            "candidate-c": 3,
        },
    )

    review_diffs = summary["review_diffs"]
    assert [diff["left_candidate"]["candidate_id"] for diff in review_diffs] == [
        "candidate-b",
        "candidate-b",
    ]
    assert [diff["right_candidate"]["candidate_id"] for diff in review_diffs] == [
        "candidate-a",
        "candidate-c",
    ]
    assert [diff["right_candidate"]["rank"] for diff in review_diffs] == [1, 3]
    assert all(diff["source_excerpt"] == "Source line." for diff in review_diffs)
    assert [diff["right_candidate"]["target_excerpt"] for diff in review_diffs] == [
        "Alpha rendering.",
        "Gamma rendering.",
    ]


@pytest.mark.unit
def test_approve_review_json_republishes_outputs_and_updates_provider_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    first = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-approve-a",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    first_payload = review_job(first.run_id)
    first_candidates = cast(list[dict[str, object]], first_payload["candidates"])
    first_candidate = first_candidates[0]

    exit_code = main(
        [
            "approve-review",
            first.run_id,
            "--candidate-id",
            cast(str, first_candidate["candidate_id"]),
            "--approved-by",
            "tester",
            "--note",
            "ship",
            "--json",
        ]
    )

    approved_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert approved_payload["status"] == "completed_after_human_review"

    second = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-approve-b",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    second_payload = review_job(second.run_id)
    second_candidates = cast(list[dict[str, object]], second_payload["candidates"])
    matching_candidate = next(
        candidate
        for candidate in second_candidates
        if cast(dict[str, object], candidate["source_transcript"])["provider_id"]
        == cast(dict[str, object], first_candidate["source_transcript"])["provider_id"]
    )
    second_result = main(
        [
            "approve-review",
            second.run_id,
            "--candidate-id",
            cast(str, matching_candidate["candidate_id"]),
            "--approved-by",
            "tester",
            "--json",
        ]
    )
    assert second_result == 0
    capsys.readouterr()

    blob_store = LocalBlobStore(tmp_path / "runtime" / "blobs")
    job = _job_context("job-review-approve-a")
    approval_path = Path(job_path(job, "approvals", "translation.json"))
    resolution_path = Path(job_path(job, "review-resolutions", "translation.json"))
    learning_path = Path(job_path(job, "learning", "transcript-approval.json"))
    transcript_path = Path(job_path(job, "published", "transcript.json"))
    translation_path = Path(job_path(job, "published", "translation.json"))

    assert blob_store.exists(str(approval_path))
    assert blob_store.exists(str(resolution_path))
    assert blob_store.exists(str(learning_path))
    assert blob_store.exists(str(transcript_path))
    assert blob_store.exists(str(translation_path))

    approved_translation = TranslationCandidate.model_validate_json(
        blob_store.read_bytes(str(translation_path))
    )
    approved_transcript = SynthesizedTranscriptArtifact.model_validate_json(
        blob_store.read_bytes(str(transcript_path))
    )
    assert approved_translation.candidate_id.startswith("human-reviewed-")
    assert approved_translation.metadata["review_mode"] == "human_review_synthesis"
    assert approved_translation.metadata["provenance_summary"]["translation_candidate_ids"] == [
        cast(str, first_candidate["candidate_id"])
    ]
    assert approved_translation.final_transcript_ref is not None
    assert approved_transcript.status == "ready"

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        first_record = store.get_run(first.run_id)
        resolution_record = store.get_human_review_resolution(first.run_id)
        stats = store.get_transcript_provider_quality_stats(
            provider_id=cast(
                str,
                cast(dict[str, object], first_candidate["source_transcript"])["provider_id"],
            ),
            source_language="en",
            target_language="zh-CN",
        )

    assert first_record is not None
    assert first_record.status == "completed_after_human_review"
    assert first_record.output_data["approval_ref"] == str(approval_path)
    assert first_record.output_data["resolution_kind"] == "approved_good"
    assert (
        first_record.output_data["final_translation_candidate_id"]
        == approved_translation.candidate_id
    )
    assert resolution_record is not None
    assert resolution_record.final_translation_candidate_id == approved_translation.candidate_id
    assert resolution_record.reviewed_span_count >= 1
    assert stats is not None
    assert stats.total_approved_outcomes == 2
    assert stats.total_review_escalations == 2
    assert stats.recent_approved_outcomes_30d == 2


@pytest.mark.unit
def test_resolve_review_validation_enforces_candidate_and_failure_tag_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-validate-resolution",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    payload = review_job(result.run_id)
    candidate_id = cast(
        str,
        cast(list[dict[str, object]], payload["candidates"])[0]["candidate_id"],
    )

    with pytest.raises(ValueError, match="candidate_id or reviewed_span_decisions is required"):
        resolve_review(result.run_id, resolution="approved_best_available")

    with pytest.raises(ValueError, match="forbidden for rejected_all"):
        resolve_review(
            result.run_id,
            resolution="rejected_all",
            candidate_id=candidate_id,
            failure_tags=("subtitle_gibberish",),
        )

    with pytest.raises(ValueError, match="at least one failure_tag"):
        resolve_review(result.run_id, resolution="rejected_all")


@pytest.mark.unit
def test_resolve_review_with_structured_span_decisions_builds_synthetic_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-structured-resolution",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    review_payload = review_job(result.run_id)
    review_spans = cast(list[dict[str, object]], review_payload["review_spans"])

    decisions: list[dict[str, object]] = []
    selected_candidate_ids: set[str] = set()
    selected_transcript_ids: set[str] = set()
    for index, span in enumerate(review_spans):
        variants = cast(list[dict[str, object]], span["variants"])
        selected_variant = variants[index % len(variants)]
        selected_candidate_ids.add(cast(str, selected_variant["candidate_id"]))
        transcript_candidate_id = cast(
            str | None, selected_variant["source_transcript_candidate_id"]
        )
        if transcript_candidate_id is not None:
            selected_transcript_ids.add(transcript_candidate_id)
        decisions.append(
            {
                "source_span_id": span["source_span_id"],
                "start_ms": span["start_ms"],
                "end_ms": span["end_ms"],
                "selected_candidate_id": selected_variant["candidate_id"],
                "selected_source_transcript_candidate_id": selected_variant[
                    "source_transcript_candidate_id"
                ],
                "selected_transcript_provider_id": selected_variant["transcript_provider_id"],
                "base_target_text": selected_variant["target_excerpt"],
                "final_target_text": selected_variant["target_excerpt"],
                "edited": False,
                "reviewer_note": "",
            }
        )

    payload = resolve_review(
        result.run_id,
        resolution="approved_good",
        reviewed_span_decisions=tuple(decisions),
        approved_by="tester",
        note="structured review",
    )

    assert payload["status"] == "completed_after_human_review"
    assert cast(str, payload["final_translation_candidate_id"]).startswith("human-reviewed-")

    blob_store = LocalBlobStore(tmp_path / "runtime" / "blobs")
    translation_path = Path(
        job_path(
            _job_context("job-review-structured-resolution"),
            "published",
            "translation.json",
        )
    )
    approved_translation = TranslationCandidate.model_validate_json(
        blob_store.read_bytes(str(translation_path))
    )
    assert approved_translation.candidate_id == payload["final_translation_candidate_id"]
    assert (
        set(approved_translation.metadata["provenance_summary"]["translation_candidate_ids"])
        == selected_candidate_ids
    )
    assert (
        set(approved_translation.metadata["provenance_summary"]["source_transcript_candidate_ids"])
        == selected_transcript_ids
    )

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        record = store.get_human_review_resolution(result.run_id)

    assert record is not None
    assert record.final_translation_candidate_id == approved_translation.candidate_id
    assert set(record.contributing_translation_candidate_ids) == selected_candidate_ids
    assert set(record.contributing_source_transcript_candidate_ids) == selected_transcript_ids


@pytest.mark.unit
def test_resolve_review_approved_best_available_persists_soft_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-soft-resolution",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    review_payload = review_job(result.run_id)
    selected = cast(list[dict[str, object]], review_payload["candidates"])[0]
    payload = resolve_review(
        result.run_id,
        resolution="approved_best_available",
        candidate_id=cast(str, selected["candidate_id"]),
        failure_tags=("literal_but_wrong_semantics",),
        approved_by="tester",
        note="best we have",
    )

    assert payload["status"] == "completed_after_human_review"
    assert payload["resolution_kind"] == "approved_best_available"
    assert payload["approval_ref"] is not None
    assert payload["approved_candidate_id"] == selected["candidate_id"]
    assert cast(str, payload["final_translation_candidate_id"]).startswith("human-reviewed-")

    refreshed_review = review_job(result.run_id)
    assert refreshed_review["resolution_kind"] == "approved_best_available"
    assert refreshed_review["residual_failure_tags"] == ["literal_but_wrong_semantics"]

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        provider_stats = store.get_transcript_provider_quality_stats(
            provider_id=cast(
                str,
                cast(dict[str, object], selected["source_transcript"])["provider_id"],
            ),
            source_language="en",
            target_language="zh-CN",
        )
        combo_stats = store.get_translation_feedback_stats(cast(str, selected["combo_key"]))

    assert provider_stats is not None
    assert provider_stats.approved_best_available_count == 1
    assert provider_stats.soft_positive_score == pytest.approx(0.35)
    assert combo_stats is not None
    assert combo_stats.approved_best_available_count == 1


@pytest.mark.unit
def test_resolve_review_rejected_all_persists_negative_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(
        RunJobRequest(
            source="input.mp4",
            job_id="job-review-rejected-all",
            metadata={"scenario": "translation_conflict_timeout"},
        )
    )
    review_payload = review_job(result.run_id)
    candidates = cast(list[dict[str, object]], review_payload["candidates"])
    payload = resolve_review(
        result.run_id,
        resolution="rejected_all",
        failure_tags=("subtitle_gibberish", "ungrounded_addition"),
        approved_by="tester",
        note="all variants broken",
    )

    assert payload["status"] == "rejected_after_human_review"
    assert payload["approval_ref"] is None
    assert payload["default_output_path"] is None

    refreshed_review = review_job(result.run_id)
    assert refreshed_review["resolution_kind"] == "rejected_all"
    assert refreshed_review["failure_tags"] == [
        "subtitle_gibberish",
        "ungrounded_addition",
    ]

    blob_store = LocalBlobStore(tmp_path / "runtime" / "blobs")
    resolution_path = Path(
        job_path(
            _job_context("job-review-rejected-all"),
            "review-resolutions",
            "translation.json",
        )
    )
    assert blob_store.exists(str(resolution_path))

    with SQLiteOperationalStore(tmp_path / "runtime" / "state.sqlite3") as store:
        record = store.get_human_review_resolution(result.run_id)
        combo_stats = store.get_translation_feedback_stats(cast(str, candidates[0]["combo_key"]))

    assert record is not None
    assert record.resolution_kind == "rejected_all"
    assert record.reviewed_span_count == 0
    assert combo_stats is not None
    assert combo_stats.rejected_all_count == 1
    assert combo_stats.failure_tag_counts["subtitle_gibberish"] == 1


@pytest.mark.unit
def test_cli_run_job_language_flags_override_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_DEFAULT_SOURCE_LANGUAGE", "en")
    monkeypatch.setenv("TA_DEFAULT_TARGET_LANGUAGE", "zh")
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(
        [
            "run-job",
            "input.wav",
            "--job-id",
            "job-cli-ja",
            "--target-language",
            "ja",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    request_payload = json.loads(
        (Path(payload["blob_root"]) / "jobs" / f"{payload['run_id']}-request.json").read_text(
            encoding="utf-8"
        )
    )

    assert exit_code == 0
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "ja"
    assert request_payload["source_language"] == "en"
    assert request_payload["target_language"] == "ja"
