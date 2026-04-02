from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.api import (
    RunJobRequest,
    RunJobResult,
    _failure_details,
    _final_status,
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
from translation_agent.graph import GraphState
from translation_agent.graph.state import RoutingFact
from translation_agent.models import JobContext
from translation_agent.storage import (
    LocalBlobStore,
    PostgresRunStore,
    SQLiteOperationalStore,
    job_path,
    job_scope_token,
)


def _job_context(job_id: str = "job-123") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-local",
        project_id="project-local",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="system@local",
        created_at=datetime(2026, 3, 31, 0, 0, tzinfo=UTC),
        profile_ref="profiles/default",
        media_key=f"source-ref:{job_id}",
    )


def _artifact_path(*parts: str) -> Path:
    return Path(job_path(_job_context(), *parts))


def _configure_real_mode_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    transcription_providers: str | None = None,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_ADAPTER_MODE", "real")
    monkeypatch.setenv("TA_ALLOW_LANGGRAPH_PY314_WARNING", "1")
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)
    monkeypatch.delenv("TA_ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("TA_SPEECHMATICS_API_KEY", raising=False)
    monkeypatch.delenv("TA_DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("TA_OPENAI_API_KEY", raising=False)
    if transcription_providers is None:
        monkeypatch.delenv("TA_TRANSCRIPTION_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("TA_TRANSCRIPTION_PROVIDERS", transcription_providers)


@pytest.mark.unit
def test_load_settings_reads_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv(
        "TA_STATE_DB_DSN",
        "postgresql://user:secret@db.example.com:5432/translation_agent?sslmode=require",
    )

    settings = load_settings()

    assert settings.data_dir == tmp_path / "runtime"
    assert settings.blob_dir == settings.data_dir / "blobs"
    assert settings.trace_dir == settings.data_dir / "traces"
    assert settings.state_db_dsn == (
        "postgresql://user:secret@db.example.com:5432/translation_agent?sslmode=require"
    )


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
    monkeypatch.delenv("TA_OPENAI_API_KEY", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.provider_config_error is not None
    assert "TA_ASSEMBLYAI_API_KEY" in result.provider_config_error


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
        "TA_SPEECHMATICS_API_KEY, TA_DEEPGRAM_API_KEY, TA_OPENAI_API_KEY"
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
        "real adapter mode requires TA_ASSEMBLYAI_API_KEY, TA_OPENAI_API_KEY"
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
        "real adapter mode requires TA_ASSEMBLYAI_API_KEY, TA_DEEPGRAM_API_KEY, TA_OPENAI_API_KEY"
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
    monkeypatch.setenv("TA_OPENAI_API_KEY", "openai")
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
    monkeypatch.delenv("TA_OPENAI_API_KEY", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.adapter_mode == "real"
    assert result.provider_config_ok is False
    assert result.provider_config_error is not None
    assert "TA_ASSEMBLYAI_API_KEY" in result.provider_config_error
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
def test_cli_run_job_plain_output_reports_run_status_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-plain"])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exit_code == 0
    assert len(lines) == 4
    assert lines[1] == "completed"
    assert lines[2] == f"sqlite: {(tmp_path / 'runtime' / 'state.sqlite3').resolve()}"
    assert Path(lines[3]).exists()


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
        lambda request: RunJobResult(
            run_id="run-failed",
            job_id=request.job_id or "job-failed",
            status="translation_failed",
            source=request.source,
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
    assert result.blob_root.exists()
    assert result.trace_path.exists()
    assert (result.blob_root / "jobs" / f"{result.run_id}-request.json").exists()
    assert (result.blob_root / _artifact_path("published", "transcript.json")).exists()
    assert (result.blob_root / _artifact_path("published", "translation.json")).exists()
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
    assert len(node_executions) == 13


@pytest.mark.unit
def test_run_job_defaults_to_local_sqlite_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = run_job(RunJobRequest(source="input.mp4", job_id="job-local"))

    assert result.status == "completed"
    assert result.state_backend == "sqlite"
    assert result.state_db_target == str((tmp_path / "runtime" / "state.sqlite3").resolve())
    assert result.failure_ref is None
    assert result.failure_summary is None
    assert result.failure_reasons == ()
    assert (tmp_path / "runtime" / "state.sqlite3").exists()
    local_job = _job_context(job_id="job-local")
    assert (result.blob_root / Path(job_path(local_job, "published", "transcript.json"))).exists()
    assert (result.blob_root / Path(job_path(local_job, "published", "translation.json"))).exists()


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

    expected_fragment = (
        "translation adjudication trusted "
        f"tl-variant-a-job-memory-a-{job_scope_token(_job_context(job_id='job-memory-a'))}"
    )
    assert any(expected_fragment in entry["content"] for entry in memory_bundle["semantic_memory"])


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
    assert payload["state_backend"] == "postgres"
    assert payload["state_db_target"] == sanitize_db_target(migrated_postgres_dsn)
    assert payload["failure_ref"] is None
    assert payload["failure_summary"] is None
    assert payload["failure_reasons"] == []
    assert Path(payload["trace_path"]).exists()

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(payload["run_id"])
        node_executions = store.list_node_executions(payload["run_id"])

    assert record is not None
    assert len(node_executions) == 13
