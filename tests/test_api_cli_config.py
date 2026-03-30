from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.api import RunJobRequest, run_job
from translation_agent.cli import main
from translation_agent.config import (
    load_settings,
    sanitize_db_target,
    validate_environment,
    validate_runtime_compatibility,
)
from translation_agent.models import JobContext
from translation_agent.storage import PostgresRunStore, job_path


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
    )


def _artifact_path(*parts: str) -> Path:
    return Path(job_path(_job_context(), *parts))


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
def test_validate_environment_fails_cleanly_without_state_db_dsn(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.state_backend == "postgres"
    assert result.adapter_mode == "fake"
    assert result.state_db_ok is False
    assert result.state_db_target == "<missing>"
    assert result.state_db_error == "TA_STATE_DB_DSN is required"
    assert result.runtime_compatibility_ok is True
    assert result.provider_config_ok is True
    for path in result.checked_paths:
        assert path.exists()


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
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["adapter_mode"] == "fake"
    assert payload["runtime_compatibility_ok"] is True
    assert payload["provider_config_ok"] is True
    assert payload["state_db_ok"] is False
    assert payload["state_db_target"] == "<missing>"


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

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(result.run_id)
        node_executions = store.list_node_executions(result.run_id)

    assert record is not None
    assert record.status == "completed"
    assert record.input_data == {
        "artifact_ref": f"jobs/{result.run_id}-request.json",
        "job_id": "job-123",
        "source": "input.mp4",
    }
    assert record.output_data is not None
    assert record.output_data["final_stage"] == "finalize_outputs"
    assert len(node_executions) == 13


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
    assert Path(payload["trace_path"]).exists()

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(payload["run_id"])
        node_executions = store.list_node_executions(payload["run_id"])

    assert record is not None
    assert len(node_executions) == 13
