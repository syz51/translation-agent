from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_agent.api import RunJobRequest, run_job
from translation_agent.cli import main
from translation_agent.config import load_settings, sanitize_db_target, validate_environment
from translation_agent.storage import PostgresRunStore


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


def test_validate_environment_fails_cleanly_without_state_db_dsn(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    result = validate_environment(load_settings())

    assert result.ok is False
    assert result.state_backend == "postgres"
    assert result.state_db_ok is False
    assert result.state_db_target == "<missing>"
    assert result.state_db_error == "TA_STATE_DB_DSN is required"
    for path in result.checked_paths:
        assert path.exists()


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


def test_cli_validate_config_json_missing_dsn(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["validate-config", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["state_db_ok"] is False
    assert payload["state_db_target"] == "<missing>"


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
    assert payload["state_db_ok"] is False
    assert payload["state_db_target"] == "postgresql://127.0.0.1:1/translation_agent"


@pytest.mark.integration
def test_run_job_bootstraps_local_artifacts_and_postgres_record(
    postgres_dsn: str, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", postgres_dsn)

    result = run_job(RunJobRequest(source="input.mp4", job_id="job-123"))

    assert result.status == "bootstrapped"
    assert result.blob_root.exists()
    assert result.trace_path.exists()
    assert (result.blob_root / "jobs" / f"{result.run_id}-request.json").exists()
    assert result.state_backend == "postgres"
    assert result.state_db_target == sanitize_db_target(postgres_dsn)

    with PostgresRunStore(postgres_dsn) as store:
        record = store.get_run(result.run_id)

    assert record is not None
    assert record.status == "bootstrapped"
    assert record.input_data == {
        "artifact_ref": f"jobs/{result.run_id}-request.json",
        "job_id": "job-123",
        "source": "input.mp4",
    }


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
def test_cli_run_job_json(postgres_dsn: str, monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_STATE_DB_DSN", postgres_dsn)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-123", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["job_id"] == "job-123"
    assert payload["state_backend"] == "postgres"
    assert payload["state_db_target"] == sanitize_db_target(postgres_dsn)
    assert Path(payload["trace_path"]).exists()

    with PostgresRunStore(postgres_dsn) as store:
        record = store.get_run(payload["run_id"])

    assert record is not None
