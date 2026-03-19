from __future__ import annotations

import json
from pathlib import Path

from translation_agent.api import RunJobRequest, run_job
from translation_agent.cli import main
from translation_agent.config import load_settings, validate_environment


def test_load_settings_reads_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    settings = load_settings()

    assert settings.data_dir == tmp_path / "runtime"
    assert settings.blob_dir == settings.data_dir / "blobs"
    assert settings.trace_dir == settings.data_dir / "traces"


def test_validate_environment_creates_runtime_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    settings = load_settings()

    result = validate_environment(settings)

    assert result.ok is True
    for path in result.checked_paths:
        assert path.exists()


def test_run_job_bootstraps_local_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))

    result = run_job(RunJobRequest(source="input.mp4"))

    assert result.status == "bootstrapped"
    assert result.blob_root.exists()
    assert result.state_db_path.exists()
    assert result.trace_path.exists()


def test_cli_validate_config_json(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))

    exit_code = main(["validate-config", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True


def test_cli_run_job_json(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))

    exit_code = main(["run-job", "input.wav", "--job-id", "job-123", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["job_id"] == "job-123"
    assert Path(payload["trace_path"]).exists()
