from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_agent.cli import main
from translation_agent.config import sanitize_db_target
from translation_agent.storage import PostgresRunStore

pytestmark = pytest.mark.contract

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def test_validate_config_json_missing_dsn_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("TA_DATA_DIR", str(runtime_dir))
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)

    exit_code = main(["validate-config", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert _normalize_validate_config_payload(payload) == _load_golden(
        "validate_config_missing_dsn.json"
    )


@pytest.mark.integration
def test_cli_run_job_json_contract(
    migrated_postgres_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("TA_DATA_DIR", str(runtime_dir))
    monkeypatch.setenv("TA_STATE_DB_DSN", migrated_postgres_dsn)

    exit_code = main(["run-job", "input.wav", "--job-id", "job-123", "--json"])

    payload = json.loads(capsys.readouterr().out)
    trace_path = Path(payload["trace_path"])
    blob_root = Path(payload["blob_root"])

    assert exit_code == 0
    assert trace_path.exists()
    assert blob_root.exists()
    assert payload["state_db_target"] == sanitize_db_target(migrated_postgres_dsn)
    assert _normalize_run_job_payload(payload) == _load_golden("run_job.json")

    with PostgresRunStore(migrated_postgres_dsn) as store:
        record = store.get_run(payload["run_id"])

    assert record is not None


def _load_golden(name: str) -> object:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def _normalize_validate_config_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["checked_paths"] = ["<data_dir>", "<blob_dir>", "<trace_dir>"]
    normalized["state_db_target"] = "<state_db_target>"
    return normalized


def _normalize_run_job_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["run_id"] = "<run_id>"
    normalized["blob_root"] = "<blob_root>"
    normalized["trace_path"] = "<trace_path>"
    normalized["state_db_target"] = "<state_db_target>"
    return normalized
