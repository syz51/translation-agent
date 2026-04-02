from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.adapters import AdapterError, FFmpegAudioExtractionAdapter, RetryPolicy
from translation_agent.adapters import ffmpeg as ffmpeg_module
from translation_agent.models import JobContext, RequestContext
from translation_agent.storage import LocalBlobStore

pytestmark = pytest.mark.unit


def _request_context(job_id: str = "job-ffmpeg") -> RequestContext:
    return RequestContext(
        run_id="run-ffmpeg",
        job=JobContext(
            job_id=job_id,
            tenant_id="tenant-1",
            project_id="project-1",
            source_video_ref="input.mp4",
            target_language="fr",
            source_language="en",
            requested_by="tester@example.com",
            created_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
            profile_ref="profiles/default",
            media_key=f"source-ref:{job_id}",
        ),
        source_artifact_ref=f"jobs/{job_id}.json",
        metadata={},
    )


def test_ffmpeg_adapter_rejects_missing_source_file(tmp_path: Path) -> None:
    adapter = FFmpegAudioExtractionAdapter(blob_store=LocalBlobStore(tmp_path / "blobs"))

    with pytest.raises(AdapterError, match="input media was not found") as exc_info:
        adapter.extract_audio(str(tmp_path / "missing.mp4"), _request_context())

    assert exc_info.value.category == "input_missing"
    assert exc_info.value.retryable is False


def test_ffmpeg_adapter_surfaces_missing_binary_and_process_failures(tmp_path: Path) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store = LocalBlobStore(tmp_path / "blobs")

    missing_binary_adapter = FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        binary="ffmpeg-test",
        command_runner=lambda command, timeout_seconds: (_ for _ in ()).throw(FileNotFoundError()),
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="binary was not found: ffmpeg-test") as missing_binary:
        missing_binary_adapter.extract_audio(
            str(source_path), _request_context("job-missing-binary")
        )

    assert missing_binary.value.category == "binary_missing"
    assert missing_binary.value.retryable is False

    failed_process_adapter = FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        command_runner=lambda command, timeout_seconds: (_ for _ in ()).throw(
            subprocess.CalledProcessError(17, list(command), stderr=b"")
        ),
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="ffmpeg failed") as process_error:
        failed_process_adapter.extract_audio(
            str(source_path), _request_context("job-process-error")
        )

    assert process_error.value.category == "process_error"
    assert process_error.value.status_code == 17


def test_default_command_runner_wraps_subprocess_completed_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], *, capture_output: bool, check: bool, timeout: float):
        captured["command"] = command
        captured["capture_output"] = capture_output
        captured["check"] = check
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, stdout=b"audio", stderr=b"ok")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)

    completed = ffmpeg_module._default_command_runner(("ffmpeg", "-version"), 3.5)

    assert completed.returncode == 0
    assert completed.stdout == b"audio"
    assert completed.stderr == b"ok"
    assert captured == {
        "command": ["ffmpeg", "-version"],
        "capture_output": True,
        "check": True,
        "timeout": 3.5,
    }
