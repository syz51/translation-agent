from __future__ import annotations

import json
import wave
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import translation_agent.graph.runtime as runtime_module
from translation_agent.adapters import (
    AdapterError,
    AssemblyAITranscriptionAdapter,
    DeepgramTranscriptionAdapter,
    FFmpegAudioExtractionAdapter,
    OpenAITranslationAdapter,
    RetryPolicy,
    SpeechmaticsTranscriptionAdapter,
    TranscriptionAdapter,
)
from translation_agent.adapters import (
    assemblyai as assemblyai_module,
)
from translation_agent.adapters import (
    deepgram as deepgram_module,
)
from translation_agent.adapters import (
    openai_translation as openai_translation_module,
)
from translation_agent.adapters import (
    speechmatics as speechmatics_module,
)
from translation_agent.adapters.common import HttpRequest, HttpResponse
from translation_agent.adapters.ffmpeg import FfmpegCompletedProcess
from translation_agent.config import Settings, validate_runtime_compatibility
from translation_agent.graph import (
    GraphState,
    RealRuntimeOverrides,
    build_phase_three_runtime,
    run_workflow,
)
from translation_agent.models import (
    AudioArtifact,
    FinalTranscriptDecision,
    JobContext,
    RequestContext,
    Segment,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.normalization import (
    CURRENT_NORMALIZATION_VERSION,
    normalize_transcript_candidate,
    normalize_translation_candidate,
)
from translation_agent.observability import NoOpTraceSink
from translation_agent.storage import (
    LocalBlobStore,
    NodeExecutionRecord,
    RunRecord,
    job_path,
    job_scope_token,
)

pytestmark = pytest.mark.unit


class InMemoryRunStore:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.node_executions: dict[str, NodeExecutionRecord] = {}

    def create_run(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        status: str = "queued",
        input_data=None,
        metadata=None,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> RunRecord:
        assert run_id is not None
        created_at = created_at or datetime.now(UTC).isoformat()
        record = RunRecord(
            run_id=run_id,
            tenant_id=tenant_id,
            project_id=project_id,
            status=status,
            created_at=created_at,
            updated_at=created_at,
            input_data=input_data,
            output_data=None,
            metadata=metadata,
            error=None,
        )
        self.runs[run_id] = record
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return list(self.runs.values())

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        output_data=None,
        metadata=None,
        error=None,
        updated_at: str | None = None,
    ) -> RunRecord:
        current = self.runs[run_id]
        record = RunRecord(
            run_id=current.run_id,
            tenant_id=current.tenant_id,
            project_id=current.project_id,
            status=current.status if status is None else status,
            created_at=current.created_at,
            updated_at=updated_at or datetime.now(UTC).isoformat(),
            input_data=current.input_data,
            output_data=current.output_data if output_data is None else output_data,
            metadata=current.metadata if metadata is None else metadata,
            error=current.error if error is None else error,
        )
        self.runs[run_id] = record
        return record

    def create_node_execution(
        self,
        *,
        run_id: str,
        node_name: str,
        status: str = "started",
        input_data=None,
        execution_id: str | None = None,
        created_at: str | None = None,
    ) -> NodeExecutionRecord:
        execution_id = execution_id or f"exec-{len(self.node_executions) + 1}"
        created_at = created_at or datetime.now(UTC).isoformat()
        record = NodeExecutionRecord(
            execution_id=execution_id,
            run_id=run_id,
            node_name=node_name,
            status=status,
            created_at=created_at,
            updated_at=created_at,
            input_data=input_data,
            output_data=None,
            error=None,
        )
        self.node_executions[execution_id] = record
        return record

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None:
        return self.node_executions.get(execution_id)

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]:
        return [record for record in self.node_executions.values() if record.run_id == run_id]

    def update_node_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        output_data=None,
        error=None,
        updated_at: str | None = None,
    ) -> NodeExecutionRecord:
        current = self.node_executions[execution_id]
        record = NodeExecutionRecord(
            execution_id=current.execution_id,
            run_id=current.run_id,
            node_name=current.node_name,
            status=current.status if status is None else status,
            created_at=current.created_at,
            updated_at=updated_at or datetime.now(UTC).isoformat(),
            input_data=current.input_data,
            output_data=current.output_data if output_data is None else output_data,
            error=current.error if error is None else error,
        )
        self.node_executions[execution_id] = record
        return record


class SequencedTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = responses
        self.requests: list[HttpRequest] = []

    def request(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected HTTP request")
        return self._responses.pop(0)


def _json_response(payload: object, *, status_code: int = 200) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def _job_context(job_id: str = "job-phase-three") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
        media_key=f"source-ref:{job_id}",
    )


def _request_context(job_id: str = "job-phase-three") -> RequestContext:
    return RequestContext(
        run_id="run-phase-three",
        job=_job_context(job_id),
        source_artifact_ref=f"jobs/{job_id}.json",
        metadata={},
    )


def _artifact_path(*parts: str) -> str:
    return job_path(_job_context(), *parts)


def _audio_artifact(job_id: str = "job-phase-three") -> AudioArtifact:
    job = _job_context(job_id)
    return AudioArtifact(
        artifact_id=f"audio-{job_id}",
        job_id=job_id,
        blob_ref=job_path(job, "artifacts", "audio.wav"),
        duration_ms=1_000,
        sample_rate_hz=16_000,
        channels=1,
        codec="pcm_s16le",
        extraction_metadata={"adapter_id": "ffmpeg"},
    )


def _transcript_candidate(job_id: str = "job-phase-three") -> TranscriptCandidate:
    return TranscriptCandidate(
        candidate_id=f"tr-openai-{job_id}",
        job_id=job_id,
        provider_id="assemblyai",
        provider_request_id="provider-job",
        language="en",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=900,
                speaker="speaker-1",
                source_text="Hello world",
            ),
        ),
        full_text="Hello world",
        speaker_map={"speaker-1": "speaker-1"},
        timing_resolution="segment",
        raw_payload_ref=_artifact_path("raw", "provider-payloads", "assemblyai.json"),
        normalization_version="raw",
        metadata={},
    )


def _retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=2,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.01,
        poll_interval_seconds=0.0,
        max_polls=4,
    )


def _real_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "adapter_mode": "real",
        "allow_langgraph_py314_warning": True,
        "state_db_dsn": "postgresql://db.example.com:5432/app",
        "assemblyai_api_key": "test-assemblyai-key",  # pragma: allowlist secret
        "speechmatics_api_key": "test-speechmatics-key",  # pragma: allowlist secret
        "deepgram_api_key": "test-deepgram-key",  # pragma: allowlist secret
        "openai_api_key": "test-openai-key",  # pragma: allowlist secret
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.contract
def test_ffmpeg_adapter_extracts_audio_and_persists_blob(tmp_path: Path) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store = LocalBlobStore(tmp_path / "blobs")

    commands: list[list[str]] = []

    def fake_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        commands.append(list(command))
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    adapter = FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        command_runner=fake_runner,
    )

    artifact = adapter.extract_audio(str(source_path), _request_context())

    assert artifact.blob_ref == _artifact_path("artifacts", "audio.wav")
    assert blob_store.read_bytes(artifact.blob_ref).startswith(b"RIFF")
    assert artifact.duration_ms == 1000
    assert commands[0][:4] == ["ffmpeg", "-y", "-i", str(source_path)]


def test_ffmpeg_adapter_retries_retryable_timeout(tmp_path: Path) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store = LocalBlobStore(tmp_path / "blobs")

    attempts = 0

    def flaky_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    adapter = FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        command_runner=flaky_runner,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
        ),
        sleep=lambda _: None,
    )

    artifact = adapter.extract_audio(str(source_path), _request_context())

    assert attempts == 2
    assert artifact.blob_ref == _artifact_path("artifacts", "audio.wav")


def test_ffmpeg_adapter_stops_after_retry_budget_exhausted(tmp_path: Path) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store = LocalBlobStore(tmp_path / "blobs")

    attempts = 0

    def timeout_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        del command, timeout_seconds
        nonlocal attempts
        attempts += 1
        raise TimeoutError

    adapter = FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        command_runner=timeout_runner,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
        ),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="timed out"):
        adapter.extract_audio(str(source_path), _request_context())

    assert attempts == 2


def test_ffmpeg_adapter_scopes_temp_and_blob_paths_for_untrusted_job_ids(tmp_path: Path) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    request_context = _request_context("../tenant-breakout")

    commands: list[list[str]] = []

    def fake_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        commands.append(list(command))
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    artifact = FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        command_runner=fake_runner,
    ).extract_audio(str(source_path), request_context)

    assert commands[0][-1].endswith(f"audio-{job_scope_token(request_context.job)}.wav")
    assert ".." not in commands[0][-1]
    assert artifact.blob_ref == job_path(request_context.job, "artifacts", "audio.wav")


@pytest.mark.contract
def test_assemblyai_adapter_retries_and_normalizes_utterances(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"error": "rate limited"}, status_code=429),
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "tr-job"}),
            _json_response(
                {
                    "id": "tr-job",
                    "status": "completed",
                    "text": " Hello   world ",
                    "utterances": [
                        {
                            "start": 0,
                            "end": 1000,
                            "speaker": "A",
                            "text": " Hello   world ",
                            "confidence": 0.98,
                        }
                    ],
                    "confidence": 0.98,
                }
            ),
        ]
    )
    adapter = AssemblyAITranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.transcribe(audio_artifact, _request_context())

    assert candidate.provider_request_id == "tr-job"
    assert candidate.full_text == "Hello world"
    assert candidate.segments[0].speaker == "speaker-a"
    assert blob_store.exists(candidate.raw_payload_ref or "")
    assert len(transport.requests) == 4


def test_assemblyai_adapter_rejects_missing_timed_segments(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "tr-job"}),
            _json_response(
                {
                    "id": "tr-job",
                    "status": "completed",
                    "text": " Hello world ",
                }
            ),
        ]
    )
    adapter = AssemblyAITranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(
        AdapterError,
        match="timed transcript segments missing; cannot produce default SRT output",
    ):
        adapter.transcribe(audio_artifact, _request_context())


def test_assemblyai_adapter_rejects_malformed_payload(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "tr-job"}),
            _json_response({"status": "completed", "text": "Hello"}),
        ]
    )
    adapter = AssemblyAITranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="missing 'id'"):
        adapter.transcribe(audio_artifact, _request_context())


def test_assemblyai_helpers_cover_sparse_utterances_and_non_object_responses() -> None:
    segments = assemblyai_module._segments_from_utterances(
        [
            "skip-me",
            {
                "start": -50,
                "end": 100,
                "speaker": " ",
                "text": " Hello   world ",
                "confidence": 0.5,
            },
        ]
    )

    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].speaker is None
    assert assemblyai_module._assemblyai_speaker_name(None) is None

    class NonObjectJsonResponse:
        status_code = 200

        def json(self) -> object:
            return []

    with pytest.raises(AdapterError, match="JSON object"):
        assemblyai_module._validated_json_response("assemblyai", NonObjectJsonResponse())


def test_assemblyai_adapter_times_out_when_polling_never_completes(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "tr-job"}),
            _json_response({"id": "tr-job", "status": "processing"}),
            _json_response({"id": "tr-job", "status": "processing"}),
        ]
    )
    adapter = AssemblyAITranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(
            max_attempts=1,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            poll_interval_seconds=0.0,
            max_polls=2,
        ),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="polling exceeded"):
        adapter.transcribe(audio_artifact, _request_context())


def test_assemblyai_adapter_retries_timeout(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")

    class TimeoutThenSuccessTransport(SequencedTransport):
        def __init__(self) -> None:
            super().__init__(
                [
                    _json_response({"upload_url": "https://upload.example/audio.wav"}),
                    _json_response({"id": "tr-job"}),
                    _json_response({"id": "tr-job", "status": "completed", "text": "Hello"}),
                ]
            )
            self._first = True

        def request(self, request: HttpRequest) -> HttpResponse:
            if self._first:
                self._first = False
                raise AdapterError(
                    provider_id="http",
                    message="request timed out",
                    category="timeout",
                    retryable=True,
                )
            return super().request(request)

    adapter = AssemblyAITranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=TimeoutThenSuccessTransport(),
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(
        AdapterError,
        match="timed transcript segments missing; cannot produce default SRT output",
    ):
        adapter.transcribe(audio_artifact, _request_context())


@pytest.mark.contract
def test_speechmatics_adapter_retries_and_normalizes_segments(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"error": "busy"}, status_code=503),
            _json_response({"id": "sm-job"}),
            _json_response({"id": "sm-job", "status": "done"}),
            _json_response(
                {
                    "results": [
                        {
                            "type": "word",
                            "alternatives": [{"content": "Hello", "confidence": 0.99}],
                            "start_time": 0.0,
                            "end_time": 0.5,
                            "speaker": "1",
                        },
                        {
                            "type": "punctuation",
                            "alternatives": [{"content": ","}],
                            "speaker": "1",
                        },
                        {
                            "type": "word",
                            "alternatives": [{"content": "world", "confidence": 0.98}],
                            "start_time": 0.5,
                            "end_time": 1.0,
                            "speaker": "1",
                        },
                    ]
                }
            ),
        ]
    )
    adapter = SpeechmaticsTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.transcribe(audio_artifact, _request_context())

    assert candidate.full_text == "Hello, world"
    assert candidate.segments[0].speaker == "speaker-1"
    assert blob_store.exists(candidate.raw_payload_ref or "")


def test_speechmatics_adapter_retries_timeout(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")

    class TimeoutThenSuccessTransport(SequencedTransport):
        def __init__(self) -> None:
            super().__init__(
                [
                    _json_response({"id": "sm-job"}),
                    _json_response({"id": "sm-job", "status": "done"}),
                    _json_response(
                        {
                            "results": [
                                {
                                    "type": "word",
                                    "alternatives": [{"content": "Hello", "confidence": 0.99}],
                                    "start_time": 0.0,
                                    "end_time": 0.5,
                                }
                            ]
                        }
                    ),
                ]
            )
            self._first = True

        def request(self, request: HttpRequest) -> HttpResponse:
            if self._first:
                self._first = False
                raise AdapterError(
                    provider_id="http",
                    message="request timed out",
                    category="timeout",
                    retryable=True,
                )
            return super().request(request)

    adapter = SpeechmaticsTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=TimeoutThenSuccessTransport(),
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.transcribe(audio_artifact, _request_context())

    assert candidate.full_text == "Hello"


def test_speechmatics_adapter_rejects_missing_results(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"id": "sm-job"}),
            _json_response({"id": "sm-job", "status": "done"}),
            _json_response({"job": {"type": "transcription"}}),
        ]
    )
    adapter = SpeechmaticsTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="missing results"):
        adapter.transcribe(audio_artifact, _request_context())


def test_speechmatics_adapter_times_out_when_job_stays_pending(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"id": "sm-job"}),
            _json_response({"id": "sm-job", "status": "running"}),
            _json_response({"id": "sm-job", "status": "running"}),
        ]
    )
    adapter = SpeechmaticsTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(
            max_attempts=1,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
            poll_interval_seconds=0.0,
            max_polls=2,
        ),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="polling exceeded"):
        adapter.transcribe(audio_artifact, _request_context())


@pytest.mark.contract
def test_deepgram_adapter_retries_and_normalizes_utterances(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response({"error": {"message": "rate limited"}}, status_code=429),
            _json_response(
                {
                    "metadata": {"request_id": "dg-job"},
                    "results": {
                        "channels": [
                            {
                                "alternatives": [
                                    {
                                        "transcript": " Hello world ",
                                    }
                                ]
                            }
                        ],
                        "utterances": [
                            {
                                "start": 0.0,
                                "end": 1.0,
                                "speaker": 0,
                                "transcript": " Hello world ",
                                "confidence": 0.96,
                            }
                        ],
                    },
                }
            ),
        ]
    )
    adapter = DeepgramTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.transcribe(audio_artifact, _request_context())

    assert candidate.provider_request_id == "dg-job"
    assert candidate.full_text == "Hello world"
    assert candidate.segments[0].speaker == "speaker-0"
    assert blob_store.exists(candidate.raw_payload_ref or "")


def test_deepgram_adapter_rejects_missing_timed_segments(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "metadata": {"request_id": "dg-job"},
                    "results": {
                        "channels": [
                            {
                                "alternatives": [
                                    {
                                        "transcript": " Hello world ",
                                    }
                                ]
                            }
                        ]
                    },
                }
            ),
        ]
    )
    adapter = DeepgramTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(
        AdapterError,
        match="timed transcript segments missing; cannot produce default SRT output",
    ):
        adapter.transcribe(audio_artifact, _request_context())


def test_deepgram_adapter_times_out_on_retryable_timeout(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")

    class TimeoutTransport:
        def request(self, request: HttpRequest) -> HttpResponse:
            del request
            raise AdapterError(
                provider_id="http",
                message="request timed out",
                category="timeout",
                retryable=True,
            )

    adapter = DeepgramTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=TimeoutTransport(),
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="request timed out"):
        adapter.transcribe(audio_artifact, _request_context())


def test_deepgram_adapter_retries_timeout(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")

    class TimeoutThenSuccessTransport(SequencedTransport):
        def __init__(self) -> None:
            super().__init__(
                [
                    _json_response(
                        {
                            "metadata": {"request_id": "dg-job"},
                            "results": {
                                "channels": [
                                    {
                                        "alternatives": [
                                            {
                                                "transcript": " Hello world ",
                                            }
                                        ]
                                    }
                                ]
                            },
                        }
                    ),
                ]
            )
            self._first = True

        def request(self, request: HttpRequest) -> HttpResponse:
            if self._first:
                self._first = False
                raise AdapterError(
                    provider_id="http",
                    message="request timed out",
                    category="timeout",
                    retryable=True,
                )
            return super().request(request)

    adapter = DeepgramTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=TimeoutThenSuccessTransport(),
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(
        AdapterError,
        match="timed transcript segments missing; cannot produce default SRT output",
    ):
        adapter.transcribe(audio_artifact, _request_context())


def test_deepgram_adapter_rejects_missing_channels(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    audio_artifact = _audio_artifact()
    blob_store.put_bytes(audio_artifact.blob_ref, b"audio-bytes")
    transport = SequencedTransport(
        [_json_response({"metadata": {"request_id": "dg-job"}, "results": {}})]
    )
    adapter = DeepgramTranscriptionAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="missing channels"):
        adapter.transcribe(audio_artifact, _request_context())


def test_deepgram_helpers_cover_malformed_shapes_and_content_type_variants() -> None:
    class NonObjectResponse:
        status_code = 200

        def json(self) -> object:
            return []

    class NonObjectTransport:
        def request(self, request: HttpRequest) -> HttpResponse:
            del request
            return cast(HttpResponse, NonObjectResponse())

    with pytest.raises(AdapterError, match="JSON object"):
        DeepgramTranscriptionAdapter(
            api_key="test-key",
            blob_store=LocalBlobStore(Path("/tmp/deepgram-blob-store")),
            transport=NonObjectTransport(),
            retry_policy=RetryPolicy(max_attempts=1),
            sleep=lambda _: None,
        )._transcribe_once(
            _audio_artifact(),
            b"audio-bytes",
            _request_context(),
        )

    with pytest.raises(AdapterError, match="missing results"):
        deepgram_module._extract_transcript_text({})
    with pytest.raises(AdapterError, match="channel payload was malformed"):
        deepgram_module._extract_transcript_text({"results": {"channels": [123]}})
    with pytest.raises(AdapterError, match="missing alternatives"):
        deepgram_module._extract_transcript_text({"results": {"channels": [{}]}})
    with pytest.raises(AdapterError, match="alternative payload was malformed"):
        deepgram_module._extract_transcript_text(
            {"results": {"channels": [{"alternatives": [123]}]}}
        )

    assert deepgram_module._extract_utterances({}) == []
    assert deepgram_module._content_type_for_blob("jobs/audio.mp3") == "audio/mpeg"
    assert deepgram_module._content_type_for_blob("jobs/audio.bin") == "application/octet-stream"
    assert deepgram_module._seconds_to_ms(None) == 0
    assert deepgram_module._seconds_to_ms("not-a-number") == 0
    assert deepgram_module._speaker_name(None) is None


@pytest.mark.contract
@pytest.mark.parametrize("prompt_variant_id", ["variant-a", "variant-b"])
def test_openai_translation_adapter_tracks_prompt_metadata_for_variants(
    tmp_path: Path, prompt_variant_id: str
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "resp-1",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-1",
                                    "target_text": "Bonjour le monde",
                                }
                            ],
                        }
                    ),
                }
            )
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",
        blob_store=blob_store,
        model_id="gpt-5.4-mini",
        prompt_version="phase-3-v1",
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.generate_translation(
        _transcript_candidate(),
        prompt_variant_id,
        _request_context(),
    )

    assert candidate.prompt_variant_id == prompt_variant_id
    assert candidate.prompt_version == "phase-3-v1"
    assert candidate.metadata["provider"]["provider_id"] == "openai"
    assert candidate.metadata["provider"]["provider_request_id"] == "resp-1"
    assert candidate.metadata["provider"]["response_id"] == "resp-1"
    assert candidate.metadata["prompt"]["variant_id"] == prompt_variant_id
    assert blob_store.exists(candidate.raw_response_ref or "")


def test_openai_translation_adapter_retries_retryable_error(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transport = SequencedTransport(
        [
            _json_response({"error": {"message": "please retry"}}, status_code=500),
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-1",
                                    "target_text": "Bonjour le monde",
                                }
                            ],
                        }
                    )
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.generate_translation(
        _transcript_candidate(), "variant-a", _request_context()
    )

    assert candidate.full_text == "Bonjour le monde"
    assert len(transport.requests) == 2


def test_openai_translation_adapter_does_not_retry_insufficient_quota(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "error": {
                        "message": "quota exceeded",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                },
                status_code=429,
            ),
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-1",
                                    "target_text": "Bonjour le monde",
                                }
                            ],
                        }
                    )
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="quota exceeded") as exc_info:
        adapter.generate_translation(_transcript_candidate(), "variant-a", _request_context())

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 429
    assert len(transport.requests) == 1


def test_openai_translation_adapter_rejects_non_json_output(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transport = SequencedTransport(
        [
            _json_response({"output_text": "not-json"}),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="valid JSON"):
        adapter.generate_translation(_transcript_candidate(), "variant-a", _request_context())


def test_openai_translation_adapter_rejects_partial_segment_payloads(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = TranscriptCandidate(
        candidate_id=f"tr-openai-job-phase-three-{job_scope_token(_job_context())}",
        job_id="job-phase-three",
        provider_id="assemblyai",
        provider_request_id="provider-job",
        language="en",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=1800,
                speaker="speaker-1",
                source_text="Hello world",
            ),
        ),
        full_text="Hello world",
        speaker_map={"speaker-1": "speaker-1"},
        timing_resolution="segment",
        raw_payload_ref=_artifact_path("raw", "provider-payloads", "assemblyai.json"),
        normalization_version="raw",
        metadata={},
    )
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [],
                        }
                    )
                }
            )
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=transport,
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    with pytest.raises(AdapterError, match="missing segment translations"):
        adapter.generate_translation(transcript, "variant-a", _request_context())


def test_openai_translation_helpers_cover_output_blocks_and_validation(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    adapter = OpenAITranslationAdapter(
        api_key="test-key",
        blob_store=blob_store,
        transport=SequencedTransport(
            [
                _json_response(
                    {
                        "id": " resp-2 ",
                        "output": [
                            {
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "full_text": " Bonjour le monde ",
                                                "segments": [
                                                    {
                                                        "segment_id": "seg-1",
                                                        "target_text": " Bonjour le monde ",
                                                    }
                                                ],
                                            }
                                        )
                                    }
                                ]
                            }
                        ],
                    }
                )
            ]
        ),
        retry_policy=_retry_policy(),
        sleep=lambda _: None,
    )

    candidate = adapter.generate_translation(
        _transcript_candidate(),
        "variant-a",
        _request_context(),
    )

    assert candidate.full_text == "Bonjour le monde"
    assert candidate.metadata["provider"]["provider_request_id"] == "resp-2"
    assert openai_translation_module._provider_request_id({"id": "   "}) is None

    with pytest.raises(AdapterError, match="output text"):
        openai_translation_module._extract_translation_payload({"output": [{"content": [{}]}]})
    with pytest.raises(AdapterError, match="JSON object"):
        openai_translation_module._parse_model_json("[]")
    with pytest.raises(AdapterError, match="missing full_text"):
        openai_translation_module._candidate_from_translation_payload(
            {"full_text": " ", "segments": [{"segment_id": "seg-1", "target_text": "Bonjour"}]},
            response_payload={},
            final_transcript=_transcript_candidate(),
            request_context=_request_context(),
            language="fr",
            prompt_variant_id="variant-a",
            prompt_version="phase-3-v1",
            model_id="gpt-5.4-mini",
            raw_response_ref=_artifact_path("raw", "provider-payloads", "openai-variant-a.json"),
        )

    class NonObjectResponse:
        status_code = 200

        def json(self) -> object:
            return []

    class NonObjectTransport:
        def request(self, request: HttpRequest) -> HttpResponse:
            del request
            return cast(HttpResponse, NonObjectResponse())

    with pytest.raises(AdapterError, match="JSON object"):
        OpenAITranslationAdapter(
            api_key="test-key",
            blob_store=blob_store,
            transport=NonObjectTransport(),
            retry_policy=RetryPolicy(max_attempts=1),
            sleep=lambda _: None,
        ).generate_translation(_transcript_candidate(), "variant-a", _request_context())


def test_openai_translation_merge_segments_ignores_invalid_entries() -> None:
    translated = openai_translation_module._merge_segments(
        _transcript_candidate().segments,
        [
            "skip-me",
            {"segment_id": "seg-1", "target_text": " Bonjour le monde "},
            {"segment_id": "seg-1", "target_text": "   "},
            {"segment_id": 1, "target_text": "ignored"},
        ],
    )

    assert translated[0].target_text == "Bonjour le monde"

    with pytest.raises(AdapterError, match="missing segment translations"):
        openai_translation_module._merge_segments(_transcript_candidate().segments, None)

    with pytest.raises(AdapterError, match="output text"):
        openai_translation_module._extract_translation_payload(
            {"output": ["skip-me", {"content": "bad-shape"}, {"content": ["skip-me"]}]}
        )


def test_speechmatics_helpers_cover_sparse_results_and_validation() -> None:
    with pytest.raises(
        AdapterError,
        match="timed transcript segments missing; cannot produce default SRT output",
    ):
        speechmatics_module._candidate_from_payload(
            {
                "results": [
                    "skip-me",
                    {"type": "entity", "alternatives": [{"content": "ignored"}]},
                ]
            },
            request_context=_request_context(),
            provider_request_id="sm-job",
            language="en",
            raw_payload_ref=_artifact_path("raw", "provider-payloads", "speechmatics.json"),
        )

    assert speechmatics_module._content_from_result({}) == ""
    assert speechmatics_module._content_from_result({"alternatives": ["skip-me"]}) == ""
    assert (
        speechmatics_module._alternative_confidence(
            {"alternatives": [{"content": "Hello", "confidence": "nope"}]}
        )
        is None
    )
    assert speechmatics_module._alternative_confidence({"alternatives": ["skip-me"]}) is None
    assert speechmatics_module._speaker_name("") is None
    assert speechmatics_module._seconds_to_ms(None) == 0
    assert speechmatics_module._seconds_to_ms("not-a-number") == 0

    class NonObjectJsonResponse:
        status_code = 200

        def json(self) -> object:
            return []

    with pytest.raises(AdapterError, match="JSON object"):
        speechmatics_module._validated_json_response("speechmatics", NonObjectJsonResponse())
    with pytest.raises(AdapterError, match="missing 'id'"):
        speechmatics_module._require_string({}, "id", provider_id="speechmatics")


def test_phase_three_normalization_helpers_canonicalize_candidates() -> None:
    transcript = TranscriptCandidate(
        candidate_id=" tr-1 ",
        job_id="job-phase-three",
        provider_id=" assemblyai ",
        provider_request_id=" req-1 ",
        language="en",
        segments=(
            Segment(
                segment_id=" seg-1 ",
                start_ms=100,
                end_ms=150,
                speaker="Speaker_2",
                source_text=" Hello   world ",
            ),
        ),
        full_text=" Hello   world ",
        speaker_map={" speaker-2 ": " Speaker_2 "},
        raw_payload_ref=f" {_artifact_path('raw', 'provider-payloads', 'assemblyai.json')} ",
        normalization_version="raw",
        metadata={"provider": "bad-shape"},
    )
    translation = TranslationCandidate(
        candidate_id=" tl-1 ",
        job_id="job-phase-three",
        source_transcript_candidate_id=" tr-1 ",
        model_id=" gpt-5.4-mini ",
        prompt_variant_id=" variant-a ",
        prompt_version=" phase-3-v1 ",
        language="fr",
        segments=(
            Segment(
                segment_id=" seg-1 ",
                start_ms=0,
                end_ms=10,
                speaker="Speaker_2",
                source_text=" Hello ",
                target_text=" Bonjour   le   monde ",
            ),
        ),
        full_text=" Bonjour   le   monde ",
        raw_response_ref=f" {_artifact_path('raw', 'provider-payloads', 'openai-variant-a.json')} ",
        normalization_version="raw",
        metadata={"prompt": "bad-shape"},
    )

    normalized_transcript = normalize_transcript_candidate(transcript)
    normalized_translation = normalize_translation_candidate(translation)

    assert normalized_transcript.normalization_version == CURRENT_NORMALIZATION_VERSION
    assert normalized_transcript.candidate_id == "tr-1"
    assert normalized_transcript.provider_id == "assemblyai"
    assert normalized_transcript.provider_request_id == "req-1"
    assert normalized_transcript.full_text == "Hello world"
    assert normalized_transcript.raw_payload_ref == (
        _artifact_path("raw", "provider-payloads", "assemblyai.json")
    )
    assert normalized_transcript.segments[0].segment_id == "seg-1"
    assert normalized_transcript.segments[0].speaker == "speaker-2"
    assert normalized_transcript.metadata["provider"]["provider_id"] == "assemblyai"
    assert normalized_transcript.metadata["provider"]["provider_request_id"] == "req-1"
    assert normalized_translation.normalization_version == CURRENT_NORMALIZATION_VERSION
    assert normalized_translation.candidate_id == "tl-1"
    assert normalized_translation.source_transcript_candidate_id == "tr-1"
    assert normalized_translation.model_id == "gpt-5.4-mini"
    assert normalized_translation.prompt_variant_id == "variant-a"
    assert normalized_translation.prompt_version == "phase-3-v1"
    assert normalized_translation.full_text == "Bonjour le monde"
    assert normalized_translation.raw_response_ref == (
        _artifact_path("raw", "provider-payloads", "openai-variant-a.json")
    )
    assert normalized_translation.segments[0].segment_id == "seg-1"
    assert normalized_translation.metadata["provider"]["provider_id"] == "openai"
    assert normalized_translation.metadata["provider"]["provider_request_id"] is None
    assert normalized_translation.metadata["prompt"]["variant_id"] == "variant-a"
    assert normalized_translation.metadata["prompt"]["version"] == "phase-3-v1"
    assert normalized_translation.metadata["prompt"]["model_id"] == "gpt-5.4-mini"


def test_phase_three_runtime_uses_configured_ffmpeg_retry_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",
        assemblyai_api_key="assembly",
        speechmatics_api_key="speech",
        deepgram_api_key="deepgram",
        openai_api_key="openai",
        adapter_retry_attempts=4,
        adapter_initial_backoff_seconds=0.5,
        adapter_max_backoff_seconds=1.5,
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert isinstance(runtime.audio_extractor, FFmpegAudioExtractionAdapter)
    assert runtime.audio_extractor._retry_policy.max_attempts == 4
    assert runtime.audio_extractor._retry_policy.initial_backoff_seconds == 0.5
    assert runtime.audio_extractor._retry_policy.max_backoff_seconds == 1.5


def test_phase_three_runtime_builds_selected_single_transcription_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = _real_settings(
        transcription_providers="assemblyai",
        speechmatics_api_key=None,
        deepgram_api_key=None,
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert [adapter.provider_id for adapter in runtime.transcription_adapters] == ["assemblyai"]


def test_phase_three_runtime_builds_selected_transcription_provider_subset_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = _real_settings(
        transcription_providers="assemblyai,deepgram",
        speechmatics_api_key=None,
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert [adapter.provider_id for adapter in runtime.transcription_adapters] == [
        "assemblyai",
        "deepgram",
    ]


def test_phase_three_runtime_unset_selector_builds_all_transcription_providers_and_openai(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = _real_settings()

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert isinstance(runtime.audio_extractor, FFmpegAudioExtractionAdapter)
    assert [adapter.provider_id for adapter in runtime.transcription_adapters] == [
        "assemblyai",
        "speechmatics",
        "deepgram",
    ]
    assert isinstance(runtime.translation_adapter, OpenAITranslationAdapter)
    assert runtime.translation_adapter.model_id == settings.translation_model_id


@pytest.mark.contract
def test_validate_runtime_compatibility_gates_python314_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "translation_agent.config._langgraph_py314_warning",
        lambda: "Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=False,
        assemblyai_api_key="assembly",
        speechmatics_api_key="speech",
        deepgram_api_key="deepgram",
        openai_api_key="openai",
    )

    error = validate_runtime_compatibility(settings)

    assert error is not None
    assert "gated on Python 3.14" in error


@pytest.mark.contract
def test_phase_three_runtime_completes_workflow_with_real_adapters(tmp_path: Path) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-123", status="running")
    source_ref = "jobs/run-123-request.json"
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store.put_bytes(source_ref, b"{}\n")

    def fake_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    assembly_transport = SequencedTransport(
        [
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "aai-job"}),
            _json_response(
                {
                    "id": "aai-job",
                    "status": "completed",
                    "text": " Hello world ",
                    "utterances": [
                        {
                            "start": 0,
                            "end": 1000,
                            "speaker": "A",
                            "text": " Hello world ",
                            "confidence": 0.98,
                        }
                    ],
                }
            ),
        ]
    )
    speechmatics_transport = SequencedTransport(
        [
            _json_response({"id": "sm-job"}),
            _json_response({"id": "sm-job", "status": "done"}),
            _json_response(
                {
                    "results": [
                        {
                            "type": "word",
                            "alternatives": [{"content": "Hello", "confidence": 0.99}],
                            "start_time": 0.0,
                            "end_time": 0.4,
                            "speaker": "1",
                        },
                        {
                            "type": "word",
                            "alternatives": [{"content": "world", "confidence": 0.98}],
                            "start_time": 0.4,
                            "end_time": 0.8,
                            "speaker": "1",
                        },
                    ]
                }
            ),
        ]
    )
    deepgram_transport = SequencedTransport(
        [
            _json_response(
                {
                    "metadata": {"request_id": "dg-job"},
                    "results": {
                        "channels": [{"alternatives": [{"transcript": "Hello world"}]}],
                        "utterances": [
                            {
                                "start": 0.0,
                                "end": 0.8,
                                "speaker": 0,
                                "transcript": "Hello world",
                                "confidence": 0.97,
                            }
                        ],
                    },
                }
            )
        ]
    )
    openai_transport = SequencedTransport(
        [
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-assemblyai-1",
                                    "target_text": "Bonjour le monde",
                                }
                            ],
                        }
                    )
                }
            ),
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Salut le monde",
                            "segments": [
                                {"segment_id": "seg-assemblyai-1", "target_text": "Salut le monde"}
                            ],
                        }
                    )
                }
            ),
        ]
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",
        assemblyai_api_key="assembly",
        speechmatics_api_key="speech",
        deepgram_api_key="deepgram",
        openai_api_key="openai",
    )
    overrides = RealRuntimeOverrides(
        audio_extractor=FFmpegAudioExtractionAdapter(
            blob_store=blob_store,
            command_runner=fake_runner,
        ),
        transcription_adapters=(
            AssemblyAITranscriptionAdapter(
                api_key="assembly",
                blob_store=blob_store,
                transport=assembly_transport,
                retry_policy=_retry_policy(),
                sleep=lambda _: None,
            ),
            SpeechmaticsTranscriptionAdapter(
                api_key="speech",
                blob_store=blob_store,
                transport=speechmatics_transport,
                retry_policy=_retry_policy(),
                sleep=lambda _: None,
            ),
            DeepgramTranscriptionAdapter(
                api_key="deepgram",
                blob_store=blob_store,
                transport=deepgram_transport,
                retry_policy=_retry_policy(),
                sleep=lambda _: None,
            ),
        ),
        translation_adapter=OpenAITranslationAdapter(
            api_key="openai",
            blob_store=blob_store,
            transport=openai_transport,
            retry_policy=_retry_policy(),
            sleep=lambda _: None,
        ),
    )
    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=overrides,
    )
    initial_state = GraphState(
        run_id="run-123",
        job=_job_context(),
        current_stage="ingest",
        source_video_ref=str(source_path),
        source_artifact_ref=source_ref,
    )

    final_state = run_workflow(initial_state, runtime)

    assert final_state.current_stage == "finalize_outputs"
    assert final_state.translation_failed is False
    assert final_state.final_translation_candidate_id is not None
    assert blob_store.exists(_artifact_path("raw", "provider-payloads", "assemblyai.json"))
    assert blob_store.exists(_artifact_path("raw", "provider-payloads", "speechmatics.json"))
    assert blob_store.exists(_artifact_path("raw", "provider-payloads", "deepgram.json"))
    assert blob_store.exists(_artifact_path("raw", "provider-payloads", "openai-variant-a.json"))
    assert blob_store.exists(
        _artifact_path(
            "candidates",
            "transcripts",
            final_state.transcript_candidate_ids[0] + ".json",
        )
    )
    assert blob_store.exists(
        _artifact_path(
            "candidates",
            "translations",
            final_state.final_translation_candidate_id + ".json",
        )
    )


@pytest.mark.contract
def test_phase_three_runtime_completes_workflow_with_assemblyai_only_real_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    blob_store = LocalBlobStore(tmp_path / "blobs")
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-assembly-only", status="running")
    source_ref = "jobs/run-assembly-only-request.json"
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store.put_bytes(source_ref, b"{}\n")

    def fake_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    assembly_transport = SequencedTransport(
        [
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "aai-job"}),
            _json_response(
                {
                    "id": "aai-job",
                    "status": "completed",
                    "text": " Hello world ",
                    "utterances": [
                        {
                            "start": 0,
                            "end": 1000,
                            "speaker": "A",
                            "text": " Hello world ",
                            "confidence": 0.98,
                        }
                    ],
                }
            ),
        ]
    )
    openai_transport = SequencedTransport(
        [
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-assemblyai-1",
                                    "target_text": "Bonjour le monde",
                                }
                            ],
                        }
                    )
                }
            ),
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Salut le monde",
                            "segments": [
                                {"segment_id": "seg-assemblyai-1", "target_text": "Salut le monde"}
                            ],
                        }
                    )
                }
            ),
        ]
    )
    settings = _real_settings(
        transcription_providers="assemblyai",
        speechmatics_api_key=None,
        deepgram_api_key=None,
    )
    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=RealRuntimeOverrides(
            audio_extractor=FFmpegAudioExtractionAdapter(
                blob_store=blob_store,
                command_runner=fake_runner,
            ),
            transcription_adapters=(
                AssemblyAITranscriptionAdapter(
                    api_key="assembly",
                    blob_store=blob_store,
                    transport=assembly_transport,
                    retry_policy=_retry_policy(),
                    sleep=lambda _: None,
                ),
            ),
            translation_adapter=OpenAITranslationAdapter(
                api_key="openai",
                blob_store=blob_store,
                transport=openai_transport,
                retry_policy=_retry_policy(),
                sleep=lambda _: None,
            ),
        ),
    )
    initial_state = GraphState(
        run_id="run-assembly-only",
        job=_job_context(),
        current_stage="ingest",
        source_video_ref=str(source_path),
        source_artifact_ref=source_ref,
    )

    final_state = run_workflow(initial_state, runtime)
    decision = FinalTranscriptDecision.model_validate_json(
        blob_store.read_bytes(_artifact_path("decisions", "transcript.json"))
    )

    assert final_state.current_stage == "finalize_outputs"
    assert final_state.translation_failed is False
    assert final_state.human_review_required is False
    assert len(final_state.transcript_candidate_ids) == 1
    assert final_state.final_transcript_candidate_id == final_state.transcript_candidate_ids[0]
    assert decision.winner_candidate_id == final_state.final_transcript_candidate_id
    assert decision.decision_mode == "automatic_finalize"
    assert decision.escalated is True
    assert decision.human_review_required is False
    assert decision.investigation_ref == _artifact_path("investigations", "transcript.json")
    assert blob_store.exists(_artifact_path("published", "transcript.json"))
    assert blob_store.exists(_artifact_path("published", "translation.json"))


def test_phase_three_runtime_continues_when_one_provider_has_no_timed_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    blob_store = LocalBlobStore(tmp_path / "blobs")
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-partial-provider-failure", status="running")
    source_ref = "jobs/run-partial-provider-failure-request.json"
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store.put_bytes(source_ref, b"{}\n")

    def fake_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    assembly_transport = SequencedTransport(
        [
            _json_response({"upload_url": "https://upload.example/audio.wav"}),
            _json_response({"id": "aai-job"}),
            _json_response(
                {
                    "id": "aai-job",
                    "status": "completed",
                    "text": " Hello world ",
                }
            ),
        ]
    )
    speechmatics_transport = SequencedTransport(
        [
            _json_response({"id": "sm-job"}),
            _json_response({"id": "sm-job", "status": "done"}),
            _json_response(
                {
                    "results": [
                        {
                            "type": "word",
                            "alternatives": [{"content": "Hello", "confidence": 0.99}],
                            "start_time": 0.0,
                            "end_time": 0.4,
                            "speaker": "1",
                        },
                        {
                            "type": "word",
                            "alternatives": [{"content": "world", "confidence": 0.98}],
                            "start_time": 0.4,
                            "end_time": 0.8,
                            "speaker": "1",
                        },
                    ]
                }
            ),
        ]
    )
    openai_transport = SequencedTransport(
        [
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-speechmatics-1",
                                    "target_text": "Bonjour",
                                },
                                {
                                    "segment_id": "seg-speechmatics-2",
                                    "target_text": "le monde",
                                },
                            ],
                        }
                    )
                }
            ),
            _json_response(
                {
                    "output_text": json.dumps(
                        {
                            "full_text": "Salut le monde",
                            "segments": [
                                {
                                    "segment_id": "seg-speechmatics-1",
                                    "target_text": "Salut",
                                },
                                {
                                    "segment_id": "seg-speechmatics-2",
                                    "target_text": "le monde",
                                },
                            ],
                        }
                    )
                }
            ),
        ]
    )
    settings = _real_settings(
        transcription_providers="assemblyai,speechmatics",
        deepgram_api_key=None,
    )
    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=RealRuntimeOverrides(
            audio_extractor=FFmpegAudioExtractionAdapter(
                blob_store=blob_store,
                command_runner=fake_runner,
            ),
            transcription_adapters=(
                AssemblyAITranscriptionAdapter(
                    api_key="assembly",
                    blob_store=blob_store,
                    transport=assembly_transport,
                    retry_policy=_retry_policy(),
                    sleep=lambda _: None,
                ),
                SpeechmaticsTranscriptionAdapter(
                    api_key="speech",
                    blob_store=blob_store,
                    transport=speechmatics_transport,
                    retry_policy=_retry_policy(),
                    sleep=lambda _: None,
                ),
            ),
            translation_adapter=OpenAITranslationAdapter(
                api_key="openai",
                blob_store=blob_store,
                transport=openai_transport,
                retry_policy=_retry_policy(),
                sleep=lambda _: None,
            ),
        ),
    )
    initial_state = GraphState(
        run_id="run-partial-provider-failure",
        job=_job_context(),
        current_stage="ingest",
        source_video_ref=str(source_path),
        source_artifact_ref=source_ref,
    )

    final_state = run_workflow(initial_state, runtime)

    assert final_state.current_stage == "finalize_outputs"
    assert final_state.translation_failed is False
    assert final_state.final_transcript_candidate_id is not None
    assert final_state.final_transcript_candidate_id.startswith("tr-speechmatics-")
    assert any(
        fact.fact_type == "transcription_provider_failed"
        and fact.value == "assemblyai"
        and fact.source_ref
        == "timed transcript segments missing; cannot produce default SRT output"
        for fact in final_state.routing_facts
    )
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    assert blob_store.exists(_artifact_path("exports", "translation.srt"))


def test_phase_three_runtime_raises_when_selected_single_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    blob_store = LocalBlobStore(tmp_path / "blobs")
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-assembly-failure", status="running")
    source_ref = "jobs/run-assembly-failure-request.json"
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video")
    blob_store.put_bytes(source_ref, b"{}\n")

    def fake_runner(command: Sequence[str], timeout_seconds: float) -> FfmpegCompletedProcess:
        with wave.open(str(Path(command[-1])), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)
        assert timeout_seconds == 120.0
        return FfmpegCompletedProcess(returncode=0, stderr=b"ok")

    class FailingAssemblyAIAdapter:
        provider_id = "assemblyai"

        def transcribe(
            self,
            audio_artifact: AudioArtifact,
            request_context: RequestContext,
        ) -> TranscriptCandidate:
            del audio_artifact, request_context
            raise RuntimeError("simulated AssemblyAI failure")

    settings = _real_settings(
        transcription_providers="assemblyai",
        speechmatics_api_key=None,
        deepgram_api_key=None,
    )
    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=RealRuntimeOverrides(
            audio_extractor=FFmpegAudioExtractionAdapter(
                blob_store=blob_store,
                command_runner=fake_runner,
            ),
            transcription_adapters=(cast("TranscriptionAdapter", FailingAssemblyAIAdapter()),),
        ),
    )
    initial_state = GraphState(
        run_id="run-assembly-failure",
        job=_job_context(),
        current_stage="ingest",
        source_video_ref=str(source_path),
        source_artifact_ref=source_ref,
    )

    with pytest.raises(RuntimeError, match="all transcription providers failed"):
        run_workflow(initial_state, runtime)

    failed_execution = next(
        record
        for record in run_store.list_node_executions("run-assembly-failure")
        if record.node_name == "fanout_transcription"
    )
    assert failed_execution.error == {
        "message": "all transcription providers failed",
        "category": "transcription_failed",
        "reason": "all_transcription_providers_failed",
        "provider_errors": [
            {
                "provider_id": "assemblyai",
                "message": "simulated AssemblyAI failure",
            }
        ],
    }


def test_build_real_transcription_adapters_uses_assemblyai_specific_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_timeouts: dict[str, float] = {}

    class StubAdapter:
        def __init__(self, provider_id: str, *, timeout_seconds: float) -> None:
            self.provider_id = provider_id
            captured_timeouts[provider_id] = timeout_seconds

    monkeypatch.setattr(
        runtime_module,
        "AssemblyAITranscriptionAdapter",
        lambda **kwargs: StubAdapter("assemblyai", timeout_seconds=kwargs["timeout_seconds"]),
    )
    monkeypatch.setattr(
        runtime_module,
        "SpeechmaticsTranscriptionAdapter",
        lambda **kwargs: StubAdapter("speechmatics", timeout_seconds=kwargs["timeout_seconds"]),
    )
    monkeypatch.setattr(
        runtime_module,
        "DeepgramTranscriptionAdapter",
        lambda **kwargs: StubAdapter("deepgram", timeout_seconds=kwargs["timeout_seconds"]),
    )

    runtime_module._build_real_transcription_adapters(
        settings=_real_settings(
            provider_timeout_seconds=30.0,
            assemblyai_timeout_seconds=300.0,
        ),
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        retry_policy=_retry_policy(),
    )

    assert captured_timeouts == {
        "assemblyai": 300.0,
        "speechmatics": 30.0,
        "deepgram": 30.0,
    }
