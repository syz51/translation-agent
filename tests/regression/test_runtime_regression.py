from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from translation_agent.api import convert_translation_json_to_srt
from translation_agent.config import (
    Settings,
    load_settings,
    resolve_transcription_providers,
    sanitize_db_target,
    validate_environment,
)
from translation_agent.graph import (
    GraphState,
    RealRuntimeOverrides,
    build_phase_three_runtime,
    build_phase_two_runtime,
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
from translation_agent.observability import NoOpTraceSink
from translation_agent.replay import ReplayAdjudicationRequest, replay_adjudication
from translation_agent.review import content_risk_class_for_scenario
from translation_agent.storage import (
    LocalBlobStore,
    NodeExecutionRecord,
    RunRecord,
    job_path,
    job_scope_token,
)

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        (
            "postgresql://user:secret@db.example.com:5432/translation_agent?sslmode=require",
            "postgresql://db.example.com:5432/translation_agent",
        ),
        (
            "postgresql://user:secret@db.example.com/translation_agent?connect_timeout=1",
            "postgresql://db.example.com/translation_agent",
        ),
        ("not-a-dsn", "<invalid>"),
        (None, "<missing>"),
    ],
)
def test_sanitize_db_target_strips_secrets_and_noise_regression(
    dsn: str | None, expected: str
) -> None:
    assert sanitize_db_target(dsn) == expected


def test_blob_store_overwrite_does_not_leave_temporary_files_regression(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    store.put_bytes("jobs/run-1/request.json", b"first")
    store.put_bytes("jobs/run-1/request.json", b"second")

    assert store.read_bytes("jobs/run-1/request.json") == b"second"
    assert store.list_keys() == ["jobs/run-1/request.json"]
    assert not any(path.name.startswith(".tmp-blob-") for path in store.root.rglob("*"))


@dataclass
class InMemoryRunStore:
    runs: dict[str, RunRecord]
    node_executions: dict[str, NodeExecutionRecord]

    def __init__(self) -> None:
        self.runs = {}
        self.node_executions = {}

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


def _job_context(
    *,
    job_id: str = "job-replay",
    tenant_id: str = "tenant-1",
    project_id: str = "project-1",
    source_language: str = "en",
    target_language: str = "fr",
) -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id=tenant_id,
        project_id=project_id,
        source_video_ref="input.mp4",
        target_language=target_language,
        source_language=source_language,
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
        media_key=f"source-ref:{job_id}",
    )


def _run_workflow(
    tmp_path: Path,
    *,
    run_id: str,
    scenario: str,
    job: JobContext | None = None,
) -> tuple[GraphState, LocalBlobStore]:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id=run_id, status="running")
    blob_store = LocalBlobStore(tmp_path / run_id / "blobs")
    source_ref = f"jobs/{run_id}-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario=scenario,
    )
    initial_state = GraphState(
        run_id=run_id,
        job=job or _job_context(),
        current_stage="ingest",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
    )
    final_state = run_workflow(initial_state, runtime)
    return final_state, blob_store


def _load_json(blob_store: LocalBlobStore, path: str) -> dict[str, object]:
    return json.loads(blob_store.read_bytes(path).decode("utf-8"))


def _normalize_scorecard(payload: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload))
    normalized["run_id"] = "<normalized>"
    normalized["trace_refs"] = ["<normalized>"]
    for fact in normalized.get("routing_facts", []):
        source_ref = fact.get("source_ref")
        if isinstance(source_ref, str) and source_ref.startswith("jobs/run-"):
            fact["source_ref"] = "<normalized-request-artifact>"
    return normalized


def _normalize_failure_manifest(payload: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload))
    normalized["run_id"] = "<normalized>"
    return normalized


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


def _configure_real_mode_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    transcription_providers: str | None = None,
    assemblyai_api_key: str | None = None,
    speechmatics_api_key: str | None = None,
    deepgram_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> None:
    monkeypatch.setenv("TA_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TA_ADAPTER_MODE", "real")
    monkeypatch.setenv("TA_ALLOW_LANGGRAPH_PY314_WARNING", "1")
    monkeypatch.delenv("TA_STATE_DB_DSN", raising=False)
    if transcription_providers is None:
        monkeypatch.delenv("TA_TRANSCRIPTION_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("TA_TRANSCRIPTION_PROVIDERS", transcription_providers)

    for env_var, value in (
        ("TA_ASSEMBLYAI_API_KEY", assemblyai_api_key),
        ("TA_SPEECHMATICS_API_KEY", speechmatics_api_key),
        ("TA_DEEPGRAM_API_KEY", deepgram_api_key),
        ("TA_OPENAI_API_KEY", openai_api_key),
    ):
        if value is None:
            monkeypatch.delenv(env_var, raising=False)
        else:
            monkeypatch.setenv(env_var, value)


class StaticAudioExtractionAdapter:
    adapter_id = "ffmpeg"

    def __init__(self, *, blob_store: LocalBlobStore) -> None:
        self._blob_store = blob_store

    def extract_audio(self, video_ref: str, job_context: RequestContext) -> AudioArtifact:
        artifact_ref = job_path(job_context.job, "artifacts", "audio.wav")
        self._blob_store.put_bytes(artifact_ref, b"RIFF")
        scope_token = job_scope_token(job_context.job)
        return AudioArtifact(
            artifact_id=f"audio-{job_context.job.job_id}-{scope_token}",
            job_id=job_context.job.job_id,
            blob_ref=artifact_ref,
            duration_ms=1_000,
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
            extraction_metadata={
                "adapter_id": self.adapter_id,
                "source_video_ref": video_ref,
            },
        )


class StaticTranscriptionAdapter:
    def __init__(self, provider_id: str, *, blob_store: LocalBlobStore) -> None:
        self.provider_id = provider_id
        self._blob_store = blob_store

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate:
        del audio_artifact
        text = f"Hello world from {self.provider_id}."
        raw_payload_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"{self.provider_id}.json",
        )
        self._blob_store.put_bytes(
            raw_payload_ref,
            (
                json.dumps(
                    {"provider": self.provider_id, "text": text},
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        scope_token = job_scope_token(request_context.job)
        return TranscriptCandidate(
            candidate_id=f"tr-{self.provider_id}-{request_context.job.job_id}-{scope_token}",
            job_id=request_context.job.job_id,
            provider_id=self.provider_id,
            provider_request_id=f"req-{self.provider_id}-{request_context.run_id}",
            language=request_context.job.source_language,
            segments=(
                Segment(
                    segment_id=f"seg-{self.provider_id}-1",
                    start_ms=0,
                    end_ms=1_000,
                    speaker="speaker-1",
                    source_text=text,
                ),
            ),
            full_text=text,
            speaker_map={"speaker-1": "Host"},
            timing_resolution="segment",
            raw_payload_ref=raw_payload_ref,
            normalization_version="raw",
            metadata={
                "provider_rank": {
                    "assemblyai": 0,
                    "speechmatics": 1,
                    "deepgram": 2,
                }[self.provider_id]
            },
        )


class FailingTranscriptionAdapter:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate:
        del audio_artifact, request_context
        raise RuntimeError(f"simulated failure for {self.provider_id}")


class StaticTranslationAdapter:
    model_id = "gpt-5.4-mini"

    def __init__(self, *, blob_store: LocalBlobStore) -> None:
        self._blob_store = blob_store

    def generate_translation(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> TranslationCandidate:
        text = {
            "variant-a": "Bonjour le monde",
            "variant-b": "Salut le monde",
        }[prompt_variant_id]
        raw_response_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"openai-{prompt_variant_id}.json",
        )
        self._blob_store.put_bytes(
            raw_response_ref,
            (
                json.dumps(
                    {"variant": prompt_variant_id, "translation": text},
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        scope_token = job_scope_token(request_context.job)
        return TranslationCandidate(
            candidate_id=f"tl-{prompt_variant_id}-{request_context.job.job_id}-{scope_token}",
            job_id=request_context.job.job_id,
            source_transcript_candidate_id=final_transcript.candidate_id,
            model_id=self.model_id,
            prompt_variant_id=prompt_variant_id,
            prompt_version="phase-3-v1",
            language=request_context.job.target_language,
            segments=(
                Segment(
                    segment_id=final_transcript.segments[0].segment_id,
                    start_ms=0,
                    end_ms=1_000,
                    speaker="speaker-1",
                    source_text=final_transcript.full_text,
                    target_text=text,
                ),
            ),
            full_text=text,
            raw_response_ref=raw_response_ref,
            normalization_version="raw",
            metadata={},
        )


def test_real_mode_unset_transcription_provider_selector_preserves_three_provider_order_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    runtime = build_phase_three_runtime(
        settings=_real_settings(),
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert [adapter.provider_id for adapter in runtime.transcription_adapters] == [
        "assemblyai",
        "speechmatics",
        "deepgram",
    ]


def test_real_mode_assemblyai_only_selector_validates_with_minimal_credentials_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(
        monkeypatch,
        tmp_path,
        transcription_providers="assemblyai",
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        openai_api_key="openai",  # pragma: allowlist secret
    )

    result = validate_environment(load_settings(env_file=None))

    assert result.ok is True
    assert result.provider_config_ok is True
    assert result.provider_config_error is None


def test_real_mode_subset_selector_normalizes_case_and_preserves_order_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_real_mode_env(
        monkeypatch,
        tmp_path,
        transcription_providers=" ASSEMBLYAI , deepgram ",
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
        openai_api_key="openai",  # pragma: allowlist secret
    )
    settings = load_settings(env_file=None)
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert resolve_transcription_providers(settings) == ("assemblyai", "deepgram")
    assert [adapter.provider_id for adapter in runtime.transcription_adapters] == [
        "assemblyai",
        "deepgram",
    ]


@pytest.mark.parametrize(
    ("configured_value", "expected_error"),
    [
        (
            "assemblyai,foo",
            "TA_TRANSCRIPTION_PROVIDERS contains unsupported providers: foo",
        ),
        (
            " , , ",
            "TA_TRANSCRIPTION_PROVIDERS must select at least one provider when set",
        ),
        (
            "assemblyai, deepgram, AssemblyAI",
            "TA_TRANSCRIPTION_PROVIDERS contains duplicate providers: assemblyai",
        ),
    ],
)
def test_real_mode_selector_rejects_invalid_provider_inputs_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_value: str,
    expected_error: str,
) -> None:
    _configure_real_mode_env(
        monkeypatch,
        tmp_path,
        transcription_providers=configured_value,
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
        openai_api_key="openai",  # pragma: allowlist secret
    )

    result = validate_environment(load_settings(env_file=None))

    assert result.ok is False
    assert result.provider_config_error == expected_error


def test_real_mode_assemblyai_only_workflow_stays_on_single_candidate_path_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-real-assembly-only")
    run_id = "run-real-assembly-only"
    run_store = InMemoryRunStore()
    run_store.create_run(run_id=run_id, status="running")
    blob_store = LocalBlobStore(tmp_path / run_id / "blobs")
    source_ref = f"jobs/{run_id}-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    runtime = build_phase_three_runtime(
        settings=_real_settings(
            transcription_providers="assemblyai",
            speechmatics_api_key=None,
            deepgram_api_key=None,
        ),
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=RealRuntimeOverrides(
            audio_extractor=StaticAudioExtractionAdapter(blob_store=blob_store),
            transcription_adapters=(
                StaticTranscriptionAdapter("assemblyai", blob_store=blob_store),
            ),
            translation_adapter=StaticTranslationAdapter(blob_store=blob_store),
        ),
    )
    final_state = run_workflow(
        GraphState(
            run_id=run_id,
            job=job,
            current_stage="ingest",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
        ),
        runtime,
    )

    decision = FinalTranscriptDecision.model_validate_json(
        blob_store.read_bytes(job_path(job, "decisions", "transcript.json"))
    )

    assert len(final_state.transcript_candidate_ids) == 1
    assert decision.decision_mode == "automatic_finalize"
    assert decision.escalated is True
    assert decision.human_review_required is False
    assert decision.investigation_ref == job_path(job, "investigations", "transcript.json")
    assert blob_store.exists(job_path(job, "published", "transcript.json"))
    assert blob_store.exists(job_path(job, "published", "translation.json"))


def test_real_mode_assemblyai_and_deepgram_subset_publishes_selected_provider_artifacts_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-real-subset")
    run_id = "run-real-subset"
    run_store = InMemoryRunStore()
    run_store.create_run(run_id=run_id, status="running")
    blob_store = LocalBlobStore(tmp_path / run_id / "blobs")
    source_ref = f"jobs/{run_id}-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    runtime = build_phase_three_runtime(
        settings=_real_settings(
            transcription_providers="assemblyai,deepgram",
            speechmatics_api_key=None,
        ),
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=RealRuntimeOverrides(
            audio_extractor=StaticAudioExtractionAdapter(blob_store=blob_store),
            transcription_adapters=(
                StaticTranscriptionAdapter("assemblyai", blob_store=blob_store),
                StaticTranscriptionAdapter("deepgram", blob_store=blob_store),
            ),
            translation_adapter=StaticTranslationAdapter(blob_store=blob_store),
        ),
    )

    final_state = run_workflow(
        GraphState(
            run_id=run_id,
            job=job,
            current_stage="ingest",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
        ),
        runtime,
    )

    assert len(final_state.transcript_candidate_ids) == 2
    assert blob_store.exists(job_path(job, "raw", "provider-payloads", "assemblyai.json"))
    assert blob_store.exists(job_path(job, "raw", "provider-payloads", "deepgram.json"))
    assert not blob_store.exists(job_path(job, "raw", "provider-payloads", "speechmatics.json"))
    assert blob_store.exists(job_path(job, "published", "translation.json"))


def test_convert_translation_json_to_srt_matches_published_export_regression(
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-convert-regression")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-convert-regression",
        scenario="happy",
        job=job,
    )

    assert final_state.translation_failed is False
    assert final_state.human_review_required is False

    source_path = blob_store.root / job_path(job, "published", "translation.json")
    rebuilt_output = tmp_path / "rebuilt" / "translation.srt"
    result = convert_translation_json_to_srt(source_path, rebuilt_output)
    published_output = blob_store.root / job_path(job, "exports", "translation.srt")

    assert result.output_path == rebuilt_output.resolve()
    assert result.subtitle_count > 0
    assert rebuilt_output.read_text(encoding="utf-8") == published_output.read_text(
        encoding="utf-8"
    )


def test_real_mode_single_selected_provider_failure_raises_expected_error_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-real-single-provider-failure")
    run_id = "run-real-single-provider-failure"
    run_store = InMemoryRunStore()
    run_store.create_run(run_id=run_id, status="running")
    blob_store = LocalBlobStore(tmp_path / run_id / "blobs")
    source_ref = f"jobs/{run_id}-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    runtime = build_phase_three_runtime(
        settings=_real_settings(
            transcription_providers="assemblyai",
            speechmatics_api_key=None,
            deepgram_api_key=None,
        ),
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        overrides=RealRuntimeOverrides(
            audio_extractor=StaticAudioExtractionAdapter(blob_store=blob_store),
            transcription_adapters=(FailingTranscriptionAdapter("assemblyai"),),
            translation_adapter=StaticTranslationAdapter(blob_store=blob_store),
        ),
    )

    with pytest.raises(RuntimeError, match="all transcription providers failed"):
        run_workflow(
            GraphState(
                run_id=run_id,
                job=job,
                current_stage="ingest",
                source_video_ref="input.mp4",
                source_artifact_ref=source_ref,
            ),
            runtime,
        )


def test_replay_scorecards_memory_and_prompt_proposals_are_stable_regression(
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-replay-happy")
    _, first_blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-a",
        scenario="happy",
        job=job,
    )
    _, second_blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-b",
        scenario="happy",
        job=job,
    )

    scorecard_path = job_path(job, "published", "scorecard.json")
    scope_token = job_scope_token(job)
    consolidation_path = job_path(
        job,
        "memory",
        "consolidations",
        f"consolidation-batch-translation_adjudication-job-replay-happy-{scope_token}.json",
    )
    prompt_path = job_path(
        job,
        "memory",
        "prompt-evolution",
        (
            "prompt-evolution-"
            f"consolidation-batch-translation_adjudication-job-replay-happy-{scope_token}.json"
        ),
    )

    assert _normalize_scorecard(_load_json(first_blob_store, scorecard_path)) == (
        _normalize_scorecard(_load_json(second_blob_store, scorecard_path))
    )
    assert _load_json(first_blob_store, consolidation_path) == _load_json(
        second_blob_store,
        consolidation_path,
    )
    assert _load_json(first_blob_store, prompt_path) == _load_json(second_blob_store, prompt_path)


def test_replay_translation_failure_manifest_is_stable_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-failure")
    _, first_blob_store = _run_workflow(
        tmp_path,
        run_id="run-failure-a",
        scenario="translation_failed",
        job=job,
    )
    _, second_blob_store = _run_workflow(
        tmp_path,
        run_id="run-failure-b",
        scenario="translation_failed",
        job=job,
    )

    failure_path = job_path(job, "published", "translation-failed.json")
    assert _normalize_failure_manifest(_load_json(first_blob_store, failure_path)) == (
        _normalize_failure_manifest(_load_json(second_blob_store, failure_path))
    )


def test_replay_adjudication_uses_persisted_candidates_reviews_and_memory_refs(
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-replay-adjudication")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-adjudication",
        scenario="translation_conflict",
        job=job,
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="translation_conflict",
    )
    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="translation",
            candidate_refs=tuple(
                job_path(job, "candidates", "translations", f"{candidate_id}.json")
                for candidate_id in final_state.translation_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "translation", f"{review_id}.json")
                for review_id in final_state.translation_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_translation"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("translation_conflict"),
        ),
    )

    stored_decision = _load_json(blob_store, job_path(job, "decisions", "translation.json"))
    replayed_decision = replayed.decision.model_dump(mode="json")

    assert replayed.decision.decision_mode == stored_decision["decision_mode"]
    assert replayed.decision.winner_candidate_id == stored_decision["winner_candidate_id"]
    assert replayed.decision.disagreement_bucket == stored_decision["disagreement_bucket"]
    assert replayed_decision["adjudication_scorecard"] == stored_decision["adjudication_scorecard"]


def test_replay_adjudication_supports_transcript_stage_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-transcript")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-transcript",
        scenario="happy",
        job=job,
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="happy",
    )

    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="transcript",
            candidate_refs=tuple(
                job_path(job, "candidates", "transcripts", f"{candidate_id}.json")
                for candidate_id in final_state.transcript_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "transcript", f"{review_id}.json")
                for review_id in final_state.transcript_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_transcript"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("happy"),
        ),
    )

    assert replayed.decision.decision_mode == "automatic_finalize"
    assert replayed.decision.winner_candidate_id == final_state.final_transcript_candidate_id


def test_replay_adjudication_preserves_timeout_escalation_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-timeout")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-timeout",
        scenario="translation_conflict_timeout",
        job=job,
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="translation_conflict_timeout",
    )

    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="translation",
            candidate_refs=tuple(
                job_path(job, "candidates", "translations", f"{candidate_id}.json")
                for candidate_id in final_state.translation_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "translation", f"{review_id}.json")
                for review_id in final_state.translation_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_translation"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("translation_conflict_timeout"),
        ),
    )

    assert replayed.decision.decision_mode == "human_review"
    assert replayed.decision.winner_candidate_id is None
    assert replayed.decision.disagreement_bucket == "unresolved"


def test_replay_adjudication_ignores_missing_timeout_artifact_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-missing-timeout")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-missing-timeout",
        scenario="translation_conflict_timeout",
        job=job,
    )
    blob_store.delete(job_path(job, "investigations", "translation.json"))
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="translation_conflict_timeout",
    )

    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="translation",
            candidate_refs=tuple(
                job_path(job, "candidates", "translations", f"{candidate_id}.json")
                for candidate_id in final_state.translation_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "translation", f"{review_id}.json")
                for review_id in final_state.translation_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_translation"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("translation_conflict_timeout"),
        ),
    )

    assert replayed.decision.decision_mode == "conflict_investigation"
    assert replayed.decision.human_review_required is False
