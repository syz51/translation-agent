from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest

from translation_agent.graph import GraphState, RoutingFact, build_phase_two_runtime, run_workflow
from translation_agent.models import (
    CanonicalTranscriptSpan,
    JobContext,
    RequestContext,
    Segment,
    SynthesizedTranscriptArtifact,
    TranscriptCandidate,
    TranscriptQualityMetrics,
    TranslationCandidate,
)
from translation_agent.nodes.translate import (
    _translation_candidate_id,
    generate_translation_candidates,
)
from translation_agent.observability import NoOpTraceSink
from translation_agent.storage import (
    LocalBlobStore,
    NodeExecutionRecord,
    RunRecord,
    SQLiteOperationalStore,
    job_path,
)

pytestmark = pytest.mark.unit


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


def _job_context(job_id: str = "job-phase-two") -> JobContext:
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


def _artifact_path(*parts: str) -> str:
    return job_path(_job_context(), *parts)


def _run_workflow(
    tmp_path: Path, *, scenario: str, job: JobContext | None = None
) -> tuple[GraphState, InMemoryRunStore, LocalBlobStore]:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-123", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-123-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario=scenario,
    )
    initial_state = GraphState(
        run_id="run-123",
        job=job or _job_context(),
        current_stage="ingest",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
    )
    final_state = run_workflow(initial_state, runtime)
    return final_state, run_store, blob_store


def test_phase_two_happy_path_executes_full_graph(tmp_path: Path) -> None:
    final_state, run_store, blob_store = _run_workflow(tmp_path, scenario="happy")

    assert final_state.current_stage == "finalize_outputs"
    assert final_state.human_review_required is False
    assert final_state.translation_failed is False
    assert len(final_state.memory_batch_ids) == 2
    assert final_state.final_translation_candidate_id is not None
    assert final_state.translation_candidate_ids == (final_state.final_translation_candidate_id,)
    assert blob_store.exists(_artifact_path("published", "transcript.json"))
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    assert len(run_store.list_node_executions("run-123")) == 16


def test_phase_two_degraded_stt_keeps_run_recoverable(tmp_path: Path) -> None:
    final_state, run_store, _ = _run_workflow(tmp_path, scenario="degraded_stt")

    assert final_state.translation_failed is False
    assert final_state.human_review_required is False
    assert any(
        fact.fact_type == "transcription_provider_failed" and fact.value == "speechmatics"
        for fact in final_state.routing_facts
    )
    assert len(run_store.list_node_executions("run-123")) == 16


def test_phase_two_dual_experiment_single_surviving_variant_still_publishes(
    tmp_path: Path,
) -> None:
    final_state, _, blob_store = _run_workflow(
        tmp_path,
        scenario="translation_single_variant",
        job=_job_context().model_copy(update={"translation_variant_policy": "dual_experiment"}),
    )

    assert final_state.translation_failed is False
    assert final_state.final_translation_candidate_id is not None
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    surviving_translation_counts = [
        fact.value
        for fact in final_state.routing_facts
        if fact.fact_type == "surviving_translation_candidates"
    ]
    assert surviving_translation_counts[-1] == "1"


def test_phase_two_translation_failure_preserves_transcript_outputs(tmp_path: Path) -> None:
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="translation_failed")

    assert final_state.translation_failed is True
    assert final_state.human_review_required is True
    assert final_state.final_translation_candidate_id is None
    assert final_state.final_translation_decision_ref is not None
    assert blob_store.exists(_artifact_path("published", "transcript.json"))
    assert not blob_store.exists(_artifact_path("published", "translation.json"))


def test_phase_two_escalation_skips_translation_path(tmp_path: Path) -> None:
    final_state, run_store, blob_store = _run_workflow(tmp_path, scenario="transcript_escalation")

    assert final_state.human_review_required is False
    assert final_state.final_translation_decision_ref is not None
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    executed_nodes = [record.node_name for record in run_store.list_node_executions("run-123")]
    assert "generate_translation_candidates" in executed_nodes
    assert len(executed_nodes) == 16


def test_phase_four_medium_disagreement_invokes_conflict_investigator(tmp_path: Path) -> None:
    final_state, _, blob_store = _run_workflow(
        tmp_path,
        scenario="translation_conflict",
        job=_job_context().model_copy(update={"translation_variant_policy": "dual_experiment"}),
    )

    assert final_state.human_review_required is False
    assert final_state.final_translation_candidate_id is not None
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    assert blob_store.exists(_artifact_path("investigations", "translation.json"))
    assert any(
        fact.fact_type == "decision_mode" and fact.value == "conflict_investigation"
        for fact in final_state.routing_facts
    )
    assert any(
        fact.fact_type == "disagreement_bucket" and fact.value == "medium"
        for fact in final_state.routing_facts
    )


def test_phase_four_high_risk_invokes_stronger_adjudicator(tmp_path: Path) -> None:
    final_state, _, blob_store = _run_workflow(
        tmp_path,
        scenario="translation_high_risk",
        job=_job_context().model_copy(update={"translation_variant_policy": "dual_experiment"}),
    )

    assert final_state.human_review_required is False
    assert final_state.final_translation_candidate_id is not None
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    assert blob_store.exists(_artifact_path("investigations", "translation.json"))
    assert any(
        fact.fact_type == "decision_mode" and fact.value == "stronger_adjudicator"
        for fact in final_state.routing_facts
    )
    assert any(
        fact.fact_type == "disagreement_bucket" and fact.value == "high"
        for fact in final_state.routing_facts
    )


def test_phase_four_translation_escalation_uses_stronger_adjudicator(
    tmp_path: Path,
) -> None:
    final_state, _, blob_store = _run_workflow(
        tmp_path,
        scenario="translation_escalation",
        job=_job_context().model_copy(update={"translation_variant_policy": "dual_experiment"}),
    )

    assert final_state.human_review_required is False
    assert final_state.final_translation_candidate_id is not None
    assert final_state.final_translation_decision_ref is not None
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    assert blob_store.exists(_artifact_path("investigations", "translation.json"))
    assert any(
        fact.fact_type == "decision_mode" and fact.value == "stronger_adjudicator"
        for fact in final_state.routing_facts
    )


def test_phase_two_transcription_fanout_runs_in_parallel(tmp_path: Path) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-123", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-123-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )

    start_barrier = threading.Barrier(3, timeout=1.0)

    class SlowAdapter:
        def __init__(self, provider_id: str, rank: int) -> None:
            self.provider_id = provider_id
            self.rank = rank

        def transcribe(self, audio_artifact, request_context):  # noqa: ANN001
            candidate, _ = self.transcribe_with_payload(audio_artifact, request_context)
            return candidate

        def transcribe_with_payload(self, audio_artifact, request_context):  # noqa: ANN001
            del audio_artifact
            start_barrier.wait()
            return (
                TranscriptCandidate(
                    candidate_id=f"tr-{self.provider_id}-{request_context.job.job_id}",
                    job_id=request_context.job.job_id,
                    provider_id=self.provider_id,
                    provider_request_id=f"req-{self.provider_id}",
                    language=request_context.job.source_language,
                    segments=(
                        Segment(
                            segment_id=f"seg-{self.provider_id}-1",
                            start_ms=0,
                            end_ms=1000,
                            speaker="speaker-1",
                            source_text=f"text-{self.provider_id}",
                        ),
                    ),
                    full_text=f"text-{self.provider_id}",
                    speaker_map={"speaker-1": "Host"},
                    timing_resolution="segment",
                    raw_payload_ref=job_path(
                        request_context.job,
                        "raw",
                        "provider-payloads",
                        f"{self.provider_id}.json",
                    ),
                    normalization_version="test",
                    metadata={"provider_rank": self.rank},
                ),
                {"provider": self.provider_id, "text": f"text-{self.provider_id}"},
            )

    runtime.transcription_adapters = (
        SlowAdapter("assemblyai", 0),
        SlowAdapter("speechmatics", 1),
        SlowAdapter("deepgram", 2),
    )

    final_state = run_workflow(
        GraphState(
            run_id="run-123",
            job=_job_context(),
            current_stage="ingest",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
        ),
        runtime,
    )

    assert final_state.current_stage == "finalize_outputs"


def test_generate_translation_candidates_defaults_to_winner_first_single_variant(
    tmp_path: Path,
) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-translation-single", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-translation-single-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )
    job = _job_context(job_id="job-translation-single")
    synthesized = SynthesizedTranscriptArtifact(
        artifact_id="synth-job-translation-single",
        job_id=job.job_id,
        run_id="run-translation-single",
        language=job.source_language,
        transcript_metadata={"blocker_tags": []},
        canonical_spans=(
            CanonicalTranscriptSpan(
                canonical_span_id="canonical-span-0001",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                supporting_candidate_ids=("tr-a",),
                supporting_provider_ids=("assemblyai",),
                metadata={},
            ),
        ),
        span_candidates=(),
        final_segments=(
            Segment(
                segment_id="seg-a",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                source_text="Alpha",
            ),
        ),
        provenance=(),
        unresolved_spans=(),
        quality_metrics=TranscriptQualityMetrics(
            canonical_span_count=1,
            supported_span_count=1,
            emitted_span_count=1,
            unresolved_span_count=0,
            overlap_count=0,
            non_monotonic_count=0,
            zero_length_count=0,
            dropped_supported_span_count=0,
            provider_support_summary={"assemblyai": 1},
        ),
        full_text="Alpha",
        status="ready",
    )
    final_transcript_ref = job_path(job, "artifacts", "final-transcript.json")
    blob_store.put_bytes(
        final_transcript_ref,
        (synthesized.model_dump_json(indent=2) + "\n").encode(),
    )

    class RawAdapter:
        model_id = "gpt-5.4-mini"
        _prompt_version = "phase-3-v1"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def generate_translation(
            self,
            final_transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> TranslationCandidate:
            candidate, _ = self.generate_translation_with_payload(
                final_transcript,
                prompt_variant_id,
                request_context,
            )
            return candidate

        def generate_translation_with_payload(
            self,
            transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> tuple[TranslationCandidate, dict[str, object]]:
            assert request_context.metadata["transcript_synthesis"]["transcript_blockers"] == []
            self.calls.append((transcript.candidate_id, prompt_variant_id))
            target_text = f"{transcript.full_text}-{prompt_variant_id}"
            candidate = TranslationCandidate(
                candidate_id=f"raw-{transcript.candidate_id}-{prompt_variant_id}",
                job_id=job.job_id,
                source_transcript_ref=final_transcript_ref,
                model_id=self.model_id,
                prompt_variant_id=prompt_variant_id,
                prompt_version="raw-prompt",
                language=job.target_language,
                segments=(transcript.segments[0].model_copy(update={"target_text": target_text}),),
                full_text=target_text,
                raw_response_ref=None,
                normalization_version="raw-test",
                metadata={},
            )
            return candidate, {"candidate": candidate.candidate_id}

    adapter = RawAdapter()
    runtime.translation_adapter = cast(Any, adapter)
    state = GraphState(
        run_id="run-translation-single",
        job=job,
        current_stage="generate_translation_candidates",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
        final_transcript_ref=final_transcript_ref,
    )

    output = generate_translation_candidates(state, runtime)
    source_token = sha256(final_transcript_ref.encode()).hexdigest()[:12]

    assert output["translation_failed"] is False
    assert adapter.calls == [("synth-job-translation-single", "variant-a")]
    assert output["raw_translation_payload_refs"] == (
        job_path(job, "raw", "translation-candidates", f"variant-a-{source_token}.json"),
    )
    assert output["raw_translation_candidate_refs"] == (
        job_path(
            job,
            "staging",
            "translations",
            f"{_translation_candidate_id('variant-a', final_transcript_ref, job.job_id)}.json",
        ),
    )


def test_generate_translation_candidates_preserves_task_order_and_partial_failures(
    tmp_path: Path,
) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-translation-order", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-translation-order-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )
    job = _job_context(job_id="job-translation-order").model_copy(
        update={"translation_variant_policy": "dual_experiment"}
    )
    synthesized_segments = (
        Segment(
            segment_id="seg-a",
            start_ms=0,
            end_ms=1_000,
            speaker="speaker-1",
            source_text="Alpha",
        ),
    )
    synthesized = SynthesizedTranscriptArtifact(
        artifact_id="synth-job-translation-order",
        job_id=job.job_id,
        run_id="run-translation-order",
        language=job.source_language,
        transcript_metadata={"blocker_tags": ["transcript_overlaps"]},
        canonical_spans=(
            CanonicalTranscriptSpan(
                canonical_span_id="canonical-span-0001",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                supporting_candidate_ids=("tr-a", "tr-b"),
                supporting_provider_ids=("assemblyai", "speechmatics"),
                metadata={},
            ),
        ),
        span_candidates=(),
        final_segments=synthesized_segments,
        provenance=(),
        unresolved_spans=(),
        quality_metrics=TranscriptQualityMetrics(
            canonical_span_count=1,
            supported_span_count=1,
            emitted_span_count=1,
            unresolved_span_count=0,
            overlap_count=0,
            non_monotonic_count=0,
            zero_length_count=0,
            dropped_supported_span_count=0,
            provider_support_summary={"assemblyai": 1, "speechmatics": 1},
        ),
        full_text="Alpha",
        status="ready",
    )
    final_transcript_ref = job_path(job, "artifacts", "final-transcript.json")
    blob_store.put_bytes(
        final_transcript_ref,
        (synthesized.model_dump_json(indent=2) + "\n").encode(),
    )

    class RawAdapter:
        model_id = "gpt-5.4-mini"
        _prompt_version = "phase-3-v1"

        def __init__(self) -> None:
            self._first_variant_started = threading.Event()
            self._second_variant_released = threading.Event()

        def generate_translation(
            self,
            final_transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> TranslationCandidate:
            candidate, _ = self.generate_translation_with_payload(
                final_transcript,
                prompt_variant_id,
                request_context,
            )
            return candidate

        def generate_translation_with_payload(
            self,
            transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> tuple[TranslationCandidate, dict[str, object]]:
            if prompt_variant_id == "variant-a":
                self._first_variant_started.set()
                assert self._second_variant_released.wait(timeout=1)
            if prompt_variant_id == "variant-b":
                assert self._first_variant_started.wait(timeout=1)
                self._second_variant_released.set()
            if prompt_variant_id == "variant-a":
                raise RuntimeError("boom-tr-b-variant-a")
            assert request_context.metadata["transcript_synthesis"]["transcript_blockers"] == [
                "transcript_overlaps"
            ]
            target_text = f"{transcript.full_text}-{prompt_variant_id}"
            candidate = TranslationCandidate(
                candidate_id=f"raw-{transcript.candidate_id}-{prompt_variant_id}",
                job_id=request_context.job.job_id,
                source_transcript_ref=final_transcript_ref,
                model_id=self.model_id,
                prompt_variant_id=prompt_variant_id,
                prompt_version="raw-prompt",
                language=job.target_language,
                segments=(transcript.segments[0].model_copy(update={"target_text": target_text}),),
                full_text=target_text,
                raw_response_ref=None,
                normalization_version="raw-test",
                metadata={},
            )
            return candidate, {"candidate": candidate.candidate_id}

    adapter = RawAdapter()
    runtime.translation_adapter = cast(Any, adapter)
    state = GraphState(
        run_id="run-translation-order",
        job=job,
        current_stage="generate_translation_candidates",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
        final_transcript_ref=final_transcript_ref,
    )

    output = generate_translation_candidates(state, runtime)
    source_token = sha256(final_transcript_ref.encode()).hexdigest()[:12]

    assert output["translation_failed"] is False
    assert output["raw_translation_payload_refs"] == (
        job_path(job, "raw", "translation-candidates", f"variant-b-{source_token}.json"),
    )
    assert output["raw_translation_candidate_refs"] == (
        job_path(
            job,
            "staging",
            "translations",
            f"{_translation_candidate_id('variant-b', final_transcript_ref, job.job_id)}.json",
        ),
    )
    routing_facts = cast(tuple[RoutingFact, ...], output["routing_facts"])
    failed_facts = [
        fact for fact in routing_facts if fact.fact_type == "translation_variant_failed"
    ]
    assert [(fact.value, fact.source_ref) for fact in failed_facts] == [
        (f"variant-a:{final_transcript_ref}", "boom-tr-b-variant-a"),
    ]


def test_generate_translation_candidates_dual_experiment_runs_both_variants_for_synthesized_input(
    tmp_path: Path,
) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-translation-experiment", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-translation-experiment-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )
    job = _job_context(job_id="job-translation-experiment").model_copy(
        update={"translation_variant_policy": "dual_experiment"}
    )
    synthesized = SynthesizedTranscriptArtifact(
        artifact_id="synth-job-translation-experiment",
        job_id=job.job_id,
        run_id="run-translation-experiment",
        language=job.source_language,
        transcript_metadata={"blocker_tags": []},
        canonical_spans=(
            CanonicalTranscriptSpan(
                canonical_span_id="canonical-span-0001",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                supporting_candidate_ids=("tr-a",),
                supporting_provider_ids=("assemblyai",),
                metadata={},
            ),
        ),
        span_candidates=(),
        final_segments=(
            Segment(
                segment_id="seg-a",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                source_text="Alpha",
            ),
        ),
        provenance=(),
        unresolved_spans=(),
        quality_metrics=TranscriptQualityMetrics(
            canonical_span_count=1,
            supported_span_count=1,
            emitted_span_count=1,
            unresolved_span_count=0,
            overlap_count=0,
            non_monotonic_count=0,
            zero_length_count=0,
            dropped_supported_span_count=0,
            provider_support_summary={"assemblyai": 1},
        ),
        full_text="Alpha",
        status="ready",
    )
    final_transcript_ref = job_path(job, "artifacts", "final-transcript.json")
    blob_store.put_bytes(
        final_transcript_ref,
        (synthesized.model_dump_json(indent=2) + "\n").encode(),
    )

    class RawAdapter:
        model_id = "gpt-5.4-mini"
        _prompt_version = "phase-3-v1"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def generate_translation(
            self,
            final_transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> TranslationCandidate:
            candidate, _ = self.generate_translation_with_payload(
                final_transcript,
                prompt_variant_id,
                request_context,
            )
            return candidate

        def generate_translation_with_payload(
            self,
            transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> tuple[TranslationCandidate, dict[str, object]]:
            assert request_context.metadata["transcript_synthesis"]["transcript_blockers"] == []
            self.calls.append((transcript.candidate_id, prompt_variant_id))
            target_text = f"{transcript.full_text}-{prompt_variant_id}"
            candidate = TranslationCandidate(
                candidate_id=f"raw-{transcript.candidate_id}-{prompt_variant_id}",
                job_id=job.job_id,
                source_transcript_ref=final_transcript_ref,
                model_id=self.model_id,
                prompt_variant_id=prompt_variant_id,
                prompt_version="raw-prompt",
                language=job.target_language,
                segments=(transcript.segments[0].model_copy(update={"target_text": target_text}),),
                full_text=target_text,
                raw_response_ref=None,
                normalization_version="raw-test",
                metadata={},
            )
            return candidate, {"candidate": candidate.candidate_id}

    adapter = RawAdapter()
    runtime.translation_adapter = cast(Any, adapter)
    state = GraphState(
        run_id="run-translation-experiment",
        job=job,
        current_stage="generate_translation_candidates",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
        final_transcript_ref=final_transcript_ref,
    )

    output = generate_translation_candidates(state, runtime)
    source_token = sha256(final_transcript_ref.encode()).hexdigest()[:12]

    assert output["translation_failed"] is False
    assert adapter.calls == [
        ("synth-job-translation-experiment", "variant-a"),
        ("synth-job-translation-experiment", "variant-b"),
    ]
    assert output["raw_translation_payload_refs"] == (
        job_path(job, "raw", "translation-candidates", f"variant-a-{source_token}.json"),
        job_path(job, "raw", "translation-candidates", f"variant-b-{source_token}.json"),
    )


def test_sqlite_runtime_store_writes_stay_on_main_thread(tmp_path: Path) -> None:
    main_thread_id = threading.get_ident()
    store = SQLiteOperationalStore(tmp_path / "state.sqlite3")
    write_threads: list[int] = []
    try:
        for method_name in (
            "create_run",
            "update_run",
            "create_node_execution",
            "update_node_execution",
            "save_transcript_candidate",
            "save_translation_candidate",
            "save_transcript_decision",
            "save_translation_decision",
            "save_batch",
        ):
            original = getattr(store, method_name)

            def wrapped(*args, __original=original, **kwargs):  # noqa: ANN002, ANN003
                write_threads.append(threading.get_ident())
                return __original(*args, **kwargs)

            setattr(store, method_name, wrapped)

        blob_store = LocalBlobStore(tmp_path / "blobs")
        source_ref = "jobs/run-sqlite-request.json"
        blob_store.put_bytes(source_ref, b"{}\n")
        store.create_run(run_id="run-sqlite", status="running")
        runtime = build_phase_two_runtime(
            blob_store=blob_store,
            run_store=store,
            trace_sink=NoOpTraceSink(),
            source_artifact_ref=source_ref,
            scenario="happy",
        )

        run_workflow(
            GraphState(
                run_id="run-sqlite",
                job=_job_context(job_id="job-sqlite"),
                current_stage="ingest",
                source_video_ref="input.mp4",
                source_artifact_ref=source_ref,
            ),
            runtime,
        )
    finally:
        store.close()

    assert write_threads
    assert set(write_threads) == {main_thread_id}
