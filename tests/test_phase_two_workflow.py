from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.graph import GraphState, build_phase_two_runtime, run_workflow
from translation_agent.models import JobContext, Segment, TranscriptCandidate
from translation_agent.observability import NoOpTraceSink
from translation_agent.storage import LocalBlobStore, NodeExecutionRecord, RunRecord, job_path

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
    tmp_path: Path, *, scenario: str
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
        job=_job_context(),
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
    assert blob_store.exists(_artifact_path("published", "transcript.json"))
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    assert len(run_store.list_node_executions("run-123")) == 13


def test_phase_two_degraded_stt_keeps_run_recoverable(tmp_path: Path) -> None:
    final_state, run_store, _ = _run_workflow(tmp_path, scenario="degraded_stt")

    assert final_state.translation_failed is False
    assert final_state.human_review_required is False
    assert any(
        fact.fact_type == "transcription_provider_failed" and fact.value == "speechmatics"
        for fact in final_state.routing_facts
    )
    assert len(run_store.list_node_executions("run-123")) == 13


def test_phase_two_single_surviving_translation_variant_still_publishes(tmp_path: Path) -> None:
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="translation_single_variant")

    assert final_state.translation_failed is False
    assert final_state.final_translation_candidate_id is not None
    assert blob_store.exists(_artifact_path("published", "translation.json"))
    surviving_translation_counts = [
        fact.value
        for fact in final_state.routing_facts
        if fact.fact_type == "surviving_translation_candidates"
    ]
    assert surviving_translation_counts[-1] == "3"


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
    assert len(executed_nodes) == 13


def test_phase_four_medium_disagreement_invokes_conflict_investigator(tmp_path: Path) -> None:
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="translation_conflict")

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
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="translation_high_risk")

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
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="translation_escalation")

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

    class SlowAdapter:
        def __init__(self, provider_id: str, rank: int) -> None:
            self.provider_id = provider_id
            self.rank = rank

        def transcribe(self, audio_artifact, request_context):  # noqa: ANN001
            del audio_artifact
            time.sleep(0.2)
            return TranscriptCandidate(
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
            )

    runtime.transcription_adapters = (
        SlowAdapter("assemblyai", 0),
        SlowAdapter("speechmatics", 1),
        SlowAdapter("deepgram", 2),
    )

    started = time.monotonic()
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
    elapsed = time.monotonic() - started

    assert final_state.current_stage == "finalize_outputs"
    assert elapsed < 0.45
