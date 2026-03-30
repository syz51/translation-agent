from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.graph import GraphState, build_phase_two_runtime, run_workflow
from translation_agent.models import (
    FinalTranslationDecision,
    JobContext,
)
from translation_agent.observability import JsonlTraceSink
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


def _job_context(
    *,
    job_id: str = "job-phase-six",
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
    )


def _run_workflow(
    tmp_path: Path,
    *,
    run_id: str,
    scenario: str,
    job: JobContext,
    blob_root: Path | None = None,
) -> tuple[GraphState, LocalBlobStore, Path]:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id=run_id, status="running")
    blob_store = LocalBlobStore(blob_root or tmp_path / "blobs")
    source_ref = f"jobs/{run_id}-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    trace_path = tmp_path / f"{run_id}.jsonl"
    with JsonlTraceSink(trace_path) as trace_sink:
        runtime = build_phase_two_runtime(
            blob_store=blob_store,
            run_store=run_store,
            trace_sink=trace_sink,
            source_artifact_ref=source_ref,
            scenario=scenario,
        )
        initial_state = GraphState(
            run_id=run_id,
            job=job,
            current_stage="ingest",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
        )
        final_state = run_workflow(initial_state, runtime)
    return final_state, blob_store, trace_path


def test_phase_six_publishes_trace_artifact_into_blob_store(tmp_path: Path) -> None:
    job = _job_context(job_id="job-trace")
    final_state, blob_store, trace_path = _run_workflow(
        tmp_path,
        run_id="run-trace",
        scenario="happy",
        job=job,
    )

    trace_ref = job_path(job, "traces", "run-trace.jsonl")
    assert trace_ref in final_state.published_artifact_refs
    assert blob_store.exists(trace_ref)
    assert blob_store.read_bytes(trace_ref) == trace_path.read_bytes()


def test_phase_six_scoped_publish_paths_isolate_same_job_id_across_tenants(tmp_path: Path) -> None:
    shared_root = tmp_path / "shared"
    tenant_a = _job_context(job_id="job-shared", tenant_id="tenant-a")
    tenant_b = _job_context(job_id="job-shared", tenant_id="tenant-b")

    _, blob_store_a, _ = _run_workflow(
        shared_root,
        run_id="run-a",
        scenario="happy",
        job=tenant_a,
        blob_root=shared_root / "blobs",
    )
    _, blob_store_b, _ = _run_workflow(
        shared_root,
        run_id="run-b",
        scenario="happy",
        job=tenant_b,
        blob_root=shared_root / "blobs",
    )

    transcript_a = job_path(tenant_a, "published", "transcript.json")
    transcript_b = job_path(tenant_b, "published", "transcript.json")

    assert transcript_a != transcript_b
    assert blob_store_a.exists(transcript_a)
    assert blob_store_b.exists(transcript_b)


def test_phase_six_translation_conflict_timeout_escalates_to_human_review(
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-timeout")
    final_state, blob_store, _ = _run_workflow(
        tmp_path,
        run_id="run-timeout",
        scenario="translation_conflict_timeout",
        job=job,
    )

    decision = FinalTranslationDecision.model_validate_json(
        blob_store.read_bytes(job_path(job, "decisions", "translation.json"))
    )
    investigation = json.loads(
        blob_store.read_bytes(job_path(job, "investigations", "translation.json")).decode("utf-8")
    )

    assert final_state.human_review_required is True
    assert decision.decision_mode == "human_review"
    assert decision.disagreement_bucket == "unresolved"
    assert investigation["status"] == "timed_out"
    assert any(
        fact.fact_type == "investigation_timeout" and fact.value == "conflict_investigator"
        for fact in final_state.routing_facts
    )


def test_phase_six_missing_blob_fetch_emits_node_failed_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job_context(job_id="job-missing-blob")
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-missing-blob", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-missing-blob-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    trace_path = tmp_path / "missing-blob.jsonl"

    def fail_read_model_artifact(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("missing blob fetch")

    monkeypatch.setattr(
        "translation_agent.nodes.transcription.read_model_artifact",
        fail_read_model_artifact,
    )

    with JsonlTraceSink(trace_path) as trace_sink:
        runtime = build_phase_two_runtime(
            blob_store=blob_store,
            run_store=run_store,
            trace_sink=trace_sink,
            source_artifact_ref=source_ref,
            scenario="happy",
        )
        initial_state = GraphState(
            run_id="run-missing-blob",
            job=job,
            current_stage="ingest",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
        )
        with pytest.raises(FileNotFoundError, match="missing blob fetch"):
            run_workflow(initial_state, runtime)

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        record["name"] == "node.failed"
        and record["attributes"]["node_name"] == "fanout_transcription"
        for record in records
    )
