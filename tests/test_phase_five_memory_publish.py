from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.graph import (
    GraphState,
    WorkflowRuntime,
    build_phase_two_runtime,
    run_workflow,
)
from translation_agent.memory import (
    DeterministicMemoryConsolidationBackend,
    DeterministicPromptEvolutionBackend,
    InMemoryLongTermMemoryStore,
    LongTermMemoryRecallBackend,
)
from translation_agent.models import (
    JobContext,
    MemoryConsolidation,
    MemoryQuery,
    MemoryWrite,
    MemoryWriteBatch,
)
from translation_agent.observability import NoOpTraceSink
from translation_agent.storage import LocalBlobStore, NodeExecutionRecord, RunRecord

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


def _job_context(job_id: str = "job-phase-five") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
    )


def _run_workflow(
    tmp_path: Path,
    *,
    scenario: str,
) -> tuple[GraphState, WorkflowRuntime, LocalBlobStore]:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-phase-five", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-phase-five-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario=scenario,
    )
    initial_state = GraphState(
        run_id="run-phase-five",
        job=_job_context(),
        current_stage="ingest",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
    )
    final_state = run_workflow(initial_state, runtime)
    return final_state, runtime, blob_store


def test_phase_five_happy_path_publishes_audit_ready_outputs(tmp_path: Path) -> None:
    final_state, runtime, blob_store = _run_workflow(tmp_path, scenario="happy")

    assert blob_store.exists("published/job-phase-five/scorecard.json")
    assert blob_store.exists("exports/job-phase-five.txt")
    assert blob_store.exists("exports/job-phase-five.json")
    assert blob_store.exists("deliveries/job-phase-five.json")
    assert blob_store.exists(
        "memory/consolidations/consolidation-batch-translation_adjudication-job-phase-five.json"
    )
    assert blob_store.exists(
        "memory/prompt-evolution/"
        "prompt-evolution-consolidation-batch-translation_adjudication-job-phase-five.json"
    )

    scorecard = json.loads(
        blob_store.read_bytes("published/job-phase-five/scorecard.json").decode("utf-8")
    )
    prompt_proposal = json.loads(
        blob_store.read_bytes(
            "memory/prompt-evolution/"
            "prompt-evolution-consolidation-batch-translation_adjudication-job-phase-five.json"
        ).decode("utf-8")
    )

    assert scorecard["translation_decision"]["disagreement_bucket"] == "low"
    assert scorecard["translation_decision"]["adjudication_scorecard"]["candidate_count"] == 2
    assert scorecard["memory_consolidation_refs"]
    assert scorecard["prompt_evolution_refs"]
    assert prompt_proposal["target_model_id"] == runtime.translation_adapter.model_id
    assert prompt_proposal["auto_activate"] is True
    assert any("memory/prompt-evolution/" in ref for ref in final_state.published_artifact_refs)


def test_phase_five_translation_failure_publishes_recoverable_manifest(tmp_path: Path) -> None:
    _, _, blob_store = _run_workflow(tmp_path, scenario="translation_failed")

    assert blob_store.exists("published/job-phase-five/translation-failed.json")
    assert not blob_store.exists("published/job-phase-five/translation.json")

    scorecard = json.loads(
        blob_store.read_bytes("published/job-phase-five/scorecard.json").decode("utf-8")
    )
    assert scorecard["translation_failed"] is True
    assert (
        scorecard["translation_failure_ref"] == "published/job-phase-five/translation-failed.json"
    )


def test_phase_five_memory_consolidation_dedupes_and_scopes_recall() -> None:
    store = InMemoryLongTermMemoryStore()
    consolidation_backend = DeterministicMemoryConsolidationBackend(store)
    recall_backend = LongTermMemoryRecallBackend(store)
    batch_one = MemoryWriteBatch(
        batch_id="batch-1",
        job_id="job-1",
        source_stage="translation_adjudication",
        semantic_writes=(
            MemoryWrite(
                kind="semantic",
                content="Prefer workflow terminology for stable UI copy.",
                metadata={"dedupe_key": "semantic:workflow"},
            ),
        ),
        episodic_writes=(
            MemoryWrite(
                kind="episodic",
                content="Low-disagreement translation finalized automatically.",
            ),
        ),
        metadata={
            "tenant_id": "tenant-a",
            "project_id": "project-1",
            "source_language": "en",
            "target_language": "fr",
        },
    )
    batch_duplicate = batch_one.model_copy(update={"batch_id": "batch-2"})
    batch_other_project = batch_one.model_copy(
        update={
            "batch_id": "batch-3",
            "metadata": {
                "tenant_id": "tenant-a",
                "project_id": "project-2",
                "source_language": "en",
                "target_language": "fr",
            },
            "semantic_writes": (
                MemoryWrite(
                    kind="semantic",
                    content="Do not bleed into the first project's recall scope.",
                    metadata={"dedupe_key": "semantic:other-project"},
                ),
            ),
        }
    )

    consolidation_one = consolidation_backend.consolidate_batch(batch_one)
    consolidation_duplicate = consolidation_backend.consolidate_batch(batch_duplicate)
    consolidation_backend.consolidate_batch(batch_other_project)
    recalled = recall_backend.recall_memory(
        MemoryQuery(
            job=_job_context(job_id="job-recall").model_copy(
                update={"tenant_id": "tenant-a", "project_id": "project-1"}
            ),
            stage="review_translations",
            query_text="workflow terminology",
        )
    )

    assert consolidation_one.semantic_memory_ids == ("semantic:batch-1:1",)
    assert consolidation_duplicate.semantic_memory_ids == ()
    assert "semantic:workflow" in consolidation_duplicate.skipped_dedupe_keys
    assert any(
        entry.content == "Prefer workflow terminology for stable UI copy."
        for entry in recalled.semantic_memory
    )
    assert all(
        entry.content != "Do not bleed into the first project's recall scope."
        for entry in recalled.semantic_memory
    )


def test_phase_five_prompt_evolution_uses_runtime_model_selection() -> None:
    backend = DeterministicPromptEvolutionBackend()
    proposal = backend.propose_prompt_evolution(
        consolidation=MemoryConsolidation(
            consolidation_id="consolidation-batch-translation",
            batch_id="batch-translation",
            job_id="job-1",
            source_stage="translation_adjudication",
            source_disagreement_bucket="low",
            source_prompt_variant_id="variant-b",
            source_prompt_version="v2",
            procedural_write_count=1,
        ),
        translation_model_id="openai-generic-model",
        evidence_ref="memory/consolidations/consolidation-batch-translation.json",
    )

    assert proposal is not None
    assert proposal.target_model_id == "openai-generic-model"
    assert proposal.target_prompt_variant_id == "variant-b"
    assert proposal.auto_activate is True
