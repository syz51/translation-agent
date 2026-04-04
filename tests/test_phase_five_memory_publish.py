from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pysubs2
import pytest

from translation_agent.graph import (
    GraphState,
    WorkflowRuntime,
    build_phase_two_runtime,
    run_workflow,
)
from translation_agent.graph.state import RoutingFact
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
from translation_agent.nodes.memory_pipeline import drain_background_memory
from translation_agent.observability import NoOpTraceSink
from translation_agent.parallelism import GlobalConcurrencyLimiter, RuntimeParallelismPolicy
from translation_agent.publish.outputs import publish_outputs
from translation_agent.storage import (
    LocalBlobStore,
    NodeExecutionRecord,
    RunRecord,
    job_path,
    job_scope_token,
    operational_job_key,
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
        media_key=f"source-ref:{job_id}",
    )


def _artifact_path(*parts: str) -> str:
    return job_path(_job_context(), *parts)


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
    scope_token = job_scope_token(_job_context())
    consolidation_id = f"consolidation-batch-translation_adjudication-job-phase-five-{scope_token}"

    assert blob_store.exists(_artifact_path("published", "scorecard.json"))
    assert blob_store.exists(_artifact_path("exports", "translation.srt"))
    assert blob_store.exists(_artifact_path("exports", "translation.json"))
    assert not blob_store.exists(_artifact_path("exports", "translation.txt"))
    assert blob_store.exists(_artifact_path("deliveries", "translation.json"))
    assert blob_store.exists(
        _artifact_path(
            "memory",
            "consolidations",
            f"{consolidation_id}.json",
        )
    )

    scorecard = json.loads(
        blob_store.read_bytes(_artifact_path("published", "scorecard.json")).decode("utf-8")
    )
    subtitles = pysubs2.SSAFile.from_string(
        blob_store.read_bytes(_artifact_path("exports", "translation.srt")).decode("utf-8"),
        format_="srt",
    )
    assert scorecard["translation_decision"]["disagreement_bucket"] == "low"
    assert scorecard["translation_decision"]["adjudication_scorecard"]["candidate_count"] == 1
    assert scorecard["export_refs"] == [
        _artifact_path("exports", "translation.srt"),
        _artifact_path("exports", "translation.json"),
    ]
    assert scorecard["memory_consolidation_refs"]
    assert len(scorecard["prompt_evolution_refs"]) == 1
    assert "project-pair-consolidation" in scorecard["prompt_evolution_refs"][0]
    assert len(subtitles.events) == 1
    assert subtitles.events[0].text == "Bonjour tout le monde depuis le workflow."
    assert "speaker-" not in subtitles.events[0].text
    assert any("/memory/prompt-evolution/" in ref for ref in final_state.published_artifact_refs)


def test_phase_five_memory_drain_preserves_batch_order_in_routing_and_manifest(
    tmp_path: Path,
) -> None:
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="happy")

    consolidated_ids = [
        fact.value
        for fact in final_state.routing_facts
        if fact.fact_type == "memory_batch_consolidated"
    ]
    manifest = json.loads(
        blob_store.read_bytes(_artifact_path("published", "artifacts.json")).decode("utf-8")
    )

    assert consolidated_ids == [
        f"consolidation-{batch_id}" for batch_id in final_state.memory_batch_ids
    ]
    assert manifest["memory_batch_refs"] == [
        _artifact_path("memory", "batches", f"{batch_id}.json")
        for batch_id in final_state.memory_batch_ids
    ]
    assert manifest["memory_consolidation_refs"] == [
        _artifact_path("memory", "consolidations", f"consolidation-{batch_id}.json")
        for batch_id in final_state.memory_batch_ids
    ]


def test_memory_drain_stays_serial_when_backend_is_not_parallel_safe(tmp_path: Path) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-memory-unsafe", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-memory-unsafe-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )
    runtime.parallelism = RuntimeParallelismPolicy(
        global_max_parallel_tokens=4,
        provider_io_token_cost=2,
        local_compute_token_cost=1,
        transcription_max_workers=None,
        translation_candidate_max_workers=None,
        translation_chunk_max_workers=None,
        review_max_workers=None,
        reference_evaluation_max_workers=None,
        memory_drain_max_workers=4,
    )
    runtime.global_concurrency_limiter = GlobalConcurrencyLimiter(4)
    job = _job_context(job_id="job-memory-unsafe")
    storage_job_id = operational_job_key(job)

    active = 0
    max_active = 0
    lock = threading.Lock()

    class UnsafeBackend:
        _store = object()

        def consolidate_batch(self, batch: MemoryWriteBatch) -> MemoryConsolidation:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                return MemoryConsolidation(
                    consolidation_id=f"consolidation-{batch.batch_id}",
                    batch_id=batch.batch_id,
                    job_id=batch.job_id,
                    source_stage=batch.source_stage,
                )
            finally:
                with lock:
                    active -= 1

    class NoProposalBackend:
        def propose_prompt_evolution(self, consolidation, **kwargs):  # noqa: ANN001
            del consolidation, kwargs
            return None

    runtime.memory_consolidation_backend = UnsafeBackend()
    runtime.prompt_evolution_backend = NoProposalBackend()

    batches = (
        MemoryWriteBatch(batch_id="batch-a", job_id=job.job_id, source_stage="translation"),
        MemoryWriteBatch(batch_id="batch-b", job_id=job.job_id, source_stage="translation"),
    )
    for batch in batches:
        runtime.memory_batch_store.save_batch(batch, storage_job_id=storage_job_id)

    drain_background_memory(
        GraphState(
            run_id="run-memory-unsafe",
            job=job,
            current_stage="background_memory_pipeline",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
            memory_batch_ids=tuple(batch.batch_id for batch in batches),
        ),
        runtime,
    )

    assert max_active == 1


def test_parallel_safe_memory_drain_respects_global_limiter(tmp_path: Path) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-memory-safe", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-memory-safe-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )
    runtime.parallelism = RuntimeParallelismPolicy(
        global_max_parallel_tokens=1,
        provider_io_token_cost=2,
        local_compute_token_cost=1,
        transcription_max_workers=None,
        translation_candidate_max_workers=None,
        translation_chunk_max_workers=None,
        review_max_workers=None,
        reference_evaluation_max_workers=None,
        memory_drain_max_workers=4,
    )
    runtime.global_concurrency_limiter = GlobalConcurrencyLimiter(1)
    job = _job_context(job_id="job-memory-safe")
    storage_job_id = operational_job_key(job)

    active = 0
    max_active = 0
    lock = threading.Lock()

    class ParallelSafeBackend:
        supports_parallel_compute = True

        def consolidate_batch(self, batch: MemoryWriteBatch) -> MemoryConsolidation:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                return MemoryConsolidation(
                    consolidation_id=f"consolidation-{batch.batch_id}",
                    batch_id=batch.batch_id,
                    job_id=batch.job_id,
                    source_stage=batch.source_stage,
                )
            finally:
                with lock:
                    active -= 1

    class NoProposalBackend:
        def propose_prompt_evolution(self, consolidation, **kwargs):  # noqa: ANN001
            del consolidation, kwargs
            return None

    runtime.memory_consolidation_backend = ParallelSafeBackend()
    runtime.prompt_evolution_backend = NoProposalBackend()

    batches = (
        MemoryWriteBatch(batch_id="batch-a", job_id=job.job_id, source_stage="translation"),
        MemoryWriteBatch(batch_id="batch-b", job_id=job.job_id, source_stage="translation"),
    )
    for batch in batches:
        runtime.memory_batch_store.save_batch(batch, storage_job_id=storage_job_id)

    drain_background_memory(
        GraphState(
            run_id="run-memory-safe",
            job=job,
            current_stage="background_memory_pipeline",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
            memory_batch_ids=tuple(batch.batch_id for batch in batches),
        ),
        runtime,
    )

    assert max_active == 1


def test_phase_five_translation_failure_publishes_recoverable_manifest(tmp_path: Path) -> None:
    _, _, blob_store = _run_workflow(tmp_path, scenario="translation_failed")

    assert blob_store.exists(_artifact_path("published", "translation-failed.json"))
    assert not blob_store.exists(_artifact_path("published", "translation.json"))

    failure_manifest = json.loads(
        blob_store.read_bytes(_artifact_path("published", "translation-failed.json")).decode(
            "utf-8"
        )
    )
    scorecard = json.loads(
        blob_store.read_bytes(_artifact_path("published", "scorecard.json")).decode("utf-8")
    )
    export_payload = json.loads(
        blob_store.read_bytes(_artifact_path("exports", "translation.json")).decode("utf-8")
    )
    delivery_payload = json.loads(
        blob_store.read_bytes(_artifact_path("deliveries", "translation.json")).decode("utf-8")
    )
    assert failure_manifest["failure_summary"] == (
        "All translation variants failed; transcript preserved for recovery."
    )
    assert failure_manifest["failure_reasons"] == [
        "variant-a: simulated translation failure for variant-a",
    ]
    assert scorecard["translation_failed"] is True
    assert export_payload["status"] == "translation_failed"
    assert delivery_payload["status"] == "translation_failed"


def test_phase_five_translation_failure_manifest_dedupes_duplicate_reasons(tmp_path: Path) -> None:
    final_state, runtime, blob_store = _run_workflow(tmp_path, scenario="translation_failed")
    duplicate_reason = RoutingFact(
        stage="translate",
        fact_type="translation_variant_failed",
        value="variant-a",
        source_ref="simulated translation failure for variant-a",
    )
    deduped_state = final_state.model_copy(
        update={"routing_facts": (*final_state.routing_facts, duplicate_reason)}
    )

    runtime.run_store.update_run(deduped_state.run_id, status="running")
    publish_outputs(deduped_state, runtime)

    failure_manifest = json.loads(
        blob_store.read_bytes(_artifact_path("published", "translation-failed.json")).decode(
            "utf-8"
        )
    )
    assert failure_manifest["failure_reasons"] == [
        "variant-a: simulated translation failure for variant-a",
    ]


def test_phase_five_background_memory_failures_do_not_block_finalization(tmp_path: Path) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-phase-five-failing-memory", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-phase-five-failing-memory-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )

    class FailingConsolidationBackend:
        def consolidate_batch(self, batch):  # noqa: ANN001
            raise RuntimeError(f"boom:{batch.batch_id}")

    runtime.memory_consolidation_backend = FailingConsolidationBackend()
    final_state = run_workflow(
        GraphState(
            run_id="run-phase-five-failing-memory",
            job=_job_context(job_id="job-phase-five-failing-memory"),
            current_stage="ingest",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
        ),
        runtime,
    )

    assert final_state.current_stage == "finalize_outputs"
    failed_job = _job_context(job_id="job-phase-five-failing-memory")
    assert blob_store.exists(job_path(failed_job, "published", "translation.json"))


def test_phase_five_multi_batch_drain_keeps_routing_and_manifest_order(
    tmp_path: Path,
) -> None:
    final_state, runtime, blob_store = _run_workflow(tmp_path, scenario="happy")
    scope_token = job_scope_token(_job_context())
    extra_batches = (
        MemoryWriteBatch(
            batch_id=f"batch-extra-a-{scope_token}",
            job_id=_job_context().job_id,
            source_stage="translation_adjudication",
            semantic_writes=(
                MemoryWrite(
                    kind="semantic",
                    content="Batch A memory.",
                    scope_kind="pair",
                    scope_key="en::fr",
                ),
            ),
            metadata={
                "tenant_id": _job_context().tenant_id,
                "project_id": _job_context().project_id,
                "source_language": _job_context().source_language,
                "target_language": _job_context().target_language,
            },
        ),
        MemoryWriteBatch(
            batch_id=f"batch-extra-b-{scope_token}",
            job_id=_job_context().job_id,
            source_stage="translation_adjudication",
            semantic_writes=(
                MemoryWrite(
                    kind="semantic",
                    content="Batch B memory.",
                    scope_kind="pair",
                    scope_key="en::fr",
                ),
            ),
            metadata={
                "tenant_id": _job_context().tenant_id,
                "project_id": _job_context().project_id,
                "source_language": _job_context().source_language,
                "target_language": _job_context().target_language,
            },
        ),
    )
    for batch in extra_batches:
        runtime.memory_batch_store.save_batch(
            batch,
            storage_job_id=operational_job_key(_job_context()),
        )
        blob_store.put_bytes(
            job_path(_job_context(), "memory", "batches", f"{batch.batch_id}.json"),
            (json.dumps(batch.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )

    drained = drain_background_memory(
        final_state.model_copy(
            update={
                "memory_batch_ids": (
                    *final_state.memory_batch_ids,
                    *(batch.batch_id for batch in extra_batches),
                )
            }
        ),
        runtime,
    )

    consolidated_values = [
        fact.value
        for fact in drained.routing_facts
        if fact.stage == "background_memory_pipeline"
        and fact.fact_type == "memory_batch_consolidated"
        and "batch-extra" in fact.value
    ]
    manifest = json.loads(
        blob_store.read_bytes(job_path(_job_context(), "published", "artifacts.json")).decode(
            "utf-8"
        )
    )
    assert consolidated_values == [
        f"consolidation-{extra_batches[0].batch_id}",
        f"consolidation-{extra_batches[1].batch_id}",
    ]
    extra_consolidation_refs = [
        ref for ref in manifest["memory_consolidation_refs"] if "batch-extra" in ref
    ]
    assert extra_consolidation_refs == [
        job_path(
            _job_context(),
            "memory",
            "consolidations",
            f"consolidation-{extra_batches[0].batch_id}.json",
        ),
        job_path(
            _job_context(),
            "memory",
            "consolidations",
            f"consolidation-{extra_batches[1].batch_id}.json",
        ),
    ]


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
                scope_kind="project_pair",
                scope_key="tenant-a::project-1::en::fr",
                metadata={"dedupe_key": "semantic:workflow"},
            ),
        ),
        episodic_writes=(
            MemoryWrite(
                kind="episodic",
                content="Low-disagreement translation finalized automatically.",
                scope_kind="project_pair",
                scope_key="tenant-a::project-1::en::fr",
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
                    scope_kind="project_pair",
                    scope_key="tenant-a::project-2::en::fr",
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

    assert proposal is None


def test_phase_five_recall_returns_procedural_memory_for_generation_context() -> None:
    store = InMemoryLongTermMemoryStore()
    consolidation_backend = DeterministicMemoryConsolidationBackend(store)
    recall_backend = LongTermMemoryRecallBackend(store)
    consolidation_backend.consolidate_batch(
        MemoryWriteBatch(
            batch_id="batch-guidance",
            job_id="job-guidance",
            source_stage="translation_human_resolution",
            procedural_writes=(
                MemoryWrite(
                    kind="procedural",
                    content="Reject mixed-script junk and unresolved transliterations.",
                    scope_kind="pair",
                    scope_key="en::fr",
                    metadata={
                        "dedupe_key": "procedural:subtitle-gibberish",
                        "failure_tags": ["subtitle_gibberish"],
                        "prompt_variant_id": "variant-a",
                        "model_id": "gpt-5.4-mini",
                        "transcript_provider_id": "assemblyai",
                    },
                ),
            ),
            metadata={
                "tenant_id": "tenant-a",
                "project_id": "project-1",
                "source_language": "en",
                "target_language": "fr",
            },
        )
    )

    recalled = recall_backend.recall_memory(
        MemoryQuery(
            job=_job_context(job_id="job-generation"),
            stage="generate_translation_guidance",
            query_text="generate_translation_guidance | en->fr | providers:assemblyai",
            provider_ids=("assemblyai",),
            prompt_variant_ids=("variant-a",),
            model_ids=("gpt-5.4-mini",),
        )
    )

    assert recalled.rules == ()
    assert [entry.content for entry in recalled.procedural_memory] == [
        "Reject mixed-script junk and unresolved transliterations."
    ]


@pytest.mark.parametrize("bucket", ["medium", "high"])
def test_phase_five_prompt_evolution_keeps_higher_disagreement_gated(bucket: str) -> None:
    backend = DeterministicPromptEvolutionBackend()

    proposal = backend.propose_prompt_evolution(
        consolidation=MemoryConsolidation(
            consolidation_id=f"consolidation-{bucket}",
            batch_id=f"batch-{bucket}",
            job_id="job-1",
            source_stage="translation_adjudication",
            source_disagreement_bucket=bucket,
            source_translation_model_id="persisted-model",
            source_prompt_variant_id="variant-a",
            source_prompt_version="v3",
            procedural_write_count=1,
        ),
        translation_model_id=None,
        evidence_ref="memory/consolidations/example.json",
    )

    assert proposal is None


def test_phase_five_prompt_evolution_returns_none_without_required_translation_inputs() -> None:
    backend = DeterministicPromptEvolutionBackend()

    missing_variant = backend.propose_prompt_evolution(
        consolidation=MemoryConsolidation(
            consolidation_id="consolidation-missing-variant",
            batch_id="batch-missing-variant",
            job_id="job-1",
            source_stage="translation_adjudication",
            source_disagreement_bucket="low",
            source_translation_model_id="persisted-model",
            source_prompt_variant_id=None,
            procedural_write_count=1,
        ),
        translation_model_id=None,
        evidence_ref="memory/consolidations/example.json",
    )
    non_translation_stage = backend.propose_prompt_evolution(
        consolidation=MemoryConsolidation(
            consolidation_id="consolidation-transcript",
            batch_id="batch-transcript",
            job_id="job-1",
            source_stage="transcript_adjudication",
            source_disagreement_bucket="low",
            source_translation_model_id="persisted-model",
            source_prompt_variant_id="variant-a",
            procedural_write_count=0,
        ),
        translation_model_id="runtime-model",
        evidence_ref="memory/consolidations/example.json",
    )

    assert missing_variant is None
    assert non_translation_stage is None
