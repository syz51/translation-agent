"""Background memory staging and post-finalization draining helpers."""

from __future__ import annotations

from dataclasses import dataclass

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.memory.staging import apply_scope_defaults, batch_metadata_for_job
from translation_agent.models import (
    JobContext,
    MemoryConsolidation,
    MemoryWriteBatch,
    PromptEvolutionProposal,
)
from translation_agent.nodes.common import (
    memory_batch_key,
    memory_consolidation_key,
    prompt_evolution_key,
    write_model_artifact,
)
from translation_agent.observability import TraceEvent
from translation_agent.parallelism import (
    ParallelTaskClass,
    concurrency_trace_attributes,
    ordered_parallel_map,
)
from translation_agent.publish.outputs import publish_outputs
from translation_agent.storage import asset_path, job_scope_token, operational_job_key


@dataclass(frozen=True, slots=True)
class _MemoryDrainComputed:
    consolidation: MemoryConsolidation
    proposal: PromptEvolutionProposal | None


def background_memory_pipeline(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Stage memory batches without blocking finalization on consolidation work."""

    if state.pending_memory_source_stage is None:
        return {"current_stage": "background_memory_pipeline"}

    storage_job_id = operational_job_key(state.job)
    if state.pending_memory_source_stage == "transcript_adjudication":
        decision = runtime.decision_store.get_transcript_decision(
            state.job.job_id,
            storage_job_id=storage_job_id,
        )
        decision_ref = state.final_transcript_decision_ref
    else:
        decision = runtime.decision_store.get_translation_decision(
            state.job.job_id,
            storage_job_id=storage_job_id,
        )
        decision_ref = state.final_translation_decision_ref

    if decision is None:
        raise RuntimeError(f"missing decision for {state.pending_memory_source_stage}")

    staged_batch = runtime.memory_staging_backend.stage_memory_candidates(
        decision,
        source_stage=state.pending_memory_source_stage,
    )
    batch = _scoped_batch(state, staged_batch, decision_ref=decision_ref)
    runtime.memory_batch_store.save_batch(batch, storage_job_id=storage_job_id)
    batch_ref = write_model_artifact(runtime, memory_batch_key(state.job, batch.batch_id), batch)

    return {
        "current_stage": "background_memory_pipeline",
        "pending_memory_source_stage": None,
        "memory_batch_ids": state.memory_batch_ids + (batch.batch_id,),
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="background_memory_pipeline",
                fact_type="memory_batch_staged",
                value=batch.batch_id,
                source_ref=batch_ref,
            ),
        ),
    }


def drain_background_memory(state: GraphState, runtime: WorkflowRuntime) -> GraphState:
    """Best-effort consolidation pass that runs after finalization has completed."""

    if not state.memory_batch_ids:
        return state

    storage_job_id = operational_job_key(state.job)
    routing_facts = list(state.routing_facts)
    pending_batches = tuple(
        batch
        for batch_id in state.memory_batch_ids
        if (batch := runtime.memory_batch_store.get_batch(batch_id)) is not None
        and batch.consolidation_status != "consolidated"
    )
    if not pending_batches:
        return state

    parallel_safe = _memory_drain_parallel_safe(runtime)
    effective_stage_workers = (
        runtime.parallelism.resolve_stage_workers(
            runtime.parallelism.memory_drain_max_workers,
            task_count=len(pending_batches),
        )
        if parallel_safe and len(pending_batches) > 1
        else 1
    )
    gathered = ordered_parallel_map(
        pending_batches,
        max_workers=effective_stage_workers,
        worker=lambda batch: _compute_memory_drain(
            batch,
            runtime,
            job=state.job,
            run_id=state.run_id,
            parallel_safe=parallel_safe,
            effective_stage_workers=effective_stage_workers,
        ),
        sort_key=lambda input_index, _batch: (input_index,),
    )

    for batch, result in zip(pending_batches, gathered, strict=True):
        try:
            if result.error is not None:
                raise result.error
            if result.value is None:  # pragma: no cover - defensive
                raise RuntimeError("missing memory drain result")

            updated_batch = batch.model_copy(update={"consolidation_status": "consolidated"})
            runtime.memory_batch_store.save_batch(updated_batch, storage_job_id=storage_job_id)
            write_model_artifact(
                runtime,
                memory_batch_key(state.job, updated_batch.batch_id),
                updated_batch,
            )
            consolidation_ref = write_model_artifact(
                runtime,
                memory_consolidation_key(state.job, result.value.consolidation.consolidation_id),
                result.value.consolidation,
            )
            routing_facts.append(
                RoutingFact(
                    stage="background_memory_pipeline",
                    fact_type="memory_batch_consolidated",
                    value=result.value.consolidation.consolidation_id,
                    source_ref=consolidation_ref,
                )
            )

            proposal = result.value.proposal
            if proposal is not None:
                proposal_ref = prompt_evolution_key(state.job, proposal.proposal_id)
                asset_proposal_ref = asset_path(
                    state.job.media_key,
                    "improvement-proposals",
                    f"{proposal.proposal_id}.json",
                )
                proposal = proposal.model_copy(
                    update={
                        "metadata": {
                            **proposal.metadata,
                            "media_key": state.job.media_key,
                            "source_language": state.job.source_language,
                            "target_language": state.job.target_language,
                            "proposal_ref": proposal_ref,
                            "asset_proposal_ref": asset_proposal_ref,
                        }
                    }
                )
                proposal_ref = write_model_artifact(runtime, proposal_ref, proposal)
                write_model_artifact(runtime, asset_proposal_ref, proposal)
                save_proposal = getattr(runtime.run_store, "save_prompt_evolution_proposal", None)
                if callable(save_proposal):
                    save_proposal(proposal)
                routing_facts.append(
                    RoutingFact(
                        stage="background_memory_pipeline",
                        fact_type="translation_prompt_evolution",
                        value=proposal.proposal_id,
                        source_ref=proposal_ref,
                    )
                )
        except Exception as exc:
            failed_batch = batch.model_copy(update={"consolidation_status": "failed"})
            runtime.memory_batch_store.save_batch(failed_batch, storage_job_id=storage_job_id)
            batch_ref = write_model_artifact(
                runtime,
                memory_batch_key(state.job, failed_batch.batch_id),
                failed_batch,
            )
            routing_facts.append(
                RoutingFact(
                    stage="background_memory_pipeline",
                    fact_type="memory_batch_failed",
                    value=failed_batch.batch_id,
                    source_ref=str(exc) or batch_ref,
                )
            )

    refreshed_state = state.model_copy(update={"routing_facts": tuple(routing_facts)})
    artifacts, manifest_ref = publish_outputs(refreshed_state, runtime)
    published_refs = tuple(
        ref
        for ref in (
            manifest_ref,
            artifacts.final_transcript_ref,
            artifacts.final_translation_ref,
            artifacts.recoverable_translation_failure_ref,
            *artifacts.reference_transcript_refs,
            *artifacts.evaluation_report_refs,
            *artifacts.regenerated_draft_refs,
            *artifacts.improvement_proposal_refs,
            *artifacts.scorecard_refs,
            *artifacts.trace_refs,
            *artifacts.export_refs,
            *artifacts.downstream_delivery_refs,
            *artifacts.memory_batch_refs,
            *artifacts.memory_consolidation_refs,
            *artifacts.prompt_evolution_refs,
        )
        if ref is not None
    )
    return refreshed_state.model_copy(update={"published_artifact_refs": published_refs})


def _scoped_batch(
    state: GraphState,
    batch: MemoryWriteBatch,
    *,
    decision_ref: str | None,
) -> MemoryWriteBatch:
    scope_key = operational_job_key(state.job)
    scope_token = job_scope_token(state.job)
    return batch.model_copy(
        update={
            "batch_id": f"{batch.batch_id}-{scope_token}",
            "decision_ref": decision_ref,
            "semantic_writes": tuple(
                apply_scope_defaults(write, job=state.job) for write in batch.semantic_writes
            ),
            "episodic_writes": tuple(
                apply_scope_defaults(write, job=state.job) for write in batch.episodic_writes
            ),
            "procedural_writes": tuple(
                apply_scope_defaults(write, job=state.job) for write in batch.procedural_writes
            ),
            "metadata": batch_metadata_for_job(
                state.job,
                **batch.metadata,
                job_scope_key=scope_key,
            ),
        }
    )


def _translation_model_id(batch: MemoryWriteBatch, runtime: WorkflowRuntime) -> str | None:
    if batch.source_stage != "translation_adjudication":
        return None
    return batch.translation_model_winner or getattr(runtime.translation_adapter, "model_id", None)


def _memory_drain_parallel_safe(runtime: WorkflowRuntime) -> bool:
    consolidation_backend = runtime.memory_consolidation_backend
    # Current deterministic consolidation writes through a shared store internally,
    # so keep draining serial unless a backend explicitly opts into safe parallel compute.
    supports_parallel_compute = bool(
        getattr(consolidation_backend, "supports_parallel_compute", False)
        or getattr(consolidation_backend, "supports_parallel_compute_only", False)
    )
    if hasattr(consolidation_backend, "_store"):
        return False
    return supports_parallel_compute


def _compute_memory_drain(
    batch: MemoryWriteBatch,
    runtime: WorkflowRuntime,
    *,
    job: JobContext,
    run_id: str,
    parallel_safe: bool,
    effective_stage_workers: int,
) -> _MemoryDrainComputed:
    trace_attributes = {
        "memory_batch_id": batch.batch_id,
        "effective_stage_workers": effective_stage_workers,
    }
    if not parallel_safe:
        consolidation = runtime.memory_consolidation_backend.consolidate_batch(batch)
        proposal = runtime.prompt_evolution_backend.propose_prompt_evolution(
            consolidation,
            translation_model_id=_translation_model_id(batch, runtime),
            evidence_ref=memory_consolidation_key(job, consolidation.consolidation_id),
        )
        return _MemoryDrainComputed(
            consolidation=consolidation,
            proposal=proposal,
        )

    with runtime.global_concurrency_limiter.acquire(
        runtime.parallelism.token_cost(ParallelTaskClass.LOCAL_COMPUTE),
        task_class=ParallelTaskClass.LOCAL_COMPUTE,
    ) as acquisition:
        trace_attributes = {
            **trace_attributes,
            **concurrency_trace_attributes(
                acquisition,
                effective_stage_workers=effective_stage_workers,
            ),
        }
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="memory.drain.started",
                attributes=trace_attributes,
            )
        )
        consolidation = runtime.memory_consolidation_backend.consolidate_batch(batch)
        proposal = runtime.prompt_evolution_backend.propose_prompt_evolution(
            consolidation,
            translation_model_id=_translation_model_id(batch, runtime),
            evidence_ref=memory_consolidation_key(job, consolidation.consolidation_id),
        )
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="memory.drain.completed",
                attributes={
                    **trace_attributes,
                    "consolidation_id": consolidation.consolidation_id,
                },
            )
        )
        return _MemoryDrainComputed(
            consolidation=consolidation,
            proposal=proposal,
        )
