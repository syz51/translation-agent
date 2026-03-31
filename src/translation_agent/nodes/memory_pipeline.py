"""Background memory staging and post-finalization draining helpers."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import MemoryWriteBatch
from translation_agent.nodes.common import (
    memory_batch_key,
    memory_consolidation_key,
    prompt_evolution_key,
    write_model_artifact,
)
from translation_agent.publish.outputs import publish_outputs
from translation_agent.storage import job_scope_token, operational_job_key


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
    for batch_id in state.memory_batch_ids:
        batch = runtime.memory_batch_store.get_batch(batch_id)
        if batch is None or batch.consolidation_status == "consolidated":
            continue
        try:
            consolidation = runtime.memory_consolidation_backend.consolidate_batch(batch)
            updated_batch = batch.model_copy(update={"consolidation_status": "consolidated"})
            runtime.memory_batch_store.save_batch(updated_batch, storage_job_id=storage_job_id)
            write_model_artifact(
                runtime,
                memory_batch_key(state.job, updated_batch.batch_id),
                updated_batch,
            )
            consolidation_ref = write_model_artifact(
                runtime,
                memory_consolidation_key(state.job, consolidation.consolidation_id),
                consolidation,
            )
            routing_facts.append(
                RoutingFact(
                    stage="background_memory_pipeline",
                    fact_type="memory_batch_consolidated",
                    value=consolidation.consolidation_id,
                    source_ref=consolidation_ref,
                )
            )
            proposal = runtime.prompt_evolution_backend.propose_prompt_evolution(
                consolidation,
                translation_model_id=_translation_model_id(updated_batch, runtime),
                evidence_ref=consolidation_ref,
            )
            if proposal is not None:
                proposal_ref = write_model_artifact(
                    runtime,
                    prompt_evolution_key(state.job, proposal.proposal_id),
                    proposal,
                )
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
            "dedupe_keys": tuple(f"{scope_key}::{key}" for key in batch.dedupe_keys),
            "metadata": {
                **batch.metadata,
                "tenant_id": state.job.tenant_id,
                "project_id": state.job.project_id,
                "source_language": state.job.source_language,
                "target_language": state.job.target_language,
                "job_scope_key": scope_key,
            },
        }
    )


def _translation_model_id(batch: MemoryWriteBatch, runtime: WorkflowRuntime) -> str | None:
    if batch.source_stage != "translation_adjudication":
        return None
    return batch.translation_model_winner or getattr(runtime.translation_adapter, "model_id", None)
