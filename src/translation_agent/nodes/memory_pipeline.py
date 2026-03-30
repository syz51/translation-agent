"""Background memory staging node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.common import (
    memory_batch_key,
    memory_consolidation_key,
    prompt_evolution_key,
    select_translation_candidates,
    write_model_artifact,
)


def background_memory_pipeline(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Stage candidate memory writes after each adjudication boundary."""

    if state.pending_memory_source_stage is None:
        return {"current_stage": "background_memory_pipeline"}

    if state.pending_memory_source_stage == "transcript_adjudication":
        decision = runtime.decision_store.get_transcript_decision(state.job.job_id)
        decision_ref = state.final_transcript_decision_ref
        winner_candidate_id = state.final_transcript_candidate_id
    else:
        decision = runtime.decision_store.get_translation_decision(state.job.job_id)
        decision_ref = state.final_translation_decision_ref
        winner_candidate_id = state.final_translation_candidate_id

    if decision is None:
        raise RuntimeError(f"missing decision for {state.pending_memory_source_stage}")

    staged_batch = runtime.memory_staging_backend.stage_memory_candidates(
        decision,
        source_stage=state.pending_memory_source_stage,
    )
    batch = staged_batch.model_copy(
        update={
            "decision_ref": decision_ref,
            "metadata": {
                **staged_batch.metadata,
                "tenant_id": state.job.tenant_id,
                "project_id": state.job.project_id,
                "source_language": state.job.source_language,
                "target_language": state.job.target_language,
            },
        }
    )
    consolidation = runtime.memory_consolidation_backend.consolidate_batch(batch)
    batch = batch.model_copy(update={"consolidation_status": "consolidated"})
    runtime.memory_batch_store.save_batch(batch)
    batch_ref = write_model_artifact(runtime, memory_batch_key(state.job, batch.batch_id), batch)
    consolidation_ref = write_model_artifact(
        runtime,
        memory_consolidation_key(state.job, consolidation.consolidation_id),
        consolidation,
    )
    proposal = runtime.prompt_evolution_backend.propose_prompt_evolution(
        consolidation,
        translation_model_id=_translation_model_id(
            state,
            runtime,
            winner_candidate_id=winner_candidate_id,
        ),
        evidence_ref=consolidation_ref,
    )
    proposal_ref = None
    if proposal is not None:
        proposal_ref = write_model_artifact(
            runtime,
            prompt_evolution_key(state.job, proposal.proposal_id),
            proposal,
        )

    routing_facts = [
        RoutingFact(
            stage="background_memory_pipeline",
            fact_type="memory_batch_staged",
            value=batch.batch_id,
            source_ref=batch_ref,
        ),
        RoutingFact(
            stage="background_memory_pipeline",
            fact_type="memory_batch_consolidated",
            value=consolidation.consolidation_id,
            source_ref=consolidation_ref,
        ),
    ]
    if proposal is not None and proposal_ref is not None:
        routing_facts.append(
            RoutingFact(
                stage="background_memory_pipeline",
                fact_type="translation_prompt_evolution",
                value=proposal.proposal_id,
                source_ref=proposal_ref,
            )
        )
    return {
        "current_stage": "background_memory_pipeline",
        "pending_memory_source_stage": None,
        "memory_batch_ids": state.memory_batch_ids + (batch.batch_id,),
        "routing_facts": state.routing_facts + tuple(routing_facts),
    }


def _translation_model_id(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    winner_candidate_id: str | None,
) -> str | None:
    if state.pending_memory_source_stage != "translation_adjudication":
        return None
    if winner_candidate_id is not None:
        candidates = select_translation_candidates(
            runtime,
            job_id=state.job.job_id,
            candidate_ids=(winner_candidate_id,),
        )
        if candidates:
            return candidates[0].model_id
    return getattr(runtime.translation_adapter, "model_id", None)
