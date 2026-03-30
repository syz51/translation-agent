"""Background memory staging node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.common import memory_batch_key, write_model_artifact


def background_memory_pipeline(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Stage candidate memory writes after each adjudication boundary."""

    if state.pending_memory_source_stage is None:
        return {"current_stage": "background_memory_pipeline"}

    if state.pending_memory_source_stage == "transcript_adjudication":
        decision = runtime.decision_store.get_transcript_decision(state.job.job_id)
    else:
        decision = runtime.decision_store.get_translation_decision(state.job.job_id)

    if decision is None:
        raise RuntimeError(f"missing decision for {state.pending_memory_source_stage}")

    batch = runtime.memory_staging_backend.stage_memory_candidates(
        decision,
        source_stage=state.pending_memory_source_stage,
    )
    runtime.memory_batch_store.save_batch(batch)
    batch_ref = write_model_artifact(runtime, memory_batch_key(batch.batch_id), batch)

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
