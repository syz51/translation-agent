"""Finalization node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.reference_evaluation import (
    run_reference_evaluation,
    update_historical_run_link,
)
from translation_agent.publish.outputs import publish_outputs


def finalize_outputs(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Publish canonical dry-run artifacts and expose their refs in graph state."""

    artifacts, manifest_ref = publish_outputs(state, runtime)
    update_historical_run_link(state, runtime, artifacts)
    resolved_state = state
    if state.job.reference_mode == "evaluate_and_regenerate":
        resolved_state = state.model_copy(update=run_reference_evaluation(state, runtime))
        artifacts, manifest_ref = publish_outputs(resolved_state, runtime)
        update_historical_run_link(resolved_state, runtime, artifacts)
    published_refs = tuple(
        ref
        for ref in (
            manifest_ref,
            artifacts.final_transcript_ref,
            artifacts.final_translation_ref,
            artifacts.recoverable_translation_failure_ref,
            *artifacts.approval_refs,
            *artifacts.learning_refs,
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

    return {
        "current_stage": "finalize_outputs",
        "reference_transcript_ref": resolved_state.reference_transcript_ref,
        "evaluation_report_ref": resolved_state.evaluation_report_ref,
        "regenerated_translation_draft_ref": resolved_state.regenerated_translation_draft_ref,
        "improvement_proposal_refs": resolved_state.improvement_proposal_refs,
        "published_artifact_refs": published_refs,
        "routing_facts": resolved_state.routing_facts
        + (
            RoutingFact(
                stage="finalize_outputs",
                fact_type="published_manifest",
                value=manifest_ref,
                source_ref=manifest_ref,
            ),
        ),
    }
