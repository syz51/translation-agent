"""Finalization node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.publish.outputs import publish_outputs


def finalize_outputs(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Publish canonical dry-run artifacts and expose their refs in graph state."""

    artifacts, manifest_ref = publish_outputs(state, runtime)
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

    return {
        "current_stage": "finalize_outputs",
        "published_artifact_refs": published_refs,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="finalize_outputs",
                fact_type="published_manifest",
                value=manifest_ref,
                source_ref=manifest_ref,
            ),
        ),
    }
