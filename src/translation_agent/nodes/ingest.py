"""Ingest node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact


def ingest_job(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Resolve the initial run facts into ref-only graph state."""

    return {
        "current_stage": "ingest",
        "source_artifact_ref": runtime.source_artifact_ref,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="ingest",
                fact_type="job_initialized",
                value=state.job.job_id,
                source_ref=runtime.source_artifact_ref,
            ),
            RoutingFact(
                stage="ingest",
                fact_type="scenario",
                value=runtime.scenario,
                source_ref=runtime.source_artifact_ref,
            ),
        ),
    }
