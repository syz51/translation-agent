"""Audio extraction node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.common import (
    audio_artifact_key,
    build_request_context,
    write_model_artifact,
)


def extract_audio(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Create and persist a deterministic stub audio artifact."""

    request_context = build_request_context(state, runtime)
    artifact = runtime.audio_extractor.extract_audio(state.source_video_ref, request_context)

    runtime.blob_store.put_bytes(
        artifact.blob_ref,
        f"stub audio for {state.job.job_id}\n".encode(),
    )
    metadata_ref = write_model_artifact(runtime, audio_artifact_key(state.job.job_id), artifact)

    return {
        "current_stage": "extract_audio",
        "audio_artifact_ref": artifact.blob_ref,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="extract_audio",
                fact_type="audio_artifact",
                value=artifact.artifact_id,
                source_ref=metadata_ref,
            ),
        ),
    }
