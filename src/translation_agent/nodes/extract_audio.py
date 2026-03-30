"""Audio extraction node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.adapters import AudioBytesExtractionAdapter
from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.common import (
    audio_artifact_key,
    build_request_context,
    write_model_artifact,
)


def extract_audio(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Extract audio, persist the blob payload, and store artifact metadata."""

    request_context = build_request_context(state, runtime)
    if isinstance(runtime.audio_extractor, AudioBytesExtractionAdapter):
        artifact, audio_bytes = runtime.audio_extractor.extract_audio_bytes(
            state.source_video_ref,
            request_context,
        )
        runtime.blob_store.put_bytes(artifact.blob_ref, audio_bytes)
    else:
        artifact = runtime.audio_extractor.extract_audio(state.source_video_ref, request_context)
    if not runtime.blob_store.exists(artifact.blob_ref):
        runtime.blob_store.put_bytes(
            artifact.blob_ref,
            f"stub audio for {state.job.job_id}\n".encode(),
        )
    metadata_ref = write_model_artifact(runtime, audio_artifact_key(state.job), artifact)

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
