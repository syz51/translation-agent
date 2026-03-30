"""Transcription fanout node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import AudioArtifact
from translation_agent.nodes.common import (
    audio_artifact_key,
    build_request_context,
    read_model_artifact,
    write_model_artifact,
)


def fanout_transcription(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Run the fake transcription providers and persist raw candidate payloads."""

    audio_artifact = read_model_artifact(
        runtime, audio_artifact_key(state.job.job_id), AudioArtifact
    )
    request_context = build_request_context(state, runtime)
    raw_refs: list[str] = []
    routing_facts = list(state.routing_facts)

    for adapter in runtime.transcription_adapters:
        try:
            candidate = adapter.transcribe(audio_artifact, request_context)
        except Exception as exc:
            routing_facts.append(
                RoutingFact(
                    stage="fanout_transcription",
                    fact_type="transcription_provider_failed",
                    value=adapter.provider_id,
                    source_ref=str(exc),
                )
            )
            continue

        raw_refs.append(write_model_artifact(runtime, candidate.raw_payload_ref or "", candidate))
        routing_facts.append(
            RoutingFact(
                stage="fanout_transcription",
                fact_type="transcription_provider_succeeded",
                value=adapter.provider_id,
                source_ref=candidate.raw_payload_ref,
            )
        )

    if not raw_refs:
        raise RuntimeError("all transcription providers failed")

    return {
        "current_stage": "fanout_transcription",
        "raw_transcript_candidate_refs": tuple(raw_refs),
        "routing_facts": tuple(routing_facts),
    }
