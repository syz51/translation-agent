"""Transcription fanout node for the deterministic dry-run workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from translation_agent.adapters import RawPayloadTranscriptionAdapter
from translation_agent.errors import TranscriptionProvidersFailedError
from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import AudioArtifact
from translation_agent.nodes.common import (
    audio_artifact_key,
    build_request_context,
    raw_transcript_candidate_key,
    read_model_artifact,
    staged_transcript_candidate_key,
    strip_private_metadata,
    write_model_artifact,
)


def fanout_transcription(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Run transcription providers and persist raw payloads plus staged candidates."""

    audio_artifact = read_model_artifact(runtime, audio_artifact_key(state.job), AudioArtifact)
    request_context = build_request_context(state, runtime)
    payload_refs: list[str] = []
    staged_refs: list[str] = []
    provider_errors: dict[str, str] = {}
    routing_facts = list(state.routing_facts)

    with ThreadPoolExecutor(max_workers=max(1, len(runtime.transcription_adapters))) as executor:
        futures = [
            (
                adapter,
                executor.submit(_transcribe_adapter, adapter, audio_artifact, request_context),
            )
            for adapter in runtime.transcription_adapters
        ]

        for adapter, future in futures:
            try:
                candidate, raw_payload = future.result()
            except Exception as exc:
                provider_errors[adapter.provider_id] = str(exc)
                routing_facts.append(
                    RoutingFact(
                        stage="fanout_transcription",
                        fact_type="transcription_provider_failed",
                        value=adapter.provider_id,
                        source_ref=str(exc),
                    )
                )
                continue

            raw_payload_ref = candidate.raw_payload_ref or raw_transcript_candidate_key(
                state.job, adapter.provider_id
            )
            if raw_payload is None:
                raw_payload = candidate.metadata.get("_raw_payload")
            if raw_payload is not None:
                payload_refs.append(write_model_artifact(runtime, raw_payload_ref, raw_payload))
            elif runtime.blob_store.exists(raw_payload_ref):
                payload_refs.append(raw_payload_ref)
            staged_candidate = candidate.model_copy(
                update={
                    "raw_payload_ref": raw_payload_ref,
                    "metadata": strip_private_metadata(candidate.metadata),
                }
            )
            staged_refs.append(
                write_model_artifact(
                    runtime,
                    staged_transcript_candidate_key(state.job, staged_candidate.candidate_id),
                    staged_candidate,
                )
            )
            routing_facts.append(
                RoutingFact(
                    stage="fanout_transcription",
                    fact_type="transcription_provider_succeeded",
                    value=adapter.provider_id,
                    source_ref=raw_payload_ref,
                )
            )

    if not staged_refs:
        raise TranscriptionProvidersFailedError(provider_errors)

    return {
        "current_stage": "fanout_transcription",
        "raw_transcript_payload_refs": tuple(payload_refs),
        "raw_transcript_candidate_refs": tuple(staged_refs),
        "routing_facts": tuple(routing_facts),
    }


def _transcribe_adapter(adapter, audio_artifact: AudioArtifact, request_context):
    raw_payload: dict[str, object] | None = None
    if isinstance(adapter, RawPayloadTranscriptionAdapter):
        candidate, raw_payload = adapter.transcribe_with_payload(audio_artifact, request_context)
    else:
        candidate = adapter.transcribe(audio_artifact, request_context)
    return candidate, raw_payload
