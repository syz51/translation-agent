"""Transcription fanout node for the deterministic dry-run workflow."""

from __future__ import annotations

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
from translation_agent.observability import TraceEvent
from translation_agent.parallelism import ordered_parallel_map


def fanout_transcription(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Run transcription providers and persist raw payloads plus staged candidates."""

    audio_artifact = read_model_artifact(runtime, audio_artifact_key(state.job), AudioArtifact)
    request_context = build_request_context(state, runtime)
    payload_refs: list[str] = []
    staged_refs: list[str] = []
    provider_errors: dict[str, str] = {}
    routing_facts = list(state.routing_facts)
    max_workers = runtime.parallelism.transcription_max_workers
    if not all(
        _supports_raw_payload_transcription(adapter) for adapter in runtime.transcription_adapters
    ):
        max_workers = 1

    gathered = ordered_parallel_map(
        runtime.transcription_adapters,
        max_workers=max_workers,
        worker=lambda adapter: _transcribe_adapter(
            adapter,
            audio_artifact,
            request_context,
            runtime=runtime,
            run_id=state.run_id,
            provider_total=len(runtime.transcription_adapters),
        ),
    )

    for adapter, result in zip(runtime.transcription_adapters, gathered, strict=True):
        if result.error is not None:
            provider_errors[adapter.provider_id] = str(result.error)
            routing_facts.append(
                RoutingFact(
                    stage="fanout_transcription",
                    fact_type="transcription_provider_failed",
                    value=adapter.provider_id,
                    source_ref=str(result.error),
                )
            )
            continue

        if result.value is None:  # pragma: no cover - defensive
            raise RuntimeError("transcription worker completed without a candidate")

        candidate, raw_payload = result.value
        raw_payload_ref = candidate.raw_payload_ref or raw_transcript_candidate_key(
            state.job, adapter.provider_id
        )
        original_raw_payload_ref = candidate.raw_payload_ref
        if raw_payload is None:
            raw_payload = candidate.metadata.get("_raw_payload")
        if raw_payload is not None:
            payload_refs.append(write_model_artifact(runtime, raw_payload_ref, raw_payload))
        elif original_raw_payload_ref is not None and runtime.blob_store.exists(
            original_raw_payload_ref
        ):
            if original_raw_payload_ref != raw_payload_ref:
                runtime.blob_store.put_bytes(
                    raw_payload_ref,
                    runtime.blob_store.read_bytes(original_raw_payload_ref),
                )
            payload_refs.append(raw_payload_ref)
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


def _transcribe_adapter(
    adapter,
    audio_artifact: AudioArtifact,
    request_context,
    *,
    runtime: WorkflowRuntime,
    run_id: str,
    provider_total: int,
):
    provider_id = adapter.provider_id
    runtime.trace_sink.record(
        TraceEvent(
            run_id=run_id,
            name="transcription.provider.started",
            attributes={
                "provider_id": provider_id,
                "provider_total": provider_total,
            },
        )
    )
    raw_payload: dict[str, object] | None = None
    try:
        if _supports_raw_payload_transcription(adapter):
            candidate, raw_payload = adapter.transcribe_with_payload(
                audio_artifact,
                request_context,
            )
        else:
            candidate = adapter.transcribe(audio_artifact, request_context)
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="transcription.provider.completed",
                attributes={
                    "provider_id": provider_id,
                    "provider_total": provider_total,
                    "candidate_id": candidate.candidate_id,
                },
            )
        )
        return candidate, raw_payload
    except Exception as exc:
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="transcription.provider.failed",
                attributes={
                    "provider_id": provider_id,
                    "provider_total": provider_total,
                    "error": str(exc),
                },
            )
        )
        raise


def _supports_raw_payload_transcription(adapter: object) -> bool:
    return isinstance(adapter, RawPayloadTranscriptionAdapter) or callable(
        getattr(adapter, "transcribe_with_payload", None)
    )
