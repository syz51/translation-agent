"""Translation generation node for the deterministic dry-run workflow."""

from __future__ import annotations

from hashlib import sha256

from translation_agent.adapters import RawPayloadTranslationAdapter
from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.common import (
    build_request_context,
    raw_translation_candidate_key,
    select_transcript_candidates,
    staged_translation_candidate_key,
    strip_private_metadata,
    write_model_artifact,
)

PROMPT_VARIANTS = ("variant-a", "variant-b")


def generate_translation_candidates(
    state: GraphState, runtime: WorkflowRuntime
) -> dict[str, object]:
    """Generate translation candidates from each surviving transcript candidate."""

    if not state.transcript_candidate_ids:
        raise RuntimeError("generate_translation_candidates requires transcript candidates")

    candidates = select_transcript_candidates(
        runtime,
        job=state.job,
        candidate_ids=state.transcript_candidate_ids,
    )
    if not candidates:
        raise RuntimeError("transcript candidates were not found")

    request_context = build_request_context(state, runtime)
    payload_refs: list[str] = []
    staged_refs: list[str] = []
    routing_facts = list(state.routing_facts)

    for transcript in candidates:
        for prompt_variant_id in PROMPT_VARIANTS:
            resolved_prompt = runtime.prompt_resolver.resolve_translation_prompt(
                base_prompt_version=getattr(
                    runtime.translation_adapter,
                    "_prompt_version",
                    "unversioned",
                ),
                prompt_variant_id=prompt_variant_id,
                model_id=runtime.translation_adapter.model_id,
                source_language=state.job.source_language,
                target_language=state.job.target_language,
                media_key=state.job.media_key,
                run_id=state.run_id,
                scope_kind="pair",
                scope_key=f"{state.job.source_language}::{state.job.target_language}",
            )
            variant_request_context = request_context.model_copy(
                update={
                    "metadata": {
                        **request_context.metadata,
                        "resolved_translation_prompt": resolved_prompt.model_dump(mode="json"),
                    }
                }
            )
            try:
                raw_payload: dict[str, object] | None = None
                if isinstance(runtime.translation_adapter, RawPayloadTranslationAdapter):
                    candidate, raw_payload = (
                        runtime.translation_adapter.generate_translation_with_payload(
                            transcript,
                            prompt_variant_id,
                            variant_request_context,
                        )
                    )
                else:
                    candidate = runtime.translation_adapter.generate_translation(
                        transcript,
                        prompt_variant_id,
                        variant_request_context,
                    )
            except Exception as exc:
                routing_facts.append(
                    RoutingFact(
                        stage="generate_translation_candidates",
                        fact_type="translation_variant_failed",
                        value=f"{prompt_variant_id}:{transcript.candidate_id}",
                        source_ref=str(exc),
                    )
                )
                continue

            raw_payload_ref = raw_translation_candidate_key(
                state.job,
                prompt_variant_id,
                transcript.candidate_id,
            )
            original_raw_payload_ref = candidate.raw_response_ref
            if raw_payload is None:
                raw_payload = candidate.metadata.get("_raw_payload")
            if raw_payload is not None:
                payload_refs.append(write_model_artifact(runtime, raw_payload_ref, raw_payload))
            elif original_raw_payload_ref is not None and runtime.blob_store.exists(
                original_raw_payload_ref
            ):
                runtime.blob_store.put_bytes(
                    raw_payload_ref,
                    runtime.blob_store.read_bytes(original_raw_payload_ref),
                )
                payload_refs.append(raw_payload_ref)
            elif runtime.blob_store.exists(raw_payload_ref):
                payload_refs.append(raw_payload_ref)
            staged_candidate = candidate.model_copy(
                update={
                    "candidate_id": _translation_candidate_id(
                        prompt_variant_id,
                        transcript.candidate_id,
                        state.job.job_id,
                    ),
                    "source_transcript_candidate_id": transcript.candidate_id,
                    "prompt_version": resolved_prompt.effective_prompt_version,
                    "raw_response_ref": raw_payload_ref,
                    "metadata": strip_private_metadata(
                        {
                            **candidate.metadata,
                            "prompt_resolver": resolved_prompt.model_dump(mode="json"),
                            "source_transcript_candidate_id": transcript.candidate_id,
                        }
                    ),
                }
            )
            staged_refs.append(
                write_model_artifact(
                    runtime,
                    staged_translation_candidate_key(state.job, staged_candidate.candidate_id),
                    staged_candidate,
                )
            )
            routing_facts.append(
                RoutingFact(
                    stage="generate_translation_candidates",
                    fact_type="translation_variant_succeeded",
                    value=f"{prompt_variant_id}:{transcript.candidate_id}",
                    source_ref=raw_payload_ref,
                )
            )

    return {
        "current_stage": "generate_translation_candidates",
        "raw_translation_payload_refs": tuple(payload_refs),
        "raw_translation_candidate_refs": tuple(staged_refs),
        "translation_failed": not staged_refs,
        "routing_facts": tuple(routing_facts),
    }


def _translation_candidate_id(
    prompt_variant_id: str,
    source_transcript_candidate_id: str,
    job_id: str,
) -> str:
    transcript_token = sha256(source_transcript_candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"tl-{prompt_variant_id}-{job_id}-{transcript_token}"
