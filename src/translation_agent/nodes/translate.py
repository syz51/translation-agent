"""Translation generation node for the deterministic dry-run workflow."""

from __future__ import annotations

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
    """Generate translation candidates from the winning transcript."""

    if state.final_transcript_candidate_id is None:
        raise RuntimeError("generate_translation_candidates requires a transcript winner")

    candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=(state.final_transcript_candidate_id,),
    )
    if not candidates:
        raise RuntimeError("winning transcript candidate was not found")

    request_context = build_request_context(state, runtime)
    transcript = candidates[0]
    payload_refs: list[str] = []
    staged_refs: list[str] = []
    routing_facts = list(state.routing_facts)

    for prompt_variant_id in PROMPT_VARIANTS:
        try:
            raw_payload: dict[str, object] | None = None
            if isinstance(runtime.translation_adapter, RawPayloadTranslationAdapter):
                candidate, raw_payload = (
                    runtime.translation_adapter.generate_translation_with_payload(
                        transcript,
                        prompt_variant_id,
                        request_context,
                    )
                )
            else:
                candidate = runtime.translation_adapter.generate_translation(
                    transcript,
                    prompt_variant_id,
                    request_context,
                )
        except Exception as exc:
            routing_facts.append(
                RoutingFact(
                    stage="generate_translation_candidates",
                    fact_type="translation_variant_failed",
                    value=prompt_variant_id,
                    source_ref=str(exc),
                )
            )
            continue

        raw_payload_ref = candidate.raw_response_ref or raw_translation_candidate_key(
            state.job, prompt_variant_id
        )
        if raw_payload is None:
            raw_payload = candidate.metadata.get("_raw_payload")
        if raw_payload is not None:
            payload_refs.append(write_model_artifact(runtime, raw_payload_ref, raw_payload))
        elif runtime.blob_store.exists(raw_payload_ref):
            payload_refs.append(raw_payload_ref)
        staged_candidate = candidate.model_copy(
            update={
                "raw_response_ref": raw_payload_ref,
                "metadata": strip_private_metadata(candidate.metadata),
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
                value=prompt_variant_id,
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
