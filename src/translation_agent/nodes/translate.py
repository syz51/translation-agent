"""Translation generation node for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.nodes.common import (
    build_request_context,
    select_transcript_candidates,
    write_model_artifact,
)

PROMPT_VARIANTS = ("variant-a", "variant-b")


def generate_translation_candidates(
    state: GraphState, runtime: WorkflowRuntime
) -> dict[str, object]:
    """Generate dry-run translation candidates from the winning transcript."""

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
    raw_refs: list[str] = []
    routing_facts = list(state.routing_facts)

    for prompt_variant_id in PROMPT_VARIANTS:
        try:
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

        raw_refs.append(write_model_artifact(runtime, candidate.raw_response_ref or "", candidate))
        routing_facts.append(
            RoutingFact(
                stage="generate_translation_candidates",
                fact_type="translation_variant_succeeded",
                value=prompt_variant_id,
                source_ref=candidate.raw_response_ref,
            )
        )

    return {
        "current_stage": "generate_translation_candidates",
        "raw_translation_candidate_refs": tuple(raw_refs),
        "translation_failed": not raw_refs,
        "routing_facts": tuple(routing_facts),
    }
