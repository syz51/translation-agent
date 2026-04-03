"""Translation generation node for the deterministic dry-run workflow."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import RequestContext, TranscriptCandidate, TranslationCandidate
from translation_agent.nodes.common import (
    build_memory_query,
    build_request_context,
    raw_translation_candidate_key,
    select_transcript_candidates,
    staged_translation_candidate_key,
    strip_private_metadata,
    write_model_artifact,
)
from translation_agent.parallelism import ordered_parallel_map

PROMPT_VARIANTS = ("variant-a", "variant-b")


@dataclass(frozen=True, slots=True)
class _TranslationCandidateTask:
    transcript: TranscriptCandidate
    prompt_variant_id: str
    resolved_prompt_payload: dict[str, object]
    request_context: RequestContext


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
    task_specs: list[_TranslationCandidateTask] = []

    for transcript in candidates:
        for prompt_variant_id in PROMPT_VARIANTS:
            guidance_bundle = runtime.memory_recall_backend.recall_memory(
                build_memory_query(
                    state,
                    stage="generate_translation_guidance",
                    candidate_ids=(transcript.candidate_id,),
                    provider_ids=(transcript.provider_id,),
                    prompt_variant_ids=(prompt_variant_id,),
                    model_ids=(runtime.translation_adapter.model_id,),
                )
            )
            historical_instructions = tuple(
                entry.content
                for entry in (*guidance_bundle.rules, *guidance_bundle.procedural_memory)
                if entry.content.strip()
            )[:4]
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
                        "historical_translation_instructions": list(historical_instructions),
                    }
                }
            )
            task_specs.append(
                _TranslationCandidateTask(
                    transcript=transcript,
                    prompt_variant_id=prompt_variant_id,
                    resolved_prompt_payload=resolved_prompt.model_dump(mode="json"),
                    request_context=variant_request_context,
                )
            )

    gathered = ordered_parallel_map(
        task_specs,
        max_workers=runtime.parallelism.translation_candidate_max_workers,
        worker=lambda task: _generate_translation_task(task, runtime),
        sort_key=lambda input_index, _task: (input_index,),
    )

    for task, result in zip(task_specs, gathered, strict=True):
        transcript = task.transcript
        prompt_variant_id = task.prompt_variant_id
        if result.error is not None:
            routing_facts.append(
                RoutingFact(
                    stage="generate_translation_candidates",
                    fact_type="translation_variant_failed",
                    value=f"{prompt_variant_id}:{transcript.candidate_id}",
                    source_ref=str(result.error),
                )
            )
            continue

        if result.value is None:  # pragma: no cover - defensive
            raise RuntimeError("translation worker completed without a candidate")

        candidate, raw_payload = result.value
        raw_payload_ref = raw_translation_candidate_key(
            state.job,
            prompt_variant_id,
            transcript.candidate_id,
        )
        original_raw_payload_ref = candidate.raw_response_ref
        if raw_payload is None:
            raw_payload = candidate.metadata.get("_raw_payload")
        if raw_payload is not None:
            if original_raw_payload_ref and original_raw_payload_ref != raw_payload_ref:
                write_model_artifact(runtime, original_raw_payload_ref, raw_payload)
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
                "candidate_id": _translation_candidate_id(
                    prompt_variant_id,
                    transcript.candidate_id,
                    state.job.job_id,
                ),
                "source_transcript_candidate_id": transcript.candidate_id,
                "prompt_version": str(
                    task.resolved_prompt_payload.get("effective_prompt_version")
                    or candidate.prompt_version
                ),
                "raw_response_ref": raw_payload_ref,
                "metadata": strip_private_metadata(
                    {
                        **candidate.metadata,
                        "prompt_resolver": task.resolved_prompt_payload,
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


def _generate_translation_task(
    task: _TranslationCandidateTask,
    runtime: WorkflowRuntime,
) -> tuple[TranslationCandidate, dict[str, object] | None]:
    generate_with_payload = getattr(
        runtime.translation_adapter,
        "generate_translation_with_payload",
        None,
    )
    if callable(generate_with_payload):
        return cast(
            "tuple[TranslationCandidate, dict[str, object] | None]",
            generate_with_payload(
                task.transcript,
                task.prompt_variant_id,
                task.request_context,
            ),
        )
    return (
        runtime.translation_adapter.generate_translation(
            task.transcript,
            task.prompt_variant_id,
            task.request_context,
        ),
        None,
    )


def _supports_raw_payload_translation(adapter: object) -> bool:
    return callable(getattr(adapter, "generate_translation_with_payload", None))
