"""Translation generation node for the deterministic dry-run workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    FinalTranscriptDecision,
    RequestContext,
    SynthesizedTranscriptArtifact,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.nodes.common import (
    build_memory_query,
    build_request_context,
    raw_translation_candidate_key,
    read_model_artifact,
    staged_translation_candidate_key,
    strip_private_metadata,
    synthesized_transcript_as_candidate,
    transcript_investigation_key,
    write_model_artifact,
)
from translation_agent.observability import TraceEvent
from translation_agent.parallelism import ordered_parallel_map

DEFAULT_PROMPT_VARIANTS = ("variant-a",)
EXPERIMENT_PROMPT_VARIANTS = ("variant-a", "variant-b")


@dataclass(frozen=True, slots=True)
class _TranslationCandidateTask:
    transcript: TranscriptCandidate
    transcript_ref: str
    transcript_artifact: SynthesizedTranscriptArtifact
    prompt_variant_id: str
    resolved_prompt_payload: dict[str, object]
    request_context: RequestContext


def generate_translation_candidates(
    state: GraphState, runtime: WorkflowRuntime
) -> dict[str, object]:
    """Generate translation candidates from the synthesized transcript artifact."""

    if state.final_transcript_ref is None:
        raise RuntimeError("generate_translation_candidates requires a synthesized transcript")
    transcript_artifact = read_model_artifact(
        runtime,
        state.final_transcript_ref,
        SynthesizedTranscriptArtifact,
    )
    transcript = synthesized_transcript_as_candidate(transcript_artifact)
    transcript_metadata = _transcript_synthesis_metadata(
        state=state,
        runtime=runtime,
        artifact=transcript_artifact,
    )
    base_request_context = build_request_context(state, runtime)
    request_context = base_request_context.model_copy(
        update={
            "metadata": {
                **base_request_context.metadata,
                "transcript_synthesis": transcript_metadata,
            }
        }
    )
    payload_refs: list[str] = []
    staged_refs: list[str] = []
    routing_facts = list(state.routing_facts)
    task_specs: list[_TranslationCandidateTask] = []

    transcript_blockers = _string_tuple(transcript_metadata.get("transcript_blockers"))
    provider_ids = tuple(transcript_artifact.quality_metrics.provider_support_summary.keys()) or (
        transcript.provider_id,
    )
    for prompt_variant_id in _selected_prompt_variants(state):
        guidance_bundle = runtime.memory_recall_backend.recall_memory(
            build_memory_query(
                state,
                stage="generate_translation_guidance",
                candidate_ids=(transcript.candidate_id,),
                provider_ids=provider_ids,
                prompt_variant_ids=(prompt_variant_id,),
                model_ids=(runtime.translation_adapter.model_id,),
                failure_tags=transcript_blockers,
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
            tenant_id=state.job.tenant_id,
            project_id=state.job.project_id,
            media_key=state.job.media_key,
            run_id=state.run_id,
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
                transcript_ref=state.final_transcript_ref,
                transcript_artifact=transcript_artifact,
                prompt_variant_id=prompt_variant_id,
                resolved_prompt_payload=resolved_prompt.model_dump(mode="json"),
                request_context=variant_request_context,
            )
        )

    effective_stage_workers = runtime.parallelism.resolve_stage_workers(
        runtime.parallelism.translation_candidate_max_workers,
        task_count=len(task_specs),
    )
    gathered = ordered_parallel_map(
        task_specs,
        max_workers=effective_stage_workers,
        worker=lambda task: _generate_translation_task(
            task,
            runtime,
            run_id=state.run_id,
            variant_total=len(task_specs),
            effective_stage_workers=effective_stage_workers,
        ),
        sort_key=lambda input_index, _task: (input_index,),
    )

    for task, result in zip(task_specs, gathered, strict=True):
        prompt_variant_id = task.prompt_variant_id
        if result.error is not None:
            routing_facts.append(
                RoutingFact(
                    stage="generate_translation_candidates",
                    fact_type="translation_variant_failed",
                    value=f"{prompt_variant_id}:{task.transcript_ref}",
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
            _translation_source_token(task.transcript_ref),
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

        transcript_synthesis = cast(
            "dict[str, object]",
            task.request_context.metadata.get("transcript_synthesis", {}),
        )
        staged_candidate = candidate.model_copy(
            update={
                "candidate_id": _translation_candidate_id(
                    prompt_variant_id,
                    task.transcript_ref,
                    state.job.job_id,
                ),
                "source_transcript_candidate_id": None,
                "source_transcript_ref": task.transcript_ref,
                "final_transcript_ref": task.transcript_ref,
                "prompt_version": str(
                    task.resolved_prompt_payload.get("effective_prompt_version")
                    or candidate.prompt_version
                ),
                "raw_response_ref": raw_payload_ref,
                "metadata": strip_private_metadata(
                    {
                        **candidate.metadata,
                        "prompt_resolver": task.resolved_prompt_payload,
                        "source_transcript_ref": task.transcript_ref,
                        "provenance_summary": {
                            "source_transcript_ref": task.transcript_ref,
                            "source_span_ids": [
                                span.canonical_span_id
                                for span in task.transcript_artifact.canonical_spans
                            ],
                            "provider_support_summary": (
                                task.transcript_artifact.quality_metrics.provider_support_summary
                            ),
                            "transcript_blockers": transcript_synthesis.get(
                                "transcript_blockers", []
                            ),
                            "transcript_synthesis_status": transcript_synthesis.get(
                                "transcript_synthesis_status"
                            ),
                            "transcript_decision_ref": transcript_synthesis.get(
                                "transcript_decision_ref"
                            ),
                            "transcript_investigation_ref": transcript_synthesis.get(
                                "transcript_investigation_ref"
                            ),
                        },
                        "transcript_synthesis": transcript_synthesis,
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
                value=f"{prompt_variant_id}:{task.transcript_ref}",
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


def _selected_prompt_variants(state: GraphState) -> tuple[str, ...]:
    if state.job.translation_variant_policy == "dual_experiment":
        return EXPERIMENT_PROMPT_VARIANTS
    return DEFAULT_PROMPT_VARIANTS


def _translation_candidate_id(
    prompt_variant_id: str,
    source_transcript_token: str,
    job_id: str,
) -> str:
    transcript_token = sha256(source_transcript_token.encode("utf-8")).hexdigest()[:12]
    return f"tl-{prompt_variant_id}-{job_id}-{transcript_token}"


def _generate_translation_task(
    task: _TranslationCandidateTask,
    runtime: WorkflowRuntime,
    *,
    run_id: str,
    variant_total: int,
    effective_stage_workers: int,
) -> tuple[TranslationCandidate, dict[str, object] | None]:
    runtime.trace_sink.record(
        TraceEvent(
            run_id=run_id,
            name="translation.variant.started",
            attributes={
                "prompt_variant_id": task.prompt_variant_id,
                "source_transcript_ref": task.transcript_ref,
                "variant_total": variant_total,
                "effective_stage_workers": effective_stage_workers,
            },
        )
    )
    generate_with_payload = getattr(
        runtime.translation_adapter,
        "generate_translation_with_payload",
        None,
    )
    try:
        if callable(generate_with_payload):
            candidate, raw_payload = cast(
                "tuple[TranslationCandidate, dict[str, object] | None]",
                generate_with_payload(
                    task.transcript,
                    task.prompt_variant_id,
                    task.request_context,
                ),
            )
        else:
            candidate, raw_payload = (
                runtime.translation_adapter.generate_translation(
                    task.transcript,
                    task.prompt_variant_id,
                    task.request_context,
                ),
                None,
            )
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="translation.variant.completed",
                attributes={
                    "prompt_variant_id": task.prompt_variant_id,
                    "source_transcript_ref": task.transcript_ref,
                    "variant_total": variant_total,
                    "effective_stage_workers": effective_stage_workers,
                    "candidate_id": candidate.candidate_id,
                },
            )
        )
        return candidate, raw_payload
    except Exception as exc:
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="translation.variant.failed",
                attributes={
                    "prompt_variant_id": task.prompt_variant_id,
                    "source_transcript_ref": task.transcript_ref,
                    "variant_total": variant_total,
                    "effective_stage_workers": effective_stage_workers,
                    "error": str(exc),
                },
            )
        )
        raise


def _translation_source_token(source_transcript_ref: str) -> str:
    return sha256(source_transcript_ref.encode("utf-8")).hexdigest()[:12]


def _transcript_synthesis_metadata(
    *,
    state: GraphState,
    runtime: WorkflowRuntime,
    artifact: SynthesizedTranscriptArtifact,
) -> dict[str, object]:
    transcript_decision_ref = state.final_transcript_decision_ref
    transcript_investigation_ref = transcript_investigation_key(state.job)
    blocker_tags = tuple(
        str(item)
        for item in artifact.transcript_metadata.get("blocker_tags", [])
        if str(item).strip()
    )
    if not blocker_tags:
        blocker_tags = blocking_failures = tuple(
            fact.value
            for fact in state.routing_facts
            if fact.fact_type == "transcript_synthesis_blocker" and fact.value.strip()
        )
        if not blocking_failures and artifact.status == "blocked":
            blocker_tags = ("unresolved_supported_spans",)

    decision_payload: dict[str, object] = {}
    if transcript_decision_ref and runtime.blob_store.exists(transcript_decision_ref):
        decision_payload = read_model_artifact(
            runtime,
            transcript_decision_ref,
            FinalTranscriptDecision,
        ).model_dump(mode="json")
    investigation_payload: dict[str, object] = {}
    if runtime.blob_store.exists(transcript_investigation_ref):
        investigation_payload = json.loads(
            runtime.blob_store.read_bytes(transcript_investigation_ref).decode("utf-8")
        )

    return {
        "final_transcript_ref": state.final_transcript_ref,
        "transcript_decision_ref": transcript_decision_ref,
        "transcript_investigation_ref": (
            transcript_investigation_ref
            if runtime.blob_store.exists(transcript_investigation_ref)
            else None
        ),
        "transcript_synthesis_status": artifact.status,
        "transcript_unresolved_span_count": artifact.quality_metrics.unresolved_span_count,
        "transcript_blockers": list(blocker_tags),
        "provider_support_summary": artifact.quality_metrics.provider_support_summary,
        "source_span_ids": [span.canonical_span_id for span in artifact.canonical_spans],
        "transcript_decision": decision_payload,
        "transcript_investigation": investigation_payload,
    }


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str))
    return ()
