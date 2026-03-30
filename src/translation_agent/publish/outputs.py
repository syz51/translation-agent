"""Dry-run publishing helpers."""

from __future__ import annotations

import json
from typing import Any

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState
from translation_agent.models import (
    FinalTranscriptDecision,
    FinalTranslationDecision,
    PublishedArtifacts,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.nodes.common import (
    published_artifacts_key,
    read_model_artifact,
    select_transcript_candidates,
    select_translation_candidates,
    translation_failure_key,
    write_model_artifact,
)


def publish_outputs(state: GraphState, runtime: WorkflowRuntime) -> tuple[PublishedArtifacts, str]:
    """Persist canonical dry-run artifacts and return the manifest ref."""

    transcript = _final_transcript(state, runtime)
    translation = _final_translation(state, runtime)
    transcript_ref = None
    translation_ref = None
    translation_failure_ref = None

    if transcript is not None:
        transcript_ref = f"published/{state.job.job_id}/transcript.json"
        write_model_artifact(runtime, transcript_ref, transcript)

    if translation is not None and not state.human_review_required:
        translation_ref = f"published/{state.job.job_id}/translation.json"
        write_model_artifact(runtime, translation_ref, translation)
    elif state.translation_failed:
        translation_failure_ref = translation_failure_key(state.job.job_id)
        write_model_artifact(
            runtime,
            translation_failure_ref,
            {
                "job_id": state.job.job_id,
                "run_id": state.run_id,
                "status": "translation_failed",
                "recoverable": True,
                "transcript_ref": transcript_ref,
                "translation_decision_ref": state.final_translation_decision_ref,
            },
        )

    trace_refs = (f"traces/{state.run_id}.jsonl",)
    memory_batch_refs = _refs_for_fact(state, "memory_batch_staged")
    memory_consolidation_refs = _refs_for_fact(state, "memory_batch_consolidated")
    prompt_evolution_refs = _refs_for_fact(state, "translation_prompt_evolution")

    export_text_ref = f"exports/{state.job.job_id}.txt"
    export_json_ref = f"exports/{state.job.job_id}.json"
    downstream_ref = f"deliveries/{state.job.job_id}.json"

    _write_text(
        runtime,
        export_text_ref,
        _render_export_text(
            state=state,
            transcript=transcript,
            translation=translation,
            translation_failed=state.translation_failed,
        ),
    )
    export_payload = _export_payload(
        state=state,
        transcript=transcript,
        translation=translation,
        transcript_ref=transcript_ref,
        translation_ref=translation_ref,
        translation_failure_ref=translation_failure_ref,
        trace_refs=trace_refs,
    )
    _write_json(runtime, export_json_ref, export_payload)
    _write_json(
        runtime,
        downstream_ref,
        {
            "job_id": state.job.job_id,
            "status": _delivery_status(state),
            "transcript_ref": transcript_ref,
            "translation_ref": translation_ref,
            "translation_failure_ref": translation_failure_ref,
        },
    )

    scorecard_ref = f"published/{state.job.job_id}/scorecard.json"
    _write_json(
        runtime,
        scorecard_ref,
        _scorecard_payload(
            state=state,
            transcript_ref=transcript_ref,
            translation_ref=translation_ref,
            translation_failure_ref=translation_failure_ref,
            trace_refs=trace_refs,
            export_refs=(export_text_ref, export_json_ref),
            downstream_refs=(downstream_ref,),
            memory_batch_refs=memory_batch_refs,
            memory_consolidation_refs=memory_consolidation_refs,
            prompt_evolution_refs=prompt_evolution_refs,
            transcript_decision=_decision(
                runtime,
                state.final_transcript_decision_ref,
                FinalTranscriptDecision,
            ),
            translation_decision=_decision(
                runtime,
                state.final_translation_decision_ref,
                FinalTranslationDecision,
            ),
        ),
    )

    artifacts = PublishedArtifacts(
        final_transcript_ref=transcript_ref,
        final_translation_ref=translation_ref,
        recoverable_translation_failure_ref=translation_failure_ref,
        scorecard_refs=(scorecard_ref,),
        trace_refs=trace_refs,
        export_refs=(export_text_ref, export_json_ref),
        downstream_delivery_refs=(downstream_ref,),
        memory_batch_refs=memory_batch_refs,
        memory_consolidation_refs=memory_consolidation_refs,
        prompt_evolution_refs=prompt_evolution_refs,
    )
    manifest_key = published_artifacts_key(state.job.job_id)
    manifest_ref = write_model_artifact(runtime, manifest_key, artifacts)
    return artifacts, manifest_ref


def _final_transcript(state: GraphState, runtime: WorkflowRuntime) -> TranscriptCandidate | None:
    if state.final_transcript_candidate_id is None:
        return None
    transcript_candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=(state.final_transcript_candidate_id,),
    )
    if transcript_candidates:
        return transcript_candidates[0]
    return None


def _final_translation(state: GraphState, runtime: WorkflowRuntime) -> TranslationCandidate | None:
    if state.final_translation_candidate_id is None:
        return None
    translation_candidates = select_translation_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=(state.final_translation_candidate_id,),
    )
    if translation_candidates:
        return translation_candidates[0]
    return None


def _scorecard_payload(
    *,
    state: GraphState,
    transcript_ref: str | None,
    translation_ref: str | None,
    translation_failure_ref: str | None,
    trace_refs: tuple[str, ...],
    export_refs: tuple[str, ...],
    downstream_refs: tuple[str, ...],
    memory_batch_refs: tuple[str, ...],
    memory_consolidation_refs: tuple[str, ...],
    prompt_evolution_refs: tuple[str, ...],
    transcript_decision: FinalTranscriptDecision | None,
    translation_decision: FinalTranslationDecision | None,
) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "job_id": state.job.job_id,
        "human_review_required": state.human_review_required,
        "translation_failed": state.translation_failed,
        "transcript_ref": transcript_ref,
        "translation_ref": translation_ref,
        "translation_failure_ref": translation_failure_ref,
        "trace_refs": list(trace_refs),
        "export_refs": list(export_refs),
        "downstream_refs": list(downstream_refs),
        "memory_batch_refs": list(memory_batch_refs),
        "memory_consolidation_refs": list(memory_consolidation_refs),
        "prompt_evolution_refs": list(prompt_evolution_refs),
        "routing_facts": [fact.model_dump(mode="json") for fact in state.routing_facts],
        "transcript_decision": (
            transcript_decision.model_dump(mode="json") if transcript_decision is not None else None
        ),
        "translation_decision": (
            translation_decision.model_dump(mode="json")
            if translation_decision is not None
            else None
        ),
    }


def _export_payload(
    *,
    state: GraphState,
    transcript: TranscriptCandidate | None,
    translation: TranslationCandidate | None,
    transcript_ref: str | None,
    translation_ref: str | None,
    translation_failure_ref: str | None,
    trace_refs: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "job_id": state.job.job_id,
        "source_language": state.job.source_language,
        "target_language": state.job.target_language,
        "transcript_ref": transcript_ref,
        "translation_ref": translation_ref,
        "translation_failure_ref": translation_failure_ref,
        "transcript_text": transcript.full_text if transcript is not None else None,
        "translation_text": translation.full_text if translation is not None else None,
        "trace_refs": list(trace_refs),
        "status": _delivery_status(state),
    }


def _render_export_text(
    *,
    state: GraphState,
    transcript: TranscriptCandidate | None,
    translation: TranslationCandidate | None,
    translation_failed: bool,
) -> str:
    transcript_text = transcript.full_text if transcript is not None else "[missing transcript]"
    translation_text = (
        translation.full_text
        if translation is not None
        else "[translation unavailable; recoverable failure]"
        if translation_failed
        else "[translation omitted pending human review]"
    )
    return (
        f"Job: {state.job.job_id}\n"
        f"Source language: {state.job.source_language}\n"
        f"Target language: {state.job.target_language}\n\n"
        "Transcript\n"
        f"{transcript_text}\n\n"
        "Translation\n"
        f"{translation_text}\n"
    )


def _decision[ModelT: FinalTranscriptDecision | FinalTranslationDecision](
    runtime: WorkflowRuntime,
    decision_ref: str | None,
    model_type: type[ModelT],
) -> ModelT | None:
    if decision_ref is None:
        return None
    return read_model_artifact(runtime, decision_ref, model_type)


def _refs_for_fact(state: GraphState, fact_type: str) -> tuple[str, ...]:
    return tuple(
        fact.source_ref
        for fact in state.routing_facts
        if fact.fact_type == fact_type and fact.source_ref is not None
    )


def _delivery_status(state: GraphState) -> str:
    if state.human_review_required:
        return "human_review_required"
    if state.translation_failed:
        return "translation_failed"
    return "completed"


def _write_json(runtime: WorkflowRuntime, key: str, payload: dict[str, Any]) -> None:
    runtime.blob_store.put_bytes(
        key,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_text(runtime: WorkflowRuntime, key: str, payload: str) -> None:
    runtime.blob_store.put_bytes(key, payload.encode("utf-8"))
