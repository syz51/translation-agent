"""Dry-run publishing helpers."""

from __future__ import annotations

import json
from pathlib import Path
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
from translation_agent.storage import job_path
from translation_agent.subtitles import render_translation_srt


def publish_outputs(state: GraphState, runtime: WorkflowRuntime) -> tuple[PublishedArtifacts, str]:
    """Persist canonical dry-run artifacts and return the manifest ref."""

    transcript = _final_transcript(state, runtime)
    translation = _final_translation(state, runtime)
    transcript_decision = _decision(
        runtime,
        state.final_transcript_decision_ref,
        FinalTranscriptDecision,
    )
    translation_decision = _decision(
        runtime,
        state.final_translation_decision_ref,
        FinalTranslationDecision,
    )
    transcript_ref = None
    translation_ref = None
    translation_failure_ref = None

    if transcript is not None:
        transcript_ref = job_path(state.job, "published", "transcript.json")
        write_model_artifact(runtime, transcript_ref, transcript)

    if translation is not None and not state.human_review_required:
        translation_ref = job_path(state.job, "published", "translation.json")
        write_model_artifact(runtime, translation_ref, translation)
    elif state.translation_failed:
        translation_failure_ref = translation_failure_key(state.job)
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
                "failure_summary": (
                    translation_decision.rationale_summary
                    if translation_decision is not None
                    else None
                ),
                "failure_reasons": _translation_failure_reasons(state),
            },
        )

    trace_refs = _publish_trace_artifact(state, runtime)
    memory_batch_refs = _ordered_memory_batch_refs(state, runtime)
    memory_consolidation_refs = _ordered_memory_consolidation_refs(state, runtime)
    prompt_evolution_refs = _refs_for_fact(state, "translation_prompt_evolution")
    learning_refs = _refs_for_fact(state, "learning_artifact")
    reference_transcript_refs = _artifact_refs(state.reference_transcript_ref)
    evaluation_report_refs = _artifact_refs(state.evaluation_report_ref)
    regenerated_draft_refs = _artifact_refs(state.regenerated_translation_draft_ref)
    improvement_proposal_refs = tuple(
        dict.fromkeys(
            (
                *state.improvement_proposal_refs,
                *_refs_for_fact(state, "reference_improvement_proposal"),
            )
        )
    )

    export_srt_ref = job_path(state.job, "exports", "translation.srt")
    export_json_ref = job_path(state.job, "exports", "translation.json")
    downstream_ref = job_path(state.job, "deliveries", "translation.json")

    export_refs: tuple[str, ...]
    if translation is not None and not state.human_review_required:
        _write_srt(runtime, export_srt_ref, render_translation_srt(translation))
        export_refs = (export_srt_ref, export_json_ref)
    else:
        runtime.blob_store.delete(export_srt_ref)
        export_refs = (export_json_ref,)
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

    scorecard_ref = job_path(state.job, "published", "scorecard.json")
    _write_json(
        runtime,
        scorecard_ref,
        _scorecard_payload(
            state=state,
            transcript_ref=transcript_ref,
            translation_ref=translation_ref,
            translation_failure_ref=translation_failure_ref,
            trace_refs=trace_refs,
            export_refs=export_refs,
            downstream_refs=(downstream_ref,),
            memory_batch_refs=memory_batch_refs,
            memory_consolidation_refs=memory_consolidation_refs,
            prompt_evolution_refs=prompt_evolution_refs,
            learning_refs=learning_refs,
            reference_transcript_refs=reference_transcript_refs,
            evaluation_report_refs=evaluation_report_refs,
            regenerated_draft_refs=regenerated_draft_refs,
            improvement_proposal_refs=improvement_proposal_refs,
            transcript_decision=transcript_decision,
            translation_decision=translation_decision,
        ),
    )

    artifacts = PublishedArtifacts(
        final_transcript_ref=transcript_ref,
        final_translation_ref=translation_ref,
        recoverable_translation_failure_ref=translation_failure_ref,
        approval_refs=_artifact_refs(state.approval_ref),
        learning_refs=learning_refs,
        scorecard_refs=(scorecard_ref,),
        trace_refs=trace_refs,
        export_refs=export_refs,
        downstream_delivery_refs=(downstream_ref,),
        memory_batch_refs=memory_batch_refs,
        memory_consolidation_refs=memory_consolidation_refs,
        prompt_evolution_refs=prompt_evolution_refs,
        reference_transcript_refs=reference_transcript_refs,
        evaluation_report_refs=evaluation_report_refs,
        regenerated_draft_refs=regenerated_draft_refs,
        improvement_proposal_refs=improvement_proposal_refs,
    )
    manifest_key = published_artifacts_key(state.job)
    manifest_ref = write_model_artifact(runtime, manifest_key, artifacts)
    return artifacts, manifest_ref


def _publish_trace_artifact(
    state: GraphState,
    runtime: WorkflowRuntime,
) -> tuple[str, ...]:
    trace_path = getattr(runtime.trace_sink, "path", None)
    if trace_path is None:
        return ()
    path = Path(trace_path)
    if not path.exists():
        return ()
    trace_ref = job_path(state.job, "traces", f"{state.run_id}.jsonl")
    runtime.blob_store.put_bytes(trace_ref, path.read_bytes())
    return (trace_ref,)


def _final_transcript(state: GraphState, runtime: WorkflowRuntime) -> TranscriptCandidate | None:
    if state.final_transcript_candidate_id is None:
        return None
    transcript_candidates = select_transcript_candidates(
        runtime,
        job=state.job,
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
        job=state.job,
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
    learning_refs: tuple[str, ...],
    reference_transcript_refs: tuple[str, ...],
    evaluation_report_refs: tuple[str, ...],
    regenerated_draft_refs: tuple[str, ...],
    improvement_proposal_refs: tuple[str, ...],
    transcript_decision: FinalTranscriptDecision | None,
    translation_decision: FinalTranslationDecision | None,
) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "job_id": state.job.job_id,
        "human_review_required": state.human_review_required,
        "review_required_stage": state.review_required_stage,
        "approval_ref": state.approval_ref,
        "approved_candidate_id": state.approved_candidate_id,
        "approved_source_transcript_candidate_id": state.approved_source_transcript_candidate_id,
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
        "learning_refs": list(learning_refs),
        "reference_transcript_refs": list(reference_transcript_refs),
        "evaluation_report_refs": list(evaluation_report_refs),
        "regenerated_draft_refs": list(regenerated_draft_refs),
        "improvement_proposal_refs": list(improvement_proposal_refs),
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
        "review_required_stage": state.review_required_stage,
        "approval_ref": state.approval_ref,
        "approved_candidate_id": state.approved_candidate_id,
        "approved_source_transcript_candidate_id": state.approved_source_transcript_candidate_id,
        "transcript_ref": transcript_ref,
        "translation_ref": translation_ref,
        "translation_failure_ref": translation_failure_ref,
        "transcript_text": transcript.full_text if transcript is not None else None,
        "translation_text": translation.full_text if translation is not None else None,
        "trace_refs": list(trace_refs),
        "status": _delivery_status(state),
    }


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


def _artifact_refs(ref: str | None) -> tuple[str, ...]:
    if ref is None:
        return ()
    return (ref,)


def _ordered_memory_batch_refs(state, runtime) -> tuple[str, ...]:  # noqa: ANN001
    candidate_refs = tuple(
        job_path(state.job, "memory", "batches", f"{batch_id}.json")
        for batch_id in state.memory_batch_ids
    )
    existing_refs = tuple(ref for ref in candidate_refs if runtime.blob_store.exists(ref))
    if existing_refs:
        return existing_refs
    return _refs_for_fact(state, "memory_batch_staged")


def _ordered_memory_consolidation_refs(state, runtime) -> tuple[str, ...]:  # noqa: ANN001
    candidate_refs = tuple(
        job_path(state.job, "memory", "consolidations", f"consolidation-{batch_id}.json")
        for batch_id in state.memory_batch_ids
    )
    existing_refs = tuple(ref for ref in candidate_refs if runtime.blob_store.exists(ref))
    if existing_refs:
        return existing_refs
    return _refs_for_fact(state, "memory_batch_consolidated")


def _translation_failure_reasons(state: GraphState) -> list[str]:
    reasons: list[str] = []
    seen_reasons: set[str] = set()
    for fact in state.routing_facts:
        if fact.fact_type != "translation_variant_failed" or fact.source_ref is None:
            continue
        variant_id = fact.value.split(":", 1)[0]
        reason = f"{variant_id}: {fact.source_ref}"
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        reasons.append(reason)
    return reasons


def _delivery_status(state: GraphState) -> str:
    if state.approval_ref is not None:
        return "completed_after_human_review"
    if state.translation_failed:
        return "translation_failed"
    if state.human_review_required:
        return "human_review_required"
    return "completed"


def _write_json(runtime: WorkflowRuntime, key: str, payload: dict[str, Any]) -> None:
    runtime.blob_store.put_bytes(
        key,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_srt(runtime: WorkflowRuntime, key: str, payload: str) -> None:
    runtime.blob_store.put_bytes(key, payload.encode("utf-8"))
