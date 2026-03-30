"""Review nodes for deterministic prose review and parser-backed bundles."""

from __future__ import annotations

from uuid import uuid4

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    CandidatePreference,
    ReviewBundle,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.models.review import ReviewStage
from translation_agent.nodes.common import (
    TRANSCRIPT_REVIEW_STAGE,
    TRANSLATION_REVIEW_STAGE,
    build_memory_query,
    select_transcript_candidates,
    select_translation_candidates,
    transcript_candidate_key,
    transcript_review_key,
    translation_candidate_key,
    translation_review_key,
    write_model_artifact,
)
from translation_agent.review import (
    PARSER_VERSION,
    build_review_context,
    build_review_prompt,
    parse_reviewer_output,
    render_reviewer_output,
    reviewer_roles_for_stage,
)


def review_transcripts(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Generate transcript reviewer prose and persist parsed bundles."""

    candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.transcript_candidate_ids,
    )
    memory_bundle = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="review_transcripts",
            candidate_ids=state.transcript_candidate_ids,
        )
    )
    if not candidates:
        raise RuntimeError("review_transcripts requires at least one normalized candidate")

    review_ids = tuple(
        _review_stage(
            state=state,
            runtime=runtime,
            stage=TRANSCRIPT_REVIEW_STAGE,
            reviewer_role=spec.reviewer_role,
            candidates=candidates,
            memory_bundle=memory_bundle,
            final_transcript=None,
        )
        for spec in reviewer_roles_for_stage(TRANSCRIPT_REVIEW_STAGE)
    )
    first_review_ref = transcript_review_key(review_ids[0])
    return {
        "current_stage": "review_transcripts",
        "transcript_review_ids": review_ids,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_transcripts",
                fact_type="review_count",
                value=str(len(review_ids)),
                source_ref=first_review_ref,
            ),
        ),
    }


def review_translations(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Generate translation reviewer prose and persist parsed bundles."""

    candidates = select_translation_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.translation_candidate_ids,
    )
    memory_bundle = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="review_translations",
            candidate_ids=state.translation_candidate_ids,
        )
    )
    if not candidates:
        return {
            "current_stage": "review_translations",
            "translation_review_ids": (),
            "routing_facts": state.routing_facts
            + (
                RoutingFact(
                    stage="review_translations",
                    fact_type="review_skipped",
                    value="no_translation_candidates",
                    source_ref=None,
                ),
            ),
        }

    final_transcript = _load_final_transcript_candidate(state, runtime)
    review_ids = tuple(
        _review_stage(
            state=state,
            runtime=runtime,
            stage=TRANSLATION_REVIEW_STAGE,
            reviewer_role=spec.reviewer_role,
            candidates=candidates,
            memory_bundle=memory_bundle,
            final_transcript=final_transcript,
        )
        for spec in reviewer_roles_for_stage(TRANSLATION_REVIEW_STAGE)
    )
    first_review_ref = translation_review_key(review_ids[0])
    return {
        "current_stage": "review_translations",
        "translation_review_ids": review_ids,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_translations",
                fact_type="review_count",
                value=str(len(review_ids)),
                source_ref=first_review_ref,
            ),
        ),
    }


def _review_stage(
    *,
    state: GraphState,
    runtime: WorkflowRuntime,
    stage: ReviewStage,
    reviewer_role: str,
    candidates: list[TranscriptCandidate] | list[TranslationCandidate],
    memory_bundle,
    final_transcript: TranscriptCandidate | None,
) -> str:
    review_context = build_review_context(
        run_id=state.run_id,
        stage=stage,
        reviewer_role=reviewer_role,
        job=state.job,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        memory_bundle=memory_bundle,
    )
    candidate_refs = tuple(
        transcript_candidate_key(candidate.candidate_id)
        if stage == TRANSCRIPT_REVIEW_STAGE
        else translation_candidate_key(candidate.candidate_id)
        for candidate in candidates
    )
    raw_payload_refs = _raw_payload_refs(stage=stage, candidates=candidates)
    prompt_text = build_review_prompt(
        review_context,
        candidate_refs=candidate_refs,
        raw_payload_refs=raw_payload_refs,
        final_transcript_ref=(
            transcript_candidate_key(final_transcript.candidate_id)
            if final_transcript is not None
            else None
        ),
    )
    raw_review_text = render_reviewer_output(
        review_context,
        candidates=candidates,
        prompt_text=prompt_text,
        final_transcript=final_transcript,
    )
    parsed = parse_reviewer_output(raw_review_text)
    review = ReviewBundle(
        review_id=_review_id(stage),
        job_id=state.job.job_id,
        stage=stage,
        reviewer_role=reviewer_role,
        candidate_preferences=(
            CandidatePreference(
                candidate_id=parsed.winner_candidate_id,
                rank=1,
                rationale=parsed.why,
            ),
        )
        if parsed.winner_candidate_id is not None
        else (),
        confidence=parsed.confidence,
        raw_review_text=raw_review_text,
        quoted_evidence=parsed.quoted_evidence,
        issue_categories=tuple(dict.fromkeys(issue.category for issue in parsed.issues)),
        suggested_fixes=parsed.suggested_fixes,
        escalation_signal=parsed.escalation_signal,
        parser_version=PARSER_VERSION,
    )
    return _persist_review(runtime, review)


def _load_final_transcript_candidate(
    state: GraphState,
    runtime: WorkflowRuntime,
) -> TranscriptCandidate:
    if state.final_transcript_candidate_id is None:
        raise RuntimeError("translation review requires a final transcript candidate")
    candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=(state.final_transcript_candidate_id,),
    )
    if not candidates:
        raise RuntimeError("final transcript candidate not found for translation review")
    return candidates[0]


def _review_id(stage: ReviewStage) -> str:
    prefix = "rev-tr" if stage == TRANSCRIPT_REVIEW_STAGE else "rev-tl"
    return f"{prefix}-{uuid4().hex}"


def _persist_review(runtime: WorkflowRuntime, review: ReviewBundle) -> str:
    key = (
        transcript_review_key(review.review_id)
        if review.stage == TRANSCRIPT_REVIEW_STAGE
        else translation_review_key(review.review_id)
    )
    write_model_artifact(runtime, key, review)
    return review.review_id


def _raw_payload_refs(
    *,
    stage: ReviewStage,
    candidates: list[TranscriptCandidate] | list[TranslationCandidate],
) -> tuple[str, ...]:
    if stage == TRANSCRIPT_REVIEW_STAGE:
        return tuple(
            candidate.raw_payload_ref
            for candidate in candidates
            if isinstance(candidate, TranscriptCandidate) and candidate.raw_payload_ref is not None
        )
    return tuple(
        candidate.raw_response_ref
        for candidate in candidates
        if isinstance(candidate, TranslationCandidate) and candidate.raw_response_ref is not None
    )
