"""Review nodes for the deterministic dry-run workflow."""

from __future__ import annotations

from uuid import uuid4

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    CandidatePreference,
    QuotedEvidence,
    ReviewBundle,
    SuggestedFix,
)
from translation_agent.nodes.common import (
    TRANSCRIPT_REVIEW_STAGE,
    TRANSLATION_REVIEW_STAGE,
    build_memory_query,
    select_transcript_candidates,
    select_translation_candidates,
    transcript_review_key,
    transcript_sort_key,
    translation_review_key,
    translation_sort_key,
    write_model_artifact,
)


def review_transcripts(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Produce deterministic transcript review bundles from normalized candidates."""

    candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.transcript_candidate_ids,
    )
    _ = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="review_transcripts",
            candidate_ids=state.transcript_candidate_ids,
        )
    )
    if not candidates:
        raise RuntimeError("review_transcripts requires at least one normalized candidate")

    preferred = sorted(candidates, key=transcript_sort_key)[0]
    review_ids = (
        _persist_review(
            runtime,
            ReviewBundle(
                review_id=f"rev-tr-{uuid4().hex}",
                job_id=state.job.job_id,
                stage=TRANSCRIPT_REVIEW_STAGE,
                reviewer_role="accuracy_reviewer",
                candidate_preferences=(
                    CandidatePreference(
                        candidate_id=preferred.candidate_id,
                        rank=1,
                        rationale="Most stable transcript candidate.",
                    ),
                ),
                confidence=0.91 if len(candidates) > 1 else 0.72,
                raw_review_text=f"Winner: {preferred.candidate_id}",
                quoted_evidence=(
                    QuotedEvidence(
                        quote=preferred.full_text,
                        candidate_id=preferred.candidate_id,
                    ),
                ),
                issue_categories=("accuracy",),
                suggested_fixes=(
                    SuggestedFix(
                        issue_category="accuracy",
                        candidate_id=preferred.candidate_id,
                        description="Retain deterministic speaker labels.",
                    ),
                ),
                escalation_signal=runtime.scenario == "transcript_escalation",
                parser_version="phase-2",
            ),
        ),
        _persist_review(
            runtime,
            ReviewBundle(
                review_id=f"rev-tr-{uuid4().hex}",
                job_id=state.job.job_id,
                stage=TRANSCRIPT_REVIEW_STAGE,
                reviewer_role="coherence_reviewer",
                candidate_preferences=(
                    CandidatePreference(
                        candidate_id=preferred.candidate_id,
                        rank=1,
                        rationale="Most coherent transcript for dry-run publishing.",
                    ),
                ),
                confidence=0.89 if len(candidates) > 1 else 0.68,
                raw_review_text=f"Winner: {preferred.candidate_id}",
                quoted_evidence=(
                    QuotedEvidence(
                        quote=preferred.full_text,
                        candidate_id=preferred.candidate_id,
                    ),
                ),
                issue_categories=("formatting",),
                suggested_fixes=(
                    SuggestedFix(
                        issue_category="formatting",
                        candidate_id=preferred.candidate_id,
                        description="Preserve sentence spacing during export.",
                    ),
                ),
                escalation_signal=runtime.scenario == "transcript_escalation",
                parser_version="phase-2",
            ),
        ),
    )

    return {
        "current_stage": "review_transcripts",
        "transcript_review_ids": review_ids,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_transcripts",
                fact_type="review_count",
                value=str(len(review_ids)),
                source_ref=transcript_review_key(review_ids[0]),
            ),
        ),
    }


def review_translations(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Produce deterministic translation review bundles from normalized candidates."""

    candidates = select_translation_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.translation_candidate_ids,
    )
    _ = runtime.memory_recall_backend.recall_memory(
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

    ordered_candidates = sorted(candidates, key=translation_sort_key)
    first = ordered_candidates[0]
    second = ordered_candidates[-1]
    disagreement = runtime.scenario == "translation_escalation" and len(ordered_candidates) > 1
    winning_candidate_id = second.candidate_id if disagreement else first.candidate_id
    winning_text = second.full_text if disagreement else first.full_text

    review_ids = (
        _persist_review(
            runtime,
            ReviewBundle(
                review_id=f"rev-tl-{uuid4().hex}",
                job_id=state.job.job_id,
                stage=TRANSLATION_REVIEW_STAGE,
                reviewer_role="faithfulness_reviewer",
                candidate_preferences=(
                    CandidatePreference(
                        candidate_id=first.candidate_id,
                        rank=1,
                        rationale="Best preserves the transcript semantics.",
                    ),
                ),
                confidence=0.9 if len(ordered_candidates) > 1 else 0.7,
                raw_review_text=f"Winner: {first.candidate_id}",
                quoted_evidence=(
                    QuotedEvidence(
                        quote=first.full_text,
                        candidate_id=first.candidate_id,
                    ),
                ),
                issue_categories=("faithfulness",),
                suggested_fixes=(
                    SuggestedFix(
                        issue_category="faithfulness",
                        candidate_id=first.candidate_id,
                        description="Preserve product terminology.",
                    ),
                ),
                escalation_signal=disagreement,
                parser_version="phase-2",
            ),
        ),
        _persist_review(
            runtime,
            ReviewBundle(
                review_id=f"rev-tl-{uuid4().hex}",
                job_id=state.job.job_id,
                stage=TRANSLATION_REVIEW_STAGE,
                reviewer_role="style_reviewer",
                candidate_preferences=(
                    CandidatePreference(
                        candidate_id=winning_candidate_id,
                        rank=1,
                        rationale="Most natural phrasing for the target language.",
                    ),
                ),
                confidence=0.84 if len(ordered_candidates) > 1 else 0.66,
                raw_review_text=f"Winner: {winning_candidate_id}",
                quoted_evidence=(
                    QuotedEvidence(
                        quote=winning_text,
                        candidate_id=winning_candidate_id,
                    ),
                ),
                issue_categories=("style",),
                suggested_fixes=(
                    SuggestedFix(
                        issue_category="style",
                        candidate_id=winning_candidate_id,
                        description="Keep audience tone consistent.",
                    ),
                ),
                escalation_signal=disagreement,
                parser_version="phase-2",
            ),
        ),
    )

    return {
        "current_stage": "review_translations",
        "translation_review_ids": review_ids,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_translations",
                fact_type="review_count",
                value=str(len(review_ids)),
                source_ref=translation_review_key(review_ids[0]),
            ),
        ),
    }


def _persist_review(runtime: WorkflowRuntime, review: ReviewBundle) -> str:
    key = (
        transcript_review_key(review.review_id)
        if review.stage == TRANSCRIPT_REVIEW_STAGE
        else translation_review_key(review.review_id)
    )
    write_model_artifact(runtime, key, review)
    return review.review_id
