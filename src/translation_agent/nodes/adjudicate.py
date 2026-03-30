"""Adjudication nodes for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import FinalTranscriptDecision, FinalTranslationDecision
from translation_agent.nodes.common import (
    TRANSCRIPT_REVIEW_STAGE,
    TRANSLATION_REVIEW_STAGE,
    load_reviews,
    select_transcript_candidates,
    select_translation_candidates,
    transcript_decision_key,
    translation_decision_key,
    translation_sort_key,
    write_model_artifact,
)


def adjudicate_transcript(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Create a deterministic transcript decision from normalized reviews."""

    candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.transcript_candidate_ids,
    )
    reviews = load_reviews(
        runtime,
        stage=TRANSCRIPT_REVIEW_STAGE,
        review_ids=state.transcript_review_ids,
    )
    preferred = _preferred_candidate_id(
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        preferred_review_ids=tuple(
            review.candidate_preferences[0].candidate_id
            for review in reviews
            if review.candidate_preferences
        ),
        fallback=candidates[0].candidate_id,
    )

    human_review_required = runtime.scenario == "transcript_escalation"
    decision_mode = "human_review" if human_review_required else "automatic_finalize"
    decision_confidence = 0.35 if human_review_required else (0.74 if len(candidates) == 1 else 0.9)
    decision = FinalTranscriptDecision(
        job_id=state.job.job_id,
        winner_candidate_id=None if human_review_required else preferred,
        decision_mode=decision_mode,
        decision_confidence=decision_confidence,
        rationale_summary=(
            "Transcript disagreement stayed unresolved in the dry-run path."
            if human_review_required
            else (
                "Single surviving transcript candidate carried the run."
                if len(candidates) == 1
                else "Transcript reviewers aligned on the canonical candidate."
            )
        ),
        review_refs=state.transcript_review_ids,
        escalated=human_review_required or len(candidates) == 1,
        human_review_required=human_review_required,
    )
    runtime.decision_store.save_transcript_decision(decision)
    decision_ref = write_model_artifact(
        runtime, transcript_decision_key(state.job.job_id), decision
    )

    return {
        "current_stage": "adjudicate_transcript",
        "final_transcript_candidate_id": preferred,
        "final_transcript_decision_ref": decision_ref,
        "pending_memory_source_stage": "transcript_adjudication",
        "escalation_pending": decision.escalated,
        "human_review_required": decision.human_review_required,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="adjudicate_transcript",
                fact_type="decision_mode",
                value=decision.decision_mode,
                source_ref=decision_ref,
            ),
        ),
    }


def adjudicate_translation(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Create a deterministic translation decision or recoverable failure state."""

    candidates = select_translation_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.translation_candidate_ids,
    )
    reviews = load_reviews(
        runtime,
        stage=TRANSLATION_REVIEW_STAGE,
        review_ids=state.translation_review_ids,
    )
    human_review_required = runtime.scenario == "translation_escalation"

    if not candidates:
        decision = FinalTranslationDecision(
            job_id=state.job.job_id,
            winner_candidate_id=None,
            decision_mode="automatic_finalize",
            decision_confidence=0.0,
            rationale_summary="All translation variants failed; transcript preserved for recovery.",
            review_refs=(),
            escalated=False,
            human_review_required=False,
            prompt_variant_winner=None,
            prompt_version_winner=None,
        )
        translation_failed = True
        winner_candidate_id = None
    else:
        preferred = _preferred_candidate_id(
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            preferred_review_ids=tuple(
                review.candidate_preferences[0].candidate_id
                for review in reviews
                if review.candidate_preferences
            ),
            fallback=sorted(candidates, key=translation_sort_key)[0].candidate_id,
        )
        winner = next(candidate for candidate in candidates if candidate.candidate_id == preferred)
        decision_mode = "human_review" if human_review_required else "automatic_finalize"
        decision_confidence = (
            0.42 if human_review_required else (0.7 if len(candidates) == 1 else 0.88)
        )
        decision = FinalTranslationDecision(
            job_id=state.job.job_id,
            winner_candidate_id=None if human_review_required else preferred,
            decision_mode=decision_mode,
            decision_confidence=decision_confidence,
            rationale_summary=(
                "Translation disagreement stayed unresolved in the dry-run path."
                if human_review_required
                else (
                    "One surviving translation variant remained publishable."
                    if len(candidates) == 1
                    else "Translation reviewers aligned on the winning variant."
                )
            ),
            review_refs=state.translation_review_ids,
            escalated=human_review_required,
            human_review_required=human_review_required,
            prompt_variant_winner=None if human_review_required else winner.prompt_variant_id,
            prompt_version_winner=None if human_review_required else winner.prompt_version,
        )
        translation_failed = False
        winner_candidate_id = preferred

    runtime.decision_store.save_translation_decision(decision)
    decision_ref = write_model_artifact(
        runtime, translation_decision_key(state.job.job_id), decision
    )

    return {
        "current_stage": "adjudicate_translation",
        "final_translation_candidate_id": winner_candidate_id,
        "final_translation_decision_ref": decision_ref,
        "pending_memory_source_stage": "translation_adjudication",
        "escalation_pending": decision.escalated,
        "human_review_required": decision.human_review_required,
        "translation_failed": translation_failed,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="adjudicate_translation",
                fact_type="decision_mode",
                value=decision.decision_mode,
                source_ref=decision_ref,
            ),
        ),
    }


def _preferred_candidate_id(
    *,
    candidate_ids: tuple[str, ...],
    preferred_review_ids: tuple[str, ...],
    fallback: str,
) -> str:
    if not candidate_ids:
        return fallback

    counts = {candidate_id: 0 for candidate_id in candidate_ids}
    for candidate_id in preferred_review_ids:
        if candidate_id in counts:
            counts[candidate_id] += 1
    return max(
        sorted(counts),
        key=lambda candidate_id: (counts[candidate_id], -len(candidate_id)),
    )
