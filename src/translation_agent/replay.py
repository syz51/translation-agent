"""Replay helpers that reconstruct adjudication from persisted artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import cast

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.models import (
    AdjudicationContext,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    JobContext,
    MemoryBundle,
    ReviewBundle,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.models.review import DisagreementBucket, ReviewStage
from translation_agent.nodes.common import read_model_artifact, translation_investigation_key
from translation_agent.review import AdjudicationOutcome, adjudicate_reviews


@dataclass(frozen=True, slots=True)
class ReplayAdjudicationRequest:
    """Ref-only replay input for deterministic adjudication."""

    run_id: str
    job: JobContext
    stage: ReviewStage
    candidate_refs: tuple[str, ...]
    review_refs: tuple[str, ...]
    memory_ref: str
    content_risk_class: str = "standard"


@dataclass(frozen=True, slots=True)
class ReplayAdjudicationResult:
    """Replayed adjudication outcome plus the reconstructed decision model."""

    outcome: AdjudicationOutcome
    decision: FinalTranscriptDecision | FinalTranslationDecision
    disagreement_bucket: DisagreementBucket


def replay_adjudication(
    runtime: WorkflowRuntime,
    request: ReplayAdjudicationRequest,
) -> ReplayAdjudicationResult:
    """Rebuild a final adjudication from stored normalized inputs and memory refs."""

    memory_bundle = read_model_artifact(runtime, request.memory_ref, MemoryBundle)
    reviews = tuple(
        read_model_artifact(runtime, review_ref, ReviewBundle) for review_ref in request.review_refs
    )

    if request.stage == "transcript":
        candidates = tuple(
            read_model_artifact(runtime, candidate_ref, TranscriptCandidate)
            for candidate_ref in request.candidate_refs
        )
    else:
        candidates = tuple(
            read_model_artifact(runtime, candidate_ref, TranslationCandidate)
            for candidate_ref in request.candidate_refs
        )

    context = AdjudicationContext(
        run_id=request.run_id,
        stage=request.stage,
        job=request.job,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        review_ids=tuple(review.review_id for review in reviews),
        memory_bundle=memory_bundle,
        content_risk_class=request.content_risk_class,
    )
    outcome = adjudicate_reviews(candidates=candidates, reviews=reviews, context=context)

    if request.stage == "transcript":
        decision = FinalTranscriptDecision(
            job_id=request.job.job_id,
            winner_candidate_id=outcome.winner_candidate_id,
            decision_mode=outcome.decision_mode,
            decision_confidence=outcome.decision_confidence,
            rationale_summary=outcome.rationale_summary,
            review_refs=tuple(review.review_id for review in reviews),
            investigation_ref=None,
            disagreement_bucket=outcome.disagreement_bucket,
            adjudication_scorecard=_scorecard(
                outcome,
                candidate_count=len(candidates),
                content_risk_class=request.content_risk_class,
            ),
            escalated=outcome.escalated,
            human_review_required=outcome.human_review_required,
        )
    else:
        translation_candidates = cast(tuple[TranslationCandidate, ...], candidates)
        outcome = _replay_translation_timeout_escalation(
            runtime=runtime,
            job=request.job,
            outcome=outcome,
        )
        winner = next(
            (
                candidate
                for candidate in translation_candidates
                if candidate.candidate_id == outcome.winner_candidate_id
            ),
            None,
        )
        decision = FinalTranslationDecision(
            job_id=request.job.job_id,
            winner_candidate_id=outcome.winner_candidate_id,
            decision_mode=outcome.decision_mode,
            decision_confidence=outcome.decision_confidence,
            rationale_summary=outcome.rationale_summary,
            review_refs=tuple(review.review_id for review in reviews),
            investigation_ref=None,
            disagreement_bucket=outcome.disagreement_bucket,
            adjudication_scorecard=_scorecard(
                outcome,
                candidate_count=len(translation_candidates),
                content_risk_class=request.content_risk_class,
            ),
            escalated=outcome.escalated,
            human_review_required=outcome.human_review_required,
            winner_model_id=winner.model_id if winner is not None else None,
            prompt_variant_winner=winner.prompt_variant_id if winner is not None else None,
            prompt_version_winner=winner.prompt_version if winner is not None else None,
        )

    return ReplayAdjudicationResult(
        outcome=outcome,
        decision=decision,
        disagreement_bucket=outcome.disagreement_bucket,
    )


def _replay_translation_timeout_escalation(
    *,
    runtime: WorkflowRuntime,
    job: JobContext,
    outcome: AdjudicationOutcome,
) -> AdjudicationOutcome:
    try:
        payload = json.loads(
            runtime.blob_store.read_bytes(translation_investigation_key(job)).decode("utf-8")
        )
    except Exception:
        return outcome
    if not isinstance(payload, dict) or payload.get("status") != "timed_out":
        return outcome
    return replace(
        outcome,
        winner_candidate_id=None,
        decision_mode="human_review",
        decision_confidence=0.0,
        rationale_summary=(
            "Translation conflict investigation timed out after medium disagreement, "
            "so the run escalated to human review."
        ),
        disagreement_bucket="unresolved",
        escalated=True,
        human_review_required=True,
        investigation_payload=payload,
    )


def _scorecard(
    outcome: AdjudicationOutcome,
    *,
    candidate_count: int,
    content_risk_class: str,
):
    assessment = outcome.assessment
    from translation_agent.models import AdjudicationScorecard

    return AdjudicationScorecard(
        candidate_count=candidate_count,
        preferred_candidate_id=assessment.preferred_candidate_id,
        average_confidence=assessment.average_confidence,
        confidence_spread=assessment.confidence_spread,
        contradictory_evidence_count=assessment.contradictory_evidence_count,
        highest_issue_severity=assessment.highest_issue_severity,
        winner_mismatch=assessment.winner_mismatch,
        escalation_signal_count=assessment.escalation_signal_count,
        total_score=assessment.total_score,
        content_risk_class=content_risk_class,
    )
