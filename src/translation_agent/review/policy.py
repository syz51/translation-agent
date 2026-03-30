"""Deterministic disagreement scoring and non-recursive escalation policy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from translation_agent.models import AdjudicationContext, ReviewBundle
from translation_agent.models.review import DecisionMode, ReviewStage
from translation_agent.review.parser import IssueSeverity, ParsedReview, parse_reviewer_output

_SEVERITY_WEIGHTS = {
    "minor": 0.5,
    "major": 1.2,
    "critical": 2.6,
}
_RISK_MULTIPLIERS = {
    "standard": 1.0,
    "high": 1.2,
    "regulated": 1.35,
    "legal": 1.45,
    "medical": 1.5,
    "critical": 1.75,
}


@dataclass(frozen=True, slots=True)
class DisagreementAssessment:
    """Scored disagreement summary used by transcript and translation adjudication."""

    decision_mode: DecisionMode
    preferred_candidate_id: str | None
    average_confidence: float
    confidence_spread: float
    contradictory_evidence_count: int
    highest_issue_severity: IssueSeverity
    winner_mismatch: bool
    escalation_signal_count: int
    total_score: float
    escalated: bool
    human_review_required: bool


@dataclass(frozen=True, slots=True)
class AdjudicationOutcome:
    """Pure adjudication result used by unit tests and workflow nodes."""

    decision_mode: DecisionMode
    winner_candidate_id: str | None
    decision_confidence: float
    rationale_summary: str
    human_review_required: bool
    escalated: bool
    investigation_payload: dict[str, object] | None
    disagreement_bucket: str
    assessment: DisagreementAssessment


def assess_review_disagreement(
    parsed_reviews: tuple[ParsedReview, ...],
    *,
    candidate_ids: tuple[str, ...],
    fallback_candidate_id: str | None,
    content_risk_class: str = "standard",
) -> DisagreementAssessment:
    """Choose the adjudication path from parsed reviewer outputs."""

    winners = [
        review.winner_candidate_id for review in parsed_reviews if review.winner_candidate_id
    ]
    preferred_candidate_id = _preferred_candidate_id(parsed_reviews, fallback_candidate_id)
    if preferred_candidate_id not in set(candidate_ids):
        preferred_candidate_id = fallback_candidate_id

    confidences = [review.confidence for review in parsed_reviews]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    confidence_spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0.0
    contradictory_evidence_count = _contradictory_evidence_count(parsed_reviews)
    highest_issue_severity = _highest_issue_severity(parsed_reviews)
    winner_mismatch = len(set(winners)) > 1
    escalation_signal_count = sum(1 for review in parsed_reviews if review.escalation_signal)

    base_score = (
        (2.4 if winner_mismatch else 0.0)
        + confidence_spread * 2.0
        + contradictory_evidence_count * 1.3
        + _SEVERITY_WEIGHTS[highest_issue_severity]
        + (0.8 if escalation_signal_count else 0.0)
    )
    total_score = round(base_score * _RISK_MULTIPLIERS.get(content_risk_class, 1.0), 4)

    human_review_required = (
        preferred_candidate_id is None
        or total_score >= 8.0
        or (
            escalation_signal_count == len(parsed_reviews)
            and winner_mismatch
            and highest_issue_severity == "critical"
        )
        or (content_risk_class in {"legal", "medical", "critical"} and total_score >= 6.4)
    )
    if human_review_required:
        decision_mode = "human_review"
    elif total_score >= 5.2:
        decision_mode = "stronger_adjudicator"
    elif total_score >= 3.0:
        decision_mode = "conflict_investigation"
    else:
        decision_mode = "automatic_finalize"

    return DisagreementAssessment(
        decision_mode=decision_mode,
        preferred_candidate_id=preferred_candidate_id,
        average_confidence=round(average_confidence, 4),
        confidence_spread=round(confidence_spread, 4),
        contradictory_evidence_count=contradictory_evidence_count,
        highest_issue_severity=highest_issue_severity,
        winner_mismatch=winner_mismatch,
        escalation_signal_count=escalation_signal_count,
        total_score=total_score,
        escalated=decision_mode != "automatic_finalize",
        human_review_required=human_review_required,
    )


def adjudicate_reviews(
    *,
    candidates,
    reviews: tuple[ReviewBundle, ...],
    context: AdjudicationContext,
) -> AdjudicationOutcome:
    """Apply deterministic review scoring outside the graph runtime."""

    fallback_candidate_id = candidates[0].candidate_id if candidates else None
    parsed_reviews = tuple(parse_reviewer_output(review.raw_review_text) for review in reviews)
    assessment = assess_review_disagreement(
        parsed_reviews,
        candidate_ids=context.candidate_ids,
        fallback_candidate_id=fallback_candidate_id,
        content_risk_class=context.content_risk_class,
    )
    candidate_count = len(tuple(candidates))
    single_candidate_escalation = candidate_count == 1 and context.stage == "transcript"
    investigation_payload = None
    if assessment.decision_mode != "automatic_finalize" or single_candidate_escalation:
        investigation_payload = {
            "stage": context.stage,
            "candidate_ids": list(context.candidate_ids),
            "review_ids": list(context.review_ids),
            "preferred_candidate_id": assessment.preferred_candidate_id,
            "winner_mismatch": assessment.winner_mismatch,
            "confidence_spread": assessment.confidence_spread,
            "contradictory_evidence_count": assessment.contradictory_evidence_count,
            "highest_issue_severity": assessment.highest_issue_severity,
            "total_score": assessment.total_score,
            "content_risk_class": context.content_risk_class,
        }
    decision_confidence = _decision_confidence(
        assessment=assessment,
        candidate_count=candidate_count,
    )
    rationale_summary = _rationale_summary(
        stage=context.stage,
        decision_mode=assessment.decision_mode,
        candidate_count=candidate_count,
        assessment=assessment,
    )
    return AdjudicationOutcome(
        decision_mode=assessment.decision_mode,
        winner_candidate_id=(
            None if assessment.human_review_required else assessment.preferred_candidate_id
        ),
        decision_confidence=decision_confidence,
        rationale_summary=rationale_summary,
        human_review_required=assessment.human_review_required,
        escalated=assessment.escalated or single_candidate_escalation,
        investigation_payload=investigation_payload,
        disagreement_bucket=_bucket_for_score(assessment),
        assessment=assessment,
    )


def content_risk_class_for_scenario(scenario: str) -> str:
    """Map deterministic dry-run scenarios onto adjudication risk classes."""

    mapping = {
        "transcript_escalation": "critical",
        "translation_high_risk": "high",
        "translation_human_review": "critical",
        "translation_escalation": "critical",
    }
    return mapping.get(scenario, "standard")


def _preferred_candidate_id(
    parsed_reviews: tuple[ParsedReview, ...],
    fallback_candidate_id: str | None,
) -> str | None:
    winner_scores: dict[str, tuple[int, float]] = {}
    for review in parsed_reviews:
        if review.winner_candidate_id is None:
            continue
        count, confidence = winner_scores.get(review.winner_candidate_id, (0, 0.0))
        winner_scores[review.winner_candidate_id] = (count + 1, confidence + review.confidence)
    if not winner_scores:
        return fallback_candidate_id
    ranked = sorted(winner_scores.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
    return ranked[0][0]


def _contradictory_evidence_count(parsed_reviews: tuple[ParsedReview, ...]) -> int:
    evidence_by_key: dict[str, set[str]] = defaultdict(set)
    for review in parsed_reviews:
        for evidence in review.quoted_evidence:
            evidence_key = evidence.segment_id or evidence.quote.lower().strip()
            candidate_id = evidence.candidate_id or review.winner_candidate_id
            if candidate_id is not None:
                evidence_by_key[evidence_key].add(candidate_id)
    return sum(1 for candidate_ids in evidence_by_key.values() if len(candidate_ids) > 1)


def _highest_issue_severity(parsed_reviews: tuple[ParsedReview, ...]) -> IssueSeverity:
    highest = "minor"
    for review in parsed_reviews:
        for issue in review.issues:
            if issue.severity == "critical":
                return "critical"
            if issue.severity == "major":
                highest = "major"
    return highest


def _bucket_for_score(assessment: DisagreementAssessment) -> str:
    if assessment.decision_mode == "human_review":
        return "unresolved"
    if assessment.decision_mode == "stronger_adjudicator":
        return "high"
    if assessment.decision_mode == "conflict_investigation":
        return "medium"
    return "low"


def _decision_confidence(
    *,
    assessment: DisagreementAssessment,
    candidate_count: int,
) -> float:
    penalty = (
        assessment.confidence_spread * 0.35
        + assessment.contradictory_evidence_count * 0.08
        + {"minor": 0.05, "major": 0.18, "critical": 0.32}[assessment.highest_issue_severity]
    )
    if assessment.decision_mode == "conflict_investigation":
        penalty += 0.12
    elif assessment.decision_mode == "stronger_adjudicator":
        penalty += 0.2
    elif assessment.decision_mode == "human_review":
        penalty += 0.42
    if candidate_count == 1:
        penalty += 0.1
    confidence = assessment.average_confidence - penalty
    return round(max(0.0, min(confidence, 0.99)), 4)


def _rationale_summary(
    *,
    stage: ReviewStage,
    decision_mode: DecisionMode,
    candidate_count: int,
    assessment: DisagreementAssessment,
) -> str:
    stage_label = "Transcript" if stage == "transcript" else "Translation"
    if decision_mode == "automatic_finalize":
        if candidate_count == 1:
            return (
                f"{stage_label} finalized from the only surviving candidate with reduced "
                "confidence."
            )
        return (
            f"{stage_label} reviewers stayed within the low-disagreement band and finalized "
            f"automatically after scoring {assessment.total_score:.2f} points."
        )
    if decision_mode == "conflict_investigation":
        return (
            f"{stage_label} disagreement landed in the medium band, so conflict investigation "
            "resolved the winner without reopening the graph."
        )
    if decision_mode == "stronger_adjudicator":
        return (
            f"{stage_label} disagreement was high enough to invoke the stronger adjudicator hook "
            "before finalization."
        )
    return (
        f"{stage_label} disagreement remained unresolved after deterministic scoring and now "
        "requires human review."
    )
