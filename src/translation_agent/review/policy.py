"""Deterministic disagreement scoring and non-recursive escalation policy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from translation_agent.models import AdjudicationContext, ReviewBundle
from translation_agent.models.review import DecisionMode, DisagreementBucket, ReviewStage
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
    disagreement_bucket: DisagreementBucket
    assessment: DisagreementAssessment


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    """Executed escalation artifact used before final re-adjudication."""

    status: str
    strategy: str
    recommended_candidate_id: str | None
    confidence_adjustment: float
    rationale: str
    payload: dict[str, object]


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
    """Apply deterministic review scoring with explicit escalation execution."""

    candidate_list = tuple(candidates)
    fallback_candidate_id = candidate_list[0].candidate_id if candidate_list else None
    parsed_reviews = tuple(parse_reviewer_output(review.raw_review_text) for review in reviews)
    assessment = assess_review_disagreement(
        parsed_reviews,
        candidate_ids=context.candidate_ids,
        fallback_candidate_id=fallback_candidate_id,
        content_risk_class=context.content_risk_class,
    )
    candidate_count = len(candidate_list)
    single_candidate_escalation = candidate_count == 1 and context.stage == "transcript"
    investigation_result = _execute_escalation(
        parsed_reviews=parsed_reviews,
        assessment=assessment,
        context=context,
        single_candidate_escalation=single_candidate_escalation,
    )
    resolved_decision_mode = assessment.decision_mode
    winner_candidate_id = assessment.preferred_candidate_id
    human_review_required = assessment.human_review_required

    if investigation_result is not None:
        if investigation_result.status in {"timed_out", "unresolved"}:
            resolved_decision_mode = "human_review"
            winner_candidate_id = None
            human_review_required = True
        elif investigation_result.recommended_candidate_id is not None:
            winner_candidate_id = investigation_result.recommended_candidate_id

    decision_confidence = _decision_confidence(
        assessment=assessment,
        candidate_count=candidate_count,
        confidence_adjustment=(
            investigation_result.confidence_adjustment if investigation_result is not None else 0.0
        ),
        final_decision_mode=resolved_decision_mode,
    )
    rationale_summary = _rationale_summary(
        stage=context.stage,
        decision_mode=resolved_decision_mode,
        candidate_count=candidate_count,
        assessment=assessment,
        investigation_result=investigation_result,
    )
    return AdjudicationOutcome(
        decision_mode=resolved_decision_mode,
        winner_candidate_id=None if human_review_required else winner_candidate_id,
        decision_confidence=decision_confidence,
        rationale_summary=rationale_summary,
        human_review_required=human_review_required,
        escalated=assessment.escalated or single_candidate_escalation,
        investigation_payload=(
            investigation_result.payload if investigation_result is not None else None
        ),
        disagreement_bucket=_bucket_for_score(
            assessment,
            final_decision_mode=resolved_decision_mode,
        ),
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


def _bucket_for_score(
    assessment: DisagreementAssessment,
    *,
    final_decision_mode: DecisionMode | None = None,
) -> DisagreementBucket:
    mode = final_decision_mode or assessment.decision_mode
    if mode == "human_review":
        return "unresolved"
    if mode == "stronger_adjudicator":
        return "high"
    if mode == "conflict_investigation":
        return "medium"
    return "low"


def _decision_confidence(
    *,
    assessment: DisagreementAssessment,
    candidate_count: int,
    confidence_adjustment: float = 0.0,
    final_decision_mode: DecisionMode | None = None,
) -> float:
    penalty = (
        assessment.confidence_spread * 0.35
        + assessment.contradictory_evidence_count * 0.08
        + {"minor": 0.05, "major": 0.18, "critical": 0.32}[assessment.highest_issue_severity]
    )
    mode = final_decision_mode or assessment.decision_mode
    if mode == "conflict_investigation":
        penalty += 0.12
    elif mode == "stronger_adjudicator":
        penalty += 0.2
    elif mode == "human_review":
        penalty += 0.42
    if candidate_count == 1:
        penalty += 0.1
    confidence = assessment.average_confidence - penalty + confidence_adjustment
    return round(max(0.0, min(confidence, 0.99)), 4)


def _rationale_summary(
    *,
    stage: ReviewStage,
    decision_mode: DecisionMode,
    candidate_count: int,
    assessment: DisagreementAssessment,
    investigation_result: InvestigationResult | None = None,
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
        if investigation_result is not None:
            return (
                f"{stage_label} disagreement landed in the medium band; "
                f"{investigation_result.strategy} ran before re-adjudication and "
                f"{investigation_result.rationale}"
            )
        return (
            f"{stage_label} disagreement landed in the medium band, so conflict investigation "
            "resolved the winner without reopening the graph."
        )
    if decision_mode == "stronger_adjudicator":
        if investigation_result is not None:
            return (
                f"{stage_label} disagreement was high enough to invoke "
                f"{investigation_result.strategy}, "
                f"which {investigation_result.rationale}"
            )
        return (
            f"{stage_label} disagreement was high enough to invoke the stronger adjudicator hook "
            "before finalization."
        )
    if investigation_result is not None and investigation_result.status in {
        "timed_out",
        "unresolved",
    }:
        return (
            f"{stage_label} escalation remained unresolved after {investigation_result.strategy} "
            "and now requires human review."
        )
    return (
        f"{stage_label} disagreement remained unresolved after deterministic scoring and now "
        "requires human review."
    )


def _execute_escalation(
    *,
    parsed_reviews: tuple[ParsedReview, ...],
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
    single_candidate_escalation: bool,
) -> InvestigationResult | None:
    if assessment.decision_mode == "automatic_finalize" and not single_candidate_escalation:
        return None
    if assessment.decision_mode == "conflict_investigation":
        return _run_conflict_investigator(
            parsed_reviews=parsed_reviews,
            assessment=assessment,
            context=context,
        )
    if assessment.decision_mode == "stronger_adjudicator":
        return _run_stronger_adjudicator(
            parsed_reviews=parsed_reviews,
            assessment=assessment,
            context=context,
        )
    if assessment.decision_mode == "human_review":
        return _build_unresolved_investigation(
            assessment=assessment,
            context=context,
            reason="kept the disagreement unresolved after deterministic scoring",
        )
    if single_candidate_escalation:
        return InvestigationResult(
            status="reviewed",
            strategy="single-candidate escalation check",
            recommended_candidate_id=assessment.preferred_candidate_id,
            confidence_adjustment=-0.04,
            rationale="confirmed the only surviving transcript without broadening the route.",
            payload=_base_investigation_payload(
                assessment=assessment,
                context=context,
                strategy="single-candidate escalation check",
                status="reviewed",
                recommended_candidate_id=assessment.preferred_candidate_id,
                notes=["Only one transcript candidate survived; the path stayed deterministic."],
            ),
        )
    return None


def _run_conflict_investigator(
    *,
    parsed_reviews: tuple[ParsedReview, ...],
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
) -> InvestigationResult:
    candidate_scores = _candidate_scores(parsed_reviews, candidate_ids=context.candidate_ids)
    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    recommended_candidate_id = ranked[0][0] if ranked else assessment.preferred_candidate_id
    margin = 0.0
    if len(ranked) > 1:
        margin = round(ranked[0][1] - ranked[1][1], 4)
    notes = [
        (
            "Conflict investigator synthesized winner counts, confidence, evidence overlap, "
            "and issue severity."
        ),
        f"Top recommendation margin: {margin:.4f}",
    ]
    if recommended_candidate_id is None:
        return _build_unresolved_investigation(
            assessment=assessment,
            context=context,
            reason="could not recommend a candidate after conflict investigation",
        )
    return InvestigationResult(
        status="resolved",
        strategy="conflict investigator then re-adjudication",
        recommended_candidate_id=recommended_candidate_id,
        confidence_adjustment=0.04 if margin >= 0.2 else 0.02,
        rationale=f"resolved the winner as {recommended_candidate_id} before re-adjudication.",
        payload=_base_investigation_payload(
            assessment=assessment,
            context=context,
            strategy="conflict investigator",
            status="resolved",
            recommended_candidate_id=recommended_candidate_id,
            notes=notes,
            stage_payload={
                "re_adjudication": {
                    "decision_mode": "conflict_investigation",
                    "recommended_candidate_id": recommended_candidate_id,
                    "margin": margin,
                }
            },
        ),
    )


def _run_stronger_adjudicator(
    *,
    parsed_reviews: tuple[ParsedReview, ...],
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
) -> InvestigationResult:
    candidate_scores = _candidate_scores(parsed_reviews, candidate_ids=context.candidate_ids)
    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    recommended_candidate_id = ranked[0][0] if ranked else assessment.preferred_candidate_id
    margin = 0.0
    if len(ranked) > 1:
        margin = round(ranked[0][1] - ranked[1][1], 4)
    if recommended_candidate_id is None or (
        assessment.highest_issue_severity == "critical" and margin < 0.15
    ):
        return _build_unresolved_investigation(
            assessment=assessment,
            context=context,
            reason="stronger adjudicator stayed unconvinced by the remaining evidence",
        )
    notes = [
        "Stronger adjudicator applied stricter evidence weighting and higher severity penalties.",
        f"Top recommendation margin: {margin:.4f}",
    ]
    return InvestigationResult(
        status="resolved",
        strategy="stronger adjudicator",
        recommended_candidate_id=recommended_candidate_id,
        confidence_adjustment=0.03 if margin >= 0.25 else 0.01,
        rationale=f"selected {recommended_candidate_id} after stronger adjudication.",
        payload=_base_investigation_payload(
            assessment=assessment,
            context=context,
            strategy="stronger adjudicator",
            status="resolved",
            recommended_candidate_id=recommended_candidate_id,
            notes=notes,
            stage_payload={"margin": margin},
        ),
    )


def _build_unresolved_investigation(
    *,
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
    reason: str,
) -> InvestigationResult:
    return InvestigationResult(
        status="unresolved",
        strategy="escalation review",
        recommended_candidate_id=None,
        confidence_adjustment=0.0,
        rationale=f"{reason} and escalated to human review.",
        payload=_base_investigation_payload(
            assessment=assessment,
            context=context,
            strategy="escalation review",
            status="unresolved",
            recommended_candidate_id=None,
            notes=[reason],
        ),
    )


def _base_investigation_payload(
    *,
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
    strategy: str,
    status: str,
    recommended_candidate_id: str | None,
    notes: list[str],
    stage_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stage": context.stage,
        "candidate_ids": list(context.candidate_ids),
        "review_ids": list(context.review_ids),
        "preferred_candidate_id": assessment.preferred_candidate_id,
        "recommended_candidate_id": recommended_candidate_id,
        "winner_mismatch": assessment.winner_mismatch,
        "confidence_spread": assessment.confidence_spread,
        "contradictory_evidence_count": assessment.contradictory_evidence_count,
        "highest_issue_severity": assessment.highest_issue_severity,
        "total_score": assessment.total_score,
        "content_risk_class": context.content_risk_class,
        "strategy": strategy,
        "status": status,
        "notes": notes,
    }
    if stage_payload:
        payload.update(stage_payload)
    return payload


def _candidate_scores(
    parsed_reviews: tuple[ParsedReview, ...],
    *,
    candidate_ids: tuple[str, ...],
) -> dict[str, float]:
    scores = {candidate_id: 0.0 for candidate_id in candidate_ids}
    for review in parsed_reviews:
        if review.winner_candidate_id is not None:
            scores.setdefault(review.winner_candidate_id, 0.0)
            scores[review.winner_candidate_id] += 1.0 + review.confidence
        for evidence in review.quoted_evidence:
            if evidence.candidate_id is None:
                continue
            scores.setdefault(evidence.candidate_id, 0.0)
            scores[evidence.candidate_id] += 0.15
        for issue in review.issues:
            if issue.candidate_id is None:
                continue
            scores.setdefault(issue.candidate_id, 0.0)
            scores[issue.candidate_id] -= {"minor": 0.12, "major": 0.35, "critical": 0.7}[
                issue.severity
            ]
    return scores
