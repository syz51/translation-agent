"""Deterministic disagreement scoring over structured review evidence."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from translation_agent.models import (
    AdjudicationContext,
    ReviewBundle,
    ReviewIssue,
    StructuredEvidence,
)
from translation_agent.models.review import DecisionMode, DisagreementBucket, ReviewStage
from translation_agent.review.parser import (
    IssueSeverity,
    ParsedReviewIssue,
    parse_reviewer_output,
)

_SEVERITY_WEIGHTS = {
    "minor": 0.5,
    "major": 1.2,
    "critical": 2.6,
}
_EVIDENCE_WEIGHTS = {
    "minor": 0.2,
    "major": 0.45,
    "critical": 0.8,
}
_RISK_MULTIPLIERS = {
    "standard": 1.0,
    "high": 1.2,
    "regulated": 1.35,
    "legal": 1.45,
    "medical": 1.5,
    "critical": 1.75,
}
_HARD_CONTRADICTION_DIMENSIONS = {"meaning", "entity", "number_date_unit", "coverage"}


@dataclass(frozen=True, slots=True)
class DisagreementAssessment:
    """Scored disagreement summary used by transcript and translation adjudication."""

    decision_mode: DecisionMode
    preferred_candidate_id: str | None
    average_confidence: float
    confidence_spread: float
    contradictory_evidence_count: int
    hard_contradiction_count: int
    blocking_hard_contradiction_count: int
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


@dataclass(frozen=True, slots=True)
class _ContradictionSummary:
    total_count: int
    hard_count: int
    blocking_hard_count: int


def assess_review_disagreement(
    reviews: tuple[ReviewBundle, ...],
    *,
    candidate_ids: tuple[str, ...],
    fallback_candidate_id: str | None,
    content_risk_class: str = "standard",
    ranking_priors: dict[str, float] | None = None,
) -> DisagreementAssessment:
    """Choose the adjudication path from structured reviewer outputs."""

    preferred_candidate_id = _preferred_candidate_id(
        reviews,
        fallback_candidate_id,
        ranking_priors=ranking_priors,
    )
    if preferred_candidate_id not in set(candidate_ids):
        preferred_candidate_id = fallback_candidate_id

    confidences = [_review_confidence(review) for review in reviews]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    confidence_spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0.0
    contradiction_summary = _contradiction_summary(reviews)
    material_refute_count = _material_refute_count(reviews)
    highest_issue_severity = _highest_issue_severity(reviews)
    winners = [winner for review in reviews if (winner := _review_winner(review)) is not None]
    winner_mismatch = len(set(winners)) > 1
    escalation_signal_count = sum(1 for review in reviews if _review_escalation_signal(review))

    base_score = (
        (2.4 if winner_mismatch else 0.0)
        + confidence_spread * 2.0
        + contradiction_summary.total_count * 1.3
        + material_refute_count * 1.2
        + _SEVERITY_WEIGHTS[highest_issue_severity]
        + (0.8 if escalation_signal_count else 0.0)
    )
    total_score = round(base_score * _RISK_MULTIPLIERS.get(content_risk_class, 1.0), 4)
    human_review_required = preferred_candidate_id is None
    if human_review_required:
        decision_mode = "human_review"
    elif contradiction_summary.blocking_hard_count >= 1:
        decision_mode = "stronger_adjudicator" if total_score >= 4.5 else "conflict_investigation"
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
        contradictory_evidence_count=contradiction_summary.total_count,
        hard_contradiction_count=contradiction_summary.hard_count,
        blocking_hard_contradiction_count=contradiction_summary.blocking_hard_count,
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
    assessment = assess_review_disagreement(
        reviews,
        candidate_ids=context.candidate_ids,
        fallback_candidate_id=fallback_candidate_id,
        content_risk_class=context.content_risk_class,
        ranking_priors=context.ranking_priors,
    )
    candidate_count = len(candidate_list)
    single_candidate_escalation = candidate_count == 1 and context.stage == "transcript"
    investigation_result = _execute_escalation(
        reviews=reviews,
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
    mapping = {
        "transcript_escalation": "critical",
        "translation_high_risk": "high",
        "translation_human_review": "critical",
        "translation_escalation": "critical",
    }
    return mapping.get(scenario, "standard")


def _preferred_candidate_id(
    reviews: tuple[ReviewBundle, ...],
    fallback_candidate_id: str | None,
    *,
    ranking_priors: dict[str, float] | None = None,
) -> str | None:
    scores: dict[str, float] = defaultdict(float)
    if ranking_priors:
        for candidate_id, boost in ranking_priors.items():
            scores[candidate_id] += max(0.0, min(float(boost), 0.35))
    for review in reviews:
        review_confidence = _review_confidence(review)
        for preference in _review_preferences(review):
            scores[preference.candidate_id] += (
                max(0.0, 1.5 - (preference.rank - 1) * 0.5) * review_confidence
            )
        for evidence in _review_evidence(review):
            weight = _EVIDENCE_WEIGHTS[evidence.severity]
            if evidence.polarity == "supports":
                scores[evidence.candidate_id] += weight
            else:
                scores[evidence.candidate_id] -= weight
        for issue in _review_issues(review):
            if issue.candidate_id is None:
                continue
            scores[issue.candidate_id] -= _SEVERITY_WEIGHTS[issue.severity] * 0.15
        winner = _review_winner(review)
        if winner is not None and not review.candidate_preferences:
            scores[winner] += review_confidence
    if not scores:
        return fallback_candidate_id
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _review_evidence(review: ReviewBundle) -> tuple[StructuredEvidence, ...]:
    if review.structured_evidence:
        return review.structured_evidence
    parsed = _parse_legacy_review(review)
    if parsed is None:
        return ()
    severity_by_candidate: dict[str, str] = {}
    for issue in parsed.issues:
        if issue.candidate_id is None:
            continue
        current = severity_by_candidate.get(issue.candidate_id, "minor")
        if issue.severity == "critical" or (issue.severity == "major" and current == "minor"):
            severity_by_candidate[issue.candidate_id] = issue.severity
    evidence: list[StructuredEvidence] = []
    for quoted in parsed.quoted_evidence:
        candidate_id = quoted.candidate_id or parsed.winner_candidate_id
        if candidate_id is None:
            continue
        severity = severity_by_candidate.get(
            candidate_id,
            "minor" if candidate_id == parsed.winner_candidate_id else "major",
        )
        evidence.append(
            StructuredEvidence(
                source_span_id=quoted.segment_id,
                candidate_id=candidate_id,
                dimension="meaning",
                polarity="supports" if candidate_id == parsed.winner_candidate_id else "refutes",
                normalized_value=None,
                severity=severity,  # type: ignore[arg-type]
                evidence_text=quoted.quote,
            )
        )
    return tuple(evidence)


def _review_preferences(review: ReviewBundle):
    if review.candidate_preferences:
        return review.candidate_preferences
    winner = _review_winner(review)
    if winner is None:
        return ()
    from translation_agent.models import CandidatePreference

    return (
        CandidatePreference(
            candidate_id=winner,
            rank=1,
            rationale="legacy winner from parsed review prose",
        ),
    )


def _review_issues(review: ReviewBundle) -> tuple[ReviewIssue, ...]:
    if review.review_issues:
        return review.review_issues
    try:
        parsed = parse_reviewer_output(review.raw_review_text)
    except Exception:
        return ()
    return tuple(_legacy_issue_to_review_issue(issue) for issue in parsed.issues)


def _review_winner(review: ReviewBundle) -> str | None:
    if review.candidate_preferences:
        return review.candidate_preferences[0].candidate_id
    parsed = _parse_legacy_review(review)
    return parsed.winner_candidate_id if parsed is not None else None


def _review_confidence(review: ReviewBundle) -> float:
    if review.candidate_preferences or review.structured_evidence:
        return review.confidence
    parsed = _parse_legacy_review(review)
    if parsed is not None:
        return parsed.confidence
    return review.confidence


def _review_escalation_signal(review: ReviewBundle) -> bool:
    if review.structured_evidence or review.escalation_signal:
        return review.escalation_signal
    parsed = _parse_legacy_review(review)
    return parsed.escalation_signal if parsed is not None else review.escalation_signal


def _parse_legacy_review(review: ReviewBundle):
    try:
        return parse_reviewer_output(review.raw_review_text)
    except Exception:
        return None


def _legacy_issue_to_review_issue(issue: ParsedReviewIssue) -> ReviewIssue:
    dimension = (
        issue.category
        if issue.category
        in {
            "meaning",
            "entity",
            "number_date_unit",
            "terminology",
            "coverage",
            "formatting",
            "style",
        }
        else "meaning"
    )
    return ReviewIssue(
        candidate_id=issue.candidate_id,
        dimension=dimension,  # type: ignore[arg-type]
        severity=issue.severity,
        description=issue.description,
    )


def _contradiction_summary(reviews: tuple[ReviewBundle, ...]) -> _ContradictionSummary:
    grouped: dict[tuple[str, str], list[StructuredEvidence]] = defaultdict(list)
    for review in reviews:
        for evidence in _review_evidence(review):
            span_id = evidence.source_span_id or _quote_fallback_key(evidence.evidence_text)
            if span_id is None:
                continue
            grouped[(span_id, evidence.dimension)].append(evidence)
    total = 0
    hard = 0
    blocking_hard = 0
    for (_, dimension), evidence_group in grouped.items():
        if not _is_contradiction(evidence_group):
            continue
        total += 1
        if dimension in _HARD_CONTRADICTION_DIMENSIONS:
            hard += 1
            if any(item.severity in {"major", "critical"} for item in evidence_group):
                blocking_hard += 1
        elif dimension == "terminology":
            if any(item.severity in {"major", "critical"} for item in evidence_group):
                total += 0
        elif dimension in {"style", "formatting"}:
            severities = {item.severity for item in evidence_group}
            if severities <= {"minor"}:
                total -= 1
    return _ContradictionSummary(
        total_count=max(total, 0),
        hard_count=hard,
        blocking_hard_count=blocking_hard,
    )


def _is_contradiction(evidence_group: list[StructuredEvidence]) -> bool:
    if len({item.candidate_id for item in evidence_group}) < 2:
        return False
    polarities = {item.polarity for item in evidence_group}
    if len(polarities) > 1:
        return True
    normalized_values = {
        item.normalized_value for item in evidence_group if item.normalized_value is not None
    }
    if "omitted" in normalized_values and "preserved" in normalized_values:
        return True
    if len(normalized_values) > 1:
        return True
    if evidence_group[0].dimension == "meaning":
        normalized_quotes = {_quote_fallback_key(item.evidence_text) for item in evidence_group}
        return len({value for value in normalized_quotes if value is not None}) > 1
    if evidence_group[0].dimension in {"style", "formatting"}:
        return all(item.severity in {"major", "critical"} for item in evidence_group)
    if evidence_group[0].dimension == "terminology":
        return (
            any(item.severity in {"major", "critical"} for item in evidence_group)
            and len(normalized_values) > 1
        )
    return False


def _quote_fallback_key(evidence_text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", evidence_text).casefold()
    normalized = "".join("<num>" if character.isdigit() else character for character in normalized)
    normalized = " ".join(normalized.split()).strip(".,!?;:'\"()[]{}")
    return normalized or None


def _highest_issue_severity(reviews: tuple[ReviewBundle, ...]) -> IssueSeverity:
    highest = "minor"
    for review in reviews:
        for issue in _review_issues(review):
            if issue.severity == "critical":
                return "critical"
            if issue.severity == "major":
                highest = "major"
        for evidence in _review_evidence(review):
            if evidence.polarity != "refutes":
                continue
            if evidence.severity == "critical":
                return "critical"
            if evidence.severity == "major":
                highest = "major"
    return highest


def _material_refute_count(reviews: tuple[ReviewBundle, ...]) -> int:
    keys = {
        (
            evidence.source_span_id or _quote_fallback_key(evidence.evidence_text),
            evidence.candidate_id,
            evidence.dimension,
        )
        for review in reviews
        for evidence in _review_evidence(review)
        if evidence.polarity == "refutes" and evidence.severity in {"major", "critical"}
    }
    return len({key for key in keys if key[0] is not None})


def _bucket_for_score(
    assessment: DisagreementAssessment,
    *,
    final_decision_mode: DecisionMode,
) -> DisagreementBucket:
    if final_decision_mode == "human_review":
        return "unresolved"
    if assessment.total_score >= 5.2 or assessment.blocking_hard_contradiction_count:
        return "high"
    if assessment.total_score >= 3.0 or assessment.contradictory_evidence_count:
        return "medium"
    return "low"


def _execute_escalation(
    *,
    reviews: tuple[ReviewBundle, ...],
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
    single_candidate_escalation: bool,
) -> InvestigationResult | None:
    if assessment.decision_mode == "automatic_finalize" and not single_candidate_escalation:
        return None
    if assessment.preferred_candidate_id is None:
        payload = {
            "status": "unresolved",
            "strategy": "no surviving candidate",
            "review_ids": list(context.review_ids),
        }
        return InvestigationResult(
            status="unresolved",
            strategy="no surviving candidate",
            recommended_candidate_id=None,
            confidence_adjustment=0.0,
            rationale="No translation candidate survived structured adjudication.",
            payload=payload,
        )
    if single_candidate_escalation:
        payload = {
            "status": "checked",
            "strategy": "single-candidate escalation check",
            "review_ids": list(context.review_ids),
        }
        return InvestigationResult(
            status="checked",
            strategy="single-candidate escalation check",
            recommended_candidate_id=assessment.preferred_candidate_id,
            confidence_adjustment=-0.12,
            rationale="Single surviving transcript candidate keeps a small confidence penalty.",
            payload=payload,
        )
    strategy = (
        "stronger adjudicator"
        if assessment.decision_mode == "stronger_adjudicator"
        else "conflict investigator"
    )
    payload = {
        "status": "resolved",
        "strategy": strategy,
        "review_ids": list(context.review_ids),
        "blocking_hard_contradictions": assessment.blocking_hard_contradiction_count,
        "hard_contradictions": assessment.hard_contradiction_count,
        "contradictions": assessment.contradictory_evidence_count,
        "winner": assessment.preferred_candidate_id,
    }
    if _requires_human_follow_up_after_escalation(assessment=assessment, context=context):
        payload["status"] = "unresolved"
        return InvestigationResult(
            status="unresolved",
            strategy=strategy,
            recommended_candidate_id=None,
            confidence_adjustment=-0.1,
            rationale="Blocking contradiction remained unresolved after machine escalation.",
            payload=payload,
        )
    return InvestigationResult(
        status="resolved",
        strategy=strategy,
        recommended_candidate_id=assessment.preferred_candidate_id,
        confidence_adjustment=-0.05 if assessment.decision_mode != "automatic_finalize" else 0.0,
        rationale="Structured contradiction normalization informed the escalation route.",
        payload=payload,
    )


def _decision_confidence(
    *,
    assessment: DisagreementAssessment,
    candidate_count: int,
    confidence_adjustment: float,
    final_decision_mode: DecisionMode,
) -> float:
    base = assessment.average_confidence
    if candidate_count <= 1:
        base -= 0.07
    if assessment.contradictory_evidence_count:
        base -= min(0.04 * assessment.contradictory_evidence_count, 0.18)
    if assessment.blocking_hard_contradiction_count:
        base -= min(0.1 * assessment.blocking_hard_contradiction_count, 0.25)
    if final_decision_mode == "human_review":
        return 0.0
    return round(max(0.0, min(base + confidence_adjustment, 0.99)), 2)


def _requires_human_follow_up_after_escalation(
    *,
    assessment: DisagreementAssessment,
    context: AdjudicationContext,
) -> bool:
    if (
        context.stage == "transcript"
        and assessment.highest_issue_severity == "critical"
        and assessment.escalation_signal_count >= 1
        and assessment.total_score >= 8.0
    ):
        return True
    return (
        assessment.highest_issue_severity == "critical"
        and assessment.winner_mismatch
        and assessment.escalation_signal_count >= 1
        and (assessment.blocking_hard_contradiction_count >= 1 or assessment.total_score >= 10.0)
    )


def _rationale_summary(
    *,
    stage: ReviewStage,
    decision_mode: DecisionMode,
    candidate_count: int,
    assessment: DisagreementAssessment,
    investigation_result: InvestigationResult | None,
) -> str:
    stage_label = "Transcript" if stage == "transcript" else "Translation"
    base = (
        f"{stage_label} adjudication reviewed {candidate_count} candidate(s), "
        f"{assessment.contradictory_evidence_count} contradiction group(s), and "
        f"{assessment.blocking_hard_contradiction_count} blocking hard contradiction(s)."
    )
    if decision_mode == "automatic_finalize":
        return base + " Structured evidence stayed below the escalation threshold."
    if decision_mode == "human_review":
        return base + " Remaining hard contradictions require human review."
    strategy = investigation_result.strategy if investigation_result is not None else "escalation"
    return base + f" The run continued through {strategy} before final selection."
