"""Canonical review and adjudication models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import ContractModel
from .jobs import JobContext
from .memory import MemoryBundle

NonEmptyStr = Annotated[str, Field(min_length=1)]
ReviewStage = Literal["transcript", "translation"]
DecisionMode = Literal[
    "automatic_finalize",
    "conflict_investigation",
    "stronger_adjudicator",
    "human_review",
]
DisagreementBucket = Literal["low", "medium", "high", "unresolved"]
ReviewDimension = Literal[
    "meaning",
    "entity",
    "number_date_unit",
    "terminology",
    "coverage",
    "formatting",
    "style",
]
EvidencePolarity = Literal["supports", "refutes"]
EvidenceSeverity = Literal["minor", "major", "critical"]


class CandidatePreference(ContractModel):
    candidate_id: NonEmptyStr
    rank: int = Field(ge=1)
    rationale: str | None = None


class QuotedEvidence(ContractModel):
    """Legacy quote-only evidence retained for replay compatibility."""

    quote: NonEmptyStr
    candidate_id: str | None = None
    segment_id: str | None = None


class StructuredEvidence(ContractModel):
    source_span_id: str | None = None
    candidate_id: NonEmptyStr
    dimension: ReviewDimension
    polarity: EvidencePolarity
    normalized_value: str | None = None
    severity: EvidenceSeverity
    evidence_text: NonEmptyStr


class ReviewIssue(ContractModel):
    candidate_id: str | None = None
    dimension: ReviewDimension
    severity: EvidenceSeverity
    description: NonEmptyStr
    source_span_id: str | None = None


class SuggestedFix(ContractModel):
    issue_category: NonEmptyStr
    candidate_id: str | None = None
    description: NonEmptyStr


class ReviewContext(ContractModel):
    """Typed context for transcript and translation reviewer calls."""

    run_id: NonEmptyStr
    stage: ReviewStage
    reviewer_role: NonEmptyStr
    job: JobContext
    candidate_ids: tuple[str, ...] = ()
    memory_bundle: MemoryBundle
    policy_ref: str | None = None


class AdjudicationContext(ContractModel):
    """Typed context for deterministic adjudication and conflict routing."""

    run_id: NonEmptyStr
    stage: ReviewStage
    job: JobContext
    candidate_ids: tuple[str, ...] = ()
    review_ids: tuple[str, ...] = ()
    memory_bundle: MemoryBundle
    investigation_ref: str | None = None
    content_risk_class: str = "standard"
    ranking_priors: dict[str, float] = Field(default_factory=dict)


class ReviewBundle(ContractModel):
    """Structured reviewer output for a single stage."""

    review_id: NonEmptyStr
    job_id: NonEmptyStr
    stage: ReviewStage
    reviewer_role: NonEmptyStr
    candidate_preferences: tuple[CandidatePreference, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    raw_review_text: str = ""
    structured_evidence: tuple[StructuredEvidence, ...] = ()
    review_issues: tuple[ReviewIssue, ...] = ()
    quoted_evidence: tuple[QuotedEvidence, ...] = ()
    issue_categories: tuple[str, ...] = ()
    suggested_fixes: tuple[SuggestedFix, ...] = ()
    escalation_signal: bool = False
    parser_version: str | None = None
    output_version: NonEmptyStr = "structured-review-v1"


class AdjudicationScorecard(ContractModel):
    """Structured inputs used to reach an adjudication decision."""

    candidate_count: int = Field(ge=0)
    preferred_candidate_id: str | None = None
    average_confidence: float = Field(ge=0.0, le=1.0)
    confidence_spread: float = Field(ge=0.0, le=1.0)
    contradictory_evidence_count: int = Field(ge=0)
    hard_contradiction_count: int = Field(default=0, ge=0)
    blocking_hard_contradiction_count: int = Field(default=0, ge=0)
    highest_issue_severity: Literal["minor", "major", "critical"]
    winner_mismatch: bool = False
    escalation_signal_count: int = Field(ge=0)
    total_score: float = Field(ge=0.0)
    content_risk_class: str = "standard"


class FinalTranscriptDecision(ContractModel):
    """Structured transcript-synthesis outcome recorded after adjudication."""

    job_id: NonEmptyStr
    winner_candidate_id: str | None = None
    transcript_artifact_ref: str | None = None
    canonical_span_ref: str | None = None
    synthesis_record_ref: str | None = None
    span_review_ref: str | None = None
    decision_mode: DecisionMode
    decision_confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: NonEmptyStr
    review_refs: tuple[str, ...] = ()
    investigation_ref: str | None = None
    disagreement_bucket: DisagreementBucket
    adjudication_scorecard: AdjudicationScorecard
    synthesis_status: Literal["complete", "blocked", "review_required"] = "complete"
    canonical_span_count: int = Field(default=0, ge=0)
    emitted_span_count: int = Field(default=0, ge=0)
    unresolved_span_count: int = Field(default=0, ge=0)
    blocker_tags: tuple[str, ...] = ()
    provider_support_summary: dict[str, int] = Field(default_factory=dict)
    provenance_refs: tuple[str, ...] = ()
    escalated: bool = False
    human_review_required: bool = False


class FinalTranslationDecision(ContractModel):
    """Deterministic translation adjudication output."""

    job_id: NonEmptyStr
    winner_candidate_id: str | None = None
    decision_mode: DecisionMode
    decision_confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: NonEmptyStr
    review_refs: tuple[str, ...] = ()
    investigation_ref: str | None = None
    disagreement_bucket: DisagreementBucket
    adjudication_scorecard: AdjudicationScorecard
    escalated: bool = False
    human_review_required: bool = False
    winner_model_id: str | None = None
    prompt_variant_winner: str | None = None
    prompt_version_winner: str | None = None
