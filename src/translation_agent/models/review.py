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


class CandidatePreference(ContractModel):
    candidate_id: NonEmptyStr
    rank: int = Field(ge=1)
    rationale: str | None = None


class QuotedEvidence(ContractModel):
    quote: NonEmptyStr
    candidate_id: str | None = None
    segment_id: str | None = None


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


class ReviewBundle(ContractModel):
    """Parsed and normalized reviewer output for a single stage."""

    review_id: NonEmptyStr
    job_id: NonEmptyStr
    stage: ReviewStage
    reviewer_role: NonEmptyStr
    candidate_preferences: tuple[CandidatePreference, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    raw_review_text: NonEmptyStr
    quoted_evidence: tuple[QuotedEvidence, ...] = ()
    issue_categories: tuple[str, ...] = ()
    suggested_fixes: tuple[SuggestedFix, ...] = ()
    escalation_signal: bool = False
    parser_version: NonEmptyStr


class AdjudicationScorecard(ContractModel):
    """Structured inputs used to reach an adjudication decision."""

    candidate_count: int = Field(ge=0)
    preferred_candidate_id: str | None = None
    average_confidence: float = Field(ge=0.0, le=1.0)
    confidence_spread: float = Field(ge=0.0, le=1.0)
    contradictory_evidence_count: int = Field(ge=0)
    highest_issue_severity: Literal["minor", "major", "critical"]
    winner_mismatch: bool = False
    escalation_signal_count: int = Field(ge=0)
    total_score: float = Field(ge=0.0)
    content_risk_class: str = "standard"


class FinalTranscriptDecision(ContractModel):
    """Deterministic transcript adjudication output."""

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
