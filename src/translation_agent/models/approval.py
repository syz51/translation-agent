"""Human review-resolution, approval, draft, and feedback aggregation models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from .base import ContractModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
ApprovalStage = Literal["translation"]
HumanSupervisionKind = Literal["approved_good", "approved_best_available", "rejected_all"]
SupervisionStrength = Literal["strong", "soft", "negative"]
FailureTag = Literal[
    "honorific_leak",
    "romanization_leak",
    "name_entity_drift",
    "ungrounded_addition",
    "subtitle_gibberish",
    "late_run_degeneration",
    "source_transcript_instability",
    "literal_but_wrong_semantics",
]


class HumanReviewedCandidateContext(ContractModel):
    """Stable candidate context captured at human-review resolution time."""

    candidate_id: NonEmptyStr
    source_transcript_candidate_id: str | None = None
    transcript_provider_id: str | None = None
    model_id: NonEmptyStr
    prompt_variant_id: NonEmptyStr
    prompt_version: NonEmptyStr
    base_prompt_version: str | None = None
    combo_key: NonEmptyStr
    prompt_resolution_mode: str | None = None
    selected_proposal_id: str | None = None


class ReviewedSpanDecision(ContractModel):
    """Final operator decision for one aligned source span."""

    source_span_id: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    selected_candidate_id: NonEmptyStr
    selected_source_transcript_candidate_id: str | None = None
    selected_transcript_provider_id: str | None = None
    base_target_text: str = ""
    final_target_text: str = ""
    edited: bool = False
    reviewer_note: str = ""


class ReviewDraftSpanDecision(ContractModel):
    """In-progress operator decision state for one span."""

    source_span_id: NonEmptyStr
    selected_base_variant_id: str | None = None
    edited_text: str | None = None
    resolution_status: Literal["unresolved", "resolved"] = "unresolved"
    dirty: bool = False
    reviewer_note: str = ""


class ReviewDraftResolution(ContractModel):
    """Persisted in-progress draft for resumable span-level review."""

    run_id: NonEmptyStr
    job_id: NonEmptyStr
    resolution_kind: HumanSupervisionKind = "approved_good"
    failure_tags: tuple[FailureTag, ...] = ()
    note: str = ""
    approved_by: str | None = None
    span_decisions: tuple[ReviewDraftSpanDecision, ...] = ()
    updated_at: datetime


class HumanReviewResolutionRecord(ContractModel):
    """Canonical human review resolution, regardless of approval outcome."""

    run_id: NonEmptyStr
    job_id: NonEmptyStr
    stage: ApprovalStage = "translation"
    resolution_kind: HumanSupervisionKind
    candidate_id: str | None = None
    final_translation_candidate_id: str | None = None
    final_translation_ref: str | None = None
    reviewed_span_count: int = Field(default=0, ge=0)
    reviewed_span_decisions: tuple[ReviewedSpanDecision, ...] = ()
    contributing_translation_candidate_ids: tuple[str, ...] = ()
    contributing_source_transcript_candidate_ids: tuple[str, ...] = ()
    contributing_transcript_provider_ids: tuple[str, ...] = ()
    approved_by: NonEmptyStr
    note: str = ""
    failure_tags: tuple[FailureTag, ...] = ()
    residual_failure_tags: tuple[FailureTag, ...] = ()
    transcript_provider_id: str | None = None
    source_transcript_candidate_id: str | None = None
    model_id: str | None = None
    prompt_variant_id: str | None = None
    prompt_version: str | None = None
    base_prompt_version: str | None = None
    combo_key: str | None = None
    source_language: NonEmptyStr
    target_language: NonEmptyStr
    tenant_id: NonEmptyStr
    project_id: NonEmptyStr
    media_key: NonEmptyStr
    machine_translation_decision_ref: NonEmptyStr
    machine_investigation_ref: str | None = None
    machine_review_refs: tuple[str, ...] = ()
    reviewed_candidates: tuple[HumanReviewedCandidateContext, ...] = ()
    resolved_at: datetime

    @property
    def supervision_strength(self) -> SupervisionStrength:
        if self.resolution_kind == "approved_good":
            return "strong"
        if self.resolution_kind == "approved_best_available":
            return "soft"
        return "negative"


class HumanApprovalRecord(ContractModel):
    """Operator approval recorded after a translation-review escalation."""

    run_id: NonEmptyStr
    job_id: NonEmptyStr
    stage: ApprovalStage = "translation"
    approved_candidate_id: str | None = None
    final_translation_candidate_id: str | None = None
    approved_source_transcript_candidate_id: str | None = None
    approved_by: NonEmptyStr
    note: str = ""
    approved_at: datetime
    machine_translation_decision_ref: NonEmptyStr
    machine_investigation_ref: str | None = None
    machine_review_refs: tuple[str, ...] = ()


class TranscriptApprovalLearningEvent(ContractModel):
    """Indirect transcript-quality signal derived from approved translations."""

    run_id: NonEmptyStr
    job_id: NonEmptyStr
    approved_translation_candidate_id: NonEmptyStr
    approved_source_transcript_candidate_id: NonEmptyStr
    transcript_provider_id: NonEmptyStr
    source_language: NonEmptyStr
    target_language: NonEmptyStr
    tenant_id: NonEmptyStr
    project_id: NonEmptyStr
    media_key: NonEmptyStr
    linked_translation_approval_ref: NonEmptyStr
    supervision_kind: NonEmptyStr = "approved_good"
    supervision_strength: NonEmptyStr = "strong"
    created_at: datetime


class TranscriptProviderQualityStats(ContractModel):
    """Operational aggregate quality counters for transcript providers."""

    provider_id: NonEmptyStr
    source_language: NonEmptyStr
    target_language: NonEmptyStr
    total_review_escalations: int = Field(default=0, ge=0)
    total_approved_outcomes: int = Field(default=0, ge=0)
    selected_via_approved_translation_count: int = Field(default=0, ge=0)
    recent_review_escalations_30d: int = Field(default=0, ge=0)
    recent_approved_outcomes_30d: int = Field(default=0, ge=0)
    approved_good_count: int = Field(default=0, ge=0)
    approved_best_available_count: int = Field(default=0, ge=0)
    rejected_all_count: int = Field(default=0, ge=0)
    recent_approved_best_available_count_30d: int = Field(default=0, ge=0)
    recent_rejected_all_count_30d: int = Field(default=0, ge=0)
    hard_positive_score: float = Field(default=0.0, ge=0.0)
    soft_positive_score: float = Field(default=0.0, ge=0.0)
    negative_feedback_score: float = Field(default=0.0, ge=0.0)
    indirect_approval_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    review_escalation_timestamps: tuple[datetime, ...] = ()
    approval_timestamps: tuple[datetime, ...] = ()
    approved_best_available_timestamps: tuple[datetime, ...] = ()
    rejected_all_timestamps: tuple[datetime, ...] = ()
    last_approved_at: datetime | None = None
    updated_at: datetime


class TranslationFeedbackStats(ContractModel):
    """Operational aggregate feedback counters for one translation combo key."""

    combo_key: NonEmptyStr
    source_language: NonEmptyStr
    target_language: NonEmptyStr
    transcript_provider_id: NonEmptyStr
    model_id: NonEmptyStr
    prompt_variant_id: NonEmptyStr
    prompt_version: NonEmptyStr
    approved_good_count: int = Field(default=0, ge=0)
    approved_best_available_count: int = Field(default=0, ge=0)
    rejected_all_count: int = Field(default=0, ge=0)
    pairwise_win_score: float = Field(default=0.0)
    pairwise_loss_score: float = Field(default=0.0)
    failure_tag_counts: dict[FailureTag, int] = Field(default_factory=dict)
    last_seen_at: datetime
    updated_at: datetime
