"""Human approval and transcript-learning models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from .base import ContractModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
ApprovalStage = Literal["translation"]


class HumanApprovalRecord(ContractModel):
    """Operator approval recorded after a translation-review escalation."""

    run_id: NonEmptyStr
    job_id: NonEmptyStr
    stage: ApprovalStage = "translation"
    approved_candidate_id: NonEmptyStr
    approved_source_transcript_candidate_id: NonEmptyStr
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
    supervision_kind: NonEmptyStr = "indirect_translation_approval"
    supervision_strength: NonEmptyStr = "indirect_strong"
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
    indirect_approval_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    review_escalation_timestamps: tuple[datetime, ...] = ()
    approval_timestamps: tuple[datetime, ...] = ()
    last_approved_at: datetime | None = None
    updated_at: datetime
