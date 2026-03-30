"""Job-scoped canonical models and typed workflow contexts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field

from .base import ContractModel

NonEmptyStr = Annotated[str, Field(min_length=1)]


class JobContext(ContractModel):
    """Single source of truth for immutable job identity and request facts."""

    job_id: NonEmptyStr
    tenant_id: NonEmptyStr
    project_id: NonEmptyStr
    source_video_ref: NonEmptyStr
    target_language: NonEmptyStr
    source_language: NonEmptyStr
    requested_by: NonEmptyStr
    created_at: datetime
    profile_ref: str | None = None


class RequestContext(ContractModel):
    """Typed context for extraction, transcription, and translation requests."""

    run_id: NonEmptyStr
    attempt: int = Field(default=1, ge=1)
    job: JobContext
    source_artifact_ref: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingContext(ContractModel):
    """Routing-only facts used to pick the next deterministic workflow step."""

    run_id: NonEmptyStr
    stage: NonEmptyStr
    job: JobContext
    available_candidate_ids: tuple[str, ...] = ()
    failed_candidate_ids: tuple[str, ...] = ()
    review_ids: tuple[str, ...] = ()
    escalation_signals: tuple[str, ...] = ()
    content_risk_class: str = "standard"
    retryable_failures_present: bool = False
