"""Canonical artifact models."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from .base import ContractModel
from .jobs import JobContext

NonEmptyStr = Annotated[str, Field(min_length=1)]


class AudioArtifact(ContractModel):
    """Canonical audio output from the extraction boundary."""

    artifact_id: NonEmptyStr
    job_id: NonEmptyStr
    blob_ref: NonEmptyStr
    duration_ms: int = Field(ge=0)
    sample_rate_hz: int = Field(ge=1)
    channels: int = Field(ge=1)
    codec: NonEmptyStr
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)


class PublishContext(ContractModel):
    """Typed context for the publishing/finalization boundary."""

    run_id: NonEmptyStr
    job: JobContext
    transcript_decision_ref: str | None = None
    translation_decision_ref: str | None = None
    trace_refs: tuple[str, ...] = ()
    export_targets: tuple[str, ...] = ()
    downstream_targets: tuple[str, ...] = ()


class PublishedArtifacts(ContractModel):
    """Stable references emitted by finalization and downstream delivery."""

    final_transcript_ref: str | None = None
    final_translation_ref: str | None = None
    recoverable_translation_failure_ref: str | None = None
    reference_transcript_refs: tuple[str, ...] = ()
    evaluation_report_refs: tuple[str, ...] = ()
    regenerated_draft_refs: tuple[str, ...] = ()
    improvement_proposal_refs: tuple[str, ...] = ()
    scorecard_refs: tuple[str, ...] = ()
    trace_refs: tuple[str, ...] = ()
    export_refs: tuple[str, ...] = ()
    downstream_delivery_refs: tuple[str, ...] = ()
    memory_batch_refs: tuple[str, ...] = ()
    memory_consolidation_refs: tuple[str, ...] = ()
    prompt_evolution_refs: tuple[str, ...] = ()
