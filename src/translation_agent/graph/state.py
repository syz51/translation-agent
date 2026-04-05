"""Lean graph state carrying only refs, identifiers, and routing facts."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from translation_agent.models.base import ContractModel
from translation_agent.models.jobs import JobContext

NonEmptyStr = Annotated[str, Field(min_length=1)]


class RoutingFact(ContractModel):
    """A single routing fact persisted in graph state."""

    stage: NonEmptyStr
    fact_type: NonEmptyStr
    value: NonEmptyStr
    source_ref: str | None = None


class GraphState(ContractModel):
    """Ref-only workflow state used between deterministic nodes."""

    run_id: NonEmptyStr
    job: JobContext
    current_stage: NonEmptyStr
    source_video_ref: NonEmptyStr
    source_artifact_ref: str | None = None
    audio_artifact_ref: str | None = None
    raw_transcript_payload_refs: tuple[str, ...] = ()
    raw_transcript_candidate_refs: tuple[str, ...] = ()
    transcript_candidate_ids: tuple[str, ...] = ()
    transcript_review_ids: tuple[str, ...] = ()
    canonical_transcript_span_ref: str | None = None
    final_transcript_ref: str | None = None
    transcript_selector_ref: str | None = None
    final_transcript_synthesis_ref: str | None = None
    transcript_span_review_ref: str | None = None
    transcript_unresolved_span_count: int = 0
    final_transcript_candidate_id: str | None = None
    final_transcript_decision_ref: str | None = None
    raw_translation_payload_refs: tuple[str, ...] = ()
    raw_translation_candidate_refs: tuple[str, ...] = ()
    translation_candidate_ids: tuple[str, ...] = ()
    translation_review_ids: tuple[str, ...] = ()
    final_translation_candidate_id: str | None = None
    final_translation_decision_ref: str | None = None
    reference_transcript_ref: str | None = None
    evaluation_report_ref: str | None = None
    regenerated_translation_draft_ref: str | None = None
    improvement_proposal_refs: tuple[str, ...] = ()
    memory_batch_ids: tuple[str, ...] = ()
    published_artifact_refs: tuple[str, ...] = ()
    routing_facts: tuple[RoutingFact, ...] = ()
    pending_memory_source_stage: str | None = None
    escalation_pending: bool = False
    human_review_required: bool = False
    review_required_stage: str | None = None
    resolution_ref: str | None = None
    resolution_kind: str | None = None
    failure_tags: tuple[str, ...] = ()
    residual_failure_tags: tuple[str, ...] = ()
    approval_ref: str | None = None
    approved_candidate_id: str | None = None
    approved_source_transcript_candidate_id: str | None = None
    transcript_failed: bool = False
    translation_failed: bool = False
