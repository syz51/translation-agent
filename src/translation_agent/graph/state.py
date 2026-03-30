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
    audio_artifact_ref: str | None = None
    transcript_candidate_ids: tuple[str, ...] = ()
    transcript_review_ids: tuple[str, ...] = ()
    final_transcript_candidate_id: str | None = None
    final_transcript_decision_ref: str | None = None
    translation_candidate_ids: tuple[str, ...] = ()
    translation_review_ids: tuple[str, ...] = ()
    final_translation_candidate_id: str | None = None
    final_translation_decision_ref: str | None = None
    memory_batch_ids: tuple[str, ...] = ()
    published_artifact_refs: tuple[str, ...] = ()
    routing_facts: tuple[RoutingFact, ...] = ()
    escalation_pending: bool = False
    human_review_required: bool = False
