"""Reference transcript, evaluation, and asset-linkage models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, model_validator

from .base import ContractModel
from .jobs import ReferenceTranscriptFormat
from .memory import (
    PromotionGateOutcome,
    PromptCompatibilityTuple,
    ProposalAggregateMetrics,
    StrongerGraderScore,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]


class ReferenceSegment(ContractModel):
    """Canonical segment emitted from a trusted external transcript."""

    segment_id: NonEmptyStr
    sequence: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = ""


class ReferenceTranscript(ContractModel):
    """Trusted transcript artifact attached to a durable media identity."""

    reference_id: NonEmptyStr
    media_key: NonEmptyStr
    asset_id: str | None = None
    source: NonEmptyStr
    format: ReferenceTranscriptFormat
    segments: tuple[ReferenceSegment, ...] = ()
    full_text: str = ""
    created_at: datetime


class AssetRecord(ContractModel):
    """Durable asset identity resolved from business and technical identifiers."""

    media_key: NonEmptyStr
    asset_id: str | None = None
    media_fingerprint: str | None = None
    first_seen_run_id: NonEmptyStr
    source_language: str | None = None
    target_language: str | None = None
    latest_reference_transcript_ref: str | None = None
    created_at: datetime
    updated_at: datetime


class HistoricalRunLink(ContractModel):
    """Per-run linkage back to a durable media asset plus published refs."""

    run_id: NonEmptyStr
    media_key: NonEmptyStr
    job_id: NonEmptyStr
    tenant_id: NonEmptyStr
    project_id: NonEmptyStr
    source_language: NonEmptyStr
    target_language: NonEmptyStr
    created_at: datetime
    transcript_ref: str | None = None
    translation_ref: str | None = None
    transcript_decision_ref: str | None = None
    translation_decision_ref: str | None = None
    evaluation_report_ref: str | None = None
    regenerated_draft_ref: str | None = None
    translation_model_id: str | None = None
    translation_prompt_variant_id: str | None = None
    translation_base_prompt_version: str | None = None
    translation_effective_prompt_version: str | None = None
    prompt_scope_kind: str | None = None
    prompt_scope_key: str | None = None
    prompt_resolution_mode: str | None = None
    prompt_proposal_id: str | None = None


class TranscriptMismatchSpan(ContractModel):
    """Aligned transcript mismatch evidence for one reference span."""

    segment_id: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    reference_text: str = ""
    candidate_text: str = ""
    similarity: float = Field(ge=0.0, le=1.0)
    omitted_tokens: tuple[str, ...] = ()
    inserted_tokens: tuple[str, ...] = ()
    evidence_ref: str | None = None


class TranscriptAlignmentReport(ContractModel):
    """Deterministic transcript scoring against a trusted reference transcript."""

    run_id: NonEmptyStr
    media_key: NonEmptyStr
    transcript_ref: str | None = None
    trusted_transcript_ref: NonEmptyStr
    coverage: float = Field(ge=0.0, le=1.0)
    omission_count: int = Field(ge=0)
    insertion_count: int = Field(ge=0)
    high_divergence_count: int = Field(ge=0)
    mismatch_spans: tuple[TranscriptMismatchSpan, ...] = ()


class TranslationScore(ContractModel):
    """Multilingual-safe translation scoring driven by a trusted source transcript."""

    run_id: NonEmptyStr
    translation_ref: str | None = None
    trusted_transcript_ref: NonEmptyStr
    faithfulness_judge: float = Field(ge=0.0, le=1.0)
    segment_coverage: float = Field(ge=0.0, le=1.0)
    entity_consistency: float = Field(ge=0.0, le=1.0)
    numeric_date_unit_consistency: float = Field(ge=0.0, le=1.0)
    glossary_compliance: float = Field(ge=0.0, le=1.0)
    omission_risk: float = Field(ge=0.0, le=1.0)
    addition_risk: float = Field(ge=0.0, le=1.0)
    fluency_judge: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_quality_score: float = Field(ge=0.0, le=1.0)
    severe_failure_bucket: bool = False
    notes: tuple[str, ...] = ()
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    terminology_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    named_entity_preservation: float | None = Field(default=None, ge=0.0, le=1.0)
    repeated_failure_terms: tuple[str, ...] = ()

    @model_validator(mode="after")
    def backfill_legacy_scores(self) -> TranslationScore:
        updates: dict[str, object] = {}
        if self.faithfulness is None:
            updates["faithfulness"] = self.faithfulness_judge
        if self.terminology_consistency is None:
            updates["terminology_consistency"] = self.glossary_compliance
        if self.named_entity_preservation is None:
            updates["named_entity_preservation"] = self.entity_consistency
        if updates:
            return self.model_copy(update=updates)
        return self


class EvaluatedRunReport(ContractModel):
    """Evaluation bundle for one historical run on the same media identity."""

    run: HistoricalRunLink
    transcript: TranscriptAlignmentReport | None = None
    translation: TranslationScore | None = None


class EvaluationFailure(ContractModel):
    stage: NonEmptyStr
    message: NonEmptyStr
    recoverable: bool = True


class EvaluationReport(ContractModel):
    """Asset-scoped evaluation artifact over historical runs."""

    evaluation_id: NonEmptyStr
    run_id: NonEmptyStr
    media_key: NonEmptyStr
    trusted_transcript_ref: NonEmptyStr
    evaluated_runs: tuple[EvaluatedRunReport, ...] = ()
    recurring_failure_patterns: tuple[str, ...] = ()
    prior_official_translation_refs: tuple[str, ...] = ()
    proposal_refs: tuple[str, ...] = ()
    regenerated_draft_ref: str | None = None
    failures: tuple[EvaluationFailure, ...] = ()
    canary_metrics: ProposalAggregateMetrics | None = None
    control_metrics: ProposalAggregateMetrics | None = None
    canary_stronger_grader: StrongerGraderScore | None = None
    control_stronger_grader: StrongerGraderScore | None = None
    gate_outcome: PromotionGateOutcome | None = None
    proposal_compatibility: tuple[PromptCompatibilityTuple, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegeneratedTranslationDraft(ContractModel):
    """Draft-only regenerated translation produced from a trusted transcript."""

    draft_id: NonEmptyStr
    run_id: NonEmptyStr
    media_key: NonEmptyStr
    trusted_transcript_ref: NonEmptyStr
    translation_candidate_id: NonEmptyStr
    full_text: str = ""
    segment_count: int = Field(ge=0)
    draft_ref: str
    prompt_variant_id: NonEmptyStr
    prompt_version: NonEmptyStr
    prompt_provenance_refs: tuple[str, ...] = ()
    generated_from_reference_transcript: bool = True
    replaces_canonical: bool = False
