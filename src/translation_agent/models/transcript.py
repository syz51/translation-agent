"""Canonical transcript and transcript-synthesis models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .base import ContractModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
TranscriptSynthesisMode = Literal[
    "select_provider_span",
    "merge_provider_spans",
    "mark_unresolved",
]
TranscriptAgentRole = Literal["selector", "reviewer", "global_adjudicator"]
TranscriptSynthesisStatus = Literal["ready", "blocked", "review_required"]
TranscriptReviewIssueType = Literal["coverage", "grounding", "timing", "provenance"]
TranscriptReviewSeverity = Literal["minor", "major", "critical"]


class Segment(ContractModel):
    """Shared segment shape for transcript and translation candidates."""

    segment_id: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: str | None = None
    source_text: str | None = None
    target_text: str | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> Segment:
        if self.end_ms < self.start_ms:
            raise ValueError("segment end_ms must be greater than or equal to start_ms")
        return self


class TranscriptCandidate(ContractModel):
    """Normalized transcript candidate returned from an STT provider."""

    candidate_id: NonEmptyStr
    job_id: NonEmptyStr
    provider_id: NonEmptyStr
    provider_request_id: str | None = None
    language: NonEmptyStr
    segments: tuple[Segment, ...] = ()
    full_text: str = ""
    speaker_map: dict[str, str] = Field(default_factory=dict)
    timing_resolution: str | None = None
    raw_payload_ref: str | None = None
    normalization_version: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalTranscriptSpan(ContractModel):
    """Canonical utterance-like span synthesized from provider-segment overlap."""

    canonical_span_id: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: str | None = None
    supporting_candidate_ids: tuple[str, ...] = ()
    supporting_provider_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> CanonicalTranscriptSpan:
        if self.end_ms <= self.start_ms:
            raise ValueError("canonical transcript spans must have positive duration")
        return self


class TranscriptSpanCandidate(ContractModel):
    """Provider evidence materialized for one canonical transcript span."""

    span_candidate_id: NonEmptyStr
    canonical_span_id: NonEmptyStr
    provider_id: NonEmptyStr
    candidate_id: NonEmptyStr
    source_segment_ids: tuple[str, ...] = ()
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    timing_overlap_ms: int = Field(ge=0)
    timing_overlap_ratio: float = Field(ge=0.0, le=1.0)
    normalized_text: str = ""
    speaker_label: str | None = None
    previous_span_text: str | None = None
    next_span_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> TranscriptSpanCandidate:
        if self.end_ms < self.start_ms:
            raise ValueError("transcript span candidate end_ms must be >= start_ms")
        return self


class TranscriptSpanDecision(ContractModel):
    """Structured selector or adjudicator decision for one canonical span."""

    canonical_span_id: NonEmptyStr
    decision_type: TranscriptSynthesisMode
    selected_candidate_ids: tuple[str, ...] = ()
    selected_span_candidate_ids: tuple[str, ...] = ()
    source_fragment_refs: tuple[str, ...] = ()
    output_text: str = ""
    speaker_label: str | None = None
    start_ms: int = Field(default=0, ge=0)
    end_ms: int = Field(default=0, ge=0)
    rationale: str = ""
    conflict_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision(self) -> TranscriptSpanDecision:
        if self.decision_type == "mark_unresolved":
            return self
        if self.end_ms <= self.start_ms:
            raise ValueError("resolved transcript span decisions must have positive duration")
        if not self.output_text.strip():
            raise ValueError("resolved transcript span decisions require output_text")
        if not self.selected_span_candidate_ids:
            raise ValueError("resolved transcript span decisions require source span evidence")
        return self


class TranscriptReviewIssue(ContractModel):
    """Reviewer issue raised against one canonical transcript span."""

    canonical_span_id: NonEmptyStr
    issue_type: TranscriptReviewIssueType
    severity: TranscriptReviewSeverity
    description: NonEmptyStr


class TranscriptSynthesisRecord(ContractModel):
    """Schema-validated output emitted by selector or global adjudicator."""

    record_id: NonEmptyStr
    job_id: NonEmptyStr
    run_id: NonEmptyStr
    agent_role: TranscriptAgentRole
    reasoning_provider: NonEmptyStr
    reasoning_model_id: NonEmptyStr
    canonical_span_count: int = Field(ge=0)
    decisions: tuple[TranscriptSpanDecision, ...] = ()
    unresolved_span_ids: tuple[str, ...] = ()
    provider_support_summary: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptSynthesisReview(ContractModel):
    """Schema-validated reviewer audit over synthesized transcript span decisions."""

    review_id: NonEmptyStr
    job_id: NonEmptyStr
    run_id: NonEmptyStr
    reasoning_provider: NonEmptyStr
    reasoning_model_id: NonEmptyStr
    accepted_span_ids: tuple[str, ...] = ()
    corrected_decisions: tuple[TranscriptSpanDecision, ...] = ()
    unresolved_span_ids: tuple[str, ...] = ()
    dropped_supported_span_ids: tuple[str, ...] = ()
    issues: tuple[TranscriptReviewIssue, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptCanonicalSpanTable(ContractModel):
    """Persisted canonical span table plus per-candidate evidence."""

    job_id: NonEmptyStr
    canonical_spans: tuple[CanonicalTranscriptSpan, ...] = ()
    span_candidates: tuple[TranscriptSpanCandidate, ...] = ()


class TranscriptSpanProvenance(ContractModel):
    """Per-span provenance emitted in the synthesized transcript artifact."""

    canonical_span_id: NonEmptyStr
    synthesis_mode: TranscriptSynthesisMode
    source_fragment_refs: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    reasoning_refs: tuple[str, ...] = ()


class TranscriptUnresolvedSpan(ContractModel):
    """Structured record of a span that remained unresolved after adjudication."""

    canonical_span_id: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    provider_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_offsets(self) -> TranscriptUnresolvedSpan:
        if self.end_ms <= self.start_ms:
            raise ValueError("unresolved transcript spans must have positive duration")
        return self


class TranscriptQualityMetrics(ContractModel):
    """Deterministic quality snapshot captured for the synthesized transcript."""

    canonical_span_count: int = Field(ge=0)
    supported_span_count: int = Field(ge=0)
    emitted_span_count: int = Field(ge=0)
    unresolved_span_count: int = Field(ge=0)
    overlap_count: int = Field(ge=0)
    non_monotonic_count: int = Field(ge=0)
    zero_length_count: int = Field(ge=0)
    dropped_supported_span_count: int = Field(ge=0)
    provider_support_summary: dict[str, int] = Field(default_factory=dict)


class SynthesizedTranscriptArtifact(ContractModel):
    """Final transcript artifact consumed by translation generation."""

    artifact_id: NonEmptyStr
    job_id: NonEmptyStr
    run_id: NonEmptyStr
    language: NonEmptyStr
    transcript_metadata: dict[str, Any] = Field(default_factory=dict)
    canonical_spans: tuple[CanonicalTranscriptSpan, ...] = ()
    span_candidates: tuple[TranscriptSpanCandidate, ...] = ()
    final_segments: tuple[Segment, ...] = ()
    provenance: tuple[TranscriptSpanProvenance, ...] = ()
    unresolved_spans: tuple[TranscriptUnresolvedSpan, ...] = ()
    quality_metrics: TranscriptQualityMetrics
    full_text: str = ""
    status: TranscriptSynthesisStatus = "ready"
