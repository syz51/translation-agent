"""Canonical translation models."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, model_validator

from .base import ContractModel
from .transcript import Segment

NonEmptyStr = Annotated[str, Field(min_length=1)]


class TranslationCandidate(ContractModel):
    """Normalized translation candidate produced from the final transcript path."""

    candidate_id: NonEmptyStr
    job_id: NonEmptyStr
    source_transcript_candidate_id: str | None = None
    final_transcript_ref: str | None = None
    model_id: NonEmptyStr
    prompt_variant_id: NonEmptyStr
    prompt_version: NonEmptyStr
    language: NonEmptyStr
    segments: tuple[Segment, ...] = ()
    full_text: str = ""
    raw_response_ref: str | None = None
    normalization_version: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_reference(self) -> TranslationCandidate:
        if self.source_transcript_candidate_id or self.final_transcript_ref:
            return self
        if self.metadata.get("review_mode") == "human_review_synthesis":
            return self
        raise ValueError(
            "translation candidates require source_transcript_candidate_id or final_transcript_ref"
        )
