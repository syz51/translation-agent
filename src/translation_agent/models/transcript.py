"""Canonical transcript models."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, model_validator

from .base import ContractModel

NonEmptyStr = Annotated[str, Field(min_length=1)]


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
