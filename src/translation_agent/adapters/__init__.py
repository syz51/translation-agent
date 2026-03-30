"""Adapter interfaces and provider implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import (
    AudioArtifact,
    RequestContext,
    TranscriptCandidate,
    TranslationCandidate,
)


@runtime_checkable
class AudioExtractionAdapter(Protocol):
    """Stable extraction boundary for video-to-audio conversion."""

    adapter_id: str

    def extract_audio(self, video_ref: str, job_context: RequestContext) -> AudioArtifact: ...


@runtime_checkable
class TranscriptionAdapter(Protocol):
    """Stable STT boundary returning canonical transcript candidates."""

    provider_id: str

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate: ...


@runtime_checkable
class TranslationAdapter(Protocol):
    """Stable translation boundary returning canonical translation candidates."""

    model_id: str

    def generate_translation(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> TranslationCandidate: ...


__all__ = [
    "AudioExtractionAdapter",
    "TranslationAdapter",
    "TranscriptionAdapter",
]
