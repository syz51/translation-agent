"""Adapter interfaces and provider implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from translation_agent.models import (
    AudioArtifact,
    RequestContext,
    TranscriptCandidate,
    TranslationCandidate,
)

from .assemblyai import AssemblyAITranscriptionAdapter
from .common import AdapterError, RetryPolicy
from .deepgram import DeepgramTranscriptionAdapter
from .ffmpeg import FFmpegAudioExtractionAdapter
from .openai_translation import OpenAITranslationAdapter
from .speechmatics import SpeechmaticsTranscriptionAdapter


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


@runtime_checkable
class AudioBytesExtractionAdapter(AudioExtractionAdapter, Protocol):
    """Optional extension for extractors that can stream audio bytes directly."""

    def extract_audio_bytes(
        self,
        video_ref: str,
        job_context: RequestContext,
    ) -> tuple[AudioArtifact, bytes]: ...


@runtime_checkable
class RawPayloadTranscriptionAdapter(TranscriptionAdapter, Protocol):
    """Optional extension for adapters that also expose the raw provider payload."""

    def transcribe_with_payload(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> tuple[TranscriptCandidate, dict[str, Any]]: ...


@runtime_checkable
class RawPayloadTranslationAdapter(TranslationAdapter, Protocol):
    """Optional extension for translation adapters that expose raw responses."""

    def generate_translation_with_payload(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> tuple[TranslationCandidate, dict[str, Any]]: ...


__all__ = [
    "AdapterError",
    "AssemblyAITranscriptionAdapter",
    "AudioBytesExtractionAdapter",
    "AudioExtractionAdapter",
    "DeepgramTranscriptionAdapter",
    "FFmpegAudioExtractionAdapter",
    "OpenAITranslationAdapter",
    "RawPayloadTranscriptionAdapter",
    "RawPayloadTranslationAdapter",
    "RetryPolicy",
    "SpeechmaticsTranscriptionAdapter",
    "TranslationAdapter",
    "TranscriptionAdapter",
]
