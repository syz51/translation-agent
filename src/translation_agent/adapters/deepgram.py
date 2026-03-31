"""Deepgram transcription adapter."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from typing import Any

from translation_agent.adapters.common import (
    AdapterError,
    HttpRequest,
    HttpTransport,
    RetryPolicy,
    StdlibHttpTransport,
    blob_filename,
    classify_http_error,
    normalize_whitespace,
    perform_with_retries,
)
from translation_agent.models import AudioArtifact, RequestContext, Segment, TranscriptCandidate
from translation_agent.storage import BlobStore, job_path, job_scope_token


class DeepgramTranscriptionAdapter:
    """Direct Deepgram prerecorded transcription adapter."""

    provider_id = "deepgram"

    def __init__(
        self,
        *,
        api_key: str,
        blob_store: BlobStore,
        base_url: str = "https://api.deepgram.com/v1/listen",
        model: str = "nova-3",
        timeout_seconds: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._blob_store = blob_store
        self._base_url = base_url
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retry_policy = retry_policy or RetryPolicy()
        self._transport = transport or StdlibHttpTransport()
        self._sleep = sleep or (lambda seconds: __import__("time").sleep(seconds))

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate:
        audio_bytes = self._blob_store.read_bytes(audio_artifact.blob_ref)
        payload = perform_with_retries(
            lambda: self._transcribe_once(audio_artifact, audio_bytes, request_context),
            provider_id=self.provider_id,
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        raw_payload_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"{self.provider_id}.json",
        )
        self._blob_store.put_bytes(
            raw_payload_ref,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return _candidate_from_payload(
            payload,
            request_context=request_context,
            language=request_context.job.source_language,
            raw_payload_ref=raw_payload_ref,
        )

    def _transcribe_once(
        self,
        audio_artifact: AudioArtifact,
        audio_bytes: bytes,
        request_context: RequestContext,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {
                "model": self._model,
                "diarize": "true",
                "utterances": "true",
                "smart_format": "true",
                "language": request_context.job.source_language,
            }
        )
        response = self._transport.request(
            HttpRequest(
                method="POST",
                url=f"{self._base_url}?{query}",
                headers={
                    "authorization": f"Token {self._api_key}",
                    "content-type": _content_type_for_blob(audio_artifact.blob_ref),
                },
                body=audio_bytes,
                timeout_seconds=self._timeout_seconds,
            )
        )
        error = classify_http_error(self.provider_id, response)
        if error is not None:
            raise error
        payload = response.json()
        if not isinstance(payload, dict):
            raise AdapterError(
                provider_id=self.provider_id,
                message="Deepgram response must be a JSON object",
                category="malformed_response",
                retryable=False,
            )
        return payload


def _candidate_from_payload(
    payload: dict[str, Any],
    *,
    request_context: RequestContext,
    language: str,
    raw_payload_ref: str,
) -> TranscriptCandidate:
    metadata = payload.get("metadata")
    request_id = metadata.get("request_id") if isinstance(metadata, dict) else None
    transcript_text = _extract_transcript_text(payload)
    utterances = _extract_utterances(payload)
    segments = tuple(
        Segment(
            segment_id=f"seg-deepgram-{index}",
            start_ms=_seconds_to_ms(utterance.get("start")),
            end_ms=_seconds_to_ms(utterance.get("end")),
            speaker=_speaker_name(utterance.get("speaker")),
            source_text=normalize_whitespace(str(utterance.get("transcript", ""))),
            annotations={
                "provider": "deepgram",
                "confidence": utterance.get("confidence"),
            },
        )
        for index, utterance in enumerate(utterances, start=1)
        if isinstance(utterance, dict)
    )
    if not segments:
        segments = (
            Segment(
                segment_id="seg-deepgram-1",
                start_ms=0,
                end_ms=0,
                speaker=None,
                source_text=transcript_text,
                annotations={"provider": "deepgram"},
            ),
        )
    return TranscriptCandidate(
        candidate_id=(
            f"tr-deepgram-{request_context.job.job_id}-{job_scope_token(request_context.job)}"
        ),
        job_id=request_context.job.job_id,
        provider_id="deepgram",
        provider_request_id=str(request_id) if request_id else None,
        language=language,
        segments=segments,
        full_text=transcript_text,
        speaker_map={
            segment.speaker: segment.speaker for segment in segments if segment.speaker is not None
        },
        timing_resolution="utterance",
        raw_payload_ref=raw_payload_ref,
        normalization_version="raw-deepgram-v1",
        metadata={
            "provider_rank": 2,
            "provider": {
                "model_info": metadata.get("model_info") if isinstance(metadata, dict) else None,
            },
        },
    )


def _extract_transcript_text(payload: dict[str, Any]) -> str:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise AdapterError(
            provider_id="deepgram",
            message="Deepgram response was missing results",
            category="malformed_response",
            retryable=False,
        )
    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        raise AdapterError(
            provider_id="deepgram",
            message="Deepgram response was missing channels",
            category="malformed_response",
            retryable=False,
        )
    first_channel = channels[0]
    if not isinstance(first_channel, dict):
        raise AdapterError(
            provider_id="deepgram",
            message="Deepgram channel payload was malformed",
            category="malformed_response",
            retryable=False,
        )
    alternatives = first_channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        raise AdapterError(
            provider_id="deepgram",
            message="Deepgram response was missing alternatives",
            category="malformed_response",
            retryable=False,
        )
    first_alternative = alternatives[0]
    if not isinstance(first_alternative, dict):
        raise AdapterError(
            provider_id="deepgram",
            message="Deepgram alternative payload was malformed",
            category="malformed_response",
            retryable=False,
        )
    return normalize_whitespace(str(first_alternative.get("transcript", "")))


def _extract_utterances(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, dict):
        return []
    utterances = results.get("utterances")
    if not isinstance(utterances, list):
        return []
    return [utterance for utterance in utterances if isinstance(utterance, dict)]


def _content_type_for_blob(blob_ref: str) -> str:
    filename = blob_filename(blob_ref, "audio.wav").lower()
    if filename.endswith(".wav"):
        return "audio/wav"
    if filename.endswith(".mp3"):
        return "audio/mpeg"
    return "application/octet-stream"


def _seconds_to_ms(value: object) -> int:
    if value is None:
        return 0
    try:
        numeric = value if isinstance(value, int | float | str) else 0.0
        return max(int(round(float(numeric) * 1000)), 0)
    except TypeError, ValueError:
        return 0


def _speaker_name(value: object) -> str | None:
    if value is None:
        return None
    return f"speaker-{str(value).strip()}"
