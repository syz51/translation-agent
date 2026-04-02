"""AssemblyAI transcription adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from translation_agent.adapters.common import (
    AdapterError,
    HttpTransport,
    RetryPolicy,
    StdlibHttpTransport,
    classify_http_error,
    json_request,
    normalize_whitespace,
    perform_with_retries,
    poll_until_complete,
    require_usable_timed_segments,
)
from translation_agent.models import AudioArtifact, RequestContext, Segment, TranscriptCandidate
from translation_agent.storage import BlobStore, job_path, job_scope_token

JsonFetcher = Callable[[str], dict[str, Any]]


class AssemblyAITranscriptionAdapter:
    """Direct AssemblyAI adapter using upload, transcript create, and poll endpoints."""

    provider_id = "assemblyai"

    def __init__(
        self,
        *,
        api_key: str,
        blob_store: BlobStore,
        base_url: str = "https://api.assemblyai.com",
        timeout_seconds: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._blob_store = blob_store
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._retry_policy = retry_policy or RetryPolicy()
        self._transport = transport or StdlibHttpTransport()
        self._sleep = sleep or (lambda seconds: __import__("time").sleep(seconds))

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate:
        candidate, raw_payload = self.transcribe_with_payload(audio_artifact, request_context)
        self._store_raw_payload(candidate.raw_payload_ref or "", raw_payload)
        return candidate

    def transcribe_with_payload(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> tuple[TranscriptCandidate, dict[str, Any]]:
        audio_bytes = self._blob_store.read_bytes(audio_artifact.blob_ref)
        upload_payload = perform_with_retries(
            lambda: self._upload_audio(audio_bytes),
            provider_id=self.provider_id,
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        transcript_job = perform_with_retries(
            lambda: self._create_transcript(upload_payload["upload_url"], request_context),
            provider_id=self.provider_id,
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        transcript_id = _require_string(transcript_job, "id", provider_id=self.provider_id)
        final_payload = poll_until_complete(
            provider_id=self.provider_id,
            fetch_status=lambda: perform_with_retries(
                lambda: self._fetch_transcript(transcript_id),
                provider_id=self.provider_id,
                retry_policy=self._retry_policy,
                sleep=self._sleep,
            ),
            pending_statuses={"queued", "processing"},
            success_statuses={"completed"},
            error_statuses={"error"},
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        raw_payload_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"{self.provider_id}.json",
        )
        return (
            _candidate_from_payload(
                final_payload,
                request_context=request_context,
                language=request_context.job.source_language,
                raw_payload_ref=raw_payload_ref,
            ),
            final_payload,
        )

    def _upload_audio(self, audio_bytes: bytes) -> dict[str, Any]:
        response = self._transport.request(
            request=_build_request(
                method="POST",
                url=f"{self._base_url}/v2/upload",
                headers=self._headers(content_type="application/octet-stream"),
                body=audio_bytes,
                timeout_seconds=self._timeout_seconds,
            )
        )
        return _validated_json_response(self.provider_id, response)

    def _create_transcript(
        self, upload_url: str, request_context: RequestContext
    ) -> dict[str, Any]:
        response = json_request(
            self._transport,
            method="POST",
            url=f"{self._base_url}/v2/transcript",
            headers=self._headers(),
            timeout_seconds=self._timeout_seconds,
            body={
                "audio_url": upload_url,
                "speech_models": ["universal-3-pro", "universal-2"],
                "speaker_labels": True,
                "language_code": request_context.job.source_language,
            },
        )
        return _validated_json_response(self.provider_id, response)

    def _fetch_transcript(self, transcript_id: str) -> dict[str, Any]:
        response = json_request(
            self._transport,
            method="GET",
            url=f"{self._base_url}/v2/transcript/{transcript_id}",
            headers=self._headers(),
            timeout_seconds=self._timeout_seconds,
        )
        return _validated_json_response(self.provider_id, response)

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers = {"authorization": self._api_key}
        if content_type is not None:
            headers["content-type"] = content_type
        return headers

    def _store_raw_payload(self, key: str, payload: dict[str, Any]) -> None:
        self._blob_store.put_bytes(key, (_json(payload) + "\n").encode("utf-8"))


def _candidate_from_payload(
    payload: dict[str, Any],
    *,
    request_context: RequestContext,
    language: str,
    raw_payload_ref: str,
) -> TranscriptCandidate:
    transcript_id = _require_string(payload, "id", provider_id="assemblyai")
    text = normalize_whitespace(str(payload.get("text", "")))
    utterances = payload.get("utterances")
    segments = require_usable_timed_segments("assemblyai", _segments_from_utterances(utterances))
    speaker_map = {
        segment.speaker: segment.speaker for segment in segments if segment.speaker is not None
    }
    return TranscriptCandidate(
        candidate_id=(
            f"tr-assemblyai-{request_context.job.job_id}-{job_scope_token(request_context.job)}"
        ),
        job_id=request_context.job.job_id,
        provider_id="assemblyai",
        provider_request_id=transcript_id,
        language=language,
        segments=segments,
        full_text=text,
        speaker_map=speaker_map,
        timing_resolution="utterance",
        raw_payload_ref=raw_payload_ref,
        normalization_version="raw-assemblyai-v1",
        metadata={
            "provider_rank": 0,
            "provider": {
                "status": payload.get("status"),
                "confidence": payload.get("confidence"),
            },
        },
    )


def _segments_from_utterances(value: object) -> tuple[Segment, ...]:
    if not isinstance(value, list):
        return ()
    segments: list[Segment] = []
    for index, utterance in enumerate(value, start=1):
        if not isinstance(utterance, dict):
            continue
        text = normalize_whitespace(str(utterance.get("text", "")))
        segments.append(
            Segment(
                segment_id=f"seg-assemblyai-{index}",
                start_ms=max(int(utterance.get("start", 0) or 0), 0),
                end_ms=max(int(utterance.get("end", 0) or 0), 0),
                speaker=_assemblyai_speaker_name(utterance.get("speaker")),
                source_text=text,
                annotations={
                    "provider": "assemblyai",
                    "confidence": utterance.get("confidence"),
                },
            )
        )
    return tuple(segments)


def _assemblyai_speaker_name(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return f"speaker-{cleaned.lower()}"


def _build_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
):
    from translation_agent.adapters.common import HttpRequest

    return HttpRequest(
        method=method,
        url=url,
        headers=headers,
        body=body,
        timeout_seconds=timeout_seconds,
    )


def _validated_json_response(provider_id: str, response) -> dict[str, Any]:
    error = classify_http_error(provider_id, response)
    if error is not None:
        raise error
    payload = response.json()
    if not isinstance(payload, dict):
        raise AdapterError(
            provider_id=provider_id,
            message="provider response must be a JSON object",
            category="malformed_response",
            retryable=False,
        )
    return payload


def _require_string(payload: dict[str, Any], key: str, *, provider_id: str) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    raise AdapterError(
        provider_id=provider_id,
        message=f"missing {key!r} in provider response",
        category="malformed_response",
        retryable=False,
    )


def _json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)
