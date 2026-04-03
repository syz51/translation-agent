"""Speechmatics transcription adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from translation_agent.adapters.common import (
    AdapterError,
    HttpRequest,
    HttpTransport,
    RetryPolicy,
    StdlibHttpTransport,
    blob_filename,
    build_multipart_form_data,
    classify_http_error,
    normalize_whitespace,
    perform_with_retries,
    poll_until_complete,
    require_usable_timed_segments,
)
from translation_agent.models import AudioArtifact, RequestContext, Segment, TranscriptCandidate
from translation_agent.storage import BlobStore, job_path, job_scope_token


class SpeechmaticsTranscriptionAdapter:
    """Direct Speechmatics batch transcription adapter."""

    provider_id = "speechmatics"

    def __init__(
        self,
        *,
        api_key: str,
        blob_store: BlobStore,
        base_url: str = "https://asr.api.speechmatics.com/v2",
        timeout_seconds: float = 60.0,
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
        self._blob_store.put_bytes(
            candidate.raw_payload_ref or "",
            (json.dumps(raw_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return candidate

    def transcribe_with_payload(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> tuple[TranscriptCandidate, dict[str, Any]]:
        audio_bytes = self._blob_store.read_bytes(audio_artifact.blob_ref)
        create_payload = perform_with_retries(
            lambda: self._create_job(audio_artifact.blob_ref, audio_bytes, request_context),
            provider_id=self.provider_id,
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        job_id = _require_string(create_payload, "id", provider_id=self.provider_id)
        perform_with_retries(
            lambda: poll_until_complete(
                provider_id=self.provider_id,
                fetch_status=lambda: self._fetch_job(job_id),
                pending_statuses={"running", "queued", "created"},
                success_statuses={"done"},
                error_statuses={"rejected", "failed", "deleted"},
                retry_policy=self._retry_policy,
                sleep=self._sleep,
            ),
            provider_id=self.provider_id,
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        transcript_payload = perform_with_retries(
            lambda: self._fetch_transcript(job_id),
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
        return (
            _candidate_from_payload(
                transcript_payload,
                request_context=request_context,
                provider_request_id=job_id,
                language=request_context.job.source_language,
                raw_payload_ref=raw_payload_ref,
            ),
            transcript_payload,
        )

    def _create_job(
        self,
        blob_ref: str,
        audio_bytes: bytes,
        request_context: RequestContext,
    ) -> dict[str, Any]:
        config = {
            "type": "transcription",
            "transcription_config": {
                "language": request_context.job.source_language,
                "diarization": "speaker",
                "operating_point": "enhanced",
            },
        }
        body, content_type = build_multipart_form_data(
            fields={"config": json.dumps(config)},
            file_field="data_file",
            filename=blob_filename(blob_ref, f"{request_context.job.job_id}.wav"),
            content=audio_bytes,
        )
        response = self._transport.request(
            HttpRequest(
                method="POST",
                url=f"{self._base_url}/jobs",
                headers={
                    "authorization": f"Bearer {self._api_key}",
                    "content-type": content_type,
                },
                body=body,
                timeout_seconds=self._timeout_seconds,
            )
        )
        return _validated_json_response(self.provider_id, response)

    def _fetch_job(self, job_id: str) -> dict[str, Any]:
        response = self._transport.request(
            HttpRequest(
                method="GET",
                url=f"{self._base_url}/jobs/{job_id}",
                headers={"authorization": f"Bearer {self._api_key}"},
                timeout_seconds=self._timeout_seconds,
            )
        )
        return _normalized_job_status_payload(_validated_json_response(self.provider_id, response))

    def _fetch_transcript(self, job_id: str) -> dict[str, Any]:
        response = self._transport.request(
            HttpRequest(
                method="GET",
                url=f"{self._base_url}/jobs/{job_id}/transcript?format=json-v2",
                headers={"authorization": f"Bearer {self._api_key}"},
                timeout_seconds=self._timeout_seconds,
            )
        )
        return _validated_json_response(self.provider_id, response)


def _candidate_from_payload(
    payload: dict[str, Any],
    *,
    request_context: RequestContext,
    provider_request_id: str,
    language: str,
    raw_payload_ref: str,
) -> TranscriptCandidate:
    segments = require_usable_timed_segments(
        "speechmatics",
        tuple(
            Segment(
                segment_id=f"seg-speechmatics-{index}",
                start_ms=_seconds_to_ms(result.get("start_time")),
                end_ms=_seconds_to_ms(result.get("end_time")),
                speaker=_speaker_name(result.get("speaker")),
                source_text=normalize_whitespace(_content_from_result(result)),
                annotations={
                    "provider": "speechmatics",
                    "type": result.get("type"),
                    "confidence": _alternative_confidence(result),
                },
            )
            for index, result in enumerate(_speech_segments(payload), start=1)
        ),
    )
    transcript_text = " ".join(segment.source_text or "" for segment in segments).strip()
    return TranscriptCandidate(
        candidate_id=(
            f"tr-speechmatics-{request_context.job.job_id}-{job_scope_token(request_context.job)}"
        ),
        job_id=request_context.job.job_id,
        provider_id="speechmatics",
        provider_request_id=provider_request_id,
        language=language,
        segments=segments,
        full_text=normalize_whitespace(transcript_text),
        speaker_map={
            segment.speaker: segment.speaker for segment in segments if segment.speaker is not None
        },
        timing_resolution="segment",
        raw_payload_ref=raw_payload_ref,
        normalization_version="raw-speechmatics-v1",
        metadata={
            "provider_rank": 1,
            "provider": {
                "job_type": payload.get("job", {}).get("type")
                if isinstance(payload.get("job"), dict)
                else None,
            },
        },
    )


def _speech_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise AdapterError(
            provider_id="speechmatics",
            message="Speechmatics transcript was missing results",
            category="malformed_response",
            retryable=False,
        )
    output: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("type") not in {None, "word", "punctuation"}:
            continue
        output.append(result)
    return _merge_punctuation(output)


def _merge_punctuation(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for result in results:
        if result.get("type") == "punctuation" and merged:
            merged[-1] = {
                **merged[-1],
                "alternatives": [
                    {
                        **merged[-1]["alternatives"][0],
                        "content": (
                            f"{merged[-1]['alternatives'][0].get('content', '')}"
                            f"{_content_from_result(result)}"
                        ),
                    }
                ],
            }
            continue
        merged.append(result)
    return merged


def _content_from_result(result: dict[str, Any]) -> str:
    alternatives = result.get("alternatives")
    if isinstance(alternatives, list) and alternatives:
        first = alternatives[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return ""


def _alternative_confidence(result: dict[str, Any]) -> float | None:
    alternatives = result.get("alternatives")
    if isinstance(alternatives, list) and alternatives:
        first = alternatives[0]
        if isinstance(first, dict):
            confidence = first.get("confidence")
            if isinstance(confidence, (int, float)):
                return float(confidence)
    return None


def _speaker_name(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return f"speaker-{str(value).strip()}"


def _seconds_to_ms(value: object) -> int:
    if value is None:
        return 0
    try:
        numeric = value if isinstance(value, int | float | str) else 0.0
        return max(int(round(float(numeric) * 1000)), 0)
    except TypeError, ValueError:
        return 0


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


def _normalized_job_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if isinstance(status, str) and status.strip():
        return payload

    job = payload.get("job")
    if isinstance(job, dict):
        nested_status = job.get("status")
        if isinstance(nested_status, str) and nested_status.strip():
            return {**payload, "status": nested_status}

    jobs = payload.get("jobs")
    if isinstance(jobs, list) and jobs:
        first_job = jobs[0]
        if isinstance(first_job, dict):
            nested_status = first_job.get("status")
            if isinstance(nested_status, str) and nested_status.strip():
                return {**payload, "status": nested_status}

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
