"""OpenAI translation adapter."""

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
    classify_http_error,
    perform_with_retries,
)
from translation_agent.models import (
    RequestContext,
    Segment,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.storage import BlobStore, job_path


class OpenAITranslationAdapter:
    """Direct OpenAI Responses API adapter for translation candidate generation."""

    def __init__(
        self,
        *,
        api_key: str,
        blob_store: BlobStore,
        model_id: str = "gpt-5.4-mini",
        prompt_version: str = "phase-3-v1",
        base_url: str = "https://api.openai.com/v1/responses",
        timeout_seconds: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._blob_store = blob_store
        self._prompt_version = prompt_version
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._retry_policy = retry_policy or RetryPolicy()
        self._transport = transport or StdlibHttpTransport()
        self._sleep = sleep or (lambda seconds: __import__("time").sleep(seconds))

    def generate_translation(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> TranslationCandidate:
        raw_payload = perform_with_retries(
            lambda: self._generate_once(final_transcript, prompt_variant_id, request_context),
            provider_id="openai",
            retry_policy=self._retry_policy,
            sleep=self._sleep,
        )
        raw_response_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"openai-{prompt_variant_id}.json",
        )
        self._blob_store.put_bytes(
            raw_response_ref,
            (json.dumps(raw_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        translation_payload = _extract_translation_payload(raw_payload)
        return _candidate_from_translation_payload(
            translation_payload,
            response_payload=raw_payload,
            final_transcript=final_transcript,
            job_id=request_context.job.job_id,
            language=request_context.job.target_language,
            prompt_variant_id=prompt_variant_id,
            prompt_version=self._prompt_version,
            model_id=self.model_id,
            raw_response_ref=raw_response_ref,
        )

    def _generate_once(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> dict[str, Any]:
        body = {
            "model": self.model_id,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _system_prompt(
                                target_language=request_context.job.target_language,
                                prompt_variant_id=prompt_variant_id,
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _user_prompt(final_transcript),
                        }
                    ],
                },
            ],
        }
        response = self._transport.request(
            HttpRequest(
                method="POST",
                url=self._base_url,
                headers={
                    "authorization": f"Bearer {self._api_key}",
                    "content-type": "application/json",
                },
                body=json.dumps(body).encode("utf-8"),
                timeout_seconds=self._timeout_seconds,
            )
        )
        error = classify_http_error("openai", response)
        if error is not None:
            raise error
        payload = response.json()
        if not isinstance(payload, dict):
            raise AdapterError(
                provider_id="openai",
                message="OpenAI response must be a JSON object",
                category="malformed_response",
                retryable=False,
            )
        return payload


def _system_prompt(*, target_language: str, prompt_variant_id: str) -> str:
    if prompt_variant_id == "variant-b":
        directive = "Preserve tone and idioms when they remain faithful."
    else:
        directive = "Prefer literal fidelity and stable terminology."
    return (
        "Translate the transcript into "
        f"{target_language}. Return JSON only with keys full_text and segments. "
        "Each segment must contain segment_id and target_text. "
        f"{directive}"
    )


def _user_prompt(final_transcript: TranscriptCandidate) -> str:
    payload = {
        "full_text": final_transcript.full_text,
        "segments": [
            {
                "segment_id": segment.segment_id,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "speaker": segment.speaker,
                "source_text": segment.source_text,
            }
            for segment in final_transcript.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def _extract_translation_payload(response_payload: dict[str, Any]) -> dict[str, Any]:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return _parse_model_json(output_text)

    output = response_payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return _parse_model_json(text)

    raise AdapterError(
        provider_id="openai",
        message="OpenAI response did not contain output text",
        category="malformed_response",
        retryable=False,
    )


def _parse_model_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError(
            provider_id="openai",
            message="model output was not valid JSON",
            category="malformed_response",
            retryable=False,
        ) from exc
    if not isinstance(payload, dict):
        raise AdapterError(
            provider_id="openai",
            message="model output must be a JSON object",
            category="malformed_response",
            retryable=False,
        )
    return payload


def _candidate_from_translation_payload(
    payload: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    final_transcript: TranscriptCandidate,
    job_id: str,
    language: str,
    prompt_variant_id: str,
    prompt_version: str,
    model_id: str,
    raw_response_ref: str,
) -> TranslationCandidate:
    full_text = payload.get("full_text")
    if not isinstance(full_text, str) or not full_text.strip():
        raise AdapterError(
            provider_id="openai",
            message="translation payload was missing full_text",
            category="malformed_response",
            retryable=False,
        )
    translated_segments = _merge_segments(
        final_transcript.segments,
        payload.get("segments"),
    )
    return TranslationCandidate(
        candidate_id=f"tl-{prompt_variant_id}-{job_id}",
        job_id=job_id,
        source_transcript_candidate_id=final_transcript.candidate_id,
        model_id=model_id,
        prompt_variant_id=prompt_variant_id,
        prompt_version=prompt_version,
        language=language,
        segments=translated_segments,
        full_text=full_text.strip(),
        raw_response_ref=raw_response_ref,
        normalization_version="raw-openai-v1",
        metadata={
            "provider": {
                "provider_id": "openai",
                "provider_request_id": _provider_request_id(response_payload),
                "response_id": _provider_request_id(response_payload),
            },
            "prompt": {
                "variant_id": prompt_variant_id,
                "version": prompt_version,
            },
        },
    )


def _merge_segments(
    source_segments: tuple[Segment, ...],
    payload_segments: object,
) -> tuple[Segment, ...]:
    translations_by_id: dict[str, str] = {}
    if isinstance(payload_segments, list):
        for item in payload_segments:
            if not isinstance(item, dict):
                continue
            segment_id = item.get("segment_id")
            target_text = item.get("target_text")
            if isinstance(segment_id, str) and isinstance(target_text, str):
                translations_by_id[segment_id] = target_text.strip()

    translated: list[Segment] = []
    for segment in source_segments:
        translated.append(
            segment.model_copy(
                update={
                    "target_text": translations_by_id.get(segment.segment_id, segment.source_text),
                }
            )
        )
    return tuple(translated)


def _provider_request_id(response_payload: dict[str, Any]) -> str | None:
    response_id = response_payload.get("id")
    if isinstance(response_id, str) and response_id.strip():
        return response_id.strip()
    return None
