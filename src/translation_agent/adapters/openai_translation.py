"""OpenAI translation adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
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
from translation_agent.parallelism import ordered_parallel_map
from translation_agent.storage import BlobStore, job_path, job_scope_token

DEFAULT_MAX_CHUNK_CHARACTERS = 5_000
DEFAULT_MAX_CHUNK_SEGMENTS = 100
DEFAULT_CONTEXT_SEGMENT_WINDOW = 2
MAX_SEGMENT_TARGET_EXPANSION_RATIO = 4
MAX_SEGMENT_TARGET_EXPANSION_MARGIN = 80
SEGMENT_DUPLICATION_LOOKAHEAD = 4
MIN_DUPLICATED_SEGMENT_TEXT_LENGTH = 24


@dataclass(frozen=True, slots=True)
class _TranslationChunk:
    chunk_key: str
    chunk_index: int
    segments: tuple[Segment, ...]
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()

    @property
    def full_text(self) -> str:
        return " ".join(
            text.strip()
            for segment in self.segments
            if isinstance((text := segment.source_text), str) and text.strip()
        )


@dataclass(frozen=True, slots=True)
class _ChunkTranslationResult:
    full_text: str
    segments: tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class _ChunkExecutionResult:
    chunk_index: int
    attempts: tuple[dict[str, Any], ...]
    segments: tuple[Segment, ...]
    chunk_texts: tuple[str, ...]
    response_ids: tuple[str, ...]


class OpenAITranslationAdapter:
    """Direct OpenAI Responses API adapter for translation candidate generation."""

    provider_id = "openai"

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
        max_chunk_workers: int = 4,
        max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
        max_chunk_segments: int = DEFAULT_MAX_CHUNK_SEGMENTS,
        context_segment_window: int = DEFAULT_CONTEXT_SEGMENT_WINDOW,
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
        self._max_chunk_workers = max(1, max_chunk_workers)
        self._max_chunk_characters = max(1, max_chunk_characters)
        self._max_chunk_segments = max(1, max_chunk_segments)
        self._context_segment_window = max(0, context_segment_window)

    def generate_translation(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> TranslationCandidate:
        candidate, raw_payload = self.generate_translation_with_payload(
            final_transcript,
            prompt_variant_id,
            request_context,
        )
        self._blob_store.put_bytes(
            candidate.raw_response_ref or "",
            (json.dumps(raw_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return candidate

    def generate_translation_with_payload(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> tuple[TranslationCandidate, dict[str, Any]]:
        resolved_prompt = _resolved_prompt_payload(request_context)
        chunks = _chunk_transcript(
            final_transcript,
            max_chunk_characters=self._max_chunk_characters,
            max_chunk_segments=self._max_chunk_segments,
            context_segment_window=self._context_segment_window,
        )
        raw_chunk_payloads: list[dict[str, Any]] = []
        translated_segments: list[Segment] = []
        chunk_full_texts: list[str] = []
        response_ids: list[str] = []
        chunk_results = ordered_parallel_map(
            chunks,
            max_workers=self._max_chunk_workers,
            worker=lambda chunk: self._translate_chunk_with_fallback(
                chunk,
                prompt_variant_id=prompt_variant_id,
                request_context=request_context,
            ),
            sort_key=lambda _input_index, chunk: (chunk.chunk_index,),
        )
        for result in chunk_results:
            if result.error is not None:
                raise result.error
            chunk_result = result.value
            if chunk_result is None:  # pragma: no cover - defensive
                raise RuntimeError("chunk translation completed without a payload")
            raw_chunk_payloads.extend(chunk_result.attempts)
            translated_segments.extend(chunk_result.segments)
            chunk_full_texts.extend(chunk_result.chunk_texts)
            response_ids.extend(chunk_result.response_ids)
        raw_response_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"openai-{prompt_variant_id}.json",
        )
        raw_payload = {
            "model": self.model_id,
            "prompt_variant_id": prompt_variant_id,
            "prompt_version": str(
                resolved_prompt.get("effective_prompt_version", self._prompt_version)
            ),
            "chunking": {
                "chunk_count": len(chunk_full_texts),
                "planned_chunk_count": len(chunks),
                "executed_request_count": len(
                    [
                        payload
                        for payload in raw_chunk_payloads
                        if payload.get("status") == "translated"
                    ]
                ),
                "max_chunk_workers": self._max_chunk_workers,
                "max_chunk_characters": self._max_chunk_characters,
                "max_chunk_segments": self._max_chunk_segments,
                "context_segment_window": self._context_segment_window,
            },
            "chunks": raw_chunk_payloads,
        }
        return (
            _candidate_from_chunk_results(
                translated_segments=tuple(translated_segments),
                full_text=" ".join(text for text in chunk_full_texts if text).strip(),
                chunk_count=len(chunk_full_texts),
                response_ids=tuple(response_ids),
                final_transcript=final_transcript,
                request_context=request_context,
                language=request_context.job.target_language,
                prompt_variant_id=prompt_variant_id,
                prompt_version=str(
                    resolved_prompt.get("effective_prompt_version", self._prompt_version)
                ),
                model_id=self.model_id,
                raw_response_ref=raw_response_ref,
            ),
            raw_payload,
        )

    def _generate_once(
        self,
        chunk: _TranslationChunk,
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
                                source_language=request_context.job.source_language,
                                target_language=request_context.job.target_language,
                                prompt_variant_id=prompt_variant_id,
                                resolved_prompt=request_context.metadata.get(
                                    "resolved_translation_prompt"
                                ),
                                historical_instructions=_historical_instructions(request_context),
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _user_prompt(chunk),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "translation_chunk",
                    "strict": True,
                    "schema": _translation_schema(),
                }
            },
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
        error = _classify_openai_http_error(response)
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

    def _translate_chunk_with_fallback(
        self,
        chunk: _TranslationChunk,
        *,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> _ChunkExecutionResult:
        raw_payload: dict[str, Any] | None = None
        try:
            raw_payload = perform_with_retries(
                lambda: self._generate_once(chunk, prompt_variant_id, request_context),
                provider_id="openai",
                retry_policy=self._retry_policy,
                sleep=self._sleep,
            )
            translation_payload = _extract_translation_payload(raw_payload)
            chunk_result = _chunk_translation_from_payload(translation_payload, chunk=chunk)
        except AdapterError as exc:
            if _should_split_chunk(chunk, exc):
                left_chunk, right_chunk = _split_chunk(
                    chunk,
                    context_segment_window=self._context_segment_window,
                )
                left_result = self._translate_chunk_with_fallback(
                    left_chunk,
                    prompt_variant_id=prompt_variant_id,
                    request_context=request_context,
                )
                right_result = self._translate_chunk_with_fallback(
                    right_chunk,
                    prompt_variant_id=prompt_variant_id,
                    request_context=request_context,
                )
                fallback_record = _chunk_attempt_record(
                    chunk,
                    status=(
                        "split_after_retryable_failure"
                        if exc.retryable
                        else "split_after_validation_failure"
                    ),
                    response=raw_payload,
                    error=exc,
                    fallback_children=(left_chunk.chunk_key, right_chunk.chunk_key),
                )
                return _ChunkExecutionResult(
                    chunk_index=chunk.chunk_index,
                    attempts=(fallback_record, *left_result.attempts, *right_result.attempts),
                    segments=(*left_result.segments, *right_result.segments),
                    chunk_texts=(*left_result.chunk_texts, *right_result.chunk_texts),
                    response_ids=(*left_result.response_ids, *right_result.response_ids),
                )
            raise
        response_id = _provider_request_id(raw_payload)
        response_ids = (response_id,) if response_id is not None else ()
        return _ChunkExecutionResult(
            chunk_index=chunk.chunk_index,
            attempts=(
                _chunk_attempt_record(
                    chunk,
                    status="translated",
                    response=raw_payload,
                ),
            ),
            segments=chunk_result.segments,
            chunk_texts=(chunk_result.full_text,),
            response_ids=response_ids,
        )


def _classify_openai_http_error(response) -> AdapterError | None:
    error = classify_http_error("openai", response)
    if error is None or response.status_code != 429:
        return error

    payload = _openai_error_payload(response.body)
    error_details = payload.get("error")
    if not isinstance(error_details, dict):
        return error
    error_type = error_details.get("type")
    error_code = error_details.get("code")
    if error_type != "insufficient_quota" and error_code != "insufficient_quota":
        return error

    return AdapterError(
        provider_id="openai",
        message=str(error_details.get("message") or error),
        category="quota_exhausted",
        retryable=False,
        status_code=response.status_code,
    )


def _openai_error_payload(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _system_prompt(
    *,
    source_language: str,
    target_language: str,
    prompt_variant_id: str,
    resolved_prompt: object | None = None,
    historical_instructions: tuple[str, ...] = (),
) -> str:
    if prompt_variant_id == "variant-b":
        directive = "Preserve tone and idioms when they remain faithful."
    else:
        directive = "Prefer literal fidelity and stable terminology."
    instructions = _resolved_prompt_instructions(resolved_prompt)
    prompt = (
        "Translate the provided transcript chunk from "
        f"{source_language} into {target_language}. Return JSON only with keys "
        "full_text and segments. Each segment must contain segment_id and "
        "target_text. Translate only the segments in chunk.segments. Use "
        "context_before and context_after only for disambiguation; do not translate "
        "or mention them. The response must include every chunk segment_id exactly "
        "once and no extra segment_ids. "
        f"{directive}"
    )
    if instructions:
        prompt += " Additional approved guidance: " + " ".join(instructions)
    if historical_instructions:
        prompt += " Historical guidance from prior operator review: " + " ".join(
            historical_instructions
        )
    return prompt


def _resolved_prompt_instructions(resolved_prompt: object | None) -> tuple[str, ...]:
    if not isinstance(resolved_prompt, dict):
        return ()
    raw_instructions = resolved_prompt.get("instructions")
    if not isinstance(raw_instructions, list):
        return ()
    return tuple(item for item in raw_instructions if isinstance(item, str) and item.strip())


def _resolved_prompt_payload(request_context: RequestContext) -> dict[str, Any]:
    payload = request_context.metadata.get("resolved_translation_prompt")
    if isinstance(payload, dict):
        return payload
    return {}


def _historical_instructions(request_context: RequestContext) -> tuple[str, ...]:
    payload = request_context.metadata.get("historical_translation_instructions")
    if not isinstance(payload, list):
        return ()
    return tuple(item for item in payload if isinstance(item, str) and item.strip())


def _user_prompt(chunk: _TranslationChunk) -> str:
    payload = {
        "chunk": {
            "chunk_index": chunk.chunk_index,
            "full_text": chunk.full_text,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "speaker": segment.speaker,
                    "source_text": segment.source_text,
                }
                for segment in chunk.segments
            ],
        },
        "context_before": list(chunk.context_before),
        "context_after": list(chunk.context_after),
    }
    return json.dumps(payload, ensure_ascii=True)


def _translation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "full_text": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "target_text": {"type": "string"},
                    },
                    "required": ["segment_id", "target_text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["full_text", "segments"],
        "additionalProperties": False,
    }


def _chunk_transcript(
    final_transcript: TranscriptCandidate,
    *,
    max_chunk_characters: int,
    max_chunk_segments: int,
    context_segment_window: int,
) -> tuple[_TranslationChunk, ...]:
    if not final_transcript.segments:
        return (_TranslationChunk(chunk_key="chunk-0", chunk_index=0, segments=()),)

    groups: list[tuple[Segment, ...]] = []
    current_group: list[Segment] = []
    current_characters = 0
    for segment in final_transcript.segments:
        segment_text = (segment.source_text or "").strip()
        projected_characters = current_characters + len(segment_text)
        if current_group and (
            len(current_group) >= max_chunk_segments or projected_characters > max_chunk_characters
        ):
            groups.append(tuple(current_group))
            current_group = []
            current_characters = 0
        current_group.append(segment)
        current_characters += len(segment_text)
    if current_group:
        groups.append(tuple(current_group))

    chunked: list[_TranslationChunk] = []
    start_index = 0
    for chunk_index, group in enumerate(groups):
        end_index = start_index + len(group)
        chunked.append(
            _TranslationChunk(
                chunk_key=f"chunk-{chunk_index}",
                chunk_index=chunk_index,
                segments=group,
                context_before=_context_texts(
                    final_transcript.segments,
                    max(start_index - context_segment_window, 0),
                    start_index,
                ),
                context_after=_context_texts(
                    final_transcript.segments,
                    end_index,
                    min(end_index + context_segment_window, len(final_transcript.segments)),
                ),
            )
        )
        start_index = end_index
    return tuple(chunked)


def _should_split_chunk(chunk: _TranslationChunk, error: AdapterError) -> bool:
    if len(chunk.segments) <= 1:
        return False
    if error.retryable:
        return True
    if error.category == "translation_validation":
        return True
    return error.category == "malformed_response" and str(error).startswith(
        "translation payload was missing segment translations"
    )


def _split_chunk(
    chunk: _TranslationChunk,
    *,
    context_segment_window: int,
) -> tuple[_TranslationChunk, _TranslationChunk]:
    midpoint = max(1, len(chunk.segments) // 2)
    left_segments = chunk.segments[:midpoint]
    right_segments = chunk.segments[midpoint:]
    if not left_segments or not right_segments:
        raise AdapterError(
            provider_id="openai",
            message="cannot split translation chunk further",
            category="chunking_error",
            retryable=False,
        )
    left_context_after = tuple(
        segment.source_text.strip()
        for segment in right_segments[:context_segment_window]
        if isinstance(segment.source_text, str) and segment.source_text.strip()
    )
    right_context_before = tuple(
        segment.source_text.strip()
        for segment in left_segments[-context_segment_window:]
        if isinstance(segment.source_text, str) and segment.source_text.strip()
    )
    return (
        _TranslationChunk(
            chunk_key=f"{chunk.chunk_key}.a",
            chunk_index=chunk.chunk_index,
            segments=left_segments,
            context_before=chunk.context_before,
            context_after=left_context_after,
        ),
        _TranslationChunk(
            chunk_key=f"{chunk.chunk_key}.b",
            chunk_index=chunk.chunk_index,
            segments=right_segments,
            context_before=right_context_before,
            context_after=chunk.context_after,
        ),
    )


def _context_texts(
    segments: tuple[Segment, ...],
    start_index: int,
    end_index: int,
) -> tuple[str, ...]:
    return tuple(
        text
        for segment in segments[start_index:end_index]
        if isinstance((text := segment.source_text), str) and text.strip()
    )


def _chunk_translation_from_payload(
    payload: dict[str, Any],
    *,
    chunk: _TranslationChunk,
) -> _ChunkTranslationResult:
    full_text = payload.get("full_text")
    if not isinstance(full_text, str) or not full_text.strip():
        raise AdapterError(
            provider_id="openai",
            message="translation payload was missing full_text",
            category="malformed_response",
            retryable=False,
        )
    return _ChunkTranslationResult(
        full_text=full_text.strip(),
        segments=_merge_segments(chunk.segments, payload.get("segments")),
    )


def _candidate_from_chunk_results(
    *,
    translated_segments: tuple[Segment, ...],
    full_text: str,
    chunk_count: int,
    response_ids: tuple[str, ...],
    final_transcript: TranscriptCandidate,
    request_context: RequestContext,
    language: str,
    prompt_variant_id: str,
    prompt_version: str,
    model_id: str,
    raw_response_ref: str,
) -> TranslationCandidate:
    provider_request_id = response_ids[0] if response_ids else None
    provider_metadata: dict[str, Any] = {
        "provider_id": "openai",
        "provider_request_id": provider_request_id,
        "response_id": provider_request_id,
    }
    if response_ids:
        provider_metadata["response_ids"] = list(response_ids)
    return TranslationCandidate(
        candidate_id=(
            f"tl-{prompt_variant_id}-{request_context.job.job_id}-"
            f"{job_scope_token(request_context.job)}"
        ),
        job_id=request_context.job.job_id,
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
            "provider": provider_metadata,
            "prompt": {
                "variant_id": prompt_variant_id,
                "version": prompt_version,
            },
            "prompt_resolver": _resolved_prompt_payload(request_context),
            "chunking": {
                "chunk_count": chunk_count,
                "response_count": len(response_ids),
            },
        },
    )


def _chunk_attempt_record(
    chunk: _TranslationChunk,
    *,
    status: str,
    response: dict[str, Any] | None = None,
    error: AdapterError | None = None,
    fallback_children: tuple[str, str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chunk_key": chunk.chunk_key,
        "chunk_index": chunk.chunk_index,
        "status": status,
        "segment_ids": [segment.segment_id for segment in chunk.segments],
        "source_text": chunk.full_text,
        "context_before": list(chunk.context_before),
        "context_after": list(chunk.context_after),
    }
    if response is not None:
        record["response"] = response
    if error is not None:
        record["error"] = error.as_metadata()
    if fallback_children is not None:
        record["fallback_children"] = list(fallback_children)
    return record


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
    request_context: RequestContext,
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
        candidate_id=(
            f"tl-{prompt_variant_id}-{request_context.job.job_id}-"
            f"{job_scope_token(request_context.job)}"
        ),
        job_id=request_context.job.job_id,
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
            "prompt_resolver": _resolved_prompt_payload(request_context),
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
                normalized_target = target_text.strip()
                if normalized_target:
                    translations_by_id[segment_id] = normalized_target

    translated: list[Segment] = []
    missing_segment_ids: list[str] = []
    for segment in source_segments:
        target_text = translations_by_id.get(segment.segment_id)
        if target_text is None:
            missing_segment_ids.append(segment.segment_id)
            continue
        _validate_segment_translation(segment, target_text)
        translated.append(
            segment.model_copy(
                update={
                    "target_text": target_text,
                }
            )
        )
    if missing_segment_ids:
        raise AdapterError(
            provider_id="openai",
            message=(
                "translation payload was missing segment translations for "
                + ", ".join(missing_segment_ids)
            ),
            category="malformed_response",
            retryable=False,
        )
    translated_tuple = tuple(translated)
    _validate_chunk_segment_alignment(translated_tuple)
    return translated_tuple


def _validate_segment_translation(segment: Segment, target_text: str) -> None:
    source_text = (segment.source_text or "").strip()
    if not source_text:
        return
    source_length = len(source_text)
    target_length = len(target_text)
    max_target_length = max(
        source_length * MAX_SEGMENT_TARGET_EXPANSION_RATIO,
        source_length + MAX_SEGMENT_TARGET_EXPANSION_MARGIN,
    )
    if target_length <= max_target_length:
        return
    raise AdapterError(
        provider_id="openai",
        message=(
            "translation payload produced implausibly long text for "
            f"{segment.segment_id}: source_length={source_length} "
            f"target_length={target_length}"
        ),
        category="translation_validation",
        retryable=False,
    )


def _validate_chunk_segment_alignment(segments: tuple[Segment, ...]) -> None:
    for index, segment in enumerate(segments):
        current_text = _normalized_translation_text(segment.target_text or "")
        if len(current_text) < MIN_DUPLICATED_SEGMENT_TEXT_LENGTH:
            continue
        for later_segment in segments[index + 1 : index + 1 + SEGMENT_DUPLICATION_LOOKAHEAD]:
            later_text = _normalized_translation_text(later_segment.target_text or "")
            if len(later_text) < MIN_DUPLICATED_SEGMENT_TEXT_LENGTH:
                continue
            if later_text in current_text:
                raise AdapterError(
                    provider_id="openai",
                    message=(
                        "translation payload duplicated later segment text: "
                        f"{segment.segment_id} contains {later_segment.segment_id}"
                    ),
                    category="translation_validation",
                    retryable=False,
                )


def _normalized_translation_text(text: str) -> str:
    return "".join(character for character in text if character.isalnum())


def _provider_request_id(response_payload: dict[str, Any]) -> str | None:
    response_id = response_payload.get("id")
    if isinstance(response_id, str) and response_id.strip():
        return response_id.strip()
    return None
