"""Canonical normalization helpers for transcript and translation candidates."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from typing import Any

from translation_agent.models import Segment, TranscriptCandidate, TranslationCandidate

CURRENT_NORMALIZATION_VERSION = "2026-03-30-phase-3"


def normalize_transcript_candidate(candidate: TranscriptCandidate) -> TranscriptCandidate:
    """Normalize transcript content, metadata, and segment ordering."""

    candidate_id = _normalize_identifier(candidate.candidate_id, fallback="transcript-candidate")
    provider_id = _normalize_identifier(candidate.provider_id, fallback="unknown-provider")
    provider_request_id = _normalize_optional_identifier(candidate.provider_request_id)
    raw_payload_ref = _normalize_ref(candidate.raw_payload_ref)
    normalized_segments = tuple(
        _normalize_segment(segment, index=index, prefer_target=False)
        for index, segment in enumerate(_sorted_segments(candidate.segments), start=1)
    )
    full_text = _collapse_text(
        segment.source_text for segment in normalized_segments if segment.source_text
    )
    metadata = _normalized_metadata(candidate.metadata)
    provider_metadata = _normalized_child_metadata(metadata, "provider")
    provider_metadata["provider_id"] = provider_id
    provider_metadata["provider_request_id"] = provider_request_id
    metadata["candidate_kind"] = "transcript"
    metadata["raw_payload_ref"] = raw_payload_ref
    metadata["speaker_labels_present"] = any(segment.speaker for segment in normalized_segments)
    return candidate.model_copy(
        update={
            "candidate_id": candidate_id,
            "provider_id": provider_id,
            "provider_request_id": provider_request_id,
            "segments": normalized_segments,
            "full_text": full_text or _normalize_text(candidate.full_text),
            "speaker_map": _normalize_speaker_map(candidate.speaker_map),
            "timing_resolution": candidate.timing_resolution or "segment",
            "raw_payload_ref": raw_payload_ref,
            "normalization_version": CURRENT_NORMALIZATION_VERSION,
            "metadata": metadata,
        }
    )


def normalize_translation_candidate(candidate: TranslationCandidate) -> TranslationCandidate:
    """Normalize translation text, prompt metadata, and segment ordering."""

    candidate_id = _normalize_identifier(candidate.candidate_id, fallback="translation-candidate")
    source_candidate_id = _normalize_optional_identifier(candidate.source_transcript_candidate_id)
    final_transcript_ref = _normalize_ref(candidate.final_transcript_ref)
    model_id = _normalize_identifier(candidate.model_id, fallback="unknown-model")
    prompt_variant_id = _normalize_identifier(candidate.prompt_variant_id, fallback="variant")
    prompt_version = _normalize_identifier(candidate.prompt_version, fallback="version")
    raw_response_ref = _normalize_ref(candidate.raw_response_ref)
    normalized_segments = tuple(
        _normalize_segment(segment, index=index, prefer_target=True)
        for index, segment in enumerate(_sorted_segments(candidate.segments), start=1)
    )
    full_text = _collapse_text(
        segment.target_text for segment in normalized_segments if segment.target_text
    )
    metadata = _normalized_metadata(candidate.metadata)
    provider_metadata = _normalized_child_metadata(metadata, "provider")
    provider_metadata["provider_id"] = _normalize_identifier(
        _metadata_string(provider_metadata, "provider_id"),
        fallback="openai",
    )
    provider_metadata["provider_request_id"] = _normalize_optional_identifier(
        _metadata_string(provider_metadata, "provider_request_id")
        or _metadata_string(provider_metadata, "response_id")
    )
    if provider_metadata["provider_request_id"] is not None:
        provider_metadata["response_id"] = provider_metadata["provider_request_id"]
    prompt_metadata = _normalized_child_metadata(metadata, "prompt")
    prompt_metadata["variant_id"] = prompt_variant_id
    prompt_metadata["version"] = prompt_version
    prompt_metadata["model_id"] = model_id
    metadata["candidate_kind"] = "translation"
    metadata["raw_response_ref"] = raw_response_ref
    metadata["source_transcript_candidate_id"] = source_candidate_id
    return candidate.model_copy(
        update={
            "candidate_id": candidate_id,
            "source_transcript_candidate_id": source_candidate_id,
            "final_transcript_ref": final_transcript_ref,
            "model_id": model_id,
            "prompt_variant_id": prompt_variant_id,
            "prompt_version": prompt_version,
            "segments": normalized_segments,
            "full_text": full_text or _normalize_text(candidate.full_text),
            "raw_response_ref": raw_response_ref,
            "normalization_version": CURRENT_NORMALIZATION_VERSION,
            "metadata": metadata,
        }
    )


def _normalize_segment(segment: Segment, *, index: int, prefer_target: bool) -> Segment:
    source_text = _normalize_text(segment.source_text)
    target_text = _normalize_text(segment.target_text)
    annotations = dict(segment.annotations)
    annotations.setdefault("normalized", True)
    annotations.setdefault("segment_index", index)
    canonical_text = target_text if prefer_target and target_text else source_text
    annotations["source_span_id"] = _source_span_id(
        segment, source_text=source_text, target_text=target_text
    )
    return segment.model_copy(
        update={
            "segment_id": _normalize_identifier(segment.segment_id, fallback=f"segment-{index}"),
            "start_ms": max(segment.start_ms, 0),
            "end_ms": max(segment.end_ms, segment.start_ms, 0),
            "speaker": _normalize_speaker(segment.speaker),
            "source_text": source_text,
            "target_text": target_text,
            "annotations": _annotated_text_lengths(annotations, canonical_text),
        }
    )


def _sorted_segments(segments: Iterable[Segment]) -> list[Segment]:
    return sorted(
        segments,
        key=lambda segment: (segment.start_ms, segment.end_ms, segment.segment_id),
    )


def _collapse_text(parts: Iterable[str]) -> str:
    values = [value for value in (_normalize_text(part) for part in parts) if value]
    return " ".join(values)


def _normalize_speaker_map(speaker_map: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in speaker_map.items():
        normalized_key = _normalize_optional_identifier(key)
        normalized_value = _normalize_speaker(value) or _normalize_optional_identifier(value)
        if normalized_key and normalized_value:
            normalized[normalized_key] = normalized_value
    return normalized


def _normalize_speaker(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.replace("_", " ").split())
    if not cleaned:
        return None
    lower = cleaned.lower()
    if lower.startswith("speaker ") or lower.startswith("speaker-"):
        suffix = cleaned.split(maxsplit=1)[1] if " " in cleaned else cleaned.split("-", 1)[1]
        suffix = "-".join(part for part in suffix.lower().split() if part)
        return f"speaker-{suffix}" if suffix else "speaker"
    return cleaned


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _normalize_identifier(value: str | None, *, fallback: str) -> str:
    normalized = _normalize_optional_identifier(value)
    return normalized or fallback


def _normalize_optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _normalize_ref(value: str | None) -> str | None:
    return _normalize_optional_identifier(value)


def _normalized_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(metadata)


def _normalized_child_metadata(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    existing = metadata.get(key)
    child = dict(existing) if isinstance(existing, dict) else {}
    metadata[key] = child
    return child


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    return None


def _annotated_text_lengths(annotations: dict[str, Any], text: str | None) -> dict[str, Any]:
    if text is not None:
        annotations.setdefault("text_length", len(text))
    return annotations


def _source_span_id(
    segment: Segment,
    *,
    source_text: str | None,
    target_text: str | None,
) -> str:
    existing = segment.annotations.get("source_span_id")
    if isinstance(existing, str) and existing.strip():
        return existing
    if segment.start_ms != 0 or segment.end_ms != 0:
        start_bucket = _round_ms(segment.start_ms)
        end_bucket = _round_ms(segment.end_ms)
        return f"span:{start_bucket}:{end_bucket}"
    text_seed = source_text or target_text or segment.segment_id
    digest = sha256(text_seed.encode("utf-8")).hexdigest()[:12]
    return f"text-span:{digest}"


def _round_ms(value: int) -> int:
    return int(round(value / 250.0) * 250)
