"""Canonical normalization helpers for transcript and translation candidates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from hashlib import sha256
from math import ceil
from typing import Any

from translation_agent.models import Segment, TranscriptCandidate, TranslationCandidate

CURRENT_NORMALIZATION_VERSION = "2026-04-03-phase-4"
MAX_TRANSCRIPT_SEGMENT_DURATION_MS = 15_000
TRANSCRIPT_PUNCTUATION = ".!?;:,。！？；：，、…"
TRANSCRIPT_WORD_RE = re.compile(r"\S+\s*")


def normalize_transcript_candidate(candidate: TranscriptCandidate) -> TranscriptCandidate:
    """Normalize transcript content, metadata, and segment ordering."""

    candidate_id = _normalize_identifier(candidate.candidate_id, fallback="transcript-candidate")
    provider_id = _normalize_identifier(candidate.provider_id, fallback="unknown-provider")
    provider_request_id = _normalize_optional_identifier(candidate.provider_request_id)
    raw_payload_ref = _normalize_ref(candidate.raw_payload_ref)
    prepared_segments = tuple(
        _normalize_segment(segment, index=index, prefer_target=False)
        for index, segment in enumerate(_sorted_segments(candidate.segments), start=1)
    )
    resegmented_segments, resegmentation_metadata = _resegment_transcript_segments(
        prepared_segments
    )
    normalized_segments = tuple(
        _normalize_segment(segment, index=index, prefer_target=False)
        for index, segment in enumerate(resegmented_segments, start=1)
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
    metadata["transcript_segment_count_before"] = len(prepared_segments)
    metadata["transcript_segment_count_after"] = len(normalized_segments)
    metadata["transcript_max_segment_duration_ms_before"] = _max_segment_duration_ms(
        prepared_segments
    )
    metadata["transcript_max_segment_duration_ms_after"] = _max_segment_duration_ms(
        normalized_segments
    )
    metadata["transcript_overlong_segment_count_before"] = _overlong_segment_count(
        prepared_segments
    )
    metadata["transcript_overlong_segment_count_after"] = _overlong_segment_count(
        normalized_segments
    )
    metadata.update(resegmentation_metadata)
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
        fallback="unknown-provider",
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


def _resegment_transcript_segments(
    segments: tuple[Segment, ...],
) -> tuple[tuple[Segment, ...], dict[str, Any]]:
    output: list[Segment] = []
    split_source_segments = 0
    generated_segments = 0

    for segment in segments:
        split_segments = _split_transcript_segment(segment)
        if len(split_segments) > 1:
            split_source_segments += 1
            generated_segments += len(split_segments) - 1
        output.extend(split_segments)

    return tuple(output), {
        "transcript_resegmented": generated_segments > 0,
        "transcript_resegmented_source_segment_count": split_source_segments,
        "transcript_resegmented_generated_segment_count": generated_segments,
    }


def _split_transcript_segment(segment: Segment) -> tuple[Segment, ...]:
    duration_ms = max(segment.end_ms - segment.start_ms, 0)
    text = (segment.source_text or "").strip()
    if duration_ms <= MAX_TRANSCRIPT_SEGMENT_DURATION_MS or len(text) < 2:
        return (segment,)

    compact_script = _contains_compact_script(text)
    chunk_count = max(ceil(duration_ms / MAX_TRANSCRIPT_SEGMENT_DURATION_MS), 2)
    max_units_per_chunk = max(
        ceil(_text_unit_count(text, compact_script=compact_script) / chunk_count),
        1,
    )
    chunks = _chunk_transcript_text(
        text,
        compact_script=compact_script,
        max_units_per_chunk=max_units_per_chunk,
    )
    if len(chunks) == 1:
        return (segment,)

    while True:
        timed_chunks = _timed_text_chunks(
            segment.start_ms,
            segment.end_ms,
            chunks,
            compact_script=compact_script,
        )
        overlong_index = next(
            (
                index
                for index, (start_ms, end_ms, _) in enumerate(timed_chunks)
                if end_ms - start_ms > MAX_TRANSCRIPT_SEGMENT_DURATION_MS
            ),
            None,
        )
        if overlong_index is None:
            break
        split_chunks = _split_text_chunk_in_two(
            chunks[overlong_index],
            compact_script=compact_script,
        )
        if len(split_chunks) == 1:
            return (segment,)
        chunks = chunks[:overlong_index] + split_chunks + chunks[overlong_index + 1 :]

    parent_source_span_id = segment.annotations.get("source_span_id")
    split_count = len(timed_chunks)
    return tuple(
        Segment(
            segment_id=f"{segment.segment_id}--part-{index}",
            start_ms=start_ms,
            end_ms=end_ms,
            speaker=segment.speaker,
            source_text=chunk,
            target_text=segment.target_text,
            annotations={
                **{
                    key: value
                    for key, value in segment.annotations.items()
                    if key != "source_span_id"
                },
                "source_segment_id": segment.segment_id,
                "parent_source_span_id": parent_source_span_id,
                "transcript_resegmented": True,
                "transcript_resegment_index": index,
                "transcript_resegment_count": split_count,
                "transcript_resegment_reason": "duration",
                "provider_start_ms": segment.start_ms,
                "provider_end_ms": segment.end_ms,
            },
        )
        for index, (start_ms, end_ms, chunk) in enumerate(timed_chunks, start=1)
    )


def _chunk_transcript_text(
    text: str,
    *,
    compact_script: bool,
    max_units_per_chunk: int,
) -> list[str]:
    clauses = _split_into_clauses(
        text,
        compact_script=compact_script,
        max_units_per_chunk=max_units_per_chunk,
    )
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        candidate = _join_parts(current, clause, compact_script=compact_script)
        if (
            current
            and _text_unit_count(candidate, compact_script=compact_script) > max_units_per_chunk
        ):
            chunks.append(current)
            current = clause
            continue
        current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def _split_into_clauses(
    text: str,
    *,
    compact_script: bool,
    max_units_per_chunk: int,
) -> list[str]:
    raw_tokens = _compact_script_tokens(text) if compact_script else _default_script_tokens(text)
    clauses: list[str] = []
    current = ""
    for token in raw_tokens:
        current = _join_parts(current, token, compact_script=compact_script)
        if token.rstrip()[-1:] in TRANSCRIPT_PUNCTUATION:
            clauses.append(current)
            current = ""
    if current:
        clauses.append(current)
    bounded: list[str] = []
    for clause in clauses or [text]:
        bounded.extend(
            _split_oversized_clause(
                clause,
                compact_script=compact_script,
                max_units_per_chunk=max_units_per_chunk,
            )
        )
    return bounded


def _split_oversized_clause(
    text: str,
    *,
    compact_script: bool,
    max_units_per_chunk: int,
) -> list[str]:
    if _text_unit_count(text, compact_script=compact_script) <= max_units_per_chunk:
        return [text]
    split_chunks = _split_text_chunk_in_two(text, compact_script=compact_script)
    if len(split_chunks) == 1:
        return [text]
    bounded: list[str] = []
    for chunk in split_chunks:
        bounded.extend(
            _split_oversized_clause(
                chunk,
                compact_script=compact_script,
                max_units_per_chunk=max_units_per_chunk,
            )
        )
    return bounded


def _timed_text_chunks(
    start_ms: int,
    end_ms: int,
    chunks: list[str],
    *,
    compact_script: bool,
) -> list[tuple[int, int, str]]:
    if not chunks:
        return []

    total_duration = max(end_ms - start_ms, len(chunks))
    weights = [max(_text_unit_count(chunk, compact_script=compact_script), 1) for chunk in chunks]
    total_weight = sum(weights)
    boundaries = [0]
    cumulative_weight = 0
    for weight in weights[:-1]:
        cumulative_weight += weight
        boundaries.append(round(total_duration * cumulative_weight / total_weight))
    boundaries.append(total_duration)

    timed: list[tuple[int, int, str]] = []
    cursor = start_ms
    remaining_budget = total_duration
    for index, chunk in enumerate(chunks):
        chunk_duration = max(boundaries[index + 1] - boundaries[index], 1)
        remaining_chunks = len(chunks) - index - 1
        chunk_duration = min(chunk_duration, max(remaining_budget - remaining_chunks, 1))
        start_time = cursor
        end_time = min(start_time + chunk_duration, end_ms - remaining_chunks)
        if end_time <= start_time:
            end_time = min(start_time + 1, end_ms)
        timed.append((start_time, end_time, chunk))
        cursor = end_time
        remaining_budget = max(end_ms - cursor, 0)

    last_start, _, last_chunk = timed[-1]
    timed[-1] = (last_start, end_ms, last_chunk)
    return timed


def _split_text_chunk_in_two(text: str, *, compact_script: bool) -> list[str]:
    if _text_unit_count(text, compact_script=compact_script) <= 1:
        return [text]
    split_index = _best_split_index(text, compact_script=compact_script)
    if split_index <= 0 or split_index >= len(text):
        return [text]
    left = text[:split_index].strip()
    right = text[split_index:].strip()
    if not left or not right:
        return [text]
    return [left, right]


def _best_split_index(text: str, *, compact_script: bool) -> int:
    target_units = max(_text_unit_count(text, compact_script=compact_script) // 2, 1)
    candidate_indices = _candidate_split_indices(text, compact_script=compact_script)
    if not candidate_indices:
        return len(text) // 2
    return min(
        candidate_indices,
        key=lambda index: abs(
            _text_unit_count(text[:index], compact_script=compact_script) - target_units
        ),
    )


def _candidate_split_indices(text: str, *, compact_script: bool) -> list[int]:
    candidates = [
        index + 1
        for index, character in enumerate(text[:-1])
        if character in TRANSCRIPT_PUNCTUATION
    ]
    if not compact_script:
        candidates.extend(index for index, character in enumerate(text[:-1]) if character == " ")
    if candidates:
        return sorted(set(candidates))
    return [len(text) // 2]


def _compact_script_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for character in text:
        current += character
        if character in TRANSCRIPT_PUNCTUATION:
            tokens.append(current.strip())
            current = ""
    if current.strip():
        tokens.append(current.strip())
    return tokens


def _default_script_tokens(text: str) -> list[str]:
    tokens = [match.group(0).strip() for match in TRANSCRIPT_WORD_RE.finditer(text)]
    return [token for token in tokens if token]


def _join_parts(left: str, right: str, *, compact_script: bool) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    separator = "" if compact_script or left.endswith(" ") or right.startswith(" ") else " "
    return f"{left.rstrip()}{separator}{right.lstrip()}".strip()


def _text_unit_count(text: str, *, compact_script: bool) -> int:
    if compact_script:
        return sum(1 for character in text if not character.isspace())
    return len(text)


def _contains_compact_script(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u9fff"
        or "\u3040" <= character <= "\u30ff"
        or "\uac00" <= character <= "\ud7af"
        for character in text
    )


def _max_segment_duration_ms(segments: Iterable[Segment]) -> int:
    return max((segment.end_ms - segment.start_ms for segment in segments), default=0)


def _overlong_segment_count(segments: Iterable[Segment]) -> int:
    return sum(
        1
        for segment in segments
        if segment.end_ms - segment.start_ms > MAX_TRANSCRIPT_SEGMENT_DURATION_MS
    )


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
