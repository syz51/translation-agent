"""Helpers for rendering persisted translation artifacts as subtitles."""

from __future__ import annotations

import re

import pysubs2

from translation_agent.models import Segment, TranslationCandidate

MAX_SUBTITLE_DURATION_MS = 7_000
SUBTITLE_GAP_MS = 84
COMPACT_SCRIPT_LANGUAGES = {"zh", "ja", "ko"}
MAX_COMPACT_CHARS_PER_LINE = 16
MAX_DEFAULT_CHARS_PER_LINE = 42
MAX_SUBTITLE_LINES = 2
COMPACT_PUNCTUATION = "。！？；：，、…"
DEFAULT_PUNCTUATION = ".!?;:,"
WORD_RE = re.compile(r"\S+\s*")


def render_translation_srt(translation: TranslationCandidate) -> str:
    """Render a translation candidate into SRT, omitting blank target segments."""

    subtitles = pysubs2.SSAFile()
    subtitles.events = _subtitle_events(translation)
    return subtitles.to_string("srt")


def subtitle_count(translation: TranslationCandidate) -> int:
    """Count the subtitle events that would be emitted for the translation."""

    return len(_subtitle_events(translation))


def subtitle_validation_errors(translation: TranslationCandidate) -> tuple[str, ...]:
    """Return deterministic subtitle export validation failures."""

    errors: list[str] = []
    blank_segment_ids = [
        segment.segment_id
        for segment in translation.segments
        if not (segment.target_text or "").strip()
    ]
    if blank_segment_ids:
        errors.append("blank_target_cues")

    events = _subtitle_events(translation)
    for left, right in zip(events, events[1:], strict=False):
        if right.start < left.end:
            errors.append("subtitle_overlaps")
            break
    return tuple(dict.fromkeys(errors))


def _subtitle_events(translation: TranslationCandidate) -> list[pysubs2.SSAEvent]:
    events: list[pysubs2.SSAEvent] = []
    compact_script = _uses_compact_script(translation)
    for segment in translation.segments:
        events.extend(_segment_to_subtitle_events(segment, compact_script=compact_script))
    return events


def _segment_to_subtitle_events(
    segment: Segment,
    *,
    compact_script: bool,
) -> list[pysubs2.SSAEvent]:
    target_text = (segment.target_text or "").strip()
    if not target_text:
        return []

    max_chars_per_line = (
        MAX_COMPACT_CHARS_PER_LINE if compact_script else MAX_DEFAULT_CHARS_PER_LINE
    )
    max_chars_per_event = max_chars_per_line * MAX_SUBTITLE_LINES
    chunks = _chunk_subtitle_text(
        target_text,
        compact_script=compact_script,
        max_chars_per_event=max_chars_per_event,
    )
    while True:
        timed_chunks = _timed_chunks(
            segment.start_ms,
            segment.end_ms,
            chunks,
            compact_script=compact_script,
        )
        overlong_index = next(
            (
                index
                for index, (start_ms, end_ms, _) in enumerate(timed_chunks)
                if end_ms - start_ms > MAX_SUBTITLE_DURATION_MS
            ),
            None,
        )
        if overlong_index is None:
            break
        split_chunks = _split_chunk_in_two(chunks[overlong_index], compact_script=compact_script)
        if len(split_chunks) == 1:
            break
        chunks = chunks[:overlong_index] + split_chunks + chunks[overlong_index + 1 :]
    return [
        pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            text=_line_break_chunk(
                chunk,
                compact_script=compact_script,
                max_chars_per_line=max_chars_per_line,
            ),
        )
        for start_ms, end_ms, chunk in timed_chunks
    ]


def _timed_chunks(
    start_ms: int,
    end_ms: int,
    chunks: list[str],
    *,
    compact_script: bool,
) -> list[tuple[int, int, str]]:
    if not chunks:
        return []

    total_duration = max(end_ms - start_ms, 1)
    gap_budget = min(
        SUBTITLE_GAP_MS * max(len(chunks) - 1, 0),
        max(total_duration - len(chunks), 0),
    )
    allocatable_duration = max(total_duration - gap_budget, len(chunks))
    weights = [
        max(_subtitle_unit_count(chunk, compact_script=compact_script), 1) for chunk in chunks
    ]
    total_weight = sum(weights)
    boundaries = [0]
    cumulative_weight = 0
    for weight in weights[:-1]:
        cumulative_weight += weight
        boundaries.append(round(allocatable_duration * cumulative_weight / total_weight))
    boundaries.append(allocatable_duration)

    timed: list[tuple[int, int, str]] = []
    cursor = start_ms
    remaining_budget = max(total_duration - (cursor - start_ms), 0)
    for index, chunk in enumerate(chunks):
        chunk_duration = boundaries[index + 1] - boundaries[index]
        remaining_chunks = len(chunks) - index - 1
        min_remaining_duration = remaining_chunks
        chunk_duration = max(chunk_duration, 1)
        chunk_duration = min(chunk_duration, max(remaining_budget - min_remaining_duration, 1))
        gap_after = SUBTITLE_GAP_MS if index < len(chunks) - 1 else 0
        start_time = cursor
        end_time = min(start_time + chunk_duration, end_ms - min_remaining_duration - gap_after)
        if end_time <= start_time:
            end_time = min(start_time + 1, end_ms)
        timed.append((start_time, end_time, chunk))
        cursor = min(end_time + gap_after, end_ms)
        remaining_budget = max(end_ms - cursor, 0)

    last_start, _, last_chunk = timed[-1]
    timed[-1] = (last_start, end_ms, last_chunk)
    return timed


def _chunk_subtitle_text(
    text: str,
    *,
    compact_script: bool,
    max_chars_per_event: int,
) -> list[str]:
    clauses = _split_into_clauses(
        text,
        compact_script=compact_script,
        max_chars_per_event=max_chars_per_event,
    )
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        candidate = _join_subtitle_parts(current, clause, compact_script=compact_script)
        if (
            current
            and _subtitle_unit_count(candidate, compact_script=compact_script) > max_chars_per_event
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
    max_chars_per_event: int,
) -> list[str]:
    raw_tokens = _compact_script_tokens(text) if compact_script else _default_script_tokens(text)
    punctuation = COMPACT_PUNCTUATION if compact_script else DEFAULT_PUNCTUATION
    clauses: list[str] = []
    current = ""
    for token in raw_tokens:
        current = _join_subtitle_parts(current, token, compact_script=compact_script)
        if token.rstrip()[-1:] in punctuation:
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
                max_chars_per_event=max_chars_per_event,
            )
        )
    return bounded


def _compact_script_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for character in text:
        current += character
        if character in COMPACT_PUNCTUATION:
            tokens.append(current.strip())
            current = ""
    if current.strip():
        tokens.append(current.strip())
    return tokens


def _default_script_tokens(text: str) -> list[str]:
    tokens = [match.group(0).strip() for match in WORD_RE.finditer(text)]
    return [token for token in tokens if token]


def _split_chunk_in_two(text: str, *, compact_script: bool) -> list[str]:
    if _subtitle_unit_count(text, compact_script=compact_script) <= 1:
        return [text]
    split_index = _best_split_index(text, compact_script=compact_script)
    if split_index <= 0 or split_index >= len(text):
        return [text]
    left = text[:split_index].strip()
    right = text[split_index:].strip()
    if not left or not right:
        return [text]
    return [left, right]


def _split_oversized_clause(
    text: str,
    *,
    compact_script: bool,
    max_chars_per_event: int,
) -> list[str]:
    if _subtitle_unit_count(text, compact_script=compact_script) <= max_chars_per_event:
        return [text]
    split_chunks = _split_chunk_in_two(text, compact_script=compact_script)
    if len(split_chunks) == 1:
        return [text]
    bounded: list[str] = []
    for chunk in split_chunks:
        bounded.extend(
            _split_oversized_clause(
                chunk,
                compact_script=compact_script,
                max_chars_per_event=max_chars_per_event,
            )
        )
    return bounded


def _line_break_chunk(
    text: str,
    *,
    compact_script: bool,
    max_chars_per_line: int,
) -> str:
    if _subtitle_unit_count(text, compact_script=compact_script) <= max_chars_per_line:
        return text
    split_index = _best_line_break_index(
        text,
        compact_script=compact_script,
        max_chars_per_line=max_chars_per_line,
    )
    if split_index <= 0 or split_index >= len(text):
        return text
    return f"{text[:split_index].strip()}\n{text[split_index:].strip()}"


def _best_line_break_index(
    text: str,
    *,
    compact_script: bool,
    max_chars_per_line: int,
) -> int:
    candidates = [
        index
        for index in _candidate_split_indices(text, compact_script=compact_script)
        if _subtitle_unit_count(text[:index], compact_script=compact_script) <= max_chars_per_line
        and _subtitle_unit_count(text[index:], compact_script=compact_script) <= max_chars_per_line
    ]
    if candidates:
        midpoint = _subtitle_unit_count(text, compact_script=compact_script) / 2
        return min(
            candidates,
            key=lambda index: abs(
                _subtitle_unit_count(text[:index], compact_script=compact_script) - midpoint
            ),
        )
    return _hard_split_index(
        text,
        compact_script=compact_script,
        max_chars_per_line=max_chars_per_line,
    )


def _hard_split_index(
    text: str,
    *,
    compact_script: bool,
    max_chars_per_line: int,
) -> int:
    units = 0
    for index, character in enumerate(text, start=1):
        if character.isspace():
            continue
        units += 1
        if units >= max_chars_per_line:
            return index
    return len(text) // 2


def _best_split_index(
    text: str,
    *,
    compact_script: bool,
    preferred_units: int | None = None,
) -> int:
    target_units = preferred_units or max(
        _subtitle_unit_count(text, compact_script=compact_script) // 2,
        1,
    )
    candidate_indices = _candidate_split_indices(text, compact_script=compact_script)
    if not candidate_indices:
        return len(text) // 2
    return min(
        candidate_indices,
        key=lambda index: abs(
            _subtitle_unit_count(text[:index], compact_script=compact_script) - target_units
        ),
    )


def _candidate_split_indices(text: str, *, compact_script: bool) -> list[int]:
    punctuation = COMPACT_PUNCTUATION if compact_script else DEFAULT_PUNCTUATION
    candidates = [
        index + 1 for index, character in enumerate(text[:-1]) if character in punctuation
    ]
    if not compact_script:
        candidates.extend(
            index for index, character in enumerate(text[:-1], start=0) if character == " "
        )
    if candidates:
        return sorted(set(candidates))
    return [len(text) // 2]


def _join_subtitle_parts(left: str, right: str, *, compact_script: bool) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    separator = "" if compact_script or left.endswith(" ") or right.startswith(" ") else " "
    return f"{left.rstrip()}{separator}{right.lstrip()}".strip()


def _subtitle_unit_count(text: str, *, compact_script: bool) -> int:
    if compact_script:
        return sum(1 for character in text if character != "\n")
    return len(text)


def _uses_compact_script(translation: TranslationCandidate) -> bool:
    language = translation.language.strip().lower()
    primary_language = language.split("-", 1)[0]
    if primary_language in COMPACT_SCRIPT_LANGUAGES:
        return True
    return any(_contains_cjk(segment.target_text or "") for segment in translation.segments)


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u9fff"
        or "\u3040" <= character <= "\u30ff"
        or "\uac00" <= character <= "\ud7af"
        for character in text
    )
