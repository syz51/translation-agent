"""Helpers for rendering persisted translation artifacts as subtitles."""

from __future__ import annotations

import pysubs2

from translation_agent.models import TranslationCandidate


def render_translation_srt(translation: TranslationCandidate) -> str:
    """Render a translation candidate into SRT, omitting blank target segments."""

    subtitles = pysubs2.SSAFile()
    subtitles.events = [
        pysubs2.SSAEvent(
            start=segment.start_ms,
            end=segment.end_ms,
            text=target_text,
        )
        for segment in translation.segments
        if (target_text := (segment.target_text or "").strip())
    ]
    return subtitles.to_string("srt")


def subtitle_count(translation: TranslationCandidate) -> int:
    """Count the subtitle events that would be emitted for the translation."""

    return sum(1 for segment in translation.segments if (segment.target_text or "").strip())
