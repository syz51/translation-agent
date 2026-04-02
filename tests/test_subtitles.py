from __future__ import annotations

from pathlib import Path

import pysubs2
import pytest
from pydantic import ValidationError

from translation_agent.api import convert_translation_json_to_srt
from translation_agent.models import Segment, TranslationCandidate
from translation_agent.subtitles import render_translation_srt


def _translation_candidate() -> TranslationCandidate:
    return TranslationCandidate(
        candidate_id="translation-candidate-1",
        job_id="job-subtitles",
        source_transcript_candidate_id="transcript-candidate-1",
        model_id="gpt-5.4-mini",
        prompt_variant_id="variant-a",
        prompt_version="v1",
        language="fr",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=1250,
                source_text="Hello world",
                target_text="Bonjour le monde",
            ),
            Segment(
                segment_id="seg-2",
                start_ms=1250,
                end_ms=2400,
                source_text="This is blank",
                target_text="   ",
            ),
        ),
        full_text="Bonjour le monde",
        raw_response_ref="raw/translation-candidate-1.json",
        normalization_version="2026-03-30",
        metadata={"source": "unit-test"},
    )


@pytest.mark.unit
def test_render_translation_srt_omits_blank_segments() -> None:
    subtitles = pysubs2.SSAFile.from_string(
        render_translation_srt(_translation_candidate()),
        format_="srt",
    )

    assert len(subtitles.events) == 1
    assert subtitles.events[0].text == "Bonjour le monde"
    assert subtitles.events[0].start == 0
    assert subtitles.events[0].end == 1250


@pytest.mark.unit
def test_convert_translation_json_to_srt_uses_default_output_path(tmp_path: Path) -> None:
    source_path = tmp_path / "published" / "translation.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        _translation_candidate().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = convert_translation_json_to_srt(source_path)
    subtitles = pysubs2.SSAFile.from_string(result.output_path.read_text(encoding="utf-8"), "srt")

    assert result.source_path == source_path.resolve()
    assert result.output_path == source_path.with_suffix(".srt").resolve()
    assert result.job_id == "job-subtitles"
    assert result.candidate_id == "translation-candidate-1"
    assert result.language == "fr"
    assert result.subtitle_count == 1
    assert len(subtitles.events) == 1
    assert subtitles.events[0].text == "Bonjour le monde"


@pytest.mark.unit
def test_convert_translation_json_to_srt_rejects_non_candidate_json(tmp_path: Path) -> None:
    source_path = tmp_path / "exports" / "translation.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        '{"job_id":"job-subtitles","translation_text":"Bonjour"}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        convert_translation_json_to_srt(source_path)
