from __future__ import annotations

from pathlib import Path

import pysubs2
import pytest
from pydantic import ValidationError

from translation_agent.api import convert_translation_json_to_srt
from translation_agent.models import Segment, TranslationCandidate
from translation_agent.subtitles import render_translation_srt, subtitle_count


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


def _long_chinese_translation_candidate() -> TranslationCandidate:
    return TranslationCandidate(
        candidate_id="translation-candidate-long-1",
        job_id="job-subtitles",
        source_transcript_candidate_id="transcript-candidate-long-1",
        model_id="gpt-5.4-mini",
        prompt_variant_id="variant-a",
        prompt_version="v1",
        language="zh",
        segments=(
            Segment(
                segment_id="seg-long-1",
                start_ms=0,
                end_ms=24000,
                source_text="Long source cue",
                target_text=(
                    "现在火候到了，火候到了。关键是如果只和再赫约会，最后不是都会给再赫投票吗？"
                    "再赫有点连接感，很帅！魅力值满满啊！再赫很有魅力！"
                    "承承虽然约会了却不受欢迎，票到现在还没来！"
                    "如果稍微不方便的话，外面吹吹风再来也可以！"
                ),
            ),
        ),
        full_text=(
            "现在火候到了，火候到了。关键是如果只和再赫约会，最后不是都会给再赫投票吗？"
            "再赫有点连接感，很帅！魅力值满满啊！再赫很有魅力！"
            "承承虽然约会了却不受欢迎，票到现在还没来！"
            "如果稍微不方便的话，外面吹吹风再来也可以！"
        ),
        raw_response_ref="raw/translation-candidate-long-1.json",
        normalization_version="2026-03-30",
        metadata={"source": "unit-test"},
    )


def _assert_readable_chinese_cues(subtitles: pysubs2.SSAFile) -> None:
    assert subtitles.events, "expected at least one subtitle event"
    assert len(subtitles.events) >= 2
    assert subtitles.events[0].start == 0
    assert subtitles.events[-1].end == 24000
    for event in subtitles.events:
        assert 0 < event.end - event.start <= 7000
        lines = event.plaintext.splitlines()
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= 16 for line in lines)


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
def test_render_translation_srt_splits_long_chinese_cues_into_readable_events() -> None:
    subtitles = pysubs2.SSAFile.from_string(
        render_translation_srt(_long_chinese_translation_candidate()),
        format_="srt",
    )

    _assert_readable_chinese_cues(subtitles)


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
def test_convert_translation_json_to_srt_splits_long_chinese_cues(tmp_path: Path) -> None:
    source_path = tmp_path / "published" / "translation.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        _long_chinese_translation_candidate().model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = convert_translation_json_to_srt(source_path)
    subtitles = pysubs2.SSAFile.from_string(result.output_path.read_text(encoding="utf-8"), "srt")

    assert result.subtitle_count == len(subtitles.events)
    _assert_readable_chinese_cues(subtitles)


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


@pytest.mark.unit
def test_render_translation_srt_resegments_long_compact_script_segments() -> None:
    translation = TranslationCandidate(
        candidate_id="translation-candidate-zh",
        job_id="job-subtitles-zh",
        source_transcript_candidate_id="transcript-candidate-zh",
        model_id="gpt-5.4-mini",
        prompt_variant_id="variant-a",
        prompt_version="v1",
        language="zh",
        segments=(
            Segment(
                segment_id="seg-zh-1",
                start_ms=0,
                end_ms=21_000,
                source_text="long source",
                target_text=(
                    "现在火候到了。火候到了。炒面。啊，谢谢。"
                    "不，关键是如果只和再赫约会，最后不是都会给再赫投票吗？"
                    "是那样啊！再赫有点连接感，很帅！魅力值满满啊！再赫很有魅力！"
                ),
            ),
        ),
        full_text="",
        raw_response_ref="raw/translation-candidate-zh.json",
        normalization_version="2026-03-30",
        metadata={"source": "unit-test"},
    )

    subtitles = pysubs2.SSAFile.from_string(render_translation_srt(translation), format_="srt")

    assert len(subtitles.events) >= 3
    assert subtitle_count(translation) == len(subtitles.events)
    assert subtitles.events[0].start == 0
    assert subtitles.events[-1].end == 21_000
    assert all(event.end > event.start for event in subtitles.events)
    assert all(event.end - event.start <= 7_000 for event in subtitles.events)
    assert all(
        len(event.text.replace("\\N", "").replace("\n", "")) <= 32 for event in subtitles.events
    )
    assert all(event.text.count("\\N") + event.text.count("\n") <= 1 for event in subtitles.events)
