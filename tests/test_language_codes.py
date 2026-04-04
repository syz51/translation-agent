from __future__ import annotations

import pytest

from translation_agent.api import RunJobRequest, _resolved_job_languages
from translation_agent.config import Settings
from translation_agent.language_codes import (
    canonicalize_language_code,
    resolve_transcription_provider_language,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en_us", "en-US"),
        ("EN-uk", "en-GB"),
        ("korean", "ko"),
        ("mandarin chinese", "zh-CN"),
        ("pt_br", "pt-BR"),
        ("simplified chinese", "zh-CN"),
        ("traditional chinese", "zh-TW"),
        ("zh", "zh-CN"),
        ("es_latam", "es-419"),
        ("zh_hans", "zh-Hans"),
        ("fr", "fr"),
    ],
)
def test_canonicalize_language_code_normalizes_common_aliases(raw: str, expected: str) -> None:
    assert canonicalize_language_code(raw) == expected


@pytest.mark.parametrize(
    ("provider_id", "raw", "expected"),
    [
        ("assemblyai", "en_us", "en"),
        ("assemblyai", "pt-BR", "pt"),
        ("assemblyai", "zh-CN", "zh"),
        ("assemblyai", "es-419", "es"),
        ("deepgram", "en_us", "en-US"),
        ("deepgram", "pt_br", "pt-BR"),
        ("deepgram", "zh_hans", "zh-Hans"),
    ],
)
def test_resolve_transcription_provider_language_bridges_provider_dialects(
    provider_id: str,
    raw: str,
    expected: str,
) -> None:
    assert resolve_transcription_provider_language(provider_id, raw) == expected


def test_resolve_transcription_provider_language_rejects_unsafe_assemblyai_mapping() -> None:
    with pytest.raises(ValueError, match="zh-HK"):
        resolve_transcription_provider_language("assemblyai", "zh-HK")


def test_resolved_job_languages_canonicalizes_request_values() -> None:
    request = RunJobRequest(
        source="input.mp4",
        source_language="en_us",
        target_language="zh_cn",
    )
    settings = Settings(
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
    )

    assert _resolved_job_languages(request, settings) == ("en-US", "zh-CN")


def test_resolved_job_languages_accepts_friendly_language_names() -> None:
    request = RunJobRequest(
        source="input.mp4",
        source_language="korean",
        target_language="simplified chinese",
    )
    settings = Settings(
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
    )

    assert _resolved_job_languages(request, settings) == ("ko", "zh-CN")
