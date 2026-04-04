"""Canonical language tags plus provider-specific transcription bridges."""

from __future__ import annotations

import re

_LANGUAGE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,31})$")
_LANGUAGE_NAME_SEPARATORS = re.compile(r"[\s_-]+")

_LANGUAGE_NAME_ALIASES = {
    "chinese": "zh-CN",
    "hangul": "ko",
    "korean": "ko",
    "mandarin": "zh-CN",
    "mandarin chinese": "zh-CN",
    "simplified chinese": "zh-CN",
    "traditional chinese": "zh-TW",
}

_CANONICAL_ALIASES = {
    "en-au": "en-AU",
    "en-ca": "en-CA",
    "en-gb": "en-GB",
    "en-ie": "en-IE",
    "en-in": "en-IN",
    "en-nz": "en-NZ",
    "en-uk": "en-GB",
    "en-us": "en-US",
    "es-419": "es-419",
    "es-latam": "es-419",
    "fr-ca": "fr-CA",
    "ko-kr": "ko-KR",
    "nl-be": "nl-BE",
    "pt-br": "pt-BR",
    "pt-pt": "pt-PT",
    "sv-se": "sv-SE",
    "th-th": "th-TH",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-Hans",
    "zh-hant": "zh-Hant",
    "zh-hk": "zh-HK",
    "zh-tw": "zh-TW",
}


def canonicalize_language_code(value: str) -> str:
    """Normalize loose language tags into a stable internal form."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("language code must not be empty")
    language_name = _LANGUAGE_NAME_SEPARATORS.sub(" ", normalized).strip().lower()
    if language_name in _LANGUAGE_NAME_ALIASES:
        return _LANGUAGE_NAME_ALIASES[language_name]
    if not _LANGUAGE_CODE_PATTERN.fullmatch(normalized):
        raise ValueError(f"unsupported language code format: {value!r}")

    normalized = normalized.replace("_", "-")
    alias = _CANONICAL_ALIASES.get(normalized.lower())
    if alias is not None:
        return alias

    parts = normalized.split("-")
    formatted_parts = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            formatted_parts.append(part.title())
            continue
        if len(part) == 2 and part.isalpha():
            formatted_parts.append(part.upper())
            continue
        if len(part) == 3 and part.isdigit():
            formatted_parts.append(part)
            continue
        formatted_parts.append(part.lower())

    canonical = "-".join(formatted_parts)
    return _CANONICAL_ALIASES.get(canonical.lower(), canonical)


def canonicalize_optional_language_code(value: str | None) -> str | None:
    if value is None:
        return None
    return canonicalize_language_code(value)


def resolve_transcription_provider_language(provider_id: str, language_code: str) -> str:
    """Translate a canonical/internal language tag into a provider request value."""

    canonical = canonicalize_language_code(language_code)
    if provider_id == "assemblyai":
        return _assemblyai_language_code(canonical)
    if provider_id == "deepgram":
        return canonical
    raise ValueError(f"unsupported transcription provider for language bridge: {provider_id}")


def _assemblyai_language_code(canonical: str) -> str:
    if canonical == "multi":
        raise ValueError("AssemblyAI does not support a multi-language source code")
    if canonical == "zh-HK":
        raise ValueError(
            "AssemblyAI has no safe Cantonese bridge for zh-HK; use zh for Mandarin audio"
        )
    if canonical.startswith("en-"):
        return "en"
    return canonical.split("-", maxsplit=1)[0]
