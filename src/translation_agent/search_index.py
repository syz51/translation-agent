"""Deterministic local embeddings and search-document helpers for memory recall."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from hashlib import sha256
from typing import Any

from translation_agent.models import MemoryEntry, MemoryQuery

EMBEDDING_MODEL_ID = "local-hash-v1"
EMBEDDING_DIMENSIONS = 16
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def embedding_metadata_for_entry(entry: MemoryEntry) -> tuple[str, list[float], str]:
    """Return the deterministic embedding payload and SQL-search document for one entry."""

    document = search_document_for_entry(entry)
    vector = embed_text(document)
    return EMBEDDING_MODEL_ID, vector, document


def embedding_metadata_for_query(query: MemoryQuery) -> tuple[list[float], str]:
    """Return the deterministic embedding payload and SQL-search document for one query."""

    document = search_document_for_query(query)
    return embed_text(document), document


def search_document_for_entry(entry: MemoryEntry) -> str:
    """Build the lexical search document used by SQL prefiltering."""

    parts: list[str] = [
        entry.kind,
        entry.memory_subtype or "",
        entry.content,
        entry.scope_kind or "",
        entry.scope_key or "",
        entry.series_id or "",
        entry.franchise_id or "",
        entry.content_type or "",
        entry.style_profile_id or "",
        " ".join(entry.speaker_ids),
        " ".join(entry.topic_tags),
        " ".join(entry.entity_keys),
        " ".join(entry.term_keys),
        _metadata_terms(entry.metadata),
        _metadata_terms(entry.typed_metadata),
    ]
    return _normalize_document(" ".join(part for part in parts if part))


def search_document_for_query(query: MemoryQuery) -> str:
    """Build the lexical search document used by SQL prefiltering."""

    asset_context = query.asset_context
    parts = [
        query.query_text,
        query.job.source_language,
        query.job.target_language,
        query.media_key or "",
        query.series_id or "",
        query.franchise_id or "",
        query.content_type or "",
        query.style_profile_id or "",
        " ".join(query.speaker_ids),
        " ".join(query.topic_tags),
        " ".join(query.entity_keys),
        " ".join(query.term_keys),
        " ".join(query.failure_tags),
        " ".join(query.escalation_reasons),
        " ".join(query.provider_ids),
        " ".join(query.prompt_variant_ids),
        " ".join(query.model_ids),
        asset_context.channel_id if asset_context is not None else "",
        asset_context.canonical_title if asset_context is not None else "",
    ]
    return _normalize_document(" ".join(part for part in parts if part))


def embed_text(text: str, *, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Hash tokens into a compact normalized vector."""

    vector = [0.0] * dimensions
    tokens = _tokens(text)
    if not tokens:
        return vector
    for index, token in enumerate(tokens, start=1):
        digest = sha256(f"{index}:{token}".encode()).digest()
        bucket = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        weight = 1.0 + min(len(token), 12) / 24.0
        vector[bucket] += sign * weight
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return [0.0] * dimensions
    return [round(value / magnitude, 6) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity in the [0, 1] range."""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    normalized = (dot + 1.0) / 2.0
    return round(max(0.0, min(normalized, 1.0)), 6)


def serialize_embedding(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def deserialize_embedding(payload: str | bytes | object) -> list[float]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            return []
    else:
        raw = payload
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except TypeError, ValueError:
            return []
    return values


def _metadata_terms(payload: dict[str, Any]) -> str:
    terms: list[str] = []
    for key, value in sorted(payload.items()):
        if isinstance(value, str):
            terms.append(f"{key} {value}")
            continue
        if isinstance(value, (list, tuple, set)):
            normalized = " ".join(str(item) for item in value)
            if normalized:
                terms.append(f"{key} {normalized}")
    return " ".join(terms)


def _normalize_document(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return [
        token
        for token in (match.group(0) for match in _TOKEN_RE.finditer(normalized))
        if token not in _STOPWORDS
    ]
