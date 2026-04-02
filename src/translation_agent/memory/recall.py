"""Memory recall interfaces and in-memory reference implementation."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol, runtime_checkable

from translation_agent.models import (
    MemoryBundle,
    MemoryEntry,
    MemoryQuery,
    MemoryScopeKind,
    ProviderCaveat,
)
from translation_agent.storage import BlobStore

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SCOPE_ORDER: tuple[MemoryScopeKind, ...] = (
    "project_pair",
    "pair",
    "source_language",
    "target_language",
    "global",
)
_REVIEW_CAPS = {"glossary": 2, "rule": 2, "semantic": 4, "episodic": 1}
_ADJUDICATION_CAPS = {"glossary": 1, "rule": 1, "semantic": 2, "episodic": 1}


@dataclass(slots=True)
class InMemoryLongTermMemoryStore:
    """Small reference store for consolidated long-term memory."""

    entries: dict[str, MemoryEntry] = field(default_factory=dict)
    dedupe_index: dict[str, str] = field(default_factory=dict)

    def put_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool:
        if dedupe_key is not None and dedupe_key in self.dedupe_index:
            return False
        self.entries[entry.memory_id] = entry
        if dedupe_key is not None:
            self.dedupe_index[dedupe_key] = entry.memory_id
        return True

    def get_entry(self, memory_id: str) -> MemoryEntry | None:
        return self.entries.get(memory_id)

    def list_entries(self) -> list[MemoryEntry]:
        return sorted(self.entries.values(), key=lambda entry: entry.memory_id)


class BlobBackedLongTermMemoryStore:
    """Blob-backed long-term store that survives separate runtime instances."""

    def __init__(
        self,
        blob_store: BlobStore,
        *,
        state_ref: str = "memory/long-term/store.json",
    ) -> None:
        self._blob_store = blob_store
        self._state_ref = state_ref
        self._loaded = False
        self._entries: dict[str, MemoryEntry] = {}
        self._dedupe_index: dict[str, str] = {}

    def put_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool:
        self._reload()
        if dedupe_key is not None and dedupe_key in self._dedupe_index:
            return False
        self._entries[entry.memory_id] = entry
        if dedupe_key is not None:
            self._dedupe_index[dedupe_key] = entry.memory_id
        self._persist()
        return True

    def get_entry(self, memory_id: str) -> MemoryEntry | None:
        self._ensure_loaded()
        return self._entries.get(memory_id)

    def list_entries(self) -> list[MemoryEntry]:
        self._ensure_loaded()
        return sorted(self._entries.values(), key=lambda entry: entry.memory_id)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._reload()

    def _reload(self) -> None:
        if not self._blob_store.exists(self._state_ref):
            self._entries = {}
            self._dedupe_index = {}
            self._loaded = True
            return
        payload = json.loads(self._blob_store.read_bytes(self._state_ref).decode("utf-8"))
        entries = payload.get("entries", [])
        dedupe_index = payload.get("dedupe_index", {})
        self._entries = {
            entry_payload["memory_id"]: MemoryEntry.model_validate(entry_payload)
            for entry_payload in entries
            if isinstance(entry_payload, dict) and isinstance(entry_payload.get("memory_id"), str)
        }
        if isinstance(dedupe_index, dict):
            self._dedupe_index = {str(key): str(value) for key, value in dedupe_index.items()}
        else:
            self._dedupe_index = {}
        self._loaded = True

    def _persist(self) -> None:
        payload = {
            "entries": [
                entry.model_dump(mode="json")
                for entry in sorted(self._entries.values(), key=lambda item: item.memory_id)
            ],
            "dedupe_index": dict(sorted(self._dedupe_index.items())),
        }
        self._blob_store.put_bytes(
            self._state_ref,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )


@runtime_checkable
class MemoryEntryStore(Protocol):
    """Minimal persistence contract for long-term memory entries."""

    def put_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool: ...

    def get_entry(self, memory_id: str) -> MemoryEntry | None: ...

    def list_entries(self) -> list[MemoryEntry]: ...


@runtime_checkable
class MemoryRecallBackend(Protocol):
    """Read-only long-term memory contract for review and adjudication."""

    def recall_memory(self, query: MemoryQuery) -> MemoryBundle: ...


class LongTermMemoryRecallBackend:
    """Recall backend with explicit scope precedence and deterministic ranking."""

    def __init__(self, store: MemoryEntryStore) -> None:
        self._store = store

    def recall_memory(self, query: MemoryQuery) -> MemoryBundle:
        caps = _caps_for_stage(query.stage)
        allowed_kinds = set(caps)
        buckets: dict[MemoryScopeKind, list[tuple[float, float, datetime, str, MemoryEntry]]] = {
            scope_kind: [] for scope_kind in _SCOPE_ORDER
        }

        for entry in self._store.list_entries():
            if not _eligible(entry, query, allowed_kinds):
                continue
            score = _bucket_score(entry, query)
            quality = float(entry.score or 0.5)
            updated_at = entry.updated_at or datetime.fromtimestamp(0, tz=UTC)
            buckets[entry.scope_kind or "global"].append(
                (score, quality, updated_at, entry.memory_id, entry)
            )

        counts = {kind: 0 for kind in caps}
        kept: dict[str, MemoryEntry] = {}
        ordered: list[MemoryEntry] = []
        for scope_kind in _SCOPE_ORDER:
            ranked = sorted(
                buckets[scope_kind],
                key=lambda item: (-item[0], -item[1], -item[2].timestamp(), item[3]),
            )
            for _, _, _, _, entry in ranked:
                kind = _bundle_kind(entry.kind)
                if counts[kind] >= caps[kind]:
                    continue
                fact_key = _fact_key(entry)
                existing = kept.get(fact_key)
                if existing is not None:
                    continue
                kept[fact_key] = entry
                counts[kind] += 1
                ordered.append(entry)
                if sum(counts.values()) >= min(query.max_items, sum(caps.values())):
                    break
            if sum(counts.values()) >= min(query.max_items, sum(caps.values())):
                break

        return MemoryBundle(
            semantic_memory=tuple(entry for entry in ordered if entry.kind == "semantic"),
            episodic_memory=tuple(entry for entry in ordered if entry.kind == "episodic"),
            glossary=tuple(entry for entry in ordered if entry.kind == "glossary"),
            rules=tuple(entry for entry in ordered if entry.kind == "rule"),
            procedural_memory=(),
            provider_caveats=(_default_provider_caveat(),),
        )


def _eligible(entry: MemoryEntry, query: MemoryQuery, allowed_kinds: set[str]) -> bool:
    if entry.scope_kind is None or entry.scope_key is None or entry.updated_at is None:
        return False
    if entry.kind not in allowed_kinds:
        return False
    if entry.lifecycle_status != "active":
        return False
    if entry.expires_at is not None and entry.expires_at <= datetime.now(UTC):
        return False
    if not _scope_compatible(entry, query):
        return False
    return True


def _scope_compatible(entry: MemoryEntry, query: MemoryQuery) -> bool:
    scope_kind = entry.scope_kind
    if scope_kind == "project_pair":
        return entry.scope_key == project_pair_scope_key(query)
    if scope_kind == "pair":
        return entry.scope_key == pair_scope_key(query)
    if scope_kind == "source_language":
        return entry.scope_key == query.job.source_language
    if scope_kind == "target_language":
        return entry.scope_key == query.job.target_language
    if scope_kind == "global":
        return entry.scope_key == "global"
    return False


def project_pair_scope_key(query: MemoryQuery) -> str:
    return (
        f"{query.job.tenant_id}::{query.job.project_id}::"
        f"{query.job.source_language}::{query.job.target_language}"
    )


def pair_scope_key(query: MemoryQuery) -> str:
    return f"{query.job.source_language}::{query.job.target_language}"


def build_scope_key(
    *,
    scope_kind: MemoryScopeKind,
    tenant_id: str,
    project_id: str,
    source_language: str,
    target_language: str,
) -> str:
    if scope_kind == "project_pair":
        return f"{tenant_id}::{project_id}::{source_language}::{target_language}"
    if scope_kind == "pair":
        return f"{source_language}::{target_language}"
    if scope_kind == "source_language":
        return source_language
    if scope_kind == "target_language":
        return target_language
    return "global"


def _bucket_score(entry: MemoryEntry, query: MemoryQuery) -> float:
    semantic_relevance = _semantic_relevance(query.query_text, entry.content)
    lexical_relevance = _lexical_relevance(query.query_text, entry.content)
    quality_score = float(entry.score or 0.5)
    recency_score = _recency_score(entry)
    return round(
        0.55 * semantic_relevance
        + 0.20 * lexical_relevance
        + 0.15 * quality_score
        + 0.10 * recency_score,
        6,
    )


def _semantic_relevance(query_text: str, content: str) -> float:
    query_tokens = set(_tokens(query_text))
    content_tokens = set(_tokens(content))
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)
    subsequence = 1.0 if " ".join(query_tokens) in content.casefold() else 0.0
    return round(min(1.0, overlap * 0.8 + subsequence * 0.2), 6)


def _lexical_relevance(query_text: str, content: str) -> float:
    query_tokens = _tokens(query_text)
    content_tokens = _tokens(content)
    if not query_tokens or not content_tokens:
        return 0.0
    query_set = set(query_tokens)
    content_set = set(content_tokens)
    return round(len(query_set & content_set) / max(len(query_set), 1), 6)


def _recency_score(entry: MemoryEntry) -> float:
    if entry.kind in {"glossary", "rule"}:
        return 1.0
    if entry.updated_at is None:
        return 0.0
    age_days = max((datetime.now(UTC) - entry.updated_at).total_seconds() / 86400.0, 0.0)
    half_life_days = 180.0 if entry.kind == "semantic" else 30.0
    return round(math.exp(-math.log(2.0) * age_days / half_life_days), 6)


def _fact_key(entry: MemoryEntry) -> str:
    if entry.kind == "glossary":
        term_key = entry.metadata.get("fact_key") or entry.metadata.get("term")
        if isinstance(term_key, str) and term_key.strip():
            return f"glossary:{_normalize_text(term_key)}"
        return f"glossary:{_normalize_text(entry.content)}"
    if entry.kind == "rule":
        rule_text = entry.metadata.get("fact_key") or entry.content
        return f"rule:{sha256(_normalize_text(str(rule_text)).encode('utf-8')).hexdigest()}"
    if entry.kind == "semantic":
        category = entry.metadata.get("category")
        category_key = _normalize_text(category) if isinstance(category, str) else ""
        content_key = sha256(_normalize_text(entry.content).encode("utf-8")).hexdigest()
        return f"semantic:{category_key}:{content_key}"
    event_id = entry.metadata.get("event_id") or entry.metadata.get("batch_id") or entry.memory_id
    return f"episodic:{event_id}"


def _bundle_kind(kind: str) -> str:
    if kind == "rule":
        return "rule"
    return kind


def _caps_for_stage(stage: str) -> dict[str, int]:
    lowered = stage.casefold()
    if "adjudicat" in lowered:
        return dict(_ADJUDICATION_CAPS)
    return dict(_REVIEW_CAPS)


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return [match.group(0) for match in _TOKEN_RE.finditer(normalized)]


def _normalize_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def _default_provider_caveat() -> ProviderCaveat:
    return ProviderCaveat(
        provider_id="speechmatics",
        note="Provider-specific caveats remain advisory and are not part of contradiction scoring.",
    )
