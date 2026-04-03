"""Memory recall interfaces and in-memory reference implementation."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from translation_agent.models import (
    MemoryBundle,
    MemoryEntry,
    MemoryQuery,
    MemoryScopeKind,
    ProviderCaveat,
)
from translation_agent.search_index import (
    cosine_similarity,
    deserialize_embedding,
    embedding_metadata_for_query,
)
from translation_agent.storage import BlobStore

if TYPE_CHECKING:
    from translation_agent.storage.operational import OperationalStore

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SCOPE_ORDER: tuple[MemoryScopeKind, ...] = (
    "asset",
    "series",
    "speaker_cluster",
    "franchise",
    "channel",
    "project_pair",
    "pair",
    "source_language",
    "target_language",
    "global",
)
_SCOPE_WEIGHTS = {
    "asset": 1.0,
    "series": 0.97,
    "speaker_cluster": 0.95,
    "franchise": 0.92,
    "channel": 0.9,
    "project_pair": 0.9,
    "pair": 0.75,
    "source_language": 0.55,
    "target_language": 0.5,
    "global": 0.35,
}
_SUBTYPE_WEIGHTS = {
    "glossary": 1.0,
    "style_rule": 0.88,
    "provider_caveat": 0.7,
    "language_convention": 0.84,
    "project_fact": 0.78,
    "failure_pattern": 0.82,
    "escalation_pattern": 0.8,
    "prompt_guidance": 0.94,
    None: 0.6,
}
_TRANSLATION_REVIEW_CAPS = {
    "glossary": 2,
    "rule": 2,
    "semantic": 4,
    "episodic": 2,
    "procedural": 2,
}
_TRANSCRIPT_REVIEW_CAPS = {"glossary": 2, "rule": 2, "semantic": 4, "episodic": 1}
_TRANSLATION_ADJUDICATION_CAPS = {
    "glossary": 1,
    "rule": 1,
    "semantic": 2,
    "episodic": 1,
    "procedural": 1,
}
_TRANSCRIPT_ADJUDICATION_CAPS = {"glossary": 1, "rule": 1, "semantic": 2, "episodic": 1}
_GENERATION_CAPS = {"rule": 2, "procedural": 2}


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


class OperationalStoreLongTermMemoryStore:
    """Operational-store-backed long-term memory repository."""

    def __init__(self, store: OperationalStore) -> None:
        self._store = store

    def put_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool:
        return self._store.put_memory_entry(entry, dedupe_key=dedupe_key)

    def get_entry(self, memory_id: str) -> MemoryEntry | None:
        return self._store.get_memory_entry(memory_id)

    def list_entries(self) -> list[MemoryEntry]:
        return self._store.list_memory_entries()

    def search_memory_entries(
        self,
        query: MemoryQuery,
        *,
        limit: int,
    ) -> list[tuple[MemoryEntry, float]]:
        search = getattr(self._store, "search_memory_entries", None)
        if callable(search):
            return cast("list[tuple[MemoryEntry, float]]", search(query, limit=limit))
        return [(entry, 0.0) for entry in self._store.list_memory_entries()]


@runtime_checkable
class MemoryEntryStore(Protocol):
    """Minimal persistence contract for long-term memory entries."""

    def put_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool: ...

    def get_entry(self, memory_id: str) -> MemoryEntry | None: ...

    def list_entries(self) -> list[MemoryEntry]: ...


@runtime_checkable
class HybridMemoryEntryStore(MemoryEntryStore, Protocol):
    """Optional SQL-backed prefilter contract."""

    def search_memory_entries(
        self,
        query: MemoryQuery,
        *,
        limit: int,
    ) -> list[tuple[MemoryEntry, float]]: ...


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
        query_embedding, _ = embedding_metadata_for_query(query)
        buckets: dict[MemoryScopeKind, list[tuple[float, datetime, str, MemoryEntry]]] = {
            scope_kind: [] for scope_kind in _SCOPE_ORDER
        }

        search_limit = max(query.max_items * 8, 32)
        candidates = (
            self._store.search_memory_entries(query, limit=search_limit)
            if isinstance(self._store, HybridMemoryEntryStore)
            else [(entry, 0.0) for entry in self._store.list_entries()]
        )
        for entry, store_score in candidates:
            if not _eligible(entry, query, allowed_kinds):
                continue
            updated_at = entry.updated_at or datetime.fromtimestamp(0, tz=UTC)
            buckets[entry.scope_kind or "global"].append(
                (
                    _ranking_score(
                        entry,
                        query,
                        store_score=store_score,
                        query_embedding=query_embedding,
                    ),
                    updated_at,
                    entry.memory_id,
                    entry,
                )
            )

        counts = {kind: 0 for kind in caps}
        kept: dict[str, MemoryEntry] = {}
        ordered: list[MemoryEntry] = []
        total_cap = min(query.max_items, sum(caps.values()))
        for scope_kind in _SCOPE_ORDER:
            ranked = sorted(
                buckets[scope_kind],
                key=lambda item: (-item[0], -item[1].timestamp(), item[2]),
            )
            for _, _, _, entry in ranked:
                kind = _bundle_kind(entry.kind)
                if counts.get(kind, 0) >= caps.get(kind, 0):
                    continue
                fact_key = _fact_key(entry)
                if fact_key in kept:
                    continue
                kept[fact_key] = entry
                counts[kind] = counts.get(kind, 0) + 1
                ordered.append(entry)
                if sum(counts.values()) >= total_cap:
                    break
            if sum(counts.values()) >= total_cap:
                break

        return MemoryBundle(
            semantic_memory=tuple(entry for entry in ordered if entry.kind == "semantic"),
            episodic_memory=tuple(entry for entry in ordered if entry.kind == "episodic"),
            glossary=tuple(entry for entry in ordered if entry.kind == "glossary"),
            rules=tuple(entry for entry in ordered if entry.kind == "rule"),
            procedural_memory=tuple(entry for entry in ordered if entry.kind == "procedural"),
            provider_caveats=(_default_provider_caveat(),),
        )


def _eligible(entry: MemoryEntry, query: MemoryQuery, allowed_kinds: set[str]) -> bool:
    if entry.scope_kind is None or entry.scope_key is None or entry.updated_at is None:
        return False
    if entry.kind not in allowed_kinds:
        return False
    if entry.kind == "procedural" and not _procedural_stage_allowed(query.stage):
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
    if scope_kind == "asset":
        return query.media_key is not None and entry.scope_key == query.media_key
    if scope_kind == "series":
        return query.series_id is not None and entry.scope_key == query.series_id
    if scope_kind == "speaker_cluster":
        return bool(query.speaker_ids) and entry.scope_key in set(query.speaker_ids)
    if scope_kind == "franchise":
        return query.franchise_id is not None and entry.scope_key == query.franchise_id
    if scope_kind == "channel":
        channel_id = query.asset_context.channel_id if query.asset_context is not None else None
        return channel_id is not None and entry.scope_key == channel_id
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
    return build_scope_key(
        scope_kind="project_pair",
        tenant_id=query.job.tenant_id,
        project_id=query.job.project_id,
        source_language=query.job.source_language,
        target_language=query.job.target_language,
    )


def pair_scope_key(query: MemoryQuery) -> str:
    return build_scope_key(
        scope_kind="pair",
        tenant_id=query.job.tenant_id,
        project_id=query.job.project_id,
        source_language=query.job.source_language,
        target_language=query.job.target_language,
    )


def build_scope_key(
    *,
    scope_kind: MemoryScopeKind,
    tenant_id: str,
    project_id: str,
    source_language: str,
    target_language: str,
) -> str:
    if scope_kind == "asset":
        return "global"
    if scope_kind == "project_pair":
        return f"{tenant_id}::{project_id}::{source_language}::{target_language}"
    if scope_kind == "pair":
        return f"{source_language}::{target_language}"
    if scope_kind == "source_language":
        return source_language
    if scope_kind == "target_language":
        return target_language
    return "global"


def _ranking_score(
    entry: MemoryEntry,
    query: MemoryQuery,
    *,
    store_score: float = 0.0,
    query_embedding: list[float] | None = None,
) -> float:
    scope_weight = _SCOPE_WEIGHTS.get(entry.scope_kind or "global", 0.0)
    subtype_weight = _SUBTYPE_WEIGHTS.get(entry.memory_subtype, 0.6)
    metadata_match = _metadata_match_score(entry, query)
    semantic_relevance = _semantic_relevance(query.query_text, entry.content)
    lexical_relevance = _lexical_relevance(query.query_text, entry.content)
    embedding_relevance = _embedding_relevance(entry, query_embedding)
    evidence_score = _evidence_score(entry)
    quality_score = float(entry.score or 0.5)
    recency_score = _recency_score(entry)
    return round(
        0.24 * scope_weight
        + 0.10 * subtype_weight
        + 0.16 * metadata_match
        + 0.10 * semantic_relevance
        + 0.05 * lexical_relevance
        + 0.10 * embedding_relevance
        + 0.10 * store_score
        + 0.10 * evidence_score
        + 0.05 * quality_score
        + 0.05 * recency_score,
        6,
    )


def _semantic_relevance(query_text: str, content: str) -> float:
    query_tokens = set(_tokens(query_text))
    content_tokens = set(_tokens(content))
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)
    subsequence = 1.0 if " ".join(sorted(query_tokens)) in content.casefold() else 0.0
    return round(min(1.0, overlap * 0.8 + subsequence * 0.2), 6)


def _lexical_relevance(query_text: str, content: str) -> float:
    query_tokens = _tokens(query_text)
    content_tokens = _tokens(content)
    if not query_tokens or not content_tokens:
        return 0.0
    query_set = set(query_tokens)
    content_set = set(content_tokens)
    return round(len(query_set & content_set) / max(len(query_set), 1), 6)


def _evidence_score(entry: MemoryEntry) -> float:
    max_support = max(
        entry.evidence_count,
        entry.supporting_run_count,
        entry.supporting_asset_count,
        entry.supporting_project_count,
        0,
    )
    contradictions = entry.contradiction_count
    if max_support <= 0 and contradictions <= 0:
        return 0.0
    support_score = min(max_support / 10.0, 1.0)
    contradiction_penalty = min(contradictions / max(max_support, 1), 1.0)
    return round(max(support_score - contradiction_penalty, 0.0), 6)


def _recency_score(entry: MemoryEntry) -> float:
    if entry.kind in {"glossary", "rule"}:
        return 1.0
    if entry.updated_at is None:
        return 0.0
    age_days = max((datetime.now(UTC) - entry.updated_at).total_seconds() / 86400.0, 0.0)
    half_life_days = 180.0 if entry.kind == "semantic" else 45.0
    return round(math.exp(-math.log(2.0) * age_days / half_life_days), 6)


def _fact_key(entry: MemoryEntry) -> str:
    subtype = entry.memory_subtype or "generic"
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
        category_key = _normalize_text(category) if isinstance(category, str) else subtype
        content_key = sha256(_normalize_text(entry.content).encode("utf-8")).hexdigest()
        return f"semantic:{category_key}:{content_key}"
    if entry.kind == "procedural":
        combo_key = entry.metadata.get("combo_key")
        if isinstance(combo_key, str) and combo_key.strip():
            return f"procedural:{subtype}:{combo_key}"
        content_key = sha256(_normalize_text(entry.content).encode("utf-8")).hexdigest()
        return f"procedural:{subtype}:{content_key}"
    event_id = entry.metadata.get("event_id") or entry.metadata.get("batch_id") or entry.memory_id
    return f"episodic:{subtype}:{event_id}"


def _bundle_kind(kind: str) -> str:
    if kind == "rule":
        return "rule"
    return kind


def _caps_for_stage(stage: str) -> dict[str, int]:
    lowered = stage.casefold()
    if "generate_translation" in lowered or "translation_generation" in lowered:
        return dict(_GENERATION_CAPS)
    if "adjudicat" in lowered and "translation" in lowered:
        return dict(_TRANSLATION_ADJUDICATION_CAPS)
    if "adjudicat" in lowered:
        return dict(_TRANSCRIPT_ADJUDICATION_CAPS)
    if "translation" in lowered:
        return dict(_TRANSLATION_REVIEW_CAPS)
    return dict(_TRANSCRIPT_REVIEW_CAPS)


def _procedural_stage_allowed(stage: str) -> bool:
    lowered = stage.casefold()
    if "generate_translation" in lowered or "translation_generation" in lowered:
        return True
    return "translation" in lowered and ("review" in lowered or "adjudicat" in lowered)


def _metadata_match_score(entry: MemoryEntry, query: MemoryQuery) -> float:
    filters = 0
    matches = 0.0
    metadata = entry.metadata
    query_asset_context = query.asset_context
    if query.media_key is not None:
        filters += 1
        if metadata.get("media_key") == query.media_key or entry.scope_key == query.media_key:
            matches += 1.0
    if query.series_id is not None:
        filters += 1
        if entry.series_id == query.series_id:
            matches += 1.0
    if query.franchise_id is not None:
        filters += 1
        if entry.franchise_id == query.franchise_id:
            matches += 1.0
    if query.speaker_ids:
        filters += 1
        if set(query.speaker_ids) & set(entry.speaker_ids):
            matches += 1.0
    if query.content_type is not None:
        filters += 1
        if entry.content_type == query.content_type:
            matches += 1.0
    if query.topic_tags:
        filters += 1
        if set(query.topic_tags) & set(entry.topic_tags):
            matches += 1.0
    if query.style_profile_id is not None:
        filters += 1
        if entry.style_profile_id == query.style_profile_id:
            matches += 1.0
    if query.entity_keys:
        filters += 1
        if set(query.entity_keys) & set(entry.entity_keys):
            matches += 1.0
    if query.term_keys:
        filters += 1
        if set(query.term_keys) & set(entry.term_keys):
            matches += 1.0
    if query_asset_context is not None and query_asset_context.channel_id is not None:
        filters += 1
        if entry.typed_metadata.get("channel_id") == query_asset_context.channel_id:
            matches += 1.0
    if query.provider_ids:
        filters += 1
        if metadata.get("transcript_provider_id") in set(query.provider_ids):
            matches += 1.0
    if query.prompt_variant_ids:
        filters += 1
        if metadata.get("prompt_variant_id") in set(query.prompt_variant_ids):
            matches += 1.0
    if query.model_ids:
        filters += 1
        if metadata.get("model_id") in set(query.model_ids):
            matches += 1.0
    if query.disagreement_bucket is not None:
        filters += 1
        if (
            metadata.get("source_disagreement_bucket") == query.disagreement_bucket
            or metadata.get("disagreement_bucket") == query.disagreement_bucket
        ):
            matches += 1.0
    if query.failure_tags:
        filters += 1
        entry_tags = metadata.get("failure_tags")
        if isinstance(entry_tags, (list, tuple, set)):
            if set(query.failure_tags) & {str(tag) for tag in entry_tags}:
                matches += 1.0
    if query.escalation_reasons:
        filters += 1
        entry_reasons = metadata.get("escalation_reasons")
        if isinstance(entry_reasons, (list, tuple, set)):
            if set(query.escalation_reasons) & {str(reason) for reason in entry_reasons}:
                matches += 1.0
    if query.glossary_misses:
        filters += 1
        glossary_misses = metadata.get("glossary_misses")
        if isinstance(glossary_misses, (list, tuple, set)):
            if set(query.glossary_misses) & {str(item) for item in glossary_misses}:
                matches += 1.0
    if query.entities:
        filters += 1
        entities = metadata.get("entities")
        if isinstance(entities, (list, tuple, set)):
            if set(query.entities) & {str(item) for item in entities}:
                matches += 1.0
    if query.numbers_dates:
        filters += 1
        numbers_dates = metadata.get("numbers_dates")
        if isinstance(numbers_dates, (list, tuple, set)):
            if set(query.numbers_dates) & {str(item) for item in numbers_dates}:
                matches += 1.0
    if query.candidate_ids:
        filters += 1
        candidate_id = metadata.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id in set(query.candidate_ids):
            matches += 1.0
    if filters == 0:
        return 0.0
    exact_bonus = 0.15 if matches == filters else 0.0
    return round(min(1.0, matches / filters + exact_bonus), 6)


def _embedding_relevance(
    entry: MemoryEntry,
    query_embedding: list[float] | None,
) -> float:
    if not query_embedding:
        return 0.0
    payload = entry.typed_metadata.get("embedding")
    if payload is None:
        return 0.0
    embedding = deserialize_embedding(payload)
    if not embedding:
        return 0.0
    return cosine_similarity(query_embedding, embedding)


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
