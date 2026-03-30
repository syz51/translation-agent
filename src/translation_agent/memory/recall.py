"""Memory recall interfaces and in-memory reference implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from translation_agent.models import MemoryBundle, MemoryEntry, MemoryQuery, ProviderCaveat


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
        return sorted(
            self.entries.values(),
            key=lambda entry: (-float(entry.score or 0.0), entry.memory_id),
        )


@runtime_checkable
class MemoryRecallBackend(Protocol):
    """Read-only long-term memory contract for review and adjudication."""

    def recall_memory(self, query: MemoryQuery) -> MemoryBundle: ...


class LongTermMemoryRecallBackend:
    """Recall backend that scopes reads to tenant/project/language metadata."""

    def __init__(self, store: InMemoryLongTermMemoryStore) -> None:
        self._store = store

    def recall_memory(self, query: MemoryQuery) -> MemoryBundle:
        semantic_entries: list[MemoryEntry] = [_default_semantic_entry(query)]
        glossary_entries: list[MemoryEntry] = [_default_glossary_entry(query)]
        rule_entries: list[MemoryEntry] = [_default_rule_entry(query)]
        episodic_entries: list[MemoryEntry] = []
        provider_caveats = (_default_provider_caveat(),)

        for entry in self._store.list_entries():
            if not _matches_scope(entry, query):
                continue
            if entry.kind == "semantic":
                semantic_entries.append(entry)
            elif entry.kind == "episodic":
                episodic_entries.append(entry)
            elif entry.kind == "glossary":
                glossary_entries.append(entry)
            elif entry.kind == "rule":
                rule_entries.append(entry)

        max_items = query.max_items
        return MemoryBundle(
            semantic_memory=tuple(semantic_entries[:max_items]),
            episodic_memory=tuple(episodic_entries[:max_items]),
            glossary=tuple(glossary_entries[:max_items]),
            rules=tuple(rule_entries[:max_items]),
            provider_caveats=provider_caveats,
        )


def _matches_scope(entry: MemoryEntry, query: MemoryQuery) -> bool:
    metadata = entry.metadata
    scoped_keys = {
        "tenant_id": query.job.tenant_id,
        "project_id": query.job.project_id,
        "source_language": query.job.source_language,
        "target_language": query.job.target_language,
    }
    for key, value in scoped_keys.items():
        scoped_value = metadata.get(key)
        if scoped_value is not None and scoped_value != value:
            return False
    return True


def _default_semantic_entry(query: MemoryQuery) -> MemoryEntry:
    return MemoryEntry(
        memory_id=f"semantic:{query.stage}:{query.job.project_id}",
        kind="semantic",
        content="Preserve named entities and product terms.",
        source_ref="memory/semantic/dry-run",
        score=0.74,
        metadata={"project_id": query.job.project_id, "tenant_id": query.job.tenant_id},
    )


def _default_glossary_entry(query: MemoryQuery) -> MemoryEntry:
    return MemoryEntry(
        memory_id=f"glossary:{query.job.target_language}",
        kind="glossary",
        content="OpenAI -> OpenAI",
        source_ref="memory/glossary/dry-run",
        metadata={"target_language": query.job.target_language},
    )


def _default_rule_entry(query: MemoryQuery) -> MemoryEntry:
    return MemoryEntry(
        memory_id=f"rule:{query.stage}",
        kind="rule",
        content=f"Prefer deterministic dry-run behavior for {query.stage}.",
        source_ref="memory/rules/dry-run",
        metadata={"tenant_id": query.job.tenant_id},
    )


def _default_provider_caveat() -> ProviderCaveat:
    return ProviderCaveat(
        provider_id="speechmatics",
        note="The fake provider can be disabled for degraded-path tests.",
    )
