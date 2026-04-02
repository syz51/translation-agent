"""Memory consolidation interfaces and deterministic reference implementation."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

from translation_agent.memory.recall import MemoryEntryStore, build_scope_key
from translation_agent.models import (
    MemoryConsolidation,
    MemoryEntry,
    MemoryScopeKind,
    MemoryWrite,
    MemoryWriteBatch,
)


@runtime_checkable
class MemoryConsolidationBackend(Protocol):
    """Background consolidation contract for long-term semantic and episodic memory."""

    def consolidate_batch(self, batch: MemoryWriteBatch) -> MemoryConsolidation: ...


class DeterministicMemoryConsolidationBackend:
    """Reference consolidation backend with stable dedupe semantics."""

    def __init__(self, store: MemoryEntryStore) -> None:
        self._store = store

    def consolidate_batch(self, batch: MemoryWriteBatch) -> MemoryConsolidation:
        semantic_memory_ids, skipped_semantic = self._persist_writes(
            batch,
            writes=batch.semantic_writes,
        )
        episodic_memory_ids, skipped_episodic = self._persist_writes(
            batch,
            writes=batch.episodic_writes,
        )
        procedural_memory_ids, skipped_procedural = self._persist_writes(
            batch,
            writes=batch.procedural_writes,
        )
        skipped = tuple(dict.fromkeys((*skipped_semantic, *skipped_episodic, *skipped_procedural)))
        return MemoryConsolidation(
            consolidation_id=f"consolidation-{batch.batch_id}",
            batch_id=batch.batch_id,
            job_id=batch.job_id,
            source_stage=batch.source_stage,
            source_decision_ref=batch.decision_ref,
            source_decision_mode=batch.decision_mode,
            source_disagreement_bucket=batch.disagreement_bucket,
            source_translation_model_id=batch.translation_model_winner,
            source_prompt_variant_id=batch.prompt_variant_winner,
            source_prompt_version=batch.prompt_version_winner,
            source_language=_metadata_string(batch.metadata, "source_language"),
            target_language=_metadata_string(batch.metadata, "target_language"),
            scope_kind=_batch_scope_kind(batch),
            scope_key=_batch_scope_key(batch),
            semantic_memory_ids=semantic_memory_ids,
            episodic_memory_ids=episodic_memory_ids,
            procedural_memory_ids=procedural_memory_ids,
            skipped_dedupe_keys=skipped,
            procedural_write_count=len(batch.procedural_writes),
        )

    def _persist_writes(
        self,
        batch: MemoryWriteBatch,
        *,
        writes: tuple[MemoryWrite, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        inserted_ids: list[str] = []
        skipped_keys: list[str] = []
        for index, write in enumerate(writes, start=1):
            dedupe_key = _dedupe_key(write, batch=batch, index=index)
            scope_kind, scope_key = _resolved_write_scope(batch=batch, write=write)
            entry = MemoryEntry(
                memory_id=f"{write.kind}:{batch.batch_id}:{index}",
                kind=write.kind,
                content=write.content,
                source_ref=write.source_ref or batch.decision_ref,
                scope_kind=scope_kind,
                scope_key=scope_key,
                updated_at=write.updated_at,
                score=write.score,
                metadata={
                    **batch.metadata,
                    **write.metadata,
                    "batch_id": batch.batch_id,
                    "job_id": batch.job_id,
                    "source_stage": batch.source_stage,
                },
            )
            if self._store.put_entry(entry, dedupe_key=dedupe_key):
                inserted_ids.append(entry.memory_id)
            elif dedupe_key is not None:
                skipped_keys.append(dedupe_key)
        return tuple(inserted_ids), tuple(skipped_keys)


def _dedupe_key(write: MemoryWrite, *, batch: MemoryWriteBatch, index: int) -> str | None:
    raw_key = write.metadata.get("dedupe_key")
    if isinstance(raw_key, str):
        return raw_key
    if batch.dedupe_keys:
        return f"{batch.dedupe_keys[0]}:{write.kind}:{index}"
    return None


def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _resolved_write_scope(
    *,
    batch: MemoryWriteBatch,
    write: MemoryWrite,
) -> tuple[MemoryScopeKind, str]:
    source_language = _metadata_string(batch.metadata, "source_language")
    target_language = _metadata_string(batch.metadata, "target_language")
    if (
        write.scope_kind == "global"
        and write.scope_key == "global"
        and source_language is not None
        and target_language is not None
    ):
        return (
            "pair",
            build_scope_key(
                scope_kind="pair",
                tenant_id=_metadata_string(batch.metadata, "tenant_id") or "",
                project_id=_metadata_string(batch.metadata, "project_id") or "",
                source_language=source_language,
                target_language=target_language,
            ),
        )
    return write.scope_kind, write.scope_key


def _batch_scope_kind(batch: MemoryWriteBatch) -> MemoryScopeKind | None:
    writes = (*batch.semantic_writes, *batch.episodic_writes, *batch.procedural_writes)
    if not writes:
        return None
    scope_kinds = {_resolved_write_scope(batch=batch, write=write)[0] for write in writes}
    if len(scope_kinds) == 1:
        return cast(MemoryScopeKind, next(iter(scope_kinds)))
    return None


def _batch_scope_key(batch: MemoryWriteBatch) -> str | None:
    writes = (*batch.semantic_writes, *batch.episodic_writes, *batch.procedural_writes)
    if not writes:
        return None
    scope_keys = {_resolved_write_scope(batch=batch, write=write)[1] for write in writes}
    if len(scope_keys) == 1:
        return next(iter(scope_keys))
    return None
