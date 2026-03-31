"""Memory consolidation interfaces and deterministic reference implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.memory.recall import MemoryEntryStore
from translation_agent.models import MemoryConsolidation, MemoryEntry, MemoryWrite, MemoryWriteBatch


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
        skipped = tuple(dict.fromkeys((*skipped_semantic, *skipped_episodic)))
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
            semantic_memory_ids=semantic_memory_ids,
            episodic_memory_ids=episodic_memory_ids,
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
            entry = MemoryEntry(
                memory_id=f"{write.kind}:{batch.batch_id}:{index}",
                kind=write.kind,
                content=write.content,
                source_ref=write.source_ref or batch.decision_ref,
                score=_score_for_write(batch),
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


def _score_for_write(batch: MemoryWriteBatch) -> float:
    if batch.decision_confidence is None:
        return 0.5
    return round(max(0.0, min(batch.decision_confidence, 0.99)), 4)
