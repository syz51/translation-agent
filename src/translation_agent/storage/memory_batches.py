"""Memory batch persistence interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import MemoryWriteBatch


@runtime_checkable
class MemoryBatchStore(Protocol):
    """Persistence contract for adjudication-time memory write batches."""

    def save_batch(self, batch: MemoryWriteBatch) -> None: ...

    def get_batch(self, batch_id: str) -> MemoryWriteBatch | None: ...

    def list_batches(self, job_id: str) -> list[MemoryWriteBatch]: ...
