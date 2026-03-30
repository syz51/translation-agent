"""Memory recall interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import MemoryBundle, MemoryQuery


@runtime_checkable
class MemoryRecallBackend(Protocol):
    """Read-only long-term memory contract for review and adjudication."""

    def recall_memory(self, query: MemoryQuery) -> MemoryBundle: ...
