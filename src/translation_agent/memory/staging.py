"""Memory staging interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import (
    FinalTranscriptDecision,
    FinalTranslationDecision,
    MemoryWriteBatch,
)


@runtime_checkable
class MemoryStagingBackend(Protocol):
    """Adjudication-boundary staging contract for candidate memory writes."""

    def stage_memory_candidates(
        self,
        decision: FinalTranscriptDecision | FinalTranslationDecision,
        *,
        source_stage: str,
    ) -> MemoryWriteBatch: ...
