"""Decision persistence interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import (
    FinalTranscriptDecision,
    FinalTranslationDecision,
    TranscriptCandidate,
    TranslationCandidate,
)


@runtime_checkable
class DecisionStore(Protocol):
    """Persistence contract for normalized candidates and adjudication outputs."""

    def save_transcript_candidate(self, candidate: TranscriptCandidate) -> None: ...

    def list_transcript_candidates(self, job_id: str) -> list[TranscriptCandidate]: ...

    def save_transcript_decision(self, decision: FinalTranscriptDecision) -> None: ...

    def get_transcript_decision(self, job_id: str) -> FinalTranscriptDecision | None: ...

    def save_translation_candidate(self, candidate: TranslationCandidate) -> None: ...

    def list_translation_candidates(self, job_id: str) -> list[TranslationCandidate]: ...

    def save_translation_decision(self, decision: FinalTranslationDecision) -> None: ...

    def get_translation_decision(self, job_id: str) -> FinalTranslationDecision | None: ...
