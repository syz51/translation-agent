"""Decision persistence interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import (
    FinalTranscriptDecision,
    FinalTranslationDecision,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.models.review import ReviewStage


@runtime_checkable
class DecisionStore(Protocol):
    """Persistence contract for normalized candidates and adjudication outputs."""

    def save_transcript_candidate(
        self,
        candidate: TranscriptCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None: ...

    def list_transcript_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranscriptCandidate]: ...

    def save_transcript_decision(
        self,
        decision: FinalTranscriptDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None: ...

    def get_transcript_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranscriptDecision | None: ...

    def save_translation_candidate(
        self,
        candidate: TranslationCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None: ...

    def list_translation_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranslationCandidate]: ...

    def save_translation_decision(
        self,
        decision: FinalTranslationDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None: ...

    def get_translation_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranslationDecision | None: ...

    def save_investigation(
        self,
        *,
        job_id: str,
        stage: ReviewStage,
        payload: dict[str, object],
        storage_job_id: str | None = None,
    ) -> None: ...

    def get_investigation(
        self,
        *,
        job_id: str,
        stage: ReviewStage,
        storage_job_id: str | None = None,
    ) -> dict[str, object] | None: ...
