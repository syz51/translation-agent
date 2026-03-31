from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.models import (
    AdjudicationScorecard,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    JobContext,
    MemoryWrite,
    MemoryWriteBatch,
    Segment,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.storage import (
    PostgresOperationalStore,
    SQLiteOperationalStore,
    job_scope_token,
    operational_job_key,
)


def _job(job_id: str = "job-operational") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
    )


def _transcript_candidate(job_id: str = "job-operational") -> TranscriptCandidate:
    return TranscriptCandidate(
        candidate_id=f"tr-{job_id}",
        job_id=job_id,
        provider_id="assemblyai",
        provider_request_id="req-1",
        language="en",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=1000,
                speaker="speaker-1",
                source_text="Hello workflow.",
            ),
        ),
        full_text="Hello workflow.",
        speaker_map={"speaker-1": "Host"},
        timing_resolution="segment",
        raw_payload_ref="raw/transcript.json",
        normalization_version="2026-03-31-test",
        metadata={"provider_rank": 0},
    )


def _translation_candidate(job_id: str = "job-operational") -> TranslationCandidate:
    return TranslationCandidate(
        candidate_id=f"tl-{job_id}",
        job_id=job_id,
        source_transcript_candidate_id=f"tr-{job_id}",
        model_id="gpt-5.4-mini",
        prompt_variant_id="variant-a",
        prompt_version="phase-5-v1",
        language="fr",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=1000,
                speaker="speaker-1",
                source_text="Hello workflow.",
                target_text="Bonjour workflow.",
            ),
        ),
        full_text="Bonjour workflow.",
        raw_response_ref="raw/translation.json",
        normalization_version="2026-03-31-test",
        metadata={},
    )


def _scorecard() -> AdjudicationScorecard:
    return AdjudicationScorecard(
        candidate_count=2,
        preferred_candidate_id="candidate-a",
        average_confidence=0.81,
        confidence_spread=0.12,
        contradictory_evidence_count=1,
        highest_issue_severity="major",
        winner_mismatch=True,
        escalation_signal_count=1,
        total_score=3.4,
        content_risk_class="standard",
    )


def _transcript_decision(job_id: str = "job-operational") -> FinalTranscriptDecision:
    return FinalTranscriptDecision(
        job_id=job_id,
        winner_candidate_id=f"tr-{job_id}",
        decision_mode="automatic_finalize",
        decision_confidence=0.77,
        rationale_summary="Transcript finalized after deterministic adjudication.",
        review_refs=("rev-1", "rev-2"),
        investigation_ref="investigations/transcript.json",
        disagreement_bucket="low",
        adjudication_scorecard=_scorecard(),
        escalated=False,
        human_review_required=False,
    )


def _translation_decision(job_id: str = "job-operational") -> FinalTranslationDecision:
    return FinalTranslationDecision(
        job_id=job_id,
        winner_candidate_id=f"tl-{job_id}",
        decision_mode="conflict_investigation",
        decision_confidence=0.63,
        rationale_summary="Conflict investigation ran before re-adjudication.",
        review_refs=("rev-3", "rev-4"),
        investigation_ref="investigations/translation.json",
        disagreement_bucket="medium",
        adjudication_scorecard=_scorecard(),
        escalated=True,
        human_review_required=False,
        winner_model_id="gpt-5.4-mini",
        prompt_variant_winner="variant-a",
        prompt_version_winner="phase-5-v1",
    )


def _memory_batch(job_id: str = "job-operational") -> MemoryWriteBatch:
    return MemoryWriteBatch(
        batch_id=f"batch-{job_id}",
        job_id=job_id,
        source_stage="translation_adjudication",
        decision_ref="decisions/translation.json",
        investigation_ref="investigations/translation.json",
        winner_candidate_id=f"tl-{job_id}",
        decision_mode="conflict_investigation",
        decision_confidence=0.63,
        disagreement_bucket="medium",
        translation_model_winner="gpt-5.4-mini",
        prompt_variant_winner="variant-a",
        prompt_version_winner="phase-5-v1",
        semantic_writes=(MemoryWrite(kind="semantic", content="Prefer workflow terminology."),),
        dedupe_keys=("semantic:workflow",),
    )


@pytest.mark.unit
def test_sqlite_operational_store_persists_candidates_decisions_investigations_and_batches(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.sqlite3"
    job = _job()
    transcript_candidate = _transcript_candidate(job.job_id)
    translation_candidate = _translation_candidate(job.job_id)
    transcript_decision = _transcript_decision(job.job_id)
    translation_decision = _translation_decision(job.job_id)
    batch = _memory_batch(job.job_id)
    investigation: dict[str, object] = {
        "status": "resolved",
        "strategy": "conflict investigator",
    }

    with SQLiteOperationalStore(db_path) as store:
        store.save_transcript_candidate(transcript_candidate)
        store.save_translation_candidate(translation_candidate)
        store.save_transcript_decision(transcript_decision)
        store.save_translation_decision(translation_decision)
        store.save_investigation(job_id=job.job_id, stage="translation", payload=investigation)
        store.save_batch(batch)

    with SQLiteOperationalStore(db_path) as reopened:
        assert reopened.list_transcript_candidates(job.job_id) == [transcript_candidate]
        assert reopened.list_translation_candidates(job.job_id) == [translation_candidate]
        assert reopened.get_transcript_decision(job.job_id) == transcript_decision
        assert reopened.get_translation_decision(job.job_id) == translation_decision
        assert reopened.get_investigation(job_id=job.job_id, stage="translation") == investigation
        assert reopened.get_batch(batch.batch_id) == batch


@pytest.mark.integration
def test_postgres_operational_store_persists_candidates_decisions_investigations_and_batches(
    migrated_postgres_dsn: str,
) -> None:
    job = _job("job-operational-postgres")
    transcript_candidate = _transcript_candidate(job.job_id)
    translation_candidate = _translation_candidate(job.job_id)
    transcript_decision = _transcript_decision(job.job_id)
    translation_decision = _translation_decision(job.job_id)
    batch = _memory_batch(job.job_id)
    investigation: dict[str, object] = {
        "status": "resolved",
        "strategy": "stronger adjudicator",
    }

    with PostgresOperationalStore(migrated_postgres_dsn) as store:
        store.save_transcript_candidate(transcript_candidate)
        store.save_translation_candidate(translation_candidate)
        store.save_transcript_decision(transcript_decision)
        store.save_translation_decision(translation_decision)
        store.save_investigation(job_id=job.job_id, stage="translation", payload=investigation)
        store.save_batch(batch)

    with PostgresOperationalStore(migrated_postgres_dsn) as reopened:
        assert reopened.list_transcript_candidates(job.job_id) == [transcript_candidate]
        assert reopened.list_translation_candidates(job.job_id) == [translation_candidate]
        assert reopened.get_transcript_decision(job.job_id) == transcript_decision
        assert reopened.get_translation_decision(job.job_id) == translation_decision
        assert reopened.get_investigation(job_id=job.job_id, stage="translation") == investigation
        assert reopened.get_batch(batch.batch_id) == batch


@pytest.mark.unit
def test_sqlite_operational_store_isolates_same_raw_job_id_by_scoped_storage_key(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.sqlite3"
    tenant_a = _job("job-shared")
    tenant_b = _job("job-shared").model_copy(update={"tenant_id": "tenant-2"})

    transcript_a = _transcript_candidate(tenant_a.job_id).model_copy(
        update={"candidate_id": f"tr-{tenant_a.job_id}-{job_scope_token(tenant_a)}"}
    )
    transcript_b = _transcript_candidate(tenant_b.job_id).model_copy(
        update={"candidate_id": f"tr-{tenant_b.job_id}-{job_scope_token(tenant_b)}"}
    )
    decision_a = _transcript_decision(tenant_a.job_id)
    decision_b = _transcript_decision(tenant_b.job_id).model_copy(
        update={"winner_candidate_id": transcript_b.candidate_id}
    )

    with SQLiteOperationalStore(db_path) as store:
        store.save_transcript_candidate(
            transcript_a,
            storage_job_id=operational_job_key(tenant_a),
        )
        store.save_transcript_candidate(
            transcript_b,
            storage_job_id=operational_job_key(tenant_b),
        )
        store.save_transcript_decision(
            decision_a,
            storage_job_id=operational_job_key(tenant_a),
        )
        store.save_transcript_decision(
            decision_b,
            storage_job_id=operational_job_key(tenant_b),
        )

    with SQLiteOperationalStore(db_path) as reopened:
        assert reopened.list_transcript_candidates(
            tenant_a.job_id,
            storage_job_id=operational_job_key(tenant_a),
        ) == [transcript_a]
        assert reopened.list_transcript_candidates(
            tenant_b.job_id,
            storage_job_id=operational_job_key(tenant_b),
        ) == [transcript_b]
        assert (
            reopened.get_transcript_decision(
                tenant_a.job_id,
                storage_job_id=operational_job_key(tenant_a),
            )
            == decision_a
        )
        assert (
            reopened.get_transcript_decision(
                tenant_b.job_id,
                storage_job_id=operational_job_key(tenant_b),
            )
            == decision_b
        )


@pytest.mark.integration
def test_postgres_operational_store_isolates_same_raw_job_id_by_scoped_storage_key(
    migrated_postgres_dsn: str,
) -> None:
    tenant_a = _job("job-shared-postgres")
    tenant_b = _job("job-shared-postgres").model_copy(update={"tenant_id": "tenant-2"})

    transcript_a = _transcript_candidate(tenant_a.job_id).model_copy(
        update={"candidate_id": f"tr-{tenant_a.job_id}-{job_scope_token(tenant_a)}"}
    )
    transcript_b = _transcript_candidate(tenant_b.job_id).model_copy(
        update={"candidate_id": f"tr-{tenant_b.job_id}-{job_scope_token(tenant_b)}"}
    )
    decision_a = _transcript_decision(tenant_a.job_id)
    decision_b = _transcript_decision(tenant_b.job_id).model_copy(
        update={"winner_candidate_id": transcript_b.candidate_id}
    )

    with PostgresOperationalStore(migrated_postgres_dsn) as store:
        store.save_transcript_candidate(
            transcript_a,
            storage_job_id=operational_job_key(tenant_a),
        )
        store.save_transcript_candidate(
            transcript_b,
            storage_job_id=operational_job_key(tenant_b),
        )
        store.save_transcript_decision(
            decision_a,
            storage_job_id=operational_job_key(tenant_a),
        )
        store.save_transcript_decision(
            decision_b,
            storage_job_id=operational_job_key(tenant_b),
        )

    with PostgresOperationalStore(migrated_postgres_dsn) as reopened:
        assert reopened.list_transcript_candidates(
            tenant_a.job_id,
            storage_job_id=operational_job_key(tenant_a),
        ) == [transcript_a]
        assert reopened.list_transcript_candidates(
            tenant_b.job_id,
            storage_job_id=operational_job_key(tenant_b),
        ) == [transcript_b]
        assert (
            reopened.get_transcript_decision(
                tenant_a.job_id,
                storage_job_id=operational_job_key(tenant_a),
            )
            == decision_a
        )
        assert (
            reopened.get_transcript_decision(
                tenant_b.job_id,
                storage_job_id=operational_job_key(tenant_b),
            )
            == decision_b
        )
