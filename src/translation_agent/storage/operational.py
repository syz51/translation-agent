"""Operational stores for local SQLite and Postgres-backed runtimes."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from translation_agent.models import (
    AssetContext,
    AssetRecord,
    AssetRelation,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    HistoricalRunLink,
    HumanReviewResolutionRecord,
    MemoryEntry,
    MemoryEvidenceEvent,
    MemoryQuery,
    MemoryWriteBatch,
    PromptEvolutionProposal,
    TranscriptCandidate,
    TranscriptProviderQualityStats,
    TranslationCandidate,
    TranslationFeedbackStats,
)
from translation_agent.models.review import ReviewStage
from translation_agent.search_index import (
    EMBEDDING_MODEL_ID,
    embedding_metadata_for_entry,
    embedding_metadata_for_query,
    serialize_embedding,
)

from .decisions import DecisionStore
from .memory_batches import MemoryBatchStore
from .runs import (
    NodeExecutionRecord,
    PostgresRunStore,
    RunRecord,
    _encode_json,
    _isoformat_timestamp,
    _require_node_execution,
    _require_run,
    _utc_now,
)

_UNSET = object()
_SQL_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_POSTGRES_SINGLETON_UPSERT_SQL = {
    "transcript_decisions": """
        INSERT INTO transcript_decisions (job_id, decision_json, created_at, updated_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            decision_json = EXCLUDED.decision_json,
            updated_at = EXCLUDED.updated_at
    """,
    "translation_decisions": """
        INSERT INTO translation_decisions (job_id, decision_json, created_at, updated_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            decision_json = EXCLUDED.decision_json,
            updated_at = EXCLUDED.updated_at
    """,
}
_POSTGRES_SINGLETON_SELECT_SQL = {
    "transcript_decisions": "SELECT decision_json FROM transcript_decisions WHERE job_id = %s",
    "translation_decisions": "SELECT decision_json FROM translation_decisions WHERE job_id = %s",
}
_POSTGRES_STAGE_UPSERT_SQL = {
    "investigations": """
        INSERT INTO investigations (job_id, stage, payload_json, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (job_id, stage) DO UPDATE SET
            payload_json = EXCLUDED.payload_json,
            updated_at = EXCLUDED.updated_at
    """
}
_POSTGRES_STAGE_SELECT_SQL = {
    "investigations": "SELECT payload_json FROM investigations WHERE job_id = %s AND stage = %s"
}
_POSTGRES_PROVIDER_STATS_UPSERT_SQL = """
    INSERT INTO transcript_provider_quality_stats (
        provider_id,
        source_language,
        target_language,
        stats_json,
        created_at,
        updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (provider_id, source_language, target_language) DO UPDATE SET
        stats_json = EXCLUDED.stats_json,
        updated_at = EXCLUDED.updated_at
"""
_POSTGRES_PROVIDER_STATS_SELECT_SQL = """
    SELECT stats_json
    FROM transcript_provider_quality_stats
    WHERE provider_id = %s AND source_language = %s AND target_language = %s
"""
_POSTGRES_HUMAN_RESOLUTION_UPSERT_SQL = """
    INSERT INTO human_review_resolutions (
        run_id,
        job_id,
        resolution_json,
        resolution_kind,
        source_language,
        target_language,
        transcript_provider_id,
        model_id,
        prompt_variant_id,
        prompt_version,
        combo_key,
        resolved_at,
        created_at,
        updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (run_id) DO UPDATE SET
        job_id = EXCLUDED.job_id,
        resolution_json = EXCLUDED.resolution_json,
        resolution_kind = EXCLUDED.resolution_kind,
        source_language = EXCLUDED.source_language,
        target_language = EXCLUDED.target_language,
        transcript_provider_id = EXCLUDED.transcript_provider_id,
        model_id = EXCLUDED.model_id,
        prompt_variant_id = EXCLUDED.prompt_variant_id,
        prompt_version = EXCLUDED.prompt_version,
        combo_key = EXCLUDED.combo_key,
        resolved_at = EXCLUDED.resolved_at,
        updated_at = EXCLUDED.updated_at
"""
_POSTGRES_HUMAN_RESOLUTION_SELECT_SQL = """
    SELECT resolution_json
    FROM human_review_resolutions
    WHERE run_id = %s
"""
_POSTGRES_HUMAN_RESOLUTION_LIST_SQL = """
    SELECT resolution_json
    FROM human_review_resolutions
    WHERE (%s IS NULL OR resolution_kind = %s)
      AND (%s IS NULL OR source_language = %s)
      AND (%s IS NULL OR target_language = %s)
      AND (%s IS NULL OR transcript_provider_id = %s)
      AND (%s IS NULL OR model_id = %s)
      AND (%s IS NULL OR prompt_variant_id = %s)
      AND (%s IS NULL OR prompt_version = %s)
      AND (%s IS NULL OR combo_key = %s)
    ORDER BY resolved_at ASC, run_id ASC
"""
_POSTGRES_TRANSLATION_FEEDBACK_UPSERT_SQL = """
    INSERT INTO translation_feedback_stats (
        combo_key,
        stats_json,
        source_language,
        target_language,
        transcript_provider_id,
        model_id,
        prompt_variant_id,
        prompt_version,
        created_at,
        updated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (combo_key) DO UPDATE SET
        stats_json = EXCLUDED.stats_json,
        source_language = EXCLUDED.source_language,
        target_language = EXCLUDED.target_language,
        transcript_provider_id = EXCLUDED.transcript_provider_id,
        model_id = EXCLUDED.model_id,
        prompt_variant_id = EXCLUDED.prompt_variant_id,
        prompt_version = EXCLUDED.prompt_version,
        updated_at = EXCLUDED.updated_at
"""
_POSTGRES_TRANSLATION_FEEDBACK_SELECT_SQL = """
    SELECT stats_json
    FROM translation_feedback_stats
    WHERE combo_key = %s
"""
_SQLITE_CANDIDATE_UPSERT_SQL = {
    "transcript_candidates": """
        INSERT INTO transcript_candidates (
            candidate_id, job_id, candidate_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            job_id = excluded.job_id,
            candidate_json = excluded.candidate_json,
            updated_at = excluded.updated_at
    """,
    "translation_candidates": """
        INSERT INTO translation_candidates (
            candidate_id, job_id, candidate_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            job_id = excluded.job_id,
            candidate_json = excluded.candidate_json,
            updated_at = excluded.updated_at
    """,
}
_SQLITE_SINGLETON_UPSERT_SQL = {
    "transcript_decisions": """
        INSERT INTO transcript_decisions (job_id, decision_json, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            decision_json = excluded.decision_json,
            updated_at = excluded.updated_at
    """,
    "translation_decisions": """
        INSERT INTO translation_decisions (job_id, decision_json, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            decision_json = excluded.decision_json,
            updated_at = excluded.updated_at
    """,
}
_SQLITE_SINGLETON_SELECT_SQL = {
    "transcript_decisions": "SELECT decision_json FROM transcript_decisions WHERE job_id = ?",
    "translation_decisions": "SELECT decision_json FROM translation_decisions WHERE job_id = ?",
}
_SQLITE_STAGE_UPSERT_SQL = {
    "investigations": """
        INSERT INTO investigations (job_id, stage, payload_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(job_id, stage) DO UPDATE SET
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
    """
}
_SQLITE_STAGE_SELECT_SQL = {
    "investigations": "SELECT payload_json FROM investigations WHERE job_id = ? AND stage = ?"
}
_SQLITE_PROVIDER_STATS_UPSERT_SQL = """
    INSERT INTO transcript_provider_quality_stats (
        provider_id,
        source_language,
        target_language,
        stats_json,
        created_at,
        updated_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(provider_id, source_language, target_language) DO UPDATE SET
        stats_json = excluded.stats_json,
        updated_at = excluded.updated_at
"""
_SQLITE_PROVIDER_STATS_SELECT_SQL = """
    SELECT stats_json
    FROM transcript_provider_quality_stats
    WHERE provider_id = ? AND source_language = ? AND target_language = ?
"""
_SQLITE_HUMAN_RESOLUTION_UPSERT_SQL = """
    INSERT INTO human_review_resolutions (
        run_id,
        job_id,
        resolution_json,
        resolution_kind,
        source_language,
        target_language,
        transcript_provider_id,
        model_id,
        prompt_variant_id,
        prompt_version,
        combo_key,
        resolved_at,
        created_at,
        updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run_id) DO UPDATE SET
        job_id = excluded.job_id,
        resolution_json = excluded.resolution_json,
        resolution_kind = excluded.resolution_kind,
        source_language = excluded.source_language,
        target_language = excluded.target_language,
        transcript_provider_id = excluded.transcript_provider_id,
        model_id = excluded.model_id,
        prompt_variant_id = excluded.prompt_variant_id,
        prompt_version = excluded.prompt_version,
        combo_key = excluded.combo_key,
        resolved_at = excluded.resolved_at,
        updated_at = excluded.updated_at
"""
_SQLITE_HUMAN_RESOLUTION_SELECT_SQL = """
    SELECT resolution_json
    FROM human_review_resolutions
    WHERE run_id = ?
"""
_SQLITE_HUMAN_RESOLUTION_LIST_SQL = """
    SELECT resolution_json
    FROM human_review_resolutions
    WHERE (? IS NULL OR resolution_kind = ?)
      AND (? IS NULL OR source_language = ?)
      AND (? IS NULL OR target_language = ?)
      AND (? IS NULL OR transcript_provider_id = ?)
      AND (? IS NULL OR model_id = ?)
      AND (? IS NULL OR prompt_variant_id = ?)
      AND (? IS NULL OR prompt_version = ?)
      AND (? IS NULL OR combo_key = ?)
    ORDER BY resolved_at ASC, run_id ASC
"""
_SQLITE_TRANSLATION_FEEDBACK_UPSERT_SQL = """
    INSERT INTO translation_feedback_stats (
        combo_key,
        stats_json,
        source_language,
        target_language,
        transcript_provider_id,
        model_id,
        prompt_variant_id,
        prompt_version,
        created_at,
        updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(combo_key) DO UPDATE SET
        stats_json = excluded.stats_json,
        source_language = excluded.source_language,
        target_language = excluded.target_language,
        transcript_provider_id = excluded.transcript_provider_id,
        model_id = excluded.model_id,
        prompt_variant_id = excluded.prompt_variant_id,
        prompt_version = excluded.prompt_version,
        updated_at = excluded.updated_at
"""
_SQLITE_TRANSLATION_FEEDBACK_SELECT_SQL = """
    SELECT stats_json
    FROM translation_feedback_stats
    WHERE combo_key = ?
"""
_SQLITE_LIST_SQL = {
    "transcript_candidates": (
        "SELECT candidate_json FROM transcript_candidates "
        "WHERE job_id = ? ORDER BY candidate_id ASC"
    ),
    "translation_candidates": (
        "SELECT candidate_json FROM translation_candidates "
        "WHERE job_id = ? ORDER BY candidate_id ASC"
    ),
    "memory_batches": (
        "SELECT batch_json FROM memory_batches WHERE job_id = ? ORDER BY batch_id ASC"
    ),
}
_POSTGRES_ASSET_SELECT_SQL = {
    "media_key": "SELECT asset_json FROM asset_records WHERE media_key = %s",
    "asset_id": "SELECT asset_json FROM asset_records WHERE asset_id = %s",
    "media_fingerprint": "SELECT asset_json FROM asset_records WHERE media_fingerprint = %s",
}
_POSTGRES_ASSET_CONTEXT_SELECT_SQL = (
    "SELECT context_json FROM asset_context_records WHERE media_key = %s"
)
_POSTGRES_ASSET_CONTEXT_LIST_SQL = (
    "SELECT context_json FROM asset_context_records ORDER BY media_key ASC"
)
_POSTGRES_ASSET_RELATION_LIST_SQL = """
    SELECT relation_json
    FROM asset_relations
    WHERE src_media_key = %s OR dst_media_key = %s
    ORDER BY relation_kind ASC, relation_id ASC
"""
_POSTGRES_MEMORY_ENTRY_SELECT_SQL = "SELECT entry_json FROM memory_entries WHERE memory_id = %s"
_POSTGRES_MEMORY_ENTRY_LIST_SQL = "SELECT entry_json FROM memory_entries ORDER BY memory_id ASC"
_POSTGRES_MEMORY_DEDUPE_SELECT_SQL = "SELECT memory_id FROM memory_entries WHERE dedupe_key = %s"
_POSTGRES_MEMORY_ENTRY_INDEX_SELECT_SQL = """
    SELECT memory_id, entry_json, dedupe_key, embedding_json
    FROM memory_entries
    ORDER BY updated_at ASC, memory_id ASC
"""
_POSTGRES_PROPOSAL_LIST_SQL = """
    SELECT proposal_json
    FROM prompt_evolution_proposals
    WHERE (%s IS NULL OR status = %s)
      AND (%s IS NULL OR prompt_family = %s)
      AND (%s IS NULL OR target_model_id = %s)
      AND (%s IS NULL OR target_language = %s)
      AND (%s IS NULL OR source_language = %s)
      AND (%s IS NULL OR prompt_variant_id = %s)
      AND (%s IS NULL OR base_prompt_version = %s)
      AND (%s IS NULL OR scope_kind = %s)
      AND (%s IS NULL OR scope_key = %s)
      AND (%s IS NULL OR media_key = %s)
    ORDER BY proposal_id ASC
"""
_SQLITE_ASSET_SELECT_SQL = {
    "media_key": "SELECT asset_json FROM asset_records WHERE media_key = ?",
    "asset_id": "SELECT asset_json FROM asset_records WHERE asset_id = ?",
    "media_fingerprint": "SELECT asset_json FROM asset_records WHERE media_fingerprint = ?",
}
_SQLITE_ASSET_CONTEXT_SELECT_SQL = (
    "SELECT context_json FROM asset_context_records WHERE media_key = ?"
)
_SQLITE_ASSET_CONTEXT_LIST_SQL = (
    "SELECT context_json FROM asset_context_records ORDER BY media_key ASC"
)
_SQLITE_ASSET_RELATION_LIST_SQL = """
    SELECT relation_json
    FROM asset_relations
    WHERE src_media_key = ? OR dst_media_key = ?
    ORDER BY relation_kind ASC, relation_id ASC
"""
_SQLITE_MEMORY_ENTRY_SELECT_SQL = "SELECT entry_json FROM memory_entries WHERE memory_id = ?"
_SQLITE_MEMORY_ENTRY_LIST_SQL = "SELECT entry_json FROM memory_entries ORDER BY memory_id ASC"
_SQLITE_MEMORY_DEDUPE_SELECT_SQL = "SELECT memory_id FROM memory_entries WHERE dedupe_key = ?"
_SQLITE_MEMORY_ENTRY_INDEX_SELECT_SQL = """
    SELECT memory_id, entry_json, dedupe_key, embedding_json
    FROM memory_entries
    ORDER BY updated_at ASC, memory_id ASC
"""
_SQLITE_PROPOSAL_LIST_SQL = """
    SELECT proposal_json
    FROM prompt_evolution_proposals
    WHERE (? IS NULL OR status = ?)
      AND (? IS NULL OR prompt_family = ?)
      AND (? IS NULL OR target_model_id = ?)
      AND (? IS NULL OR target_language = ?)
      AND (? IS NULL OR source_language = ?)
      AND (? IS NULL OR prompt_variant_id = ?)
      AND (? IS NULL OR base_prompt_version = ?)
      AND (? IS NULL OR scope_kind = ?)
      AND (? IS NULL OR scope_key = ?)
      AND (? IS NULL OR media_key = ?)
    ORDER BY proposal_id ASC
"""


class OperationalStore(DecisionStore, MemoryBatchStore, Protocol):
    """Combined operational persistence contract used by public entrypoints."""

    def close(self) -> None: ...

    def __enter__(self) -> OperationalStore: ...

    def __exit__(self, exc_type, exc, tb) -> None: ...

    def create_run(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        status: str = "queued",
        input_data: Any = None,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def list_runs(self) -> list[RunRecord]: ...

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        output_data: Any = _UNSET,
        metadata: Mapping[str, Any] | None | object = _UNSET,
        error: Any = _UNSET,
        updated_at: str | None = None,
    ) -> RunRecord: ...

    def create_node_execution(
        self,
        *,
        run_id: str,
        node_name: str,
        status: str = "started",
        input_data: Any = None,
        execution_id: str | None = None,
        created_at: str | None = None,
    ) -> NodeExecutionRecord: ...

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None: ...

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]: ...

    def update_node_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        output_data: Any = _UNSET,
        error: Any = _UNSET,
        updated_at: str | None = None,
    ) -> NodeExecutionRecord: ...

    def resolve_asset(
        self,
        *,
        asset_id: str | None,
        media_fingerprint: str | None,
        first_seen_run_id: str,
        source_language: str,
        target_language: str,
    ) -> AssetRecord: ...

    def get_asset(self, media_key: str) -> AssetRecord | None: ...

    def save_asset_record(self, asset: AssetRecord) -> AssetRecord: ...

    def get_asset_context(self, media_key: str) -> AssetContext | None: ...

    def list_asset_contexts(self) -> list[AssetContext]: ...

    def save_asset_context(self, asset_context: AssetContext) -> AssetContext: ...

    def save_asset_relation(self, relation: AssetRelation) -> AssetRelation: ...

    def list_asset_relations(self, media_key: str) -> list[AssetRelation]: ...

    def put_memory_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool: ...

    def get_memory_entry(self, memory_id: str) -> MemoryEntry | None: ...

    def list_memory_entries(self) -> list[MemoryEntry]: ...

    def search_memory_entries(
        self,
        query: MemoryQuery,
        *,
        limit: int,
    ) -> list[tuple[MemoryEntry, float]]: ...

    def backfill_memory_embeddings(self, *, limit: int | None = None) -> int: ...

    def save_memory_evidence_event(self, event: MemoryEvidenceEvent) -> MemoryEvidenceEvent: ...

    def upsert_historical_run_link(self, link: HistoricalRunLink) -> None: ...

    def list_historical_run_links(
        self,
        media_key: str,
        *,
        exclude_run_id: str | None = None,
    ) -> list[HistoricalRunLink]: ...

    def save_prompt_evolution_proposal(self, proposal: PromptEvolutionProposal) -> None: ...

    def list_prompt_evolution_proposals(
        self,
        *,
        status: str | None = None,
        prompt_family: str | None = None,
        target_model_id: str | None = None,
        target_language: str | None = None,
        source_language: str | None = None,
        prompt_variant_id: str | None = None,
        base_prompt_version: str | None = None,
        scope_kind: str | None = None,
        scope_key: str | None = None,
        media_key: str | None = None,
    ) -> list[PromptEvolutionProposal]: ...

    def save_transcript_provider_quality_stats(
        self, stats: TranscriptProviderQualityStats
    ) -> None: ...

    def get_transcript_provider_quality_stats(
        self,
        *,
        provider_id: str,
        source_language: str,
        target_language: str,
    ) -> TranscriptProviderQualityStats | None: ...

    def save_human_review_resolution(self, resolution: HumanReviewResolutionRecord) -> None: ...

    def get_human_review_resolution(self, run_id: str) -> HumanReviewResolutionRecord | None: ...

    def list_human_review_resolutions(
        self,
        *,
        resolution_kind: str | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        transcript_provider_id: str | None = None,
        model_id: str | None = None,
        prompt_variant_id: str | None = None,
        prompt_version: str | None = None,
        combo_key: str | None = None,
    ) -> list[HumanReviewResolutionRecord]: ...

    def save_translation_feedback_stats(self, stats: TranslationFeedbackStats) -> None: ...

    def get_translation_feedback_stats(self, combo_key: str) -> TranslationFeedbackStats | None: ...


class PostgresOperationalStore(PostgresRunStore):
    """Postgres operational store extended with durable workflow entities."""

    def __enter__(self) -> PostgresOperationalStore:
        return self

    def save_transcript_candidate(
        self,
        candidate: TranscriptCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        payload = candidate.model_dump(mode="json")
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO transcript_candidates (
                    candidate_id, job_id, candidate_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (candidate_id) DO UPDATE SET
                    job_id = EXCLUDED.job_id,
                    candidate_json = EXCLUDED.candidate_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    candidate.candidate_id,
                    storage_job_id or candidate.job_id,
                    _encode_json(payload),
                    now,
                    now,
                ),
            )

    def list_transcript_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranscriptCandidate]:
        rows = self._conn.execute(
            """
            SELECT candidate_json
            FROM transcript_candidates
            WHERE job_id = %s
            ORDER BY candidate_id ASC
            """,
            (storage_job_id or job_id,),
        ).fetchall()
        return [
            TranscriptCandidate.model_validate(_decode_db_json(row["candidate_json"]))
            for row in rows
        ]

    def save_transcript_decision(
        self,
        decision: FinalTranscriptDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._save_singleton_model(
            table="transcript_decisions",
            job_id=storage_job_id or decision.job_id,
            payload=decision.model_dump(mode="json"),
        )

    def get_transcript_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranscriptDecision | None:
        payload = self._get_singleton_model(
            table="transcript_decisions",
            job_id=storage_job_id or job_id,
        )
        if payload is None:
            return None
        return FinalTranscriptDecision.model_validate(payload)

    def save_translation_candidate(
        self,
        candidate: TranslationCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        payload = candidate.model_dump(mode="json")
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO translation_candidates (
                    candidate_id, job_id, candidate_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (candidate_id) DO UPDATE SET
                    job_id = EXCLUDED.job_id,
                    candidate_json = EXCLUDED.candidate_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    candidate.candidate_id,
                    storage_job_id or candidate.job_id,
                    _encode_json(payload),
                    now,
                    now,
                ),
            )

    def list_translation_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranslationCandidate]:
        rows = self._conn.execute(
            """
            SELECT candidate_json
            FROM translation_candidates
            WHERE job_id = %s
            ORDER BY candidate_id ASC
            """,
            (storage_job_id or job_id,),
        ).fetchall()
        return [
            TranslationCandidate.model_validate(_decode_db_json(row["candidate_json"]))
            for row in rows
        ]

    def save_translation_decision(
        self,
        decision: FinalTranslationDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._save_singleton_model(
            table="translation_decisions",
            job_id=storage_job_id or decision.job_id,
            payload=decision.model_dump(mode="json"),
        )

    def get_translation_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranslationDecision | None:
        payload = self._get_singleton_model(
            table="translation_decisions",
            job_id=storage_job_id or job_id,
        )
        if payload is None:
            return None
        return FinalTranslationDecision.model_validate(payload)

    def save_investigation(
        self,
        *,
        job_id: str,
        stage: ReviewStage,
        payload: dict[str, object],
        storage_job_id: str | None = None,
    ) -> None:
        self._save_stage_payload(
            table="investigations",
            job_id=storage_job_id or job_id,
            stage=stage,
            payload=payload,
        )

    def get_investigation(
        self,
        *,
        job_id: str,
        stage: ReviewStage,
        storage_job_id: str | None = None,
    ) -> dict[str, object] | None:
        payload = self._get_stage_payload(
            table="investigations",
            job_id=storage_job_id or job_id,
            stage=stage,
        )
        if payload is None:
            return None
        return cast(dict[str, object], payload)

    def save_batch(
        self,
        batch: MemoryWriteBatch,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        payload = batch.model_dump(mode="json")
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO memory_batches (
                    batch_id, job_id, batch_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (batch_id) DO UPDATE SET
                    job_id = EXCLUDED.job_id,
                    batch_json = EXCLUDED.batch_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (batch.batch_id, storage_job_id or batch.job_id, _encode_json(payload), now, now),
            )

    def get_batch(self, batch_id: str) -> MemoryWriteBatch | None:
        row = self._conn.execute(
            "SELECT batch_json FROM memory_batches WHERE batch_id = %s",
            (batch_id,),
        ).fetchone()
        if row is None:
            return None
        return MemoryWriteBatch.model_validate(_decode_db_json(row["batch_json"]))

    def list_batches(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[MemoryWriteBatch]:
        rows = self._conn.execute(
            """
            SELECT batch_json
            FROM memory_batches
            WHERE job_id = %s
            ORDER BY batch_id ASC
            """,
            (storage_job_id or job_id,),
        ).fetchall()
        return [MemoryWriteBatch.model_validate(_decode_db_json(row["batch_json"])) for row in rows]

    def save_prompt_evolution_proposal(self, proposal: PromptEvolutionProposal) -> None:
        payload = proposal.model_dump(mode="json")
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO prompt_evolution_proposals (
                    proposal_id,
                    proposal_json,
                    status,
                    prompt_family,
                    target_model_id,
                    target_language,
                    source_language,
                    prompt_variant_id,
                    base_prompt_version,
                    scope_kind,
                    scope_key,
                    media_key,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (proposal_id) DO UPDATE SET
                    proposal_json = EXCLUDED.proposal_json,
                    status = EXCLUDED.status,
                    prompt_family = EXCLUDED.prompt_family,
                    target_model_id = EXCLUDED.target_model_id,
                    target_language = EXCLUDED.target_language,
                    source_language = EXCLUDED.source_language,
                    prompt_variant_id = EXCLUDED.prompt_variant_id,
                    base_prompt_version = EXCLUDED.base_prompt_version,
                    scope_kind = EXCLUDED.scope_kind,
                    scope_key = EXCLUDED.scope_key,
                    media_key = EXCLUDED.media_key,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    proposal.proposal_id,
                    _encode_json(payload),
                    proposal.status,
                    proposal.prompt_family,
                    proposal.target_model_id,
                    _proposal_metadata_value(proposal, "target_language"),
                    _proposal_metadata_value(proposal, "source_language"),
                    _proposal_compatibility_value(proposal, "prompt_variant_id"),
                    _proposal_compatibility_value(proposal, "base_prompt_version"),
                    _proposal_compatibility_value(proposal, "scope_kind"),
                    _proposal_compatibility_value(proposal, "scope_key"),
                    _proposal_metadata_value(proposal, "media_key"),
                    now,
                    now,
                ),
            )

    def list_prompt_evolution_proposals(
        self,
        *,
        status: str | None = None,
        prompt_family: str | None = None,
        target_model_id: str | None = None,
        target_language: str | None = None,
        source_language: str | None = None,
        prompt_variant_id: str | None = None,
        base_prompt_version: str | None = None,
        scope_kind: str | None = None,
        scope_key: str | None = None,
        media_key: str | None = None,
    ) -> list[PromptEvolutionProposal]:
        status = _normalized_proposal_status(status)
        query = ["SELECT proposal_json FROM prompt_evolution_proposals WHERE 1=1"]
        params: list[Any] = []
        for column, value in (
            ("status", status),
            ("prompt_family", prompt_family),
            ("target_model_id", target_model_id),
            ("target_language", target_language),
            ("source_language", source_language),
            ("prompt_variant_id", prompt_variant_id),
            ("base_prompt_version", base_prompt_version),
            ("scope_kind", scope_kind),
            ("scope_key", scope_key),
            ("media_key", media_key),
        ):
            if value is None:
                continue
            query.append(f"AND {column} = %s")
            params.append(value)
        query.append("ORDER BY proposal_id ASC")
        rows = self._conn.execute(" ".join(query), tuple(params)).fetchall()
        return [
            PromptEvolutionProposal.model_validate(_decode_db_json(row["proposal_json"]))
            for row in rows
        ]

    def save_transcript_provider_quality_stats(self, stats: TranscriptProviderQualityStats) -> None:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                _POSTGRES_PROVIDER_STATS_UPSERT_SQL,
                (
                    stats.provider_id,
                    stats.source_language,
                    stats.target_language,
                    _encode_json(stats.model_dump(mode="json")),
                    now,
                    now,
                ),
            )

    def get_transcript_provider_quality_stats(
        self,
        *,
        provider_id: str,
        source_language: str,
        target_language: str,
    ) -> TranscriptProviderQualityStats | None:
        row = self._conn.execute(
            _POSTGRES_PROVIDER_STATS_SELECT_SQL,
            (provider_id, source_language, target_language),
        ).fetchone()
        if row is None:
            return None
        return TranscriptProviderQualityStats.model_validate(_decode_db_json(row["stats_json"]))

    def save_human_review_resolution(self, resolution: HumanReviewResolutionRecord) -> None:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                _POSTGRES_HUMAN_RESOLUTION_UPSERT_SQL,
                (
                    resolution.run_id,
                    resolution.job_id,
                    _encode_json(resolution.model_dump(mode="json")),
                    resolution.resolution_kind,
                    resolution.source_language,
                    resolution.target_language,
                    resolution.transcript_provider_id,
                    resolution.model_id,
                    resolution.prompt_variant_id,
                    resolution.prompt_version,
                    resolution.combo_key,
                    resolution.resolved_at,
                    now,
                    now,
                ),
            )

    def get_human_review_resolution(self, run_id: str) -> HumanReviewResolutionRecord | None:
        row = self._conn.execute(_POSTGRES_HUMAN_RESOLUTION_SELECT_SQL, (run_id,)).fetchone()
        if row is None:
            return None
        return HumanReviewResolutionRecord.model_validate(_decode_db_json(row["resolution_json"]))

    def list_human_review_resolutions(
        self,
        *,
        resolution_kind: str | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        transcript_provider_id: str | None = None,
        model_id: str | None = None,
        prompt_variant_id: str | None = None,
        prompt_version: str | None = None,
        combo_key: str | None = None,
    ) -> list[HumanReviewResolutionRecord]:
        query = ["SELECT resolution_json FROM human_review_resolutions WHERE 1=1"]
        params: list[Any] = []
        for column, value in (
            ("resolution_kind", resolution_kind),
            ("source_language", source_language),
            ("target_language", target_language),
            ("transcript_provider_id", transcript_provider_id),
            ("model_id", model_id),
            ("prompt_variant_id", prompt_variant_id),
            ("prompt_version", prompt_version),
            ("combo_key", combo_key),
        ):
            if value is None:
                continue
            query.append(f"AND {column} = %s")
            params.append(value)
        query.append("ORDER BY resolved_at ASC, run_id ASC")
        rows = self._conn.execute(" ".join(query), tuple(params)).fetchall()
        return [
            HumanReviewResolutionRecord.model_validate(_decode_db_json(row["resolution_json"]))
            for row in rows
        ]

    def save_translation_feedback_stats(self, stats: TranslationFeedbackStats) -> None:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                _POSTGRES_TRANSLATION_FEEDBACK_UPSERT_SQL,
                (
                    stats.combo_key,
                    _encode_json(stats.model_dump(mode="json")),
                    stats.source_language,
                    stats.target_language,
                    stats.transcript_provider_id,
                    stats.model_id,
                    stats.prompt_variant_id,
                    stats.prompt_version,
                    now,
                    now,
                ),
            )

    def get_translation_feedback_stats(self, combo_key: str) -> TranslationFeedbackStats | None:
        row = self._conn.execute(_POSTGRES_TRANSLATION_FEEDBACK_SELECT_SQL, (combo_key,)).fetchone()
        if row is None:
            return None
        return TranslationFeedbackStats.model_validate(_decode_db_json(row["stats_json"]))

    def resolve_asset(
        self,
        *,
        asset_id: str | None,
        media_fingerprint: str | None,
        first_seen_run_id: str,
        source_language: str,
        target_language: str,
    ) -> AssetRecord:
        return _resolve_asset_record(
            resolver=_PostgresAssetResolver(self._conn),
            asset_id=asset_id,
            media_fingerprint=media_fingerprint,
            first_seen_run_id=first_seen_run_id,
            source_language=source_language,
            target_language=target_language,
        )

    def get_asset(self, media_key: str) -> AssetRecord | None:
        row = self._conn.execute(_POSTGRES_ASSET_SELECT_SQL["media_key"], (media_key,)).fetchone()
        if row is None:
            return None
        return AssetRecord.model_validate(_decode_db_json(row["asset_json"]))

    def save_asset_record(self, asset: AssetRecord) -> AssetRecord:
        return _PostgresAssetResolver(self._conn).save(asset)

    def get_asset_context(self, media_key: str) -> AssetContext | None:
        row = self._conn.execute(_POSTGRES_ASSET_CONTEXT_SELECT_SQL, (media_key,)).fetchone()
        if row is None:
            return None
        return AssetContext.model_validate(_decode_db_json(row["context_json"]))

    def list_asset_contexts(self) -> list[AssetContext]:
        rows = self._conn.execute(_POSTGRES_ASSET_CONTEXT_LIST_SQL).fetchall()
        return [AssetContext.model_validate(_decode_db_json(row["context_json"])) for row in rows]

    def save_asset_context(self, asset_context: AssetContext) -> AssetContext:
        now = _utc_now()
        payload = asset_context.model_copy(update={"updated_at": datetime.now(UTC)}).model_dump(
            mode="json"
        )
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO asset_context_records (
                    media_key,
                    context_json,
                    canonical_title,
                    series_id,
                    franchise_id,
                    channel_id,
                    content_type,
                    style_profile_id,
                    metadata_confidence,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (media_key) DO UPDATE SET
                    context_json = EXCLUDED.context_json,
                    canonical_title = EXCLUDED.canonical_title,
                    series_id = EXCLUDED.series_id,
                    franchise_id = EXCLUDED.franchise_id,
                    channel_id = EXCLUDED.channel_id,
                    content_type = EXCLUDED.content_type,
                    style_profile_id = EXCLUDED.style_profile_id,
                    metadata_confidence = EXCLUDED.metadata_confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    asset_context.media_key,
                    _encode_json(payload),
                    asset_context.canonical_title,
                    asset_context.series_id,
                    asset_context.franchise_id,
                    asset_context.channel_id,
                    asset_context.content_type,
                    asset_context.style_profile_id,
                    asset_context.metadata_confidence,
                    now,
                    now,
                ),
            )
        saved = self.get_asset_context(asset_context.media_key)
        if saved is None:
            raise RuntimeError(f"asset context {asset_context.media_key} was not persisted")
        return saved

    def save_asset_relation(self, relation: AssetRelation) -> AssetRelation:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO asset_relations (
                    relation_id,
                    src_media_key,
                    dst_media_key,
                    relation_kind,
                    confidence,
                    relation_json,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (relation_id) DO UPDATE SET
                    src_media_key = EXCLUDED.src_media_key,
                    dst_media_key = EXCLUDED.dst_media_key,
                    relation_kind = EXCLUDED.relation_kind,
                    confidence = EXCLUDED.confidence,
                    relation_json = EXCLUDED.relation_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    relation.relation_id,
                    relation.src_media_key,
                    relation.dst_media_key,
                    relation.relation_kind,
                    relation.confidence,
                    _encode_json(relation.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
        return relation

    def list_asset_relations(self, media_key: str) -> list[AssetRelation]:
        rows = self._conn.execute(
            _POSTGRES_ASSET_RELATION_LIST_SQL,
            (media_key, media_key),
        ).fetchall()
        return [AssetRelation.model_validate(_decode_db_json(row["relation_json"])) for row in rows]

    def put_memory_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool:
        if dedupe_key is not None:
            existing = self._conn.execute(
                _POSTGRES_MEMORY_DEDUPE_SELECT_SQL,
                (dedupe_key,),
            ).fetchone()
            if existing is not None:
                return False
        embedding_model_id, embedding, search_document = embedding_metadata_for_entry(entry)
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO memory_entries (
                    memory_id,
                    entry_json,
                    dedupe_key,
                    scope_kind,
                    scope_key,
                    series_id,
                    franchise_id,
                    content_type,
                    style_profile_id,
                    promotion_status,
                    lifecycle_status,
                    validation_status,
                    typed_metadata,
                    search_document,
                    embedding_model_id,
                    embedding_json,
                    embedding_updated_at,
                    created_at,
                    updated_at
                ) VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    to_tsvector('simple', %s),
                    %s,
                    %s::jsonb,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (memory_id) DO UPDATE SET
                    entry_json = EXCLUDED.entry_json,
                    dedupe_key = EXCLUDED.dedupe_key,
                    scope_kind = EXCLUDED.scope_kind,
                    scope_key = EXCLUDED.scope_key,
                    series_id = EXCLUDED.series_id,
                    franchise_id = EXCLUDED.franchise_id,
                    content_type = EXCLUDED.content_type,
                    style_profile_id = EXCLUDED.style_profile_id,
                    promotion_status = EXCLUDED.promotion_status,
                    lifecycle_status = EXCLUDED.lifecycle_status,
                    validation_status = EXCLUDED.validation_status,
                    typed_metadata = EXCLUDED.typed_metadata,
                    search_document = EXCLUDED.search_document,
                    embedding_model_id = EXCLUDED.embedding_model_id,
                    embedding_json = EXCLUDED.embedding_json,
                    embedding_updated_at = EXCLUDED.embedding_updated_at,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    entry.memory_id,
                    _encode_json(entry.model_dump(mode="json")),
                    dedupe_key,
                    entry.scope_kind,
                    entry.scope_key,
                    entry.series_id,
                    entry.franchise_id,
                    entry.content_type,
                    entry.style_profile_id,
                    entry.promotion_status,
                    entry.lifecycle_status,
                    entry.validation_status,
                    _encode_json(
                        {
                            **entry.typed_metadata,
                            "embedding": embedding,
                            "embedding_model_id": embedding_model_id,
                        }
                    ),
                    search_document,
                    embedding_model_id,
                    serialize_embedding(embedding),
                    _isoformat_timestamp(entry.updated_at or datetime.now(UTC)),
                    _isoformat_timestamp(entry.updated_at or datetime.now(UTC)),
                    _isoformat_timestamp(entry.updated_at or datetime.now(UTC)),
                ),
            )
        return True

    def get_memory_entry(self, memory_id: str) -> MemoryEntry | None:
        row = self._conn.execute(_POSTGRES_MEMORY_ENTRY_SELECT_SQL, (memory_id,)).fetchone()
        if row is None:
            return None
        return MemoryEntry.model_validate(_decode_db_json(row["entry_json"]))

    def list_memory_entries(self) -> list[MemoryEntry]:
        rows = self._conn.execute(_POSTGRES_MEMORY_ENTRY_LIST_SQL).fetchall()
        return [MemoryEntry.model_validate(_decode_db_json(row["entry_json"])) for row in rows]

    def search_memory_entries(
        self,
        query: MemoryQuery,
        *,
        limit: int,
    ) -> list[tuple[MemoryEntry, float]]:
        search_document = _sql_search_document(query)
        channel_id = query.asset_context.channel_id if query.asset_context is not None else None
        channel_filter = json.dumps({"channel_id": channel_id}) if channel_id is not None else None
        params: list[object] = [
            _memory_scope_payload(query),
            search_document,
            search_document,
            query.series_id,
            query.series_id,
            query.franchise_id,
            query.franchise_id,
            query.content_type,
            query.content_type,
            query.style_profile_id,
            query.style_profile_id,
            channel_filter,
            channel_filter,
            max(limit, 1),
        ]
        sql = """
            WITH candidate_scopes AS (
                SELECT scope_kind, scope_key
                FROM jsonb_to_recordset(%s::jsonb) AS scope(scope_kind text, scope_key text)
            )
            SELECT
                entry_json,
                embedding_json,
                LEAST(
                    1.0,
                    COALESCE(ts_rank_cd(search_document, websearch_to_tsquery('simple', %s)), 0.0)
                    + GREATEST(similarity(COALESCE(entry_json->>'content', ''), %s), 0.0) * 0.45
                    + CASE WHEN %s::text IS NOT NULL AND series_id = %s::text THEN 0.15 ELSE 0 END
                    + CASE
                        WHEN %s::text IS NOT NULL AND franchise_id = %s::text THEN 0.12
                        ELSE 0
                    END
                    + CASE
                        WHEN %s::text IS NOT NULL AND content_type = %s::text THEN 0.08
                        ELSE 0
                    END
                    + CASE
                        WHEN %s::text IS NOT NULL AND style_profile_id = %s::text THEN 0.08
                        ELSE 0
                    END
                    + CASE
                        WHEN %s::jsonb IS NOT NULL AND typed_metadata @> %s::jsonb THEN 0.08
                        ELSE 0
                    END
                ) AS recall_score
            FROM memory_entries
            WHERE lifecycle_status = 'active'
              AND EXISTS (
                    SELECT 1
                    FROM candidate_scopes
                    WHERE candidate_scopes.scope_kind = memory_entries.scope_kind
                      AND candidate_scopes.scope_key = memory_entries.scope_key
              )
            ORDER BY recall_score DESC, updated_at DESC, memory_id ASC
            LIMIT %s
        """
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [
            (
                _entry_with_runtime_embedding(
                    MemoryEntry.model_validate(_decode_db_json(row["entry_json"])),
                    row["embedding_json"],
                ),
                float(row["recall_score"] or 0.0),
            )
            for row in rows
        ]

    def backfill_memory_embeddings(self, *, limit: int | None = None) -> int:
        rows = self._conn.execute(_POSTGRES_MEMORY_ENTRY_INDEX_SELECT_SQL).fetchall()
        updated = 0
        for row in rows[:limit] if limit is not None else rows:
            entry = MemoryEntry.model_validate(_decode_db_json(row["entry_json"]))
            embedding_model_id, embedding, search_document = embedding_metadata_for_entry(entry)
            with self._conn.transaction():
                self._conn.execute(
                    """
                    UPDATE memory_entries
                    SET typed_metadata = %s,
                        search_document = to_tsvector('simple', %s),
                        embedding_model_id = %s,
                        embedding_json = %s::jsonb,
                        embedding_updated_at = %s,
                        updated_at = %s
                    WHERE memory_id = %s
                    """,
                    (
                        _encode_json(
                            {
                                **entry.typed_metadata,
                                "embedding": embedding,
                                "embedding_model_id": embedding_model_id,
                            }
                        ),
                        search_document,
                        embedding_model_id,
                        serialize_embedding(embedding),
                        _isoformat_timestamp(entry.updated_at or datetime.now(UTC)),
                        _isoformat_timestamp(entry.updated_at or datetime.now(UTC)),
                        entry.memory_id,
                    ),
                )
            updated += 1
        return updated

    def save_memory_evidence_event(self, event: MemoryEvidenceEvent) -> MemoryEvidenceEvent:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO memory_evidence_events (
                    event_id,
                    memory_id,
                    event_kind,
                    run_id,
                    job_id,
                    media_key,
                    stage,
                    source_ref,
                    event_json,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET
                    memory_id = EXCLUDED.memory_id,
                    event_kind = EXCLUDED.event_kind,
                    run_id = EXCLUDED.run_id,
                    job_id = EXCLUDED.job_id,
                    media_key = EXCLUDED.media_key,
                    stage = EXCLUDED.stage,
                    source_ref = EXCLUDED.source_ref,
                    event_json = EXCLUDED.event_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    event.event_id,
                    event.memory_id,
                    event.event_kind,
                    event.run_id,
                    event.job_id,
                    event.media_key,
                    event.stage,
                    event.source_ref,
                    _encode_json(event.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
        return event

    def upsert_historical_run_link(self, link: HistoricalRunLink) -> None:
        now = _utc_now()
        existing_row = self._conn.execute(
            "SELECT link_json FROM historical_run_links WHERE run_id = %s",
            (link.run_id,),
        ).fetchone()
        if existing_row is not None:
            existing = HistoricalRunLink.model_validate(_decode_db_json(existing_row["link_json"]))
            link = _merge_historical_run_link(existing, link)
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO historical_run_links (
                    run_id, media_key, link_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    media_key = EXCLUDED.media_key,
                    link_json = EXCLUDED.link_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    link.run_id,
                    link.media_key,
                    _encode_json(link.model_dump(mode="json")),
                    now,
                    now,
                ),
            )

    def list_historical_run_links(
        self,
        media_key: str,
        *,
        exclude_run_id: str | None = None,
    ) -> list[HistoricalRunLink]:
        if exclude_run_id is None:
            rows = self._conn.execute(
                """
                SELECT link_json
                FROM historical_run_links
                WHERE media_key = %s
                ORDER BY created_at ASC, run_id ASC
                """,
                (media_key,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT link_json
                FROM historical_run_links
                WHERE media_key = %s AND run_id <> %s
                ORDER BY created_at ASC, run_id ASC
                """,
                (media_key, exclude_run_id),
            ).fetchall()
        return [HistoricalRunLink.model_validate(_decode_db_json(row["link_json"])) for row in rows]

    def _save_singleton_model(
        self,
        *,
        table: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> None:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                _POSTGRES_SINGLETON_UPSERT_SQL[table],
                (job_id, _encode_json(payload), now, now),
            )

    def _get_singleton_model(
        self,
        *,
        table: str,
        job_id: str,
    ) -> Any | None:
        row = self._conn.execute(
            _POSTGRES_SINGLETON_SELECT_SQL[table],
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_db_json(row["decision_json"])

    def _save_stage_payload(
        self,
        *,
        table: str,
        job_id: str,
        stage: ReviewStage,
        payload: dict[str, object],
    ) -> None:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                _POSTGRES_STAGE_UPSERT_SQL[table],
                (job_id, stage, _encode_json(payload), now, now),
            )

    def _get_stage_payload(
        self,
        *,
        table: str,
        job_id: str,
        stage: ReviewStage,
    ) -> Any | None:
        row = self._conn.execute(
            _POSTGRES_STAGE_SELECT_SQL[table],
            (job_id, stage),
        ).fetchone()
        if row is None:
            return None
        return _decode_db_json(row["payload_json"])

    def _get_asset_by(self, field_name: str, value: str | None) -> AssetRecord | None:
        if value is None:
            return None
        row = self._conn.execute(
            _POSTGRES_ASSET_SELECT_SQL[field_name],
            (value,),
        ).fetchone()
        if row is None:
            return None
        return AssetRecord.model_validate(_decode_db_json(row["asset_json"]))

    def _save_asset_record(self, record: AssetRecord) -> AssetRecord:
        payload = record.model_dump(mode="json")
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO asset_records (
                    media_key,
                    asset_id,
                    media_fingerprint,
                    asset_json,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (media_key) DO UPDATE SET
                    asset_id = EXCLUDED.asset_id,
                    media_fingerprint = EXCLUDED.media_fingerprint,
                    asset_json = EXCLUDED.asset_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    record.media_key,
                    record.asset_id,
                    record.media_fingerprint,
                    _encode_json(payload),
                    now,
                    now,
                ),
            )
        return record


class SQLiteOperationalStore(AbstractContextManager["SQLiteOperationalStore"]):
    """SQLite operational store used by the local-first reference runtime."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._bootstrap_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteOperationalStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def create_run(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        status: str = "queued",
        input_data: Any = None,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> RunRecord:
        from uuid import uuid4

        run_id = run_id or f"run_{uuid4().hex}"
        created_at = created_at or _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, tenant_id, project_id, status, created_at, updated_at,
                    input_data_json, output_data_json, metadata_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    tenant_id,
                    project_id,
                    status,
                    created_at,
                    created_at,
                    _encode_sqlite_json(input_data),
                    None,
                    _encode_sqlite_json(metadata),
                    None,
                ),
            )
        return _require_run(self.get_run(run_id), run_id)

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return _sqlite_row_to_run(row) if row else None

    def list_runs(self) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC"
        ).fetchall()
        return [_sqlite_row_to_run(row) for row in rows]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        output_data: Any = _UNSET,
        metadata: Mapping[str, Any] | None | object = _UNSET,
        error: Any = _UNSET,
        updated_at: str | None = None,
    ) -> RunRecord:
        current = self.get_run(run_id)
        if current is None:
            raise KeyError(run_id)
        updated_at = updated_at or _utc_now()
        new_output = current.output_data if output_data is _UNSET else output_data
        new_metadata = current.metadata if metadata is _UNSET else metadata
        new_error = current.error if error is _UNSET else error
        with self._conn:
            self._conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    updated_at = ?,
                    output_data_json = ?,
                    metadata_json = ?,
                    error_json = ?
                WHERE run_id = ?
                """,
                (
                    current.status if status is None else status,
                    updated_at,
                    _encode_sqlite_json(new_output),
                    _encode_sqlite_json(new_metadata),
                    _encode_sqlite_json(new_error),
                    run_id,
                ),
            )
        return _require_run(self.get_run(run_id), run_id)

    def create_node_execution(
        self,
        *,
        run_id: str,
        node_name: str,
        status: str = "started",
        input_data: Any = None,
        execution_id: str | None = None,
        created_at: str | None = None,
    ) -> NodeExecutionRecord:
        from uuid import uuid4

        execution_id = execution_id or f"exec_{uuid4().hex}"
        created_at = created_at or _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO node_executions (
                    execution_id, run_id, node_name, status, created_at, updated_at,
                    input_data_json, output_data_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    run_id,
                    node_name,
                    status,
                    created_at,
                    created_at,
                    _encode_sqlite_json(input_data),
                    None,
                    None,
                ),
            )
        return _require_node_execution(self.get_node_execution(execution_id), execution_id)

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM node_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        return _sqlite_row_to_node_execution(row) if row else None

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM node_executions
            WHERE run_id = ?
            ORDER BY created_at ASC, execution_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [_sqlite_row_to_node_execution(row) for row in rows]

    def update_node_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        output_data: Any = _UNSET,
        error: Any = _UNSET,
        updated_at: str | None = None,
    ) -> NodeExecutionRecord:
        current = self.get_node_execution(execution_id)
        if current is None:
            raise KeyError(execution_id)
        updated_at = updated_at or _utc_now()
        new_output = current.output_data if output_data is _UNSET else output_data
        new_error = current.error if error is _UNSET else error
        with self._conn:
            self._conn.execute(
                """
                UPDATE node_executions
                SET status = ?, updated_at = ?, output_data_json = ?, error_json = ?
                WHERE execution_id = ?
                """,
                (
                    current.status if status is None else status,
                    updated_at,
                    _encode_sqlite_json(new_output),
                    _encode_sqlite_json(new_error),
                    execution_id,
                ),
            )
        return _require_node_execution(self.get_node_execution(execution_id), execution_id)

    def save_transcript_candidate(
        self,
        candidate: TranscriptCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._save_candidate(
            table="transcript_candidates",
            candidate_id=candidate.candidate_id,
            job_id=storage_job_id or candidate.job_id,
            payload=candidate.model_dump(mode="json"),
        )

    def list_transcript_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranscriptCandidate]:
        return [
            TranscriptCandidate.model_validate(payload)
            for payload in self._list_payloads(
                table="transcript_candidates",
                job_id=storage_job_id or job_id,
                column="candidate_json",
            )
        ]

    def save_transcript_decision(
        self,
        decision: FinalTranscriptDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._save_singleton_model(
            table="transcript_decisions",
            job_id=storage_job_id or decision.job_id,
            payload=decision.model_dump(mode="json"),
        )

    def get_transcript_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranscriptDecision | None:
        payload = self._get_singleton_payload(
            table="transcript_decisions",
            job_id=storage_job_id or job_id,
        )
        if payload is None:
            return None
        return FinalTranscriptDecision.model_validate(payload)

    def save_translation_candidate(
        self,
        candidate: TranslationCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._save_candidate(
            table="translation_candidates",
            candidate_id=candidate.candidate_id,
            job_id=storage_job_id or candidate.job_id,
            payload=candidate.model_dump(mode="json"),
        )

    def list_translation_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranslationCandidate]:
        return [
            TranslationCandidate.model_validate(payload)
            for payload in self._list_payloads(
                table="translation_candidates",
                job_id=storage_job_id or job_id,
                column="candidate_json",
            )
        ]

    def save_translation_decision(
        self,
        decision: FinalTranslationDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._save_singleton_model(
            table="translation_decisions",
            job_id=storage_job_id or decision.job_id,
            payload=decision.model_dump(mode="json"),
        )

    def get_translation_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranslationDecision | None:
        payload = self._get_singleton_payload(
            table="translation_decisions",
            job_id=storage_job_id or job_id,
        )
        if payload is None:
            return None
        return FinalTranslationDecision.model_validate(payload)

    def save_investigation(
        self,
        *,
        job_id: str,
        stage: ReviewStage,
        payload: dict[str, object],
        storage_job_id: str | None = None,
    ) -> None:
        self._save_stage_payload(
            table="investigations",
            job_id=storage_job_id or job_id,
            stage=stage,
            payload=payload,
        )

    def get_investigation(
        self,
        *,
        job_id: str,
        stage: ReviewStage,
        storage_job_id: str | None = None,
    ) -> dict[str, object] | None:
        payload = self._get_stage_payload(
            table="investigations",
            job_id=storage_job_id or job_id,
            stage=stage,
        )
        if payload is None:
            return None
        return cast(dict[str, object], payload)

    def save_batch(
        self,
        batch: MemoryWriteBatch,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        payload = batch.model_dump(mode="json")
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_batches (
                    batch_id, job_id, batch_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    job_id = excluded.job_id,
                    batch_json = excluded.batch_json,
                    updated_at = excluded.updated_at
                """,
                (
                    batch.batch_id,
                    storage_job_id or batch.job_id,
                    _encode_sqlite_json(payload),
                    now,
                    now,
                ),
            )

    def get_batch(self, batch_id: str) -> MemoryWriteBatch | None:
        row = self._conn.execute(
            "SELECT batch_json FROM memory_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            return None
        return MemoryWriteBatch.model_validate(_decode_sqlite_json(row["batch_json"]))

    def list_batches(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[MemoryWriteBatch]:
        return [
            MemoryWriteBatch.model_validate(payload)
            for payload in self._list_payloads(
                table="memory_batches",
                job_id=storage_job_id or job_id,
                column="batch_json",
            )
        ]

    def resolve_asset(
        self,
        *,
        asset_id: str | None,
        media_fingerprint: str | None,
        first_seen_run_id: str,
        source_language: str,
        target_language: str,
    ) -> AssetRecord:
        return _resolve_asset_record(
            resolver=_SQLiteAssetResolver(self._conn),
            asset_id=asset_id,
            media_fingerprint=media_fingerprint,
            first_seen_run_id=first_seen_run_id,
            source_language=source_language,
            target_language=target_language,
        )

    def get_asset(self, media_key: str) -> AssetRecord | None:
        row = self._conn.execute(_SQLITE_ASSET_SELECT_SQL["media_key"], (media_key,)).fetchone()
        if row is None:
            return None
        return AssetRecord.model_validate(_decode_sqlite_json(row["asset_json"]))

    def save_asset_record(self, asset: AssetRecord) -> AssetRecord:
        return _SQLiteAssetResolver(self._conn).save(asset)

    def get_asset_context(self, media_key: str) -> AssetContext | None:
        row = self._conn.execute(_SQLITE_ASSET_CONTEXT_SELECT_SQL, (media_key,)).fetchone()
        if row is None:
            return None
        return AssetContext.model_validate(_decode_sqlite_json(row["context_json"]))

    def list_asset_contexts(self) -> list[AssetContext]:
        rows = self._conn.execute(_SQLITE_ASSET_CONTEXT_LIST_SQL).fetchall()
        return [
            AssetContext.model_validate(_decode_sqlite_json(row["context_json"])) for row in rows
        ]

    def save_asset_context(self, asset_context: AssetContext) -> AssetContext:
        now = _utc_now()
        payload = asset_context.model_copy(update={"updated_at": datetime.now(UTC)}).model_dump(
            mode="json"
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO asset_context_records (
                    media_key,
                    context_json,
                    canonical_title,
                    series_id,
                    franchise_id,
                    channel_id,
                    content_type,
                    style_profile_id,
                    metadata_confidence,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(media_key) DO UPDATE SET
                    context_json = excluded.context_json,
                    canonical_title = excluded.canonical_title,
                    series_id = excluded.series_id,
                    franchise_id = excluded.franchise_id,
                    channel_id = excluded.channel_id,
                    content_type = excluded.content_type,
                    style_profile_id = excluded.style_profile_id,
                    metadata_confidence = excluded.metadata_confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    asset_context.media_key,
                    _encode_sqlite_json(payload),
                    asset_context.canonical_title,
                    asset_context.series_id,
                    asset_context.franchise_id,
                    asset_context.channel_id,
                    asset_context.content_type,
                    asset_context.style_profile_id,
                    asset_context.metadata_confidence,
                    now,
                    now,
                ),
            )
        saved = self.get_asset_context(asset_context.media_key)
        if saved is None:
            raise RuntimeError(f"asset context {asset_context.media_key} was not persisted")
        return saved

    def save_asset_relation(self, relation: AssetRelation) -> AssetRelation:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO asset_relations (
                    relation_id,
                    src_media_key,
                    dst_media_key,
                    relation_kind,
                    confidence,
                    relation_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relation_id) DO UPDATE SET
                    src_media_key = excluded.src_media_key,
                    dst_media_key = excluded.dst_media_key,
                    relation_kind = excluded.relation_kind,
                    confidence = excluded.confidence,
                    relation_json = excluded.relation_json,
                    updated_at = excluded.updated_at
                """,
                (
                    relation.relation_id,
                    relation.src_media_key,
                    relation.dst_media_key,
                    relation.relation_kind,
                    relation.confidence,
                    _encode_sqlite_json(relation.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
        return relation

    def list_asset_relations(self, media_key: str) -> list[AssetRelation]:
        rows = self._conn.execute(
            _SQLITE_ASSET_RELATION_LIST_SQL,
            (media_key, media_key),
        ).fetchall()
        return [
            AssetRelation.model_validate(_decode_sqlite_json(row["relation_json"])) for row in rows
        ]

    def put_memory_entry(self, entry: MemoryEntry, *, dedupe_key: str | None = None) -> bool:
        if dedupe_key is not None:
            existing = self._conn.execute(
                _SQLITE_MEMORY_DEDUPE_SELECT_SQL,
                (dedupe_key,),
            ).fetchone()
            if existing is not None:
                return False
        embedding_model_id, embedding, search_document = embedding_metadata_for_entry(entry)
        now = _isoformat_timestamp(entry.updated_at or datetime.now(UTC))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_entries (
                    memory_id,
                    entry_json,
                    dedupe_key,
                    scope_kind,
                    scope_key,
                    series_id,
                    franchise_id,
                    content_type,
                    style_profile_id,
                    promotion_status,
                    lifecycle_status,
                    validation_status,
                    typed_metadata,
                    search_document,
                    embedding_model_id,
                    embedding_json,
                    embedding_updated_at,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    entry_json = excluded.entry_json,
                    dedupe_key = excluded.dedupe_key,
                    scope_kind = excluded.scope_kind,
                    scope_key = excluded.scope_key,
                    series_id = excluded.series_id,
                    franchise_id = excluded.franchise_id,
                    content_type = excluded.content_type,
                    style_profile_id = excluded.style_profile_id,
                    promotion_status = excluded.promotion_status,
                    lifecycle_status = excluded.lifecycle_status,
                    validation_status = excluded.validation_status,
                    typed_metadata = excluded.typed_metadata,
                    search_document = excluded.search_document,
                    embedding_model_id = excluded.embedding_model_id,
                    embedding_json = excluded.embedding_json,
                    embedding_updated_at = excluded.embedding_updated_at,
                    updated_at = excluded.updated_at
                """,
                (
                    entry.memory_id,
                    _encode_sqlite_json(entry.model_dump(mode="json")),
                    dedupe_key,
                    entry.scope_kind,
                    entry.scope_key,
                    entry.series_id,
                    entry.franchise_id,
                    entry.content_type,
                    entry.style_profile_id,
                    entry.promotion_status,
                    entry.lifecycle_status,
                    entry.validation_status,
                    _encode_sqlite_json(
                        {
                            **entry.typed_metadata,
                            "embedding": embedding,
                            "embedding_model_id": embedding_model_id,
                        }
                    ),
                    search_document,
                    embedding_model_id,
                    serialize_embedding(embedding),
                    now,
                    now,
                    now,
                ),
            )
        return True

    def get_memory_entry(self, memory_id: str) -> MemoryEntry | None:
        row = self._conn.execute(_SQLITE_MEMORY_ENTRY_SELECT_SQL, (memory_id,)).fetchone()
        if row is None:
            return None
        return MemoryEntry.model_validate(_decode_sqlite_json(row["entry_json"]))

    def list_memory_entries(self) -> list[MemoryEntry]:
        rows = self._conn.execute(_SQLITE_MEMORY_ENTRY_LIST_SQL).fetchall()
        return [MemoryEntry.model_validate(_decode_sqlite_json(row["entry_json"])) for row in rows]

    def search_memory_entries(
        self,
        query: MemoryQuery,
        *,
        limit: int,
    ) -> list[tuple[MemoryEntry, float]]:
        search_document = _sql_search_document(query)
        channel_id = query.asset_context.channel_id if query.asset_context is not None else None
        params: list[object] = [
            _memory_scope_payload(query),
            f"%{search_document}%",
            query.series_id,
            query.series_id,
            query.franchise_id,
            query.franchise_id,
            query.content_type,
            query.content_type,
            query.style_profile_id,
            query.style_profile_id,
            channel_id,
            channel_id,
            max(limit, 1),
        ]
        sql = """
            WITH candidate_scopes(scope_kind, scope_key) AS (
                SELECT
                    json_extract(value, '$.scope_kind'),
                    json_extract(value, '$.scope_key')
                FROM json_each(?)
            )
            SELECT
                entry_json,
                embedding_json,
                MIN(
                    1.0,
                    CASE WHEN search_document LIKE ? THEN 0.55 ELSE 0 END
                    + CASE WHEN ? IS NOT NULL AND series_id = ? THEN 0.15 ELSE 0 END
                    + CASE WHEN ? IS NOT NULL AND franchise_id = ? THEN 0.12 ELSE 0 END
                    + CASE WHEN ? IS NOT NULL AND content_type = ? THEN 0.08 ELSE 0 END
                    + CASE WHEN ? IS NOT NULL AND style_profile_id = ? THEN 0.08 ELSE 0 END
                    + CASE
                        WHEN ? IS NOT NULL AND instr(COALESCE(typed_metadata, ''), ?) > 0
                        THEN 0.08
                        ELSE 0
                    END
                ) AS recall_score
            FROM memory_entries
            WHERE lifecycle_status = 'active'
              AND EXISTS (
                    SELECT 1
                    FROM candidate_scopes
                    WHERE candidate_scopes.scope_kind = memory_entries.scope_kind
                      AND candidate_scopes.scope_key = memory_entries.scope_key
              )
            ORDER BY recall_score DESC, updated_at DESC, memory_id ASC
            LIMIT ?
        """
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [
            (
                _entry_with_runtime_embedding(
                    MemoryEntry.model_validate(_decode_sqlite_json(row["entry_json"])),
                    row["embedding_json"],
                ),
                float(row["recall_score"] or 0.0),
            )
            for row in rows
        ]

    def backfill_memory_embeddings(self, *, limit: int | None = None) -> int:
        rows = self._conn.execute(_SQLITE_MEMORY_ENTRY_INDEX_SELECT_SQL).fetchall()
        updated = 0
        for row in rows[:limit] if limit is not None else rows:
            entry = MemoryEntry.model_validate(_decode_sqlite_json(row["entry_json"]))
            embedding_model_id, embedding, search_document = embedding_metadata_for_entry(entry)
            timestamp = _isoformat_timestamp(entry.updated_at or datetime.now(UTC))
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE memory_entries
                    SET typed_metadata = ?,
                        search_document = ?,
                        embedding_model_id = ?,
                        embedding_json = ?,
                        embedding_updated_at = ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        _encode_sqlite_json(
                            {
                                **entry.typed_metadata,
                                "embedding": embedding,
                                "embedding_model_id": embedding_model_id,
                            }
                        ),
                        search_document,
                        embedding_model_id,
                        serialize_embedding(embedding),
                        timestamp,
                        timestamp,
                        entry.memory_id,
                    ),
                )
            updated += 1
        return updated

    def save_memory_evidence_event(self, event: MemoryEvidenceEvent) -> MemoryEvidenceEvent:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_evidence_events (
                    event_id,
                    memory_id,
                    event_kind,
                    run_id,
                    job_id,
                    media_key,
                    stage,
                    source_ref,
                    event_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    memory_id = excluded.memory_id,
                    event_kind = excluded.event_kind,
                    run_id = excluded.run_id,
                    job_id = excluded.job_id,
                    media_key = excluded.media_key,
                    stage = excluded.stage,
                    source_ref = excluded.source_ref,
                    event_json = excluded.event_json,
                    updated_at = excluded.updated_at
                """,
                (
                    event.event_id,
                    event.memory_id,
                    event.event_kind,
                    event.run_id,
                    event.job_id,
                    event.media_key,
                    event.stage,
                    event.source_ref,
                    _encode_sqlite_json(event.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
        return event

    def upsert_historical_run_link(self, link: HistoricalRunLink) -> None:
        now = _utc_now()
        existing_row = self._conn.execute(
            "SELECT link_json FROM historical_run_links WHERE run_id = ?",
            (link.run_id,),
        ).fetchone()
        if existing_row is not None:
            existing = HistoricalRunLink.model_validate(
                _decode_sqlite_json(existing_row["link_json"])
            )
            link = _merge_historical_run_link(existing, link)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO historical_run_links (
                    run_id, media_key, link_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    media_key = excluded.media_key,
                    link_json = excluded.link_json,
                    updated_at = excluded.updated_at
                """,
                (
                    link.run_id,
                    link.media_key,
                    _encode_sqlite_json(link.model_dump(mode="json")),
                    now,
                    now,
                ),
            )

    def list_historical_run_links(
        self,
        media_key: str,
        *,
        exclude_run_id: str | None = None,
    ) -> list[HistoricalRunLink]:
        params: list[str] = [media_key]
        query = "SELECT link_json FROM historical_run_links WHERE media_key = ?"
        if exclude_run_id is not None:
            query += " AND run_id <> ?"
            params.append(exclude_run_id)
        query += " ORDER BY created_at ASC, run_id ASC"
        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [
            HistoricalRunLink.model_validate(_decode_sqlite_json(row["link_json"])) for row in rows
        ]

    def save_prompt_evolution_proposal(self, proposal: PromptEvolutionProposal) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO prompt_evolution_proposals (
                    proposal_id,
                    proposal_json,
                    status,
                    prompt_family,
                    target_model_id,
                    target_language,
                    source_language,
                    prompt_variant_id,
                    base_prompt_version,
                    scope_kind,
                    scope_key,
                    media_key,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    proposal_json = excluded.proposal_json,
                    status = excluded.status,
                    prompt_family = excluded.prompt_family,
                    target_model_id = excluded.target_model_id,
                    target_language = excluded.target_language,
                    source_language = excluded.source_language,
                    prompt_variant_id = excluded.prompt_variant_id,
                    base_prompt_version = excluded.base_prompt_version,
                    scope_kind = excluded.scope_kind,
                    scope_key = excluded.scope_key,
                    media_key = excluded.media_key,
                    updated_at = excluded.updated_at
                """,
                (
                    proposal.proposal_id,
                    _encode_sqlite_json(proposal.model_dump(mode="json")),
                    proposal.status,
                    proposal.prompt_family,
                    proposal.target_model_id,
                    _proposal_metadata_value(proposal, "target_language"),
                    _proposal_metadata_value(proposal, "source_language"),
                    _proposal_compatibility_value(proposal, "prompt_variant_id"),
                    _proposal_compatibility_value(proposal, "base_prompt_version"),
                    _proposal_compatibility_value(proposal, "scope_kind"),
                    _proposal_compatibility_value(proposal, "scope_key"),
                    _proposal_metadata_value(proposal, "media_key"),
                    now,
                    now,
                ),
            )

    def list_prompt_evolution_proposals(
        self,
        *,
        status: str | None = None,
        prompt_family: str | None = None,
        target_model_id: str | None = None,
        target_language: str | None = None,
        source_language: str | None = None,
        prompt_variant_id: str | None = None,
        base_prompt_version: str | None = None,
        scope_kind: str | None = None,
        scope_key: str | None = None,
        media_key: str | None = None,
    ) -> list[PromptEvolutionProposal]:
        status = _normalized_proposal_status(status)
        rows = self._conn.execute(
            _SQLITE_PROPOSAL_LIST_SQL,
            (
                status,
                status,
                prompt_family,
                prompt_family,
                target_model_id,
                target_model_id,
                target_language,
                target_language,
                source_language,
                source_language,
                prompt_variant_id,
                prompt_variant_id,
                base_prompt_version,
                base_prompt_version,
                scope_kind,
                scope_kind,
                scope_key,
                scope_key,
                media_key,
                media_key,
            ),
        ).fetchall()
        return [
            PromptEvolutionProposal.model_validate(_decode_sqlite_json(row["proposal_json"]))
            for row in rows
        ]

    def save_transcript_provider_quality_stats(self, stats: TranscriptProviderQualityStats) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                _SQLITE_PROVIDER_STATS_UPSERT_SQL,
                (
                    stats.provider_id,
                    stats.source_language,
                    stats.target_language,
                    _encode_sqlite_json(stats.model_dump(mode="json")),
                    now,
                    now,
                ),
            )

    def get_transcript_provider_quality_stats(
        self,
        *,
        provider_id: str,
        source_language: str,
        target_language: str,
    ) -> TranscriptProviderQualityStats | None:
        row = self._conn.execute(
            _SQLITE_PROVIDER_STATS_SELECT_SQL,
            (provider_id, source_language, target_language),
        ).fetchone()
        if row is None:
            return None
        return TranscriptProviderQualityStats.model_validate(_decode_sqlite_json(row["stats_json"]))

    def save_human_review_resolution(self, resolution: HumanReviewResolutionRecord) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                _SQLITE_HUMAN_RESOLUTION_UPSERT_SQL,
                (
                    resolution.run_id,
                    resolution.job_id,
                    _encode_sqlite_json(resolution.model_dump(mode="json")),
                    resolution.resolution_kind,
                    resolution.source_language,
                    resolution.target_language,
                    resolution.transcript_provider_id,
                    resolution.model_id,
                    resolution.prompt_variant_id,
                    resolution.prompt_version,
                    resolution.combo_key,
                    resolution.resolved_at.isoformat(),
                    now,
                    now,
                ),
            )

    def get_human_review_resolution(self, run_id: str) -> HumanReviewResolutionRecord | None:
        row = self._conn.execute(_SQLITE_HUMAN_RESOLUTION_SELECT_SQL, (run_id,)).fetchone()
        if row is None:
            return None
        return HumanReviewResolutionRecord.model_validate(
            _decode_sqlite_json(row["resolution_json"])
        )

    def list_human_review_resolutions(
        self,
        *,
        resolution_kind: str | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        transcript_provider_id: str | None = None,
        model_id: str | None = None,
        prompt_variant_id: str | None = None,
        prompt_version: str | None = None,
        combo_key: str | None = None,
    ) -> list[HumanReviewResolutionRecord]:
        rows = self._conn.execute(
            _SQLITE_HUMAN_RESOLUTION_LIST_SQL,
            (
                resolution_kind,
                resolution_kind,
                source_language,
                source_language,
                target_language,
                target_language,
                transcript_provider_id,
                transcript_provider_id,
                model_id,
                model_id,
                prompt_variant_id,
                prompt_variant_id,
                prompt_version,
                prompt_version,
                combo_key,
                combo_key,
            ),
        ).fetchall()
        return [
            HumanReviewResolutionRecord.model_validate(_decode_sqlite_json(row["resolution_json"]))
            for row in rows
        ]

    def save_translation_feedback_stats(self, stats: TranslationFeedbackStats) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                _SQLITE_TRANSLATION_FEEDBACK_UPSERT_SQL,
                (
                    stats.combo_key,
                    _encode_sqlite_json(stats.model_dump(mode="json")),
                    stats.source_language,
                    stats.target_language,
                    stats.transcript_provider_id,
                    stats.model_id,
                    stats.prompt_variant_id,
                    stats.prompt_version,
                    now,
                    now,
                ),
            )

    def get_translation_feedback_stats(self, combo_key: str) -> TranslationFeedbackStats | None:
        row = self._conn.execute(_SQLITE_TRANSLATION_FEEDBACK_SELECT_SQL, (combo_key,)).fetchone()
        if row is None:
            return None
        return TranslationFeedbackStats.model_validate(_decode_sqlite_json(row["stats_json"]))

    def _bootstrap_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                tenant_id TEXT,
                project_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                input_data_json TEXT,
                output_data_json TEXT,
                metadata_json TEXT,
                error_json TEXT
            );

            CREATE TABLE IF NOT EXISTS node_executions (
                execution_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                node_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                input_data_json TEXT,
                output_data_json TEXT,
                error_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_node_executions_run_id_created_at
            ON node_executions(run_id, created_at, execution_id);

            CREATE TABLE IF NOT EXISTS transcript_candidates (
                candidate_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_transcript_candidates_job_id_candidate_id
            ON transcript_candidates(job_id, candidate_id);

            CREATE TABLE IF NOT EXISTS translation_candidates (
                candidate_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_translation_candidates_job_id_candidate_id
            ON translation_candidates(job_id, candidate_id);

            CREATE TABLE IF NOT EXISTS transcript_decisions (
                job_id TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS translation_decisions (
                job_id TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS investigations (
                job_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (job_id, stage)
            );

            CREATE TABLE IF NOT EXISTS memory_batches (
                batch_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                batch_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_batches_job_id_batch_id
            ON memory_batches(job_id, batch_id);

            CREATE TABLE IF NOT EXISTS asset_records (
                media_key TEXT PRIMARY KEY,
                asset_id TEXT UNIQUE,
                media_fingerprint TEXT UNIQUE,
                asset_json TEXT NOT NULL,
                first_seen_run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_asset_records_asset_id
            ON asset_records(asset_id);

            CREATE INDEX IF NOT EXISTS idx_asset_records_media_fingerprint
            ON asset_records(media_fingerprint);

            CREATE TABLE IF NOT EXISTS asset_context_records (
                media_key TEXT PRIMARY KEY REFERENCES asset_records(media_key) ON DELETE CASCADE,
                context_json TEXT NOT NULL,
                canonical_title TEXT,
                series_id TEXT,
                franchise_id TEXT,
                channel_id TEXT,
                content_type TEXT,
                style_profile_id TEXT,
                metadata_confidence TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_asset_context_records_series
            ON asset_context_records(series_id);

            CREATE INDEX IF NOT EXISTS idx_asset_context_records_franchise
            ON asset_context_records(franchise_id);

            CREATE TABLE IF NOT EXISTS asset_relations (
                relation_id TEXT PRIMARY KEY,
                src_media_key TEXT NOT NULL REFERENCES asset_records(media_key) ON DELETE CASCADE,
                dst_media_key TEXT NOT NULL REFERENCES asset_records(media_key) ON DELETE CASCADE,
                relation_kind TEXT NOT NULL,
                confidence REAL NOT NULL,
                relation_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_asset_relations_src_kind
            ON asset_relations(src_media_key, relation_kind);

            CREATE INDEX IF NOT EXISTS idx_asset_relations_dst_kind
            ON asset_relations(dst_media_key, relation_kind);

            CREATE TABLE IF NOT EXISTS memory_entries (
                memory_id TEXT PRIMARY KEY,
                entry_json TEXT NOT NULL,
                dedupe_key TEXT UNIQUE,
                scope_kind TEXT,
                scope_key TEXT,
                series_id TEXT,
                franchise_id TEXT,
                content_type TEXT,
                style_profile_id TEXT,
                promotion_status TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                typed_metadata TEXT NOT NULL DEFAULT '{}',
                search_document TEXT NOT NULL DEFAULT '',
                embedding_model_id TEXT,
                embedding_json TEXT NOT NULL DEFAULT '[]',
                embedding_updated_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_entries_scope
            ON memory_entries(scope_kind, scope_key);

            CREATE INDEX IF NOT EXISTS idx_memory_entries_series
            ON memory_entries(series_id);

            CREATE INDEX IF NOT EXISTS idx_memory_entries_franchise
            ON memory_entries(franchise_id);

            CREATE INDEX IF NOT EXISTS idx_memory_entries_content_type
            ON memory_entries(content_type);

            CREATE INDEX IF NOT EXISTS idx_memory_entries_style_profile
            ON memory_entries(style_profile_id);

            CREATE TABLE IF NOT EXISTS memory_evidence_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memory_entries(memory_id) ON DELETE CASCADE,
                event_kind TEXT NOT NULL,
                run_id TEXT,
                job_id TEXT,
                media_key TEXT,
                stage TEXT,
                source_ref TEXT,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_evidence_events_memory_id
            ON memory_evidence_events(memory_id, created_at);

            CREATE TABLE IF NOT EXISTS historical_run_links (
                run_id TEXT PRIMARY KEY,
                media_key TEXT NOT NULL REFERENCES asset_records(media_key) ON DELETE CASCADE,
                link_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_historical_run_links_media_key_created_at
            ON historical_run_links(media_key, created_at, run_id);

            CREATE TABLE IF NOT EXISTS prompt_evolution_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_json TEXT NOT NULL,
                status TEXT NOT NULL,
                prompt_family TEXT NOT NULL,
                target_model_id TEXT NOT NULL,
                target_language TEXT,
                source_language TEXT,
                prompt_variant_id TEXT,
                base_prompt_version TEXT,
                scope_kind TEXT,
                scope_key TEXT,
                media_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prompt_evolution_proposals_scope
            ON prompt_evolution_proposals(
                status,
                prompt_family,
                target_model_id,
                source_language,
                target_language,
                prompt_variant_id,
                base_prompt_version,
                scope_kind,
                scope_key
            );

            CREATE TABLE IF NOT EXISTS transcript_provider_quality_stats (
                provider_id TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (provider_id, source_language, target_language)
            );

            CREATE TABLE IF NOT EXISTS human_review_resolutions (
                run_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                resolution_json TEXT NOT NULL,
                resolution_kind TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                transcript_provider_id TEXT,
                model_id TEXT,
                prompt_variant_id TEXT,
                prompt_version TEXT,
                combo_key TEXT,
                resolved_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_human_review_resolutions_lookup
            ON human_review_resolutions(
                resolution_kind,
                source_language,
                target_language,
                transcript_provider_id,
                model_id,
                prompt_variant_id,
                prompt_version,
                combo_key
            );

            CREATE TABLE IF NOT EXISTS translation_feedback_stats (
                combo_key TEXT PRIMARY KEY,
                stats_json TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                transcript_provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_variant_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_translation_feedback_stats_lookup
            ON translation_feedback_stats(
                source_language,
                target_language,
                transcript_provider_id,
                model_id,
                prompt_variant_id,
                prompt_version
            );
            """
        )
        self._conn.commit()

    def _save_candidate(
        self,
        *,
        table: str,
        candidate_id: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                _SQLITE_CANDIDATE_UPSERT_SQL[table],
                (candidate_id, job_id, _encode_sqlite_json(payload), now, now),
            )

    def _save_singleton_model(
        self,
        *,
        table: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                _SQLITE_SINGLETON_UPSERT_SQL[table],
                (job_id, _encode_sqlite_json(payload), now, now),
            )

    def _get_singleton_payload(
        self,
        *,
        table: str,
        job_id: str,
    ) -> Any | None:
        row = self._conn.execute(
            _SQLITE_SINGLETON_SELECT_SQL[table],
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_sqlite_json(row["decision_json"])

    def _save_stage_payload(
        self,
        *,
        table: str,
        job_id: str,
        stage: ReviewStage,
        payload: dict[str, object],
    ) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                _SQLITE_STAGE_UPSERT_SQL[table],
                (job_id, stage, _encode_sqlite_json(payload), now, now),
            )

    def _get_stage_payload(
        self,
        *,
        table: str,
        job_id: str,
        stage: ReviewStage,
    ) -> Any | None:
        row = self._conn.execute(
            _SQLITE_STAGE_SELECT_SQL[table],
            (job_id, stage),
        ).fetchone()
        if row is None:
            return None
        return _decode_sqlite_json(row["payload_json"])

    def _list_payloads(
        self,
        *,
        table: str,
        job_id: str,
        column: str,
    ) -> list[Any]:
        rows = self._conn.execute(
            _SQLITE_LIST_SQL[table],
            (job_id,),
        ).fetchall()
        return [_decode_sqlite_json(row[column]) for row in rows]


class _AssetResolver(Protocol):
    def get_by_media_key(self, media_key: str) -> AssetRecord | None: ...

    def get_by_asset_id(self, asset_id: str) -> AssetRecord | None: ...

    def get_by_media_fingerprint(self, media_fingerprint: str) -> AssetRecord | None: ...

    def save(self, asset: AssetRecord) -> AssetRecord: ...


class _PostgresAssetResolver:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def get_by_media_key(self, media_key: str) -> AssetRecord | None:
        return self._get_one("media_key", media_key)

    def get_by_asset_id(self, asset_id: str) -> AssetRecord | None:
        return self._get_one("asset_id", asset_id)

    def get_by_media_fingerprint(self, media_fingerprint: str) -> AssetRecord | None:
        return self._get_one("media_fingerprint", media_fingerprint)

    def save(self, asset: AssetRecord) -> AssetRecord:
        now = _utc_now()
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO asset_records (
                    media_key,
                    asset_id,
                    media_fingerprint,
                    asset_json,
                    first_seen_run_id,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (media_key) DO UPDATE SET
                    asset_id = EXCLUDED.asset_id,
                    media_fingerprint = EXCLUDED.media_fingerprint,
                    asset_json = EXCLUDED.asset_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    asset.media_key,
                    asset.asset_id,
                    asset.media_fingerprint,
                    _encode_json(asset.model_dump(mode="json")),
                    asset.first_seen_run_id,
                    _isoformat_timestamp(asset.created_at),
                    now,
                ),
            )
        resolved = self.get_by_media_key(asset.media_key)
        if resolved is None:
            raise RuntimeError(f"asset {asset.media_key} was not persisted")
        return resolved

    def _get_one(self, field_name: str, value: str) -> AssetRecord | None:
        row = self._conn.execute(_POSTGRES_ASSET_SELECT_SQL[field_name], (value,)).fetchone()
        if row is None:
            return None
        return AssetRecord.model_validate(_decode_db_json(row["asset_json"]))


class _SQLiteAssetResolver:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_by_media_key(self, media_key: str) -> AssetRecord | None:
        return self._get_one("media_key", media_key)

    def get_by_asset_id(self, asset_id: str) -> AssetRecord | None:
        return self._get_one("asset_id", asset_id)

    def get_by_media_fingerprint(self, media_fingerprint: str) -> AssetRecord | None:
        return self._get_one("media_fingerprint", media_fingerprint)

    def save(self, asset: AssetRecord) -> AssetRecord:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO asset_records (
                    media_key,
                    asset_id,
                    media_fingerprint,
                    asset_json,
                    first_seen_run_id,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(media_key) DO UPDATE SET
                    asset_id = excluded.asset_id,
                    media_fingerprint = excluded.media_fingerprint,
                    asset_json = excluded.asset_json,
                    updated_at = excluded.updated_at
                """,
                (
                    asset.media_key,
                    asset.asset_id,
                    asset.media_fingerprint,
                    _encode_sqlite_json(asset.model_dump(mode="json")),
                    asset.first_seen_run_id,
                    _isoformat_timestamp(asset.created_at),
                    now,
                ),
            )
        resolved = self.get_by_media_key(asset.media_key)
        if resolved is None:
            raise RuntimeError(f"asset {asset.media_key} was not persisted")
        return resolved

    def _get_one(self, field_name: str, value: str) -> AssetRecord | None:
        row = self._conn.execute(_SQLITE_ASSET_SELECT_SQL[field_name], (value,)).fetchone()
        if row is None:
            return None
        return AssetRecord.model_validate(_decode_sqlite_json(row["asset_json"]))


def _sql_search_document(query: MemoryQuery) -> str:
    query_embedding_document = embedding_metadata_for_query(query)[1]
    tokens = [match.group(0) for match in _SQL_TOKEN_RE.finditer(query_embedding_document)]
    if not tokens:
        return "memory"
    return " ".join(tokens[:24])


def _memory_scope_payload(query: MemoryQuery) -> str:
    return json.dumps(
        [
            {"scope_kind": scope_kind, "scope_key": scope_key}
            for scope_kind, scope_key in _memory_scope_pairs(query)
        ]
    )


def _memory_scope_pairs(query: MemoryQuery) -> list[tuple[str, str]]:
    scopes: list[tuple[str, str]] = []
    if query.media_key is not None:
        scopes.append(("asset", query.media_key))
    if query.series_id is not None:
        scopes.append(("series", query.series_id))
    for speaker_id in dict.fromkeys(query.speaker_ids):
        scopes.append(("speaker_cluster", speaker_id))
    if query.franchise_id is not None:
        scopes.append(("franchise", query.franchise_id))
    if query.asset_context is not None and query.asset_context.channel_id is not None:
        scopes.append(("channel", query.asset_context.channel_id))
    scopes.extend(
        [
            (
                "project_pair",
                f"{query.job.tenant_id}::{query.job.project_id}::{query.job.source_language}::{query.job.target_language}",
            ),
            ("pair", f"{query.job.source_language}::{query.job.target_language}"),
            ("source_language", query.job.source_language),
            ("target_language", query.job.target_language),
            ("global", "global"),
        ]
    )
    return scopes


def _entry_with_runtime_embedding(entry: MemoryEntry, embedding_payload: object) -> MemoryEntry:
    return entry.model_copy(
        update={
            "typed_metadata": {
                **entry.typed_metadata,
                "embedding": embedding_payload,
                "embedding_model_id": EMBEDDING_MODEL_ID,
            }
        }
    )


def _resolve_asset_record(
    *,
    resolver: _AssetResolver,
    asset_id: str | None,
    media_fingerprint: str | None,
    first_seen_run_id: str,
    source_language: str,
    target_language: str,
) -> AssetRecord:
    normalized_asset_id = _normalized_optional_identifier(asset_id)
    normalized_media_fingerprint = _normalized_optional_identifier(media_fingerprint)

    asset_match = (
        resolver.get_by_asset_id(normalized_asset_id) if normalized_asset_id is not None else None
    )
    fingerprint_match = (
        resolver.get_by_media_fingerprint(normalized_media_fingerprint)
        if normalized_media_fingerprint is not None
        else None
    )
    if (
        asset_match is not None
        and fingerprint_match is not None
        and asset_match.media_key != fingerprint_match.media_key
    ):
        raise ValueError(
            "conflicting asset_id and media_fingerprint mappings require operator action"
        )

    existing = asset_match or fingerprint_match
    now = datetime.now(UTC)
    if existing is not None:
        if normalized_asset_id is not None and existing.asset_id not in {None, normalized_asset_id}:
            raise ValueError("asset_id conflicts with existing asset mapping")
        if normalized_media_fingerprint is not None and existing.media_fingerprint not in {
            None,
            normalized_media_fingerprint,
        }:
            raise ValueError("media_fingerprint conflicts with existing asset mapping")
        return resolver.save(
            existing.model_copy(
                update={
                    "asset_id": existing.asset_id or normalized_asset_id,
                    "media_fingerprint": existing.media_fingerprint or normalized_media_fingerprint,
                    "source_language": existing.source_language or source_language,
                    "target_language": existing.target_language or target_language,
                    "updated_at": now,
                }
            )
        )

    media_key = _build_media_key(
        asset_id=normalized_asset_id,
        media_fingerprint=normalized_media_fingerprint,
        fallback_seed=first_seen_run_id,
    )
    return resolver.save(
        AssetRecord(
            media_key=media_key,
            asset_id=normalized_asset_id,
            media_fingerprint=normalized_media_fingerprint,
            first_seen_run_id=first_seen_run_id,
            source_language=source_language,
            target_language=target_language,
            created_at=now,
            updated_at=now,
        )
    )


def _build_media_key(
    *,
    asset_id: str | None,
    media_fingerprint: str | None,
    fallback_seed: str,
) -> str:
    if asset_id is not None:
        return f"asset-id:{asset_id}"
    if media_fingerprint is not None:
        return f"media-fingerprint:{media_fingerprint}"
    return f"source-ref:{fallback_seed}"


def _normalized_optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _proposal_metadata_value(
    proposal: PromptEvolutionProposal,
    key: str,
) -> str | None:
    compatibility_value = _proposal_compatibility_value(proposal, key)
    if compatibility_value is not None:
        return compatibility_value
    value = proposal.metadata.get(key)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _merge_historical_run_link(
    existing: HistoricalRunLink,
    incoming: HistoricalRunLink,
) -> HistoricalRunLink:
    merged: dict[str, Any] = existing.model_dump(mode="python")
    for key, value in incoming.model_dump(mode="python").items():
        if value is None:
            continue
        merged[key] = value
    return HistoricalRunLink.model_validate(merged)


def _normalized_proposal_status(status: str | None) -> str | None:
    if status == "approved":
        return "active"
    if status == "rejected":
        return "rolled_back"
    return status


def _proposal_compatibility_value(
    proposal: PromptEvolutionProposal,
    key: str,
) -> str | None:
    compatibility = proposal.compatibility
    if compatibility is None:
        return None
    mapping = {
        "prompt_family": compatibility.prompt_family,
        "model_id": compatibility.model_id,
        "prompt_variant_id": compatibility.prompt_variant_id,
        "base_prompt_version": compatibility.base_prompt_version,
        "source_language": compatibility.source_language,
        "target_language": compatibility.target_language,
        "scope_kind": compatibility.scope_kind,
        "scope_key": compatibility.scope_key,
    }
    value = mapping.get(key)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _sqlite_row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        tenant_id=row["tenant_id"],
        project_id=row["project_id"],
        status=row["status"],
        created_at=_isoformat_timestamp(row["created_at"]),
        updated_at=_isoformat_timestamp(row["updated_at"]),
        input_data=_decode_sqlite_json(row["input_data_json"]),
        output_data=_decode_sqlite_json(row["output_data_json"]),
        metadata=_decode_sqlite_json(row["metadata_json"]),
        error=_decode_sqlite_json(row["error_json"]),
    )


def _sqlite_row_to_node_execution(row: sqlite3.Row) -> NodeExecutionRecord:
    return NodeExecutionRecord(
        execution_id=row["execution_id"],
        run_id=row["run_id"],
        node_name=row["node_name"],
        status=row["status"],
        created_at=_isoformat_timestamp(row["created_at"]),
        updated_at=_isoformat_timestamp(row["updated_at"]),
        input_data=_decode_sqlite_json(row["input_data_json"]),
        output_data=_decode_sqlite_json(row["output_data_json"]),
        error=_decode_sqlite_json(row["error_json"]),
    )


def _encode_sqlite_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _decode_sqlite_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _decode_db_json(value: Any) -> Any:
    if value is None:
        return None
    return value
