"""Operational stores for local SQLite and Postgres-backed runtimes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, cast

from translation_agent.models import (
    FinalTranscriptDecision,
    FinalTranslationDecision,
    MemoryWriteBatch,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.models.review import ReviewStage

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


class OperationalStore(DecisionStore, MemoryBatchStore, Protocol):
    """Combined operational persistence contract used by public entrypoints."""

    def close(self) -> None: ...

    def __enter__(self): ...

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
