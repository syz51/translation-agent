from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    tenant_id: str | None
    project_id: str | None
    status: str
    created_at: str
    updated_at: str
    input_data: Any
    output_data: Any
    metadata: Any
    error: Any


@dataclass(frozen=True, slots=True)
class NodeExecutionRecord:
    execution_id: str
    run_id: str
    node_name: str
    status: str
    created_at: str
    updated_at: str
    input_data: Any
    output_data: Any
    error: Any


_UNSET = object()


class PostgresRunStore:
    """Postgres-backed operational store for Phase 0 runs."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: Any = psycopg.connect(
            self.dsn,
            autocommit=True,
            row_factory=cast(Any, dict_row),
        )
        self._initialize_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PostgresRunStore:
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
        run_id = run_id or f"run_{uuid4().hex}"
        created_at = created_at or _utc_now()
        payload = (
            run_id,
            tenant_id,
            project_id,
            status,
            created_at,
            created_at,
            _encode_json(input_data),
            None,
            _encode_json(metadata),
            None,
        )
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, tenant_id, project_id, status, created_at, updated_at,
                    input_data_json, output_data_json, metadata_json, error_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                payload,
            )
        return _require_run(self.get_run(run_id), run_id)

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC"
        ).fetchall()
        return [_row_to_run(row) for row in rows]

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
        new_status = current.status if status is None else status
        new_output = current.output_data if output_data is _UNSET else output_data
        new_metadata = current.metadata if metadata is _UNSET else metadata
        new_error = current.error if error is _UNSET else error
        with self._conn.transaction():
            self._conn.execute(
                """
                UPDATE runs
                SET status = %s,
                    updated_at = %s,
                    output_data_json = %s,
                    metadata_json = %s,
                    error_json = %s
                WHERE run_id = %s
                """,
                (
                    new_status,
                    updated_at,
                    _encode_json(new_output),
                    _encode_json(new_metadata),
                    _encode_json(new_error),
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
        execution_id = execution_id or f"exec_{uuid4().hex}"
        created_at = created_at or _utc_now()
        payload = (
            execution_id,
            run_id,
            node_name,
            status,
            created_at,
            created_at,
            _encode_json(input_data),
            None,
            None,
        )
        with self._conn.transaction():
            self._conn.execute(
                """
                INSERT INTO node_executions (
                    execution_id, run_id, node_name, status, created_at, updated_at,
                    input_data_json, output_data_json, error_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                payload,
            )
        return _require_node_execution(self.get_node_execution(execution_id), execution_id)

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM node_executions WHERE execution_id = %s",
            (execution_id,),
        ).fetchone()
        return _row_to_node_execution(row) if row else None

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM node_executions
            WHERE run_id = %s
            ORDER BY created_at ASC, execution_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [_row_to_node_execution(row) for row in rows]

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
        new_status = current.status if status is None else status
        new_output = current.output_data if output_data is _UNSET else output_data
        new_error = current.error if error is _UNSET else error
        with self._conn.transaction():
            self._conn.execute(
                """
                UPDATE node_executions
                SET status = %s, updated_at = %s, output_data_json = %s, error_json = %s
                WHERE execution_id = %s
                """,
                (
                    new_status,
                    updated_at,
                    _encode_json(new_output),
                    _encode_json(new_error),
                    execution_id,
                ),
            )
        return _require_node_execution(self.get_node_execution(execution_id), execution_id)

    def _initialize_schema(self) -> None:
        with self._conn.transaction():
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    project_id TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    input_data_json JSONB,
                    output_data_json JSONB,
                    metadata_json JSONB,
                    error_json JSONB
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_executions (
                    execution_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    node_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    input_data_json JSONB,
                    output_data_json JSONB,
                    error_json JSONB
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_node_executions_run_id_created_at
                ON node_executions(run_id, created_at, execution_id)
                """
            )


def _row_to_run(row: Mapping[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        tenant_id=row["tenant_id"],
        project_id=row["project_id"],
        status=row["status"],
        created_at=_isoformat_timestamp(row["created_at"]),
        updated_at=_isoformat_timestamp(row["updated_at"]),
        input_data=_decode_json(row["input_data_json"]),
        output_data=_decode_json(row["output_data_json"]),
        metadata=_decode_json(row["metadata_json"]),
        error=_decode_json(row["error_json"]),
    )


def _row_to_node_execution(row: Mapping[str, Any]) -> NodeExecutionRecord:
    return NodeExecutionRecord(
        execution_id=row["execution_id"],
        run_id=row["run_id"],
        node_name=row["node_name"],
        status=row["status"],
        created_at=_isoformat_timestamp(row["created_at"]),
        updated_at=_isoformat_timestamp(row["updated_at"]),
        input_data=_decode_json(row["input_data_json"]),
        output_data=_decode_json(row["output_data_json"]),
        error=_decode_json(row["error_json"]),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _isoformat_timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    return value.isoformat()


def _encode_json(value: Any) -> Jsonb | None:
    if value is None:
        return None
    return Jsonb(value)


def _decode_json(value: Any) -> Any:
    if value is None:
        return None
    return value


def _require_run(record: RunRecord | None, run_id: str) -> RunRecord:
    if record is None:
        raise RuntimeError(f"run {run_id} was not persisted")
    return record


def _require_node_execution(
    record: NodeExecutionRecord | None, execution_id: str
) -> NodeExecutionRecord:
    if record is None:
        raise RuntimeError(f"node execution {execution_id} was not persisted")
    return record
