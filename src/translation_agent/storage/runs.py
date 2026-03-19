from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


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


class SQLiteRunStore:
    """SQLite-backed operational store for Phase 0 runs."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteRunStore:
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
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, tenant_id, project_id, status, created_at, updated_at,
                    input_data_json, output_data_json, metadata_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
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
        with self._conn:
            self._conn.execute(
                """
                UPDATE runs
                SET status = ?, updated_at = ?, output_data_json = ?, metadata_json = ?, error_json = ?
                WHERE run_id = ?
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
        return self.get_run(run_id)

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
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO node_executions (
                    execution_id, run_id, node_name, status, created_at, updated_at,
                    input_data_json, output_data_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return self.get_node_execution(execution_id)

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM node_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        return _row_to_node_execution(row) if row else None

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM node_executions
            WHERE run_id = ?
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
        with self._conn:
            self._conn.execute(
                """
                UPDATE node_executions
                SET status = ?, updated_at = ?, output_data_json = ?, error_json = ?
                WHERE execution_id = ?
                """,
                (
                    new_status,
                    updated_at,
                    _encode_json(new_output),
                    _encode_json(new_error),
                    execution_id,
                ),
            )
        return self.get_node_execution(execution_id)

    def _initialize_schema(self) -> None:
        with self._conn:
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
                """
            )


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        tenant_id=row["tenant_id"],
        project_id=row["project_id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        input_data=_decode_json(row["input_data_json"]),
        output_data=_decode_json(row["output_data_json"]),
        metadata=_decode_json(row["metadata_json"]),
        error=_decode_json(row["error_json"]),
    )


def _row_to_node_execution(row: sqlite3.Row) -> NodeExecutionRecord:
    return NodeExecutionRecord(
        execution_id=row["execution_id"],
        run_id=row["run_id"],
        node_name=row["node_name"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        input_data=_decode_json(row["input_data_json"]),
        output_data=_decode_json(row["output_data_json"]),
        error=_decode_json(row["error_json"]),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _decode_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)

