from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest
from psycopg.types.json import Jsonb

from translation_agent.storage import PostgresRunStore
from translation_agent.storage.migrations import upgrade_database

pytestmark = [pytest.mark.integration, pytest.mark.migration]

_CREATE_RUNS_TABLE = """
CREATE TABLE runs (
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

_CREATE_NODE_EXECUTIONS_TABLE = """
CREATE TABLE node_executions (
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


def test_upgrade_database_bootstraps_schema_on_empty_database(postgres_dsn: str) -> None:
    upgrade_database(postgres_dsn)

    with PostgresRunStore(postgres_dsn) as store:
        tables = {
            row["tablename"]
            for row in store._conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
            ).fetchall()
        }
        indexes = {
            row["indexname"]
            for row in store._conn.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            ).fetchall()
        }
        revision = store._conn.execute("SELECT version_num FROM alembic_version").fetchone()
        run = store.create_run(input_data={"source": "legacy.mp4"})
        node = store.create_node_execution(run_id=run.run_id, node_name="ingest")

    assert {"runs", "node_executions"} <= tables
    assert "idx_node_executions_run_id_created_at" in indexes
    assert revision is not None
    assert revision["version_num"] == "0001_runtime_bootstrap"
    assert node.run_id == run.run_id


def test_upgrade_database_preserves_existing_legacy_data(postgres_dsn: str) -> None:
    _create_legacy_schema(postgres_dsn)
    now = datetime.now(UTC)

    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, tenant_id, project_id, status, created_at, updated_at,
                input_data_json, output_data_json, metadata_json, error_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                "legacy-run",
                "tenant-legacy",
                "project-legacy",
                "completed",
                now,
                now,
                Jsonb({"source": "legacy.mp4"}),
                Jsonb({"artifact_ref": "jobs/legacy-run-request.json"}),
                Jsonb({"phase": 0}),
                Jsonb({"code": "none"}),
            ),
        )
        conn.execute(
            """
            INSERT INTO node_executions (
                execution_id, run_id, node_name, status, created_at, updated_at,
                input_data_json, output_data_json, error_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                "legacy-exec",
                "legacy-run",
                "ingest",
                "succeeded",
                now,
                now,
                Jsonb({"step": "ingest"}),
                Jsonb({"ok": True}),
                Jsonb({"code": "none"}),
            ),
        )

    upgrade_database(postgres_dsn)

    with PostgresRunStore(postgres_dsn) as store:
        indexes = {
            row["indexname"]
            for row in store._conn.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            ).fetchall()
        }
        run = store.get_run("legacy-run")
        executions = store.list_node_executions("legacy-run")

    assert run is not None
    assert run.status == "completed"
    assert run.metadata == {"phase": 0}
    assert "idx_node_executions_run_id_created_at" in indexes
    assert executions[0].execution_id == "legacy-exec"
    assert executions[0].output_data == {"ok": True}


def _create_legacy_schema(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(_CREATE_RUNS_TABLE)
        conn.execute(_CREATE_NODE_EXECUTIONS_TABLE)
