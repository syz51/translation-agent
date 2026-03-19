from __future__ import annotations

import psycopg
import pytest

from translation_agent.storage import PostgresRunStore


pytestmark = pytest.mark.integration


def test_store_bootstraps_schema_on_first_connection(postgres_dsn: str) -> None:
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

    assert {"runs", "node_executions"} <= tables
    assert "idx_node_executions_run_id_created_at" in indexes


def test_create_get_update_list_run_records(postgres_dsn: str) -> None:
    with PostgresRunStore(postgres_dsn) as store:
        run = store.create_run(
            tenant_id="tenant-1",
            project_id="project-1",
            input_data={"source": "video.mp4"},
            metadata={"phase": 0},
        )
        updated = store.update_run(
            run.run_id,
            status="completed",
            output_data={"artifacts": 1},
            error={"code": "none"},
        )

    with PostgresRunStore(postgres_dsn) as reopened:
        loaded = reopened.get_run(run.run_id)
        listed = reopened.list_runs()

    assert loaded == updated
    assert [record.run_id for record in listed] == [run.run_id]
    assert loaded is not None
    assert loaded.input_data == {"source": "video.mp4"}
    assert loaded.metadata == {"phase": 0}
    assert loaded.output_data == {"artifacts": 1}
    assert loaded.error == {"code": "none"}
    assert isinstance(loaded.created_at, str)
    assert isinstance(loaded.updated_at, str)


def test_create_get_update_list_node_execution_records(postgres_dsn: str) -> None:
    with PostgresRunStore(postgres_dsn) as store:
        run = store.create_run(input_data={"source": "video.mp4"})
        first = store.create_node_execution(
            run_id=run.run_id,
            node_name="ingest_job",
            input_data={"step": "start"},
        )
        second = store.create_node_execution(
            run_id=run.run_id,
            node_name="translate_job",
            input_data={"step": "translate"},
        )
        updated = store.update_node_execution(
            first.execution_id,
            status="succeeded",
            output_data={"ok": True},
            error={"code": "none"},
        )

    with PostgresRunStore(postgres_dsn) as reopened:
        loaded = reopened.get_node_execution(first.execution_id)
        listed = reopened.list_node_executions(run.run_id)

    assert loaded == updated
    assert [record.execution_id for record in listed] == [first.execution_id, second.execution_id]
    assert loaded is not None
    assert loaded.input_data == {"step": "start"}
    assert loaded.output_data == {"ok": True}
    assert loaded.error == {"code": "none"}
    assert isinstance(loaded.created_at, str)
    assert isinstance(loaded.updated_at, str)


def test_node_execution_foreign_key_behavior(postgres_dsn: str) -> None:
    with PostgresRunStore(postgres_dsn) as store:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            store.create_node_execution(
                run_id="missing-run",
                node_name="ingest_job",
                input_data={"step": "start"},
            )

        run = store.create_run(input_data={"source": "video.mp4"})
        node = store.create_node_execution(
            run_id=run.run_id,
            node_name="ingest_job",
            input_data={"step": "start"},
        )

        with store._conn.transaction():
            store._conn.execute("DELETE FROM runs WHERE run_id = %s", (run.run_id,))

        deleted_node = store.get_node_execution(node.execution_id)

    assert deleted_node is None
