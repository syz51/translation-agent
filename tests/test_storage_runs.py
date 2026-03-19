from pathlib import Path

from translation_agent.storage import SQLiteRunStore


def test_run_and_node_execution_persist(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.sqlite3"

    with SQLiteRunStore(db_path) as store:
        run = store.create_run(
            tenant_id="tenant-1",
            project_id="project-1",
            input_data={"source": "video.mp4"},
            metadata={"phase": 0},
        )
        node = store.create_node_execution(
            run_id=run.run_id,
            node_name="ingest_job",
            input_data={"step": "start"},
        )

        updated_run = store.update_run(
            run.run_id,
            status="completed",
            output_data={"artifacts": 1},
        )
        updated_node = store.update_node_execution(
            node.execution_id,
            status="succeeded",
            output_data={"ok": True},
        )

    reopened = SQLiteRunStore(db_path)
    try:
        loaded_run = reopened.get_run(run.run_id)
        loaded_node = reopened.get_node_execution(node.execution_id)

        assert loaded_run == updated_run
        assert loaded_node == updated_node
        assert [record.run_id for record in reopened.list_runs()] == [run.run_id]
        assert [record.execution_id for record in reopened.list_node_executions(run.run_id)] == [
            node.execution_id
        ]
    finally:
        reopened.close()

