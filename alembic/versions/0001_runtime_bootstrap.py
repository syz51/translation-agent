from __future__ import annotations

from alembic import op

revision = "0001_runtime_bootstrap"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
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
    op.execute(
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
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_node_executions_run_id_created_at
        ON node_executions(run_id, created_at, execution_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_node_executions_run_id_created_at")
    op.execute("DROP TABLE IF EXISTS node_executions")
    op.execute("DROP TABLE IF EXISTS runs")
