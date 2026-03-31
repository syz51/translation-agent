from __future__ import annotations

from alembic import op

revision = "0002_operational_entities"
down_revision = "0001_runtime_bootstrap"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_candidates (
            candidate_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            candidate_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_transcript_candidates_job_id_candidate_id
        ON transcript_candidates(job_id, candidate_id)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS translation_candidates (
            candidate_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            candidate_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_translation_candidates_job_id_candidate_id
        ON translation_candidates(job_id, candidate_id)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_decisions (
            job_id TEXT PRIMARY KEY,
            decision_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS translation_decisions (
            job_id TEXT PRIMARY KEY,
            decision_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS investigations (
            job_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            payload_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (job_id, stage)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_batches (
            batch_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            batch_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_batches_job_id_batch_id
        ON memory_batches(job_id, batch_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_memory_batches_job_id_batch_id")
    op.execute("DROP TABLE IF EXISTS memory_batches")
    op.execute("DROP TABLE IF EXISTS investigations")
    op.execute("DROP TABLE IF EXISTS translation_decisions")
    op.execute("DROP TABLE IF EXISTS transcript_decisions")
    op.execute("DROP INDEX IF EXISTS idx_translation_candidates_job_id_candidate_id")
    op.execute("DROP TABLE IF EXISTS translation_candidates")
    op.execute("DROP INDEX IF EXISTS idx_transcript_candidates_job_id_candidate_id")
    op.execute("DROP TABLE IF EXISTS transcript_candidates")
