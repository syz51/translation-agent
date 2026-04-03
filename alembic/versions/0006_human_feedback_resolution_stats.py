from __future__ import annotations

from alembic import op

revision = "0006_human_feedback_stats"
down_revision = "0005_provider_quality_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS human_review_resolutions (
            run_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            resolution_json JSONB NOT NULL,
            resolution_kind TEXT NOT NULL,
            source_language TEXT NOT NULL,
            target_language TEXT NOT NULL,
            transcript_provider_id TEXT,
            model_id TEXT,
            prompt_variant_id TEXT,
            prompt_version TEXT,
            combo_key TEXT,
            resolved_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
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
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS translation_feedback_stats (
            combo_key TEXT PRIMARY KEY,
            stats_json JSONB NOT NULL,
            source_language TEXT NOT NULL,
            target_language TEXT NOT NULL,
            transcript_provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_variant_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_translation_feedback_stats_lookup
        ON translation_feedback_stats(
            source_language,
            target_language,
            transcript_provider_id,
            model_id,
            prompt_variant_id,
            prompt_version
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_translation_feedback_stats_lookup")
    op.execute("DROP TABLE IF EXISTS translation_feedback_stats")
    op.execute("DROP INDEX IF EXISTS idx_human_review_resolutions_lookup")
    op.execute("DROP TABLE IF EXISTS human_review_resolutions")
