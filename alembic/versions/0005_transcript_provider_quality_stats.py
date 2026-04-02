from __future__ import annotations

from alembic import op

revision = "0005_provider_quality_stats"
down_revision = "0004_prompt_proposal_compat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_provider_quality_stats (
            provider_id TEXT NOT NULL,
            source_language TEXT NOT NULL,
            target_language TEXT NOT NULL,
            stats_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (provider_id, source_language, target_language)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS transcript_provider_quality_stats")
