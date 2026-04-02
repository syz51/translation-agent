from __future__ import annotations

from alembic import op

revision = "0003_reference_evaluation_assets"
down_revision = "0002_operational_entities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_records (
            media_key TEXT PRIMARY KEY,
            asset_id TEXT UNIQUE,
            media_fingerprint TEXT UNIQUE,
            asset_json JSONB NOT NULL,
            first_seen_run_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_records_asset_id
        ON asset_records(asset_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_records_media_fingerprint
        ON asset_records(media_fingerprint)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_run_links (
            run_id TEXT PRIMARY KEY,
            media_key TEXT NOT NULL REFERENCES asset_records(media_key) ON DELETE CASCADE,
            link_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_historical_run_links_media_key_created_at
        ON historical_run_links(media_key, created_at, run_id)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_evolution_proposals (
            proposal_id TEXT PRIMARY KEY,
            proposal_json JSONB NOT NULL,
            status TEXT NOT NULL,
            target_model_id TEXT NOT NULL,
            target_language TEXT,
            source_language TEXT,
            media_key TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prompt_evolution_proposals_scope
        ON prompt_evolution_proposals(status, target_model_id, target_language, media_key)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_prompt_evolution_proposals_scope")
    op.execute("DROP TABLE IF EXISTS prompt_evolution_proposals")
    op.execute("DROP INDEX IF EXISTS idx_historical_run_links_media_key_created_at")
    op.execute("DROP TABLE IF EXISTS historical_run_links")
    op.execute("DROP INDEX IF EXISTS idx_asset_records_media_fingerprint")
    op.execute("DROP INDEX IF EXISTS idx_asset_records_asset_id")
    op.execute("DROP TABLE IF EXISTS asset_records")
