from __future__ import annotations

from alembic import op

revision = "0007_asset_context_memory_store"
down_revision = "0006_human_feedback_stats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS vector;
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'pgvector extension unavailable; continuing without vector columns';
        END
        $$;
        """
    )
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_context_records (
            media_key TEXT PRIMARY KEY REFERENCES asset_records(media_key) ON DELETE CASCADE,
            context_json JSONB NOT NULL,
            canonical_title TEXT,
            series_id TEXT,
            franchise_id TEXT,
            channel_id TEXT,
            content_type TEXT,
            style_profile_id TEXT,
            metadata_confidence TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_relations (
            relation_id TEXT PRIMARY KEY,
            src_media_key TEXT NOT NULL REFERENCES asset_records(media_key) ON DELETE CASCADE,
            dst_media_key TEXT NOT NULL REFERENCES asset_records(media_key) ON DELETE CASCADE,
            relation_kind TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            relation_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            memory_id TEXT PRIMARY KEY,
            entry_json JSONB NOT NULL,
            dedupe_key TEXT UNIQUE,
            scope_kind TEXT,
            scope_key TEXT,
            series_id TEXT,
            franchise_id TEXT,
            content_type TEXT,
            style_profile_id TEXT,
            promotion_status TEXT NOT NULL,
            lifecycle_status TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            typed_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            search_document TSVECTOR,
            embedding_model_id TEXT,
            embedding_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            embedding_updated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_evidence_events (
            event_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL REFERENCES memory_entries(memory_id) ON DELETE CASCADE,
            event_kind TEXT NOT NULL,
            run_id TEXT,
            job_id TEXT,
            media_key TEXT,
            stage TEXT,
            source_ref TEXT,
            event_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_context_records_series
        ON asset_context_records(series_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_context_records_franchise
        ON asset_context_records(franchise_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_context_records_title_trgm
        ON asset_context_records USING GIN (canonical_title gin_trgm_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_relations_src_kind
        ON asset_relations(src_media_key, relation_kind)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_asset_relations_dst_kind
        ON asset_relations(dst_media_key, relation_kind)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_scope
        ON memory_entries(scope_kind, scope_key)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_series
        ON memory_entries(series_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_franchise
        ON memory_entries(franchise_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_content_type
        ON memory_entries(content_type)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_style_profile
        ON memory_entries(style_profile_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_promotion_lifecycle
        ON memory_entries(promotion_status, lifecycle_status)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_typed_metadata
        ON memory_entries USING GIN (typed_metadata)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_entries_search_document
        ON memory_entries USING GIN (search_document)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_evidence_events_memory_id
        ON memory_evidence_events(memory_id, created_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_memory_evidence_events_memory_id")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_search_document")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_typed_metadata")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_promotion_lifecycle")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_style_profile")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_content_type")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_franchise")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_series")
    op.execute("DROP INDEX IF EXISTS idx_memory_entries_scope")
    op.execute("DROP INDEX IF EXISTS idx_asset_relations_dst_kind")
    op.execute("DROP INDEX IF EXISTS idx_asset_relations_src_kind")
    op.execute("DROP INDEX IF EXISTS idx_asset_context_records_title_trgm")
    op.execute("DROP INDEX IF EXISTS idx_asset_context_records_franchise")
    op.execute("DROP INDEX IF EXISTS idx_asset_context_records_series")
    op.execute("DROP TABLE IF EXISTS memory_evidence_events")
    op.execute("DROP TABLE IF EXISTS memory_entries")
    op.execute("DROP TABLE IF EXISTS asset_relations")
    op.execute("DROP TABLE IF EXISTS asset_context_records")
