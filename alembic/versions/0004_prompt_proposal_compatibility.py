from __future__ import annotations

from alembic import op

revision = "0004_prompt_proposal_compat"
down_revision = "0003_reference_evaluation_assets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        ADD COLUMN IF NOT EXISTS prompt_family TEXT NOT NULL DEFAULT 'translation'
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        ADD COLUMN IF NOT EXISTS prompt_variant_id TEXT
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        ADD COLUMN IF NOT EXISTS base_prompt_version TEXT
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        ADD COLUMN IF NOT EXISTS scope_kind TEXT
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        ADD COLUMN IF NOT EXISTS scope_key TEXT
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_prompt_evolution_proposals_scope")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prompt_evolution_proposals_scope
        ON prompt_evolution_proposals(
            status,
            prompt_family,
            target_model_id,
            source_language,
            target_language,
            prompt_variant_id,
            base_prompt_version,
            scope_kind,
            scope_key
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_prompt_evolution_proposals_scope")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prompt_evolution_proposals_scope
        ON prompt_evolution_proposals(status, target_model_id, target_language, media_key)
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        DROP COLUMN IF EXISTS scope_key
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        DROP COLUMN IF EXISTS scope_kind
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        DROP COLUMN IF EXISTS base_prompt_version
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        DROP COLUMN IF EXISTS prompt_variant_id
        """
    )
    op.execute(
        """
        ALTER TABLE prompt_evolution_proposals
        DROP COLUMN IF EXISTS prompt_family
        """
    )
