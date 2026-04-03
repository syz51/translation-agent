from __future__ import annotations

from datetime import UTC, datetime

import pytest

from translation_agent.api import backfill_memory_embeddings
from translation_agent.config import Settings
from translation_agent.models import MemoryEntry
from translation_agent.storage import SQLiteOperationalStore

pytestmark = pytest.mark.unit


def test_backfill_memory_embeddings_updates_sqlite_runtime(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime")
    with SQLiteOperationalStore(settings.state_db_path) as store:
        asset = store.resolve_asset(
            asset_id="asset-backfill",
            media_fingerprint="sha256:asset-backfill",
            first_seen_run_id="run-backfill",
            source_language="en",
            target_language="fr",
        )
        store.put_memory_entry(
            MemoryEntry(
                memory_id="memory-backfill-1",
                kind="semantic",
                memory_subtype="project_fact",
                content="Keep workflow terminology stable across the series.",
                scope_kind="asset",
                scope_key=asset.media_key,
                updated_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
                score=0.8,
                metadata={"media_key": asset.media_key},
            )
        )

    result = backfill_memory_embeddings(settings=settings)

    assert result.updated_entries == 1
    assert result.state_backend == "sqlite"
