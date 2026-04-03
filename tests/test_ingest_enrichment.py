from __future__ import annotations

from pathlib import Path

import pytest

from translation_agent.api import RunJobRequest, run_job
from translation_agent.config import Settings
from translation_agent.storage import SQLiteOperationalStore

pytestmark = pytest.mark.unit


def test_run_job_derives_asset_context_and_same_series_relations(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime")
    first_source = tmp_path / "Space Market S01E01 Pilot.mp4"
    second_source = tmp_path / "Space Market S01E02 Earnings.mp4"
    first_source.write_bytes(b"episode-one")
    second_source.write_bytes(b"episode-two")

    first_result = run_job(
        RunJobRequest(
            source=str(first_source),
            asset_id="asset-episode-1",
            target_language="fr",
        ),
        settings=settings,
    )
    second_result = run_job(
        RunJobRequest(
            source=str(second_source),
            asset_id="asset-episode-2",
            target_language="fr",
        ),
        settings=settings,
    )

    with SQLiteOperationalStore(settings.state_db_path) as store:
        first_run = store.get_run(first_result.run_id)
        second_run = store.get_run(second_result.run_id)
        assert first_run is not None
        assert second_run is not None
        first_media_key = str(first_run.input_data["media_key"])
        second_media_key = str(second_run.input_data["media_key"])
        first_context = store.get_asset_context(first_media_key)
        second_context = store.get_asset_context(second_media_key)
        relations = store.list_asset_relations(second_media_key)

    assert first_context is not None
    assert second_context is not None
    assert first_context.series_id == "space-market"
    assert second_context.series_id == "space-market"
    assert second_context.content_type == "episode"
    assert second_context.episode_number == 2
    assert second_context.style_profile_id is not None
    assert any(relation.relation_kind == "same_series" for relation in relations)
