from __future__ import annotations

from pathlib import Path

import pytest

from translation_agent.config import sanitize_db_target
from translation_agent.storage import LocalBlobStore

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        (
            "postgresql://user:secret@db.example.com:5432/translation_agent?sslmode=require",
            "postgresql://db.example.com:5432/translation_agent",
        ),
        (
            "postgresql://user:secret@db.example.com/translation_agent?connect_timeout=1",
            "postgresql://db.example.com/translation_agent",
        ),
        ("not-a-dsn", "<invalid>"),
        (None, "<missing>"),
    ],
)
def test_sanitize_db_target_strips_secrets_and_noise_regression(
    dsn: str | None, expected: str
) -> None:
    assert sanitize_db_target(dsn) == expected


def test_blob_store_overwrite_does_not_leave_temporary_files_regression(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    store.put_bytes("jobs/run-1/request.json", b"first")
    store.put_bytes("jobs/run-1/request.json", b"second")

    assert store.read_bytes("jobs/run-1/request.json") == b"second"
    assert store.list_keys() == ["jobs/run-1/request.json"]
    assert not any(path.name.startswith(".tmp-blob-") for path in store.root.rglob("*"))
