from pathlib import Path

import pytest

from translation_agent.storage import LocalBlobStore

pytestmark = pytest.mark.unit


def test_blob_round_trip_and_listing(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    entry = store.put_bytes("audio/run-1/raw.bin", b"payload")

    assert entry.key == "audio/run-1/raw.bin"
    assert entry.path.read_bytes() == b"payload"
    assert store.read_bytes("audio/run-1/raw.bin") == b"payload"
    assert store.list_keys() == ["audio/run-1/raw.bin"]
    assert store.list_keys(prefix="audio/") == ["audio/run-1/raw.bin"]


def test_blob_rejects_traversal(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    with pytest.raises(ValueError):
        store.put_bytes("../escape.bin", b"nope")
