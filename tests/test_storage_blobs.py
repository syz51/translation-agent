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


@pytest.mark.unit
def test_blob_delete_and_iter_entries_respect_prefix(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    store.put_bytes("audio/run-1/raw.bin", b"raw")
    store.put_bytes("audio/run-1/processed.bin", b"processed")
    store.put_bytes("trace/run-1.jsonl", b"trace")

    entries = list(store.iter_entries(prefix="audio/run-1/"))

    assert [entry.key for entry in entries] == [
        "audio/run-1/processed.bin",
        "audio/run-1/raw.bin",
    ]
    assert [entry.size_bytes for entry in entries] == [9, 3]

    store.delete("audio/run-1/raw.bin")
    store.delete("audio/run-1/raw.bin")

    assert store.exists("audio/run-1/raw.bin") is False
    assert store.list_keys(prefix="audio/") == ["audio/run-1/processed.bin"]


@pytest.mark.unit
@pytest.mark.parametrize("key", ["", "/absolute.bin"])
def test_blob_rejects_invalid_key_shapes(tmp_path: Path, key: str) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    with pytest.raises(ValueError):
        store.put_bytes(key, b"invalid")
