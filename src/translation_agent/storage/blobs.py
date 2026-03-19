from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BlobEntry:
    key: str
    path: Path
    size_bytes: int


class LocalBlobStore:
    """Local filesystem blob store for Phase 0."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> BlobEntry:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=path.parent, prefix=".tmp-blob-"
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
        return BlobEntry(key=key, path=path, size_bytes=path.stat().st_size)

    def read_bytes(self, key: str) -> bytes:
        return self._path_for_key(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path_for_key(key).exists()

    def delete(self, key: str) -> None:
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()

    def list_keys(self, prefix: str | None = None) -> list[str]:
        keys = [self._relative_key(path) for path in sorted(self.root.rglob("*")) if path.is_file()]
        if prefix is None:
            return keys
        return [key for key in keys if key.startswith(prefix)]

    def iter_entries(self, prefix: str | None = None) -> Iterator[BlobEntry]:
        for key in self.list_keys(prefix=prefix):
            path = self._path_for_key(key)
            yield BlobEntry(key=key, path=path, size_bytes=path.stat().st_size)

    def _path_for_key(self, key: str) -> Path:
        relative = self._validate_key(key)
        return self.root / relative

    def _relative_key(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _validate_key(self, key: str) -> Path:
        if not key:
            raise ValueError("blob key must be non-empty")
        relative = Path(key)
        if relative.is_absolute():
            raise ValueError("blob key must be relative")
        if any(part in {"..", ".", ""} for part in relative.parts):
            raise ValueError("blob key must not traverse directories")
        return relative
