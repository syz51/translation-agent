"""Helpers for durable media identity resolution."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def compute_media_fingerprint(source: str) -> str | None:
    """Return a stable content fingerprint when the source is readable locally."""

    path = Path(source).expanduser()
    if not path.exists() or not path.is_file():
        return None

    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def fallback_media_key_seed(source: str) -> str:
    """Return a deterministic fallback seed when no durable identifiers are available."""

    normalized_source = str(Path(source).expanduser())
    return sha256(normalized_source.encode("utf-8")).hexdigest()[:24]
