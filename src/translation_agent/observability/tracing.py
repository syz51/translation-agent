from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .events import _json_default


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


class TraceSink(ABC):
    @abstractmethod
    def record(self, event: TraceEvent) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> TraceSink:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class NoOpTraceSink(TraceSink):
    def record(self, event: TraceEvent) -> None:
        return None


class JsonlTraceSink(TraceSink):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._lock = Lock()

    def record(self, event: TraceEvent) -> None:
        payload = json.dumps(event.to_record(), default=_json_default, sort_keys=True)
        with self._lock:
            self._handle.write(payload)
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()
