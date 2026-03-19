from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

try:  # Optional dependency: keep the code ready for structlog when installed.
    structlog = import_module("structlog")
except ModuleNotFoundError:  # pragma: no cover - exercised by environment, not logic.
    structlog = None

STRUCTLOG_AVAILABLE = structlog is not None


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class StructuredEvent:
    event: str
    level: str = "info"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    fields: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {
            "event": self.event,
            "level": self.level,
            "timestamp": self.timestamp.isoformat(),
        }
        record.update(self.fields)
        return record


def configure_structured_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s", force=True)
    if structlog is None:
        return

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(serializer=json.dumps, sort_keys=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_structured_logger(name: str | None = None) -> Any:
    if structlog is not None:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def log_structured_event(
    logger: Any,
    event: str,
    *,
    level: str = "info",
    **fields: Any,
) -> StructuredEvent:
    structured_event = StructuredEvent(event=event, level=level, fields=dict(fields))
    payload = json.dumps(structured_event.to_record(), default=_json_default, sort_keys=True)

    if structlog is not None and hasattr(logger, "bind"):
        bound_logger = logger.bind(**fields)
        getattr(bound_logger, level, bound_logger.info)(event)
    else:
        getattr(logger, level, logger.info)(payload)

    return structured_event
