"""Observability primitives for structured logging and tracing."""

from .events import (
    STRUCTLOG_AVAILABLE,
    StructuredEvent,
    configure_structured_logging,
    get_structured_logger,
    log_structured_event,
)
from .tracing import CompositeTraceSink, JsonlTraceSink, NoOpTraceSink, TraceEvent, TraceSink

__all__ = [
    "STRUCTLOG_AVAILABLE",
    "StructuredEvent",
    "configure_structured_logging",
    "get_structured_logger",
    "log_structured_event",
    "CompositeTraceSink",
    "JsonlTraceSink",
    "NoOpTraceSink",
    "TraceEvent",
    "TraceSink",
]
