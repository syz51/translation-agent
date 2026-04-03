from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import cast
from unittest import TestCase, main

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translation_agent.observability import (  # noqa: E402
    STRUCTLOG_AVAILABLE,
    CompositeTraceSink,
    JsonlTraceSink,
    NoOpTraceSink,
    StructuredEvent,
    TraceEvent,
    configure_structured_logging,
    get_structured_logger,
    log_structured_event,
)
from translation_agent.observability import events as events_module  # noqa: E402

pytestmark = pytest.mark.unit


class ObservabilityTests(TestCase):
    def test_structured_event_flattens_fields(self) -> None:
        event = StructuredEvent(event="translation.started", fields={"run_id": "run-1", "step": 2})

        record = event.to_record()

        self.assertEqual(record["event"], "translation.started")
        self.assertEqual(record["level"], "info")
        self.assertEqual(record["run_id"], "run-1")
        self.assertEqual(record["step"], 2)

    def test_jsonl_trace_sink_writes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sink_path = Path(tmp_dir) / "trace.jsonl"

            with JsonlTraceSink(sink_path) as sink:
                sink.record(TraceEvent(name="node.enter", attributes={"node": "extract"}))

            payload = sink_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(payload), 1)
            record = json.loads(payload[0])
            self.assertEqual(record["name"], "node.enter")
            self.assertEqual(record["attributes"], {"node": "extract"})
            self.assertIsNone(record["trace_id"])

    def test_noop_trace_sink_ignores_events(self) -> None:
        sink = NoOpTraceSink()

        sink.record(TraceEvent(name="node.exit"))

    def test_structured_logging_falls_back_without_structlog(self) -> None:
        configure_structured_logging()
        logger = logging.getLogger("translation-agent.tests")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        try:
            log_structured_event(logger, "job.started", run_id="run-1", source="test")
        finally:
            logger.removeHandler(handler)

        self.assertIn(STRUCTLOG_AVAILABLE, {True, False})
        payload = json.loads(stream.getvalue().strip())
        self.assertEqual(payload["event"], "job.started")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["source"], "test")

    def test_get_structured_logger_is_available(self) -> None:
        logger = get_structured_logger("translation-agent.tests")
        self.assertIsNotNone(logger)


def test_json_default_and_trace_helpers_cover_optional_paths(tmp_path: Path) -> None:
    now = datetime(2026, 3, 31, 12, 0, tzinfo=UTC)

    assert events_module._json_default(now) == now.isoformat()
    assert events_module._json_default(tmp_path) == str(tmp_path)
    assert events_module._json_default(123) == "123"

    event = TraceEvent(
        name="node.exit",
        timestamp=now,
        attributes={"path": tmp_path},
        run_id="run-1",
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id="parent-1",
    )
    record = event.to_record()

    assert record["timestamp"] == now.isoformat()
    assert record["trace_id"] == "trace-1"

    with NoOpTraceSink() as sink:
        assert sink.path is None
        sink.record(event)
        sink.close()


def test_jsonl_trace_sink_exposes_path_and_close_is_idempotent(tmp_path: Path) -> None:
    sink_path = tmp_path / "trace" / "events.jsonl"
    sink = JsonlTraceSink(sink_path)

    assert sink.path == sink_path
    sink.record(TraceEvent(name="node.enter"))
    sink.close()
    sink.close()

    payload = sink_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(payload) == 1


def test_composite_trace_sink_fan_outs_to_jsonl_and_live_sink(tmp_path: Path) -> None:
    sink_path = tmp_path / "trace" / "events.jsonl"
    live_events: list[str] = []

    class LiveSink:
        @property
        def path(self) -> None:
            return None

        def record(self, event: TraceEvent) -> None:
            live_events.append(event.name)

        def close(self) -> None:
            live_events.append("closed")

    sink = CompositeTraceSink(JsonlTraceSink(sink_path), LiveSink())
    sink.record(TraceEvent(name="run.started", run_id="run-1"))
    sink.close()

    payload = sink_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(payload) == 1
    assert live_events == ["run.started", "closed"]
    assert sink.path == sink_path


def test_structlog_paths_are_exercised(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeBoundLogger:
        def warning(self, event: str) -> None:
            calls["event"] = event
            calls["level"] = "warning"

        def info(self, event: str) -> None:
            calls["fallback"] = event

    class FakeLogger:
        def bind(self, **fields: object) -> FakeBoundLogger:
            calls["fields"] = fields
            return FakeBoundLogger()

    class FakeStructlog:
        class stdlib:
            filter_by_level = object()
            add_log_level = object()
            BoundLogger = object

            @staticmethod
            def LoggerFactory() -> str:
                return "factory"

        class processors:
            @staticmethod
            def TimeStamper(fmt: str, utc: bool) -> tuple[str, str, bool]:
                return ("timestamp", fmt, utc)

            @staticmethod
            def JSONRenderer(serializer, sort_keys: bool) -> tuple[str, bool]:
                return ("renderer", sort_keys)

        @staticmethod
        def configure(**kwargs: object) -> None:
            calls["configure"] = kwargs

        @staticmethod
        def get_logger(name: str | None = None) -> FakeLogger:
            calls["logger_name"] = name
            return FakeLogger()

    monkeypatch.setattr(events_module, "structlog", FakeStructlog)

    events_module.configure_structured_logging(logging.WARNING)
    logger = events_module.get_structured_logger("translation-agent.tests.structlog")
    event = events_module.log_structured_event(
        logger,
        "job.warning",
        level="warning",
        run_id="run-1",
    )

    assert event.event == "job.warning"
    assert calls["logger_name"] == "translation-agent.tests.structlog"
    assert calls["fields"] == {"run_id": "run-1"}
    assert calls["event"] == "job.warning"
    assert calls["level"] == "warning"
    assert "processors" in cast(dict[str, object], calls["configure"])


if __name__ == "__main__":
    main()
