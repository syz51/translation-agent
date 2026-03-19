from __future__ import annotations

import json
import logging
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translation_agent.observability import (  # noqa: E402
    STRUCTLOG_AVAILABLE,
    JsonlTraceSink,
    NoOpTraceSink,
    StructuredEvent,
    TraceEvent,
    configure_structured_logging,
    get_structured_logger,
    log_structured_event,
)


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


if __name__ == "__main__":
    main()
