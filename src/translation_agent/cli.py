"""CLI entrypoint for the translation agent dry-run workflow."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from translation_agent.api import (
    RunJobRequest,
    RunJobResult,
    approve_review,
    convert_translation_json_to_srt,
    get_run_status,
    list_runs,
    resolve_review,
    resume_transcription,
    resume_translation,
    review_job,
    run_job,
    save_review_draft,
)
from translation_agent.config import load_settings, sanitize_db_target, validate_environment
from translation_agent.observability import TraceEvent, TraceSink
from translation_agent.run_status import (
    PhaseCounters,
    RunStatusAccumulator,
    RunStatusSnapshot,
    is_terminal_run_status,
)
from translation_agent.storage.migrations import upgrade_database
from translation_agent.tui import ReviewTerminalApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate local runtime config")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    migrate_parser = subparsers.add_parser("migrate-db", help="Apply Postgres migrations")
    migrate_parser.add_argument("--revision", default="head")

    list_runs_parser = subparsers.add_parser("list-runs", help="List persisted workflow runs")
    list_runs_parser.add_argument("--json", action="store_true", dest="as_json")

    show_run_parser = subparsers.add_parser("show-run", help="Show one persisted run snapshot")
    show_run_parser.add_argument("run_id")
    show_run_parser.add_argument("--json", action="store_true", dest="as_json")

    watch_run_parser = subparsers.add_parser(
        "watch-run",
        help="Watch one persisted run until it reaches a terminal state",
    )
    watch_run_parser.add_argument("run_id")
    watch_run_parser.add_argument("--interval", type=float, default=0.5)

    convert_parser = subparsers.add_parser(
        "convert-json-to-srt",
        help="Convert a persisted translation JSON artifact into SRT",
    )
    convert_parser.add_argument("source")
    convert_parser.add_argument("--output")
    convert_parser.add_argument("--json", action="store_true", dest="as_json")

    run_parser = subparsers.add_parser("run-job", help="Execute the local dry-run workflow")
    run_parser.add_argument("source")
    run_parser.add_argument("--job-id")
    run_parser.add_argument("--asset-id")
    run_parser.add_argument("--reference-transcript-source")
    run_parser.add_argument("--reference-transcript-format", choices=["srt"])
    run_parser.add_argument(
        "--reference-mode",
        choices=["none", "evaluate_and_regenerate"],
        default="none",
    )
    run_parser.add_argument("--source-language")
    run_parser.add_argument("--target-language")
    run_parser.add_argument(
        "--review",
        choices=["auto", "always", "never"],
        default="auto",
    )
    run_parser.add_argument("--json", action="store_true", dest="as_json")

    review_parser = subparsers.add_parser(
        "review-job",
        help="Open the exception-only flagged-span compare tool for a pending translation review",
    )
    review_parser.add_argument("run_id")
    review_parser.add_argument("--json", action="store_true", dest="as_json")

    resume_parser = subparsers.add_parser(
        "resume-translation",
        help="Resume translation from persisted transcript candidates of a prior run",
    )
    resume_parser.add_argument("run_id")
    resume_parser.add_argument(
        "--review",
        choices=["auto", "always", "never"],
        default="auto",
    )
    resume_parser.add_argument("--json", action="store_true", dest="as_json")

    resume_transcription_parser = subparsers.add_parser(
        "resume-transcription",
        help="Resume transcription from persisted audio and rerun selected providers",
    )
    resume_transcription_parser.add_argument("run_id")
    resume_transcription_parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        default=[],
        help="Provider ID to rerun; repeat to target multiple providers",
    )
    resume_transcription_parser.add_argument(
        "--review",
        choices=["auto", "always", "never"],
        default="auto",
    )
    resume_transcription_parser.add_argument("--json", action="store_true", dest="as_json")

    approve_parser = subparsers.add_parser(
        "approve-review",
        help="Compatibility surface: publish a pending review by choosing one base candidate",
    )
    approve_parser.add_argument("run_id")
    approve_parser.add_argument("--candidate-id", required=True)
    approve_parser.add_argument("--approved-by")
    approve_parser.add_argument("--note")
    approve_parser.add_argument("--json", action="store_true", dest="as_json")

    resolve_parser = subparsers.add_parser(
        "resolve-review",
        help="Compatibility surface: resolve a pending review with internal supervision enums",
    )
    resolve_parser.add_argument("run_id")
    resolve_parser.add_argument(
        "--resolution",
        required=True,
        choices=["approved_good", "approved_best_available", "rejected_all"],
    )
    resolve_parser.add_argument("--candidate-id")
    resolve_parser.add_argument(
        "--failure-tag",
        action="append",
        dest="failure_tags",
        default=[],
    )
    resolve_parser.add_argument(
        "--reviewed-span-decisions-path",
        help="Path to a JSON array of ReviewedSpanDecision payloads",
    )
    resolve_parser.add_argument("--approved-by")
    resolve_parser.add_argument("--note")
    resolve_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        settings = load_settings()
        result = validate_environment(settings)
        payload = {
            "ok": result.ok,
            "checked_paths": [str(path) for path in result.checked_paths],
            "state_backend": result.state_backend,
            "state_db_ok": result.state_db_ok,
            "state_db_target": result.state_db_target,
            "adapter_mode": result.adapter_mode,
            "runtime_compatibility_ok": result.runtime_compatibility_ok,
            "runtime_compatibility_error": result.runtime_compatibility_error,
            "provider_config_ok": result.provider_config_ok,
            "provider_config_error": result.provider_config_error,
            "state_db_error": result.state_db_error,
        }
        if args.as_json:
            print(json.dumps(payload))
        else:
            db_connectivity = (
                "database connectivity ok" if result.state_db_ok else "database connectivity failed"
            )
            print("configuration valid" if result.ok else "configuration invalid")
            for path in payload["checked_paths"]:
                print(path)
            print(f"{result.state_backend}: {result.state_db_target}")
            print(f"adapter_mode: {result.adapter_mode}")
            print(db_connectivity)
            print(
                "runtime compatibility ok"
                if result.runtime_compatibility_ok
                else "runtime compatibility failed"
            )
            print(
                "provider configuration ok"
                if result.provider_config_ok
                else "provider configuration failed"
            )
            if result.state_db_error:
                print(result.state_db_error)
        return 0 if result.ok else 1

    if args.command == "migrate-db":
        settings = load_settings()
        if settings.state_db_dsn is None:
            print("TA_STATE_DB_DSN is required for migrate-db")
            return 1
        upgrade_database(settings.state_db_dsn, revision=args.revision)
        print(f"migrated {sanitize_db_target(settings.state_db_dsn)} to {args.revision}")
        return 0

    if args.command == "list-runs":
        settings = load_settings()
        records = list_runs(settings=settings)
        payload = [asdict(record) for record in records]
        if args.as_json:
            print(json.dumps(_json_ready(payload)))
        else:
            _print_run_listing(payload)
        return 0

    if args.command == "show-run":
        settings = load_settings()
        snapshot = get_run_status(args.run_id, settings=settings)
        if args.as_json:
            print(json.dumps(_json_ready(asdict(snapshot))))
        else:
            _print_run_status_snapshot(snapshot)
        return 0

    if args.command == "watch-run":
        settings = load_settings()
        return _watch_run(args.run_id, interval_seconds=args.interval, settings=settings)

    if args.command == "convert-json-to-srt":
        result = convert_translation_json_to_srt(args.source, output_path=args.output)
        payload = {
            key: str(value) if hasattr(value, "__fspath__") else value
            for key, value in asdict(result).items()
        }
        if args.as_json:
            print(json.dumps(payload))
        else:
            print(result.output_path)
            print(f"subtitles: {result.subtitle_count}")
        return 0

    if args.command == "run-job":
        settings = load_settings()
        result = _run_with_optional_live_panel(
            run_job,
            settings=settings,
            enabled=not args.as_json,
            request=RunJobRequest(
                source=args.source,
                job_id=args.job_id,
                asset_id=args.asset_id,
                source_language=args.source_language,
                target_language=args.target_language,
                reference_transcript_source=args.reference_transcript_source,
                reference_transcript_format=args.reference_transcript_format,
                reference_mode=args.reference_mode,
                review_mode=args.review,
            ),
        )
        payload = _json_ready(asdict(result))
        if args.as_json:
            print(json.dumps(payload))
        else:
            _print_run_job_result(result)
            if result.review_required_stage == "translation":
                if _should_enter_interactive_review(mode=args.review):
                    return _interactive_review_flow(result.run_id, settings=settings)
                if args.review == "always" and not _has_tty():
                    print("interactive review requires a real TTY")
                    return 2
                print("review_required_stage: translation")
                for command in result.resume_commands:
                    print(command)
        return 0

    if args.command == "review-job":
        settings = load_settings()
        payload = review_job(args.run_id, settings=settings)
        if args.as_json:
            print(json.dumps(_json_ready(payload)))
            return 0
        return _interactive_review_flow(args.run_id, settings=settings, initial_payload=payload)

    if args.command == "resume-translation":
        settings = load_settings()
        result = _run_with_optional_live_panel(
            resume_translation,
            settings=settings,
            enabled=not args.as_json,
            run_id=args.run_id,
            review_mode=args.review,
        )
        payload = _json_ready(asdict(result))
        if args.as_json:
            print(json.dumps(payload))
        else:
            _print_run_job_result(result)
            if result.review_required_stage == "translation":
                if _should_enter_interactive_review(mode=args.review):
                    return _interactive_review_flow(result.run_id, settings=settings)
                if args.review == "always" and not _has_tty():
                    print("interactive review requires a real TTY")
                    return 2
                print("review_required_stage: translation")
                for command in result.resume_commands:
                    print(command)
        return 0

    if args.command == "resume-transcription":
        settings = load_settings()
        result = _run_with_optional_live_panel(
            resume_transcription,
            settings=settings,
            enabled=not args.as_json,
            run_id=args.run_id,
            provider_ids=tuple(args.providers),
            review_mode=args.review,
        )
        payload = _json_ready(asdict(result))
        if args.as_json:
            print(json.dumps(payload))
        else:
            _print_run_job_result(result)
            if result.review_required_stage == "translation":
                if _should_enter_interactive_review(mode=args.review):
                    return _interactive_review_flow(result.run_id, settings=settings)
                if args.review == "always" and not _has_tty():
                    print("interactive review requires a real TTY")
                    return 2
                print("review_required_stage: translation")
                for command in result.resume_commands:
                    print(command)
        return 0

    if args.command == "approve-review":
        settings = load_settings()
        payload = approve_review(
            args.run_id,
            candidate_id=args.candidate_id,
            approved_by=args.approved_by,
            note=args.note,
            settings=settings,
        )
        if args.as_json:
            print(json.dumps(_json_ready(payload)))
        else:
            print(payload["run_id"])
            print(payload["status"])
            print(f"approved_candidate_id: {payload['approved_candidate_id']}")
            print(
                "approved_source_transcript_candidate_id: "
                f"{payload['approved_source_transcript_candidate_id']}"
            )
            print(f"approval_ref: {payload['approval_ref']}")
            print(f"default_output_path: {payload['default_output_path']}")
        return 0

    if args.command == "resolve-review":
        settings = load_settings()
        reviewed_span_decisions = ()
        if args.reviewed_span_decisions_path:
            reviewed_span_decisions = tuple(
                item
                for item in json.loads(
                    Path(args.reviewed_span_decisions_path).read_text(encoding="utf-8")
                )
                if isinstance(item, dict)
            )
        payload = resolve_review(
            args.run_id,
            resolution=args.resolution,
            candidate_id=args.candidate_id,
            reviewed_span_decisions=reviewed_span_decisions,
            failure_tags=tuple(args.failure_tags),
            approved_by=args.approved_by,
            note=args.note,
            settings=settings,
        )
        if args.as_json:
            print(json.dumps(_json_ready(payload)))
        else:
            print(payload["run_id"])
            print(payload["status"])
            print(f"resolution_kind: {payload['resolution_kind']}")
            print(f"resolution_ref: {payload['resolution_ref']}")
            if payload.get("approval_ref") is not None:
                print(f"approval_ref: {payload['approval_ref']}")
            if payload.get("approved_candidate_id") is not None:
                print(f"approved_candidate_id: {payload['approved_candidate_id']}")
            if payload.get("default_output_path") is not None:
                print(f"default_output_path: {payload['default_output_path']}")
            if payload.get("failure_tags"):
                failure_tags = payload["failure_tags"]
                if isinstance(failure_tags, list):
                    print("failure_tags: " + ", ".join(str(tag) for tag in failure_tags))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


def _interactive_review_flow(
    run_id: str,
    *,
    settings,
    initial_payload: dict[str, Any] | None = None,
) -> int:
    if not _has_tty():
        print("interactive review requires a real TTY")
        return 2
    payload = initial_payload or review_job(run_id, settings=settings)
    app = ReviewTerminalApp(
        run_id=run_id,
        payload=payload,
        settings=settings,
        save_review_draft=save_review_draft,
        resolve_review=resolve_review,
    )
    app.run()
    return 0


class _LiveRunStatusPanel(TraceSink):
    def __init__(self, *, output, refresh_interval: float = 0.2) -> None:
        self._output = output
        self._refresh_interval = refresh_interval
        self._accumulator = RunStatusAccumulator()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread_started = False

    def record(self, event: TraceEvent) -> None:
        with self._lock:
            self._accumulator.apply_trace_event(event.to_record())
            if not self._thread_started:
                self._thread.start()
                self._thread_started = True
        if self._is_terminal_event(event):
            self.close()

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._thread_started and self._thread.is_alive():
            self._thread.join(timeout=max(self._refresh_interval * 2, 0.1))
        with self._lock:
            snapshot = self._accumulator.snapshot()
        self._draw(snapshot)

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(self._refresh_interval):
            with self._lock:
                snapshot = self._accumulator.snapshot()
            self._draw(snapshot)

    def _draw(self, snapshot: RunStatusSnapshot) -> None:
        self._output.write("\x1b[2J\x1b[H")
        self._output.write(_format_run_status_panel(snapshot))
        self._output.flush()

    @staticmethod
    def _is_terminal_event(event: TraceEvent) -> bool:
        return event.name in {"run.completed", "run.failed"}


def _run_with_optional_live_panel(
    fn,
    *,
    settings,
    enabled: bool = True,
    **kwargs: Any,
) -> RunJobResult:
    if not enabled or not _has_tty():
        return _invoke_with_optional_live_sink(fn, settings=settings, **kwargs)
    panel = _LiveRunStatusPanel(output=sys.stdout)
    try:
        return _invoke_with_optional_live_sink(
            fn,
            settings=settings,
            live_trace_sink=panel,
            **kwargs,
        )
    finally:
        panel.close()
        sys.stdout.write("\n")
        sys.stdout.flush()


def _invoke_with_optional_live_sink(
    fn,
    *,
    settings,
    live_trace_sink: TraceSink | None = None,
    **kwargs: Any,
) -> RunJobResult:
    signature = inspect.signature(fn)
    normalized_kwargs = _normalize_invocation_kwargs(signature, kwargs)
    parameters = signature.parameters.values()
    accepts_live_trace_sink = any(
        parameter.name == "live_trace_sink" for parameter in parameters
    ) or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if live_trace_sink is not None and accepts_live_trace_sink:
        return fn(settings=settings, live_trace_sink=live_trace_sink, **normalized_kwargs)
    return fn(settings=settings, **normalized_kwargs)


def _normalize_invocation_kwargs(
    signature: inspect.Signature,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(kwargs)
    parameters = signature.parameters
    if "run_id" in normalized and "run_id" not in parameters and "source_run_id" in parameters:
        normalized["source_run_id"] = normalized.pop("run_id")
    if (
        "source_run_id" in normalized
        and "source_run_id" not in parameters
        and "run_id" in parameters
    ):
        normalized["run_id"] = normalized.pop("source_run_id")
    return normalized


def _watch_run(run_id: str, *, interval_seconds: float, settings) -> int:
    interval = max(0.1, interval_seconds)
    if not _has_tty():
        while True:
            snapshot = get_run_status(run_id, settings=settings)
            if is_terminal_run_status(snapshot.status):
                _print_run_status_snapshot(snapshot)
                return 0
            time.sleep(interval)

    while True:
        snapshot = get_run_status(run_id, settings=settings)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(_format_run_status_panel(snapshot))
        sys.stdout.flush()
        if is_terminal_run_status(snapshot.status):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return 0
        time.sleep(interval)


def _print_run_job_result(result: RunJobResult) -> None:
    print(result.run_id)
    print(result.status)
    print(f"source_language: {result.source_language}")
    print(f"target_language: {result.target_language}")
    print(f"{result.state_backend}: {result.state_db_target}")
    print(result.trace_path)
    if result.default_output_path is not None:
        print(f"default_output_path: {result.default_output_path}")
    if result.failure_summary:
        print(result.failure_summary)
    for reason in result.failure_reasons:
        print(reason)


def _print_run_status_snapshot(snapshot: RunStatusSnapshot) -> None:
    print(snapshot.run_id)
    print(snapshot.status)
    print(f"elapsed_seconds: {snapshot.elapsed_seconds:.1f}")
    print(f"current_stage: {snapshot.current_stage or '-'}")
    if snapshot.active_node is not None:
        print(f"active_node: {snapshot.active_node}")
    print(snapshot.trace_path)
    for counter_line in _counter_lines(snapshot):
        print(counter_line)
    if snapshot.recent_events:
        print("recent_events:")
        for event in snapshot.recent_events:
            print(f"- {_event_time_label(event.timestamp)} {event.message}")


def _format_run_status_panel(snapshot: RunStatusSnapshot) -> str:
    lines = [
        f"run: {snapshot.run_id or '-'}",
        f"status: {snapshot.status} | elapsed: {snapshot.elapsed_seconds:.1f}s",
        f"stage: {snapshot.current_stage or '-'} | active_node: {snapshot.active_node or '-'}",
        f"trace: {snapshot.trace_path}",
    ]
    lines.extend(_counter_lines(snapshot))
    if snapshot.recent_events:
        lines.append("recent events:")
        for event in snapshot.recent_events:
            lines.append(f"- {_event_time_label(event.timestamp)} {event.message}")
    return "\n".join(lines) + "\n"


def _counter_lines(snapshot: RunStatusSnapshot) -> list[str]:
    lines: list[str] = []
    if snapshot.transcription_providers is not None:
        lines.append(_format_counter_line("providers", snapshot.transcription_providers))
    if snapshot.translation_variants is not None:
        lines.append(_format_counter_line("translation_variants", snapshot.translation_variants))
    if snapshot.review_bundles is not None:
        lines.append(_format_counter_line("review_bundles", snapshot.review_bundles))
    return lines


def _format_counter_line(label: str, counters: PhaseCounters) -> str:
    total = "-" if counters.total is None else str(counters.total)
    return (
        f"{label}: active={counters.active} "
        f"completed={counters.completed} failed={counters.failed} total={total}"
    )


def _event_time_label(timestamp: str) -> str:
    if len(timestamp) >= 19:
        return timestamp[11:19]
    return timestamp


def _should_enter_interactive_review(*, mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return _has_tty()
    return _has_tty()


def _has_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_run_listing(records: list[dict[str, Any]]) -> None:
    if not records:
        print("no runs found")
        return
    for record in records:
        input_data = record.get("input_data")
        job_id = None
        source = None
        if isinstance(input_data, dict):
            job_id = input_data.get("job_id")
            source = input_data.get("source")
        line = (
            f"{record['run_id']} "
            f"status={record['status']} "
            f"job_id={job_id or '-'} "
            f"created_at={record['created_at']}"
        )
        if source:
            line = f"{line} source={source}"
        print(line)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value
