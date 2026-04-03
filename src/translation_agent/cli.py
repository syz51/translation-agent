"""CLI entrypoint for the translation agent dry-run workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from translation_agent.api import (
    RunJobRequest,
    approve_review,
    convert_translation_json_to_srt,
    list_runs,
    resolve_review,
    resume_transcription,
    resume_translation,
    review_job,
    run_job,
    save_review_draft,
)
from translation_agent.config import load_settings, sanitize_db_target, validate_environment
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
        help="Inspect a pending translation review and optionally finalize a winner",
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
        help="Finalize a pending translation review with one winning candidate",
    )
    approve_parser.add_argument("run_id")
    approve_parser.add_argument("--candidate-id", required=True)
    approve_parser.add_argument("--approved-by")
    approve_parser.add_argument("--note")
    approve_parser.add_argument("--json", action="store_true", dest="as_json")

    resolve_parser = subparsers.add_parser(
        "resolve-review",
        help="Resolve a pending translation review with graded human supervision",
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
        result = run_job(
            RunJobRequest(
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
            settings=settings,
        )
        payload = _json_ready(asdict(result))
        if args.as_json:
            print(json.dumps(payload))
        else:
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
        result = resume_translation(args.run_id, review_mode=args.review, settings=settings)
        payload = _json_ready(asdict(result))
        if args.as_json:
            print(json.dumps(payload))
        else:
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
        result = resume_transcription(
            args.run_id,
            provider_ids=tuple(args.providers),
            review_mode=args.review,
            settings=settings,
        )
        payload = _json_ready(asdict(result))
        if args.as_json:
            print(json.dumps(payload))
        else:
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
