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
    review_job,
    run_job,
)
from translation_agent.config import load_settings, sanitize_db_target, validate_environment
from translation_agent.storage.migrations import upgrade_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate local runtime config")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    migrate_parser = subparsers.add_parser("migrate-db", help="Apply Postgres migrations")
    migrate_parser.add_argument("--revision", default="head")

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
        help="Inspect a pending translation review for an existing run",
    )
    review_parser.add_argument("run_id")
    review_parser.add_argument("--json", action="store_true", dest="as_json")

    approve_parser = subparsers.add_parser(
        "approve-review",
        help="Approve a pending translation review candidate",
    )
    approve_parser.add_argument("run_id")
    approve_parser.add_argument("--candidate-id", required=True)
    approve_parser.add_argument("--approved-by")
    approve_parser.add_argument("--note")
    approve_parser.add_argument("--json", action="store_true", dest="as_json")
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

    parser.error(f"unsupported command: {args.command}")
    return 2


def _interactive_review_flow(
    run_id: str,
    *,
    settings,
    initial_payload: dict[str, Any] | None = None,
) -> int:
    payload = initial_payload or review_job(run_id, settings=settings)
    print(payload["run_id"])
    print(payload["status"])
    if payload.get("summary"):
        print(payload["summary"])
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        if payload.get("review_available") is False:
            print("no pending translation review")
        return 0

    _print_human_review_summary(payload)
    while True:
        _print_candidate_list(candidates)
        raw_command = input("review command ([number]=details, a <number>=approve, q=quit): ")
        raw_command = raw_command.strip()
        if raw_command.lower() in {"q", "quit", "exit"}:
            return 0
        if raw_command.isdigit():
            index = int(raw_command) - 1
            if 0 <= index < len(candidates):
                _print_candidate_details(candidates[index], payload)
            continue
        if raw_command.lower().startswith("a "):
            index_text = raw_command[2:].strip()
            if not index_text.isdigit():
                print("approval target must be a candidate number")
                continue
            index = int(index_text) - 1
            if not (0 <= index < len(candidates)):
                print("candidate number out of range")
                continue
            candidate = candidates[index]
            approved_by = input("approved_by (blank uses current user): ").strip() or None
            note = input("note (optional): ").strip() or None
            approved = approve_review(
                run_id,
                candidate_id=str(candidate["candidate_id"]),
                approved_by=approved_by,
                note=note,
                settings=settings,
            )
            print(approved["status"])
            print(f"approval_ref: {approved['approval_ref']}")
            print(f"default_output_path: {approved['default_output_path']}")
            return 0
        print("unknown command")


def _print_candidate_list(candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        source = candidate.get("source_transcript", {})
        provider = source.get("provider_id") if isinstance(source, dict) else None
        contradiction_count = candidate.get("contradiction_count", 0)
        blocking_count = candidate.get("blocking_hard_contradiction_count", 0)
        print(
            f"{candidate['rank']}. {candidate['candidate_id']} "
            f"[{candidate['prompt_variant_id']}] "
            f"transcript={candidate['source_transcript_candidate_id']} "
            f"provider={provider or 'unknown'} "
            f"contradictions={contradiction_count} "
            f"blocking={blocking_count}"
        )


def _print_candidate_details(candidate: dict[str, Any], payload: dict[str, Any]) -> None:
    print(candidate["candidate_id"])
    print(f"prompt_variant_id: {candidate['prompt_variant_id']}")
    print(f"prompt_version: {candidate['prompt_version']}")
    print(f"model_id: {candidate['model_id']}")
    print(f"translation_preview_json_path: {candidate['translation_preview_json_path']}")
    print(f"translation_preview_srt_path: {candidate['translation_preview_srt_path']}")
    source = candidate.get("source_transcript", {})
    if isinstance(source, dict):
        print(f"source_transcript_candidate_id: {source.get('candidate_id')}")
        print(f"source_transcript_provider_id: {source.get('provider_id')}")
        if source.get("transcript_preview_json_path"):
            print(f"transcript_preview_json_path: {source['transcript_preview_json_path']}")
        if source.get("transcript_preview_txt_path"):
            print(f"transcript_preview_txt_path: {source['transcript_preview_txt_path']}")
    review_summary = payload.get("transcript_review_summary", {})
    if isinstance(review_summary, dict):
        print(f"transcript_decision_ref: {review_summary.get('decision_ref')}")
        print(f"transcript_investigation_ref: {review_summary.get('investigation_ref')}")
    _print_reviewer_preferences(candidate)
    _print_candidate_contradictions(candidate)


def _print_human_review_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("human_review_summary", {})
    if not isinstance(summary, dict):
        return
    contradiction_count = summary.get("contradiction_count")
    blocking_count = summary.get("blocking_hard_contradiction_count")
    if contradiction_count is not None or blocking_count is not None:
        print(
            "review_summary: "
            f"contradictions={contradiction_count or 0} "
            f"blocking_hard={blocking_count or 0}"
        )
    preferences = summary.get("reviewer_preferences")
    if not isinstance(preferences, list) or not preferences:
        return
    print("reviewer_preferences:")
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        preferred_candidate_id = preference.get("preferred_candidate_id") or "none"
        print(
            f"- {preference.get('reviewer_role')}: "
            f"preferred={preferred_candidate_id} "
            f"confidence={preference.get('confidence')}"
        )


def _print_reviewer_preferences(candidate: dict[str, Any]) -> None:
    preferences = candidate.get("reviewer_preferences", [])
    if not isinstance(preferences, list) or not preferences:
        return
    print("candidate_reviewer_preferences:")
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        rationale = preference.get("rationale")
        rationale_suffix = f" rationale={rationale}" if rationale else ""
        print(
            f"- {preference.get('reviewer_role')}: "
            f"rank={preference.get('rank')} "
            f"confidence={preference.get('confidence')}{rationale_suffix}"
        )


def _print_candidate_contradictions(candidate: dict[str, Any]) -> None:
    contradictions = candidate.get("contradictions", [])
    if not isinstance(contradictions, list) or not contradictions:
        print("contradictions: none")
        return
    print("contradictions:")
    for index, contradiction in enumerate(contradictions, start=1):
        if not isinstance(contradiction, dict):
            continue
        reviewer_roles = contradiction.get("reviewer_roles")
        reviewer_text = ", ".join(reviewer_roles) if isinstance(reviewer_roles, list) else "unknown"
        span_text = (
            contradiction.get("time_range") or contradiction.get("source_span_id") or "unknown"
        )
        print(
            f"{index}. {span_text} | {contradiction.get('dimension')} | "
            f"{contradiction.get('severity')} | reviewers={reviewer_text}"
        )
        print(f"   note: {contradiction.get('evidence_text')}")
        if contradiction.get("normalized_value"):
            print(f"   normalized_value: {contradiction['normalized_value']}")
        if contradiction.get("source_excerpt"):
            print(f"   source_excerpt: {contradiction['source_excerpt']}")
        if contradiction.get("target_excerpt"):
            print(f"   target_excerpt: {contradiction['target_excerpt']}")


def _should_enter_interactive_review(*, mode: str) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return _has_tty()
    return _has_tty()


def _has_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


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
