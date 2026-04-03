"""CLI entrypoint for the translation agent dry-run workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
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
)
from translation_agent.config import load_settings, sanitize_db_target, validate_environment
from translation_agent.storage.migrations import upgrade_database

_NO_OVERLAPPING_EXCERPT = "[no overlapping excerpt]"
_REVIEW_DIFF_SEPARATOR = " | "
_MIN_REVIEW_PANE_WIDTH = 40


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
        payload = resolve_review(
            args.run_id,
            resolution=args.resolution,
            candidate_id=args.candidate_id,
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
    payload = initial_payload or review_job(run_id, settings=settings)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        if payload.get("review_available") is False:
            print("no pending translation review")
        return 0
    review_diffs = payload.get("review_diffs", [])
    if not isinstance(review_diffs, list):
        review_diffs = []
    if not review_diffs:
        _print_review_header(payload)
        _print_candidate_list(candidates, review_diffs)
        print("no review diffs available")
        return 0

    diff_index = 0
    while True:
        diff = review_diffs[diff_index]
        if not isinstance(diff, dict):
            print("review diff payload is invalid")
            return 1
        _print_review_header(payload)
        _print_candidate_list(candidates, review_diffs)
        _print_review_diff(diff, diff_index=diff_index, diff_count=len(review_diffs))
        raw_command = input(
            "review command (n=next, p=previous, l=scoreboard, f l=finalize with left, "
            "f r=finalize with right, q=quit): "
        )
        raw_command = raw_command.strip()
        if raw_command.lower() in {"q", "quit", "exit"}:
            return 0
        if raw_command.lower() == "n":
            if diff_index >= len(review_diffs) - 1:
                print("already at last diff")
                continue
            diff_index += 1
            continue
        if raw_command.lower() == "p":
            if diff_index <= 0:
                print("already at first diff")
                continue
            diff_index -= 1
            continue
        if raw_command.lower() == "l":
            _print_candidate_list(candidates, review_diffs)
            continue
        normalized_command = raw_command.lower()
        if normalized_command in {"f l", "f r", "a l", "a r"}:
            side_key = "left_candidate" if normalized_command.endswith("l") else "right_candidate"
            side = diff.get(side_key)
            if not isinstance(side, dict) or not side.get("candidate_id"):
                print("approval target is unavailable for this diff")
                continue
            print(
                "final approval selects this candidate for the whole review; "
                "it does not advance to the next diff"
            )
            confirm = input("finalize review with this candidate? [y/N]: ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("approval cancelled")
                continue
            approved_by = input("approved_by (blank uses current user): ").strip() or None
            note = input("note (optional): ").strip() or None
            approved = approve_review(
                run_id,
                candidate_id=str(side["candidate_id"]),
                approved_by=approved_by,
                note=note,
                settings=settings,
            )
            print(approved["status"])
            print(f"approval_ref: {approved['approval_ref']}")
            print(f"default_output_path: {approved['default_output_path']}")
            return 0
        print("unknown command")


def _print_review_header(payload: dict[str, Any]) -> None:
    print(payload["run_id"])
    print(payload["status"])
    if payload.get("summary"):
        print(payload["summary"])
    _print_human_review_summary(payload)


def _print_candidate_list(
    candidates: list[dict[str, Any]],
    review_diffs: list[dict[str, Any]],
) -> None:
    diff_counts = _candidate_diff_counts(review_diffs)
    print("candidate_scoreboard:")
    for candidate in candidates:
        source = candidate.get("source_transcript", {})
        provider = source.get("provider_id") if isinstance(source, dict) else None
        contradiction_count = candidate.get("contradiction_count", 0)
        blocking_count = candidate.get("blocking_hard_contradiction_count", 0)
        diff_count = diff_counts.get(str(candidate["candidate_id"]), 0)
        print(
            f"{candidate['rank']}. {candidate['candidate_id']} "
            f"prompt={candidate['prompt_variant_id']} "
            f"provider={provider or 'unknown'} "
            f"transcript={candidate['source_transcript_candidate_id']} "
            f"contradictions={contradiction_count} "
            f"blocking={blocking_count} "
            f"diffs={diff_count}"
        )


def _candidate_diff_counts(review_diffs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diff in review_diffs:
        if not isinstance(diff, dict):
            continue
        for side_key in ("left_candidate", "right_candidate"):
            side = diff.get(side_key)
            if not isinstance(side, dict):
                continue
            candidate_id = side.get("candidate_id")
            if not isinstance(candidate_id, str):
                continue
            counts[candidate_id] = counts.get(candidate_id, 0) + 1
    return counts


def _print_review_diff(
    diff: dict[str, Any],
    *,
    diff_index: int,
    diff_count: int,
) -> None:
    print(f"diff {diff_index + 1}/{diff_count}")
    print(
        f"source: {diff.get('time_range') or diff.get('source_span_id') or 'unknown'} "
        f"| dimension={diff.get('dimension')} "
        f"| severity={diff.get('severity')} "
        f"| blocking_hard={diff.get('blocking_hard_contradiction')}"
    )
    reviewer_roles = diff.get("reviewer_roles")
    if isinstance(reviewer_roles, list) and reviewer_roles:
        print(f"reviewers: {', '.join(str(role) for role in reviewer_roles)}")
    if diff.get("normalized_value"):
        print(f"normalized_value: {diff['normalized_value']}")
    print(f"evidence: {diff.get('evidence_text')}")
    print(f"source_excerpt: {diff.get('source_excerpt') or _NO_OVERLAPPING_EXCERPT}")
    left_candidate = diff.get("left_candidate", {})
    right_candidate = diff.get("right_candidate", {})
    if not isinstance(left_candidate, dict) or not isinstance(right_candidate, dict):
        return
    _print_candidate_panes(left_candidate, right_candidate)
    print("commands: n=next p=previous l=scoreboard f l=finalize left f r=finalize right q=quit")


def _print_candidate_panes(
    left_candidate: dict[str, Any],
    right_candidate: dict[str, Any],
) -> None:
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    pane_width = (terminal_width - len(_REVIEW_DIFF_SEPARATOR)) // 2
    if pane_width < _MIN_REVIEW_PANE_WIDTH:
        _print_candidate_panes_stacked(left_candidate, right_candidate)
        return

    left_lines = _pane_lines("- ", "LEFT", left_candidate, pane_width)
    right_lines = _pane_lines("+ ", "RIGHT", right_candidate, pane_width)
    row_count = max(len(left_lines), len(right_lines))
    for index in range(row_count):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        print(f"{left.ljust(pane_width)}{_REVIEW_DIFF_SEPARATOR}{right.ljust(pane_width)}".rstrip())


def _print_candidate_panes_stacked(
    left_candidate: dict[str, Any],
    right_candidate: dict[str, Any],
) -> None:
    for line in _pane_lines("", "LEFT", left_candidate, 120):
        print(line)
    for line in _pane_lines("", "RIGHT", right_candidate, 120):
        print(line)


def _pane_lines(
    prefix: str,
    title: str,
    side: dict[str, Any],
    width: int,
) -> list[str]:
    lines: list[str] = []
    lines.extend(_wrap_prefixed(f"{prefix}{title} candidate", width))
    lines.extend(_wrap_prefixed(f"{prefix}candidate_id: {side.get('candidate_id')}", width))
    lines.extend(_wrap_prefixed(f"{prefix}rank: {side.get('rank')}", width))
    lines.extend(
        _wrap_prefixed(f"{prefix}prompt_variant_id: {side.get('prompt_variant_id')}", width)
    )
    lines.extend(_wrap_prefixed(f"{prefix}model_id: {side.get('model_id')}", width))
    lines.extend(
        _wrap_prefixed(
            f"{prefix}source_transcript_candidate_id: {side.get('source_transcript_candidate_id')}",
            width,
        )
    )
    lines.extend(
        _wrap_prefixed(
            f"{prefix}target_excerpt: {side.get('target_excerpt') or _NO_OVERLAPPING_EXCERPT}",
            width,
        )
    )
    return lines


def _wrap_prefixed(text: str, width: int) -> list[str]:
    wrapped = textwrap.wrap(
        text,
        width=max(width, 20),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [text]


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
