"""CLI entrypoint for the translation agent dry-run workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from translation_agent.api import RunJobRequest, run_job
from translation_agent.config import load_settings, sanitize_db_target, validate_environment
from translation_agent.storage.migrations import upgrade_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate local runtime config")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    migrate_parser = subparsers.add_parser("migrate-db", help="Apply Postgres migrations")
    migrate_parser.add_argument("--revision", default="head")

    run_parser = subparsers.add_parser("run-job", help="Execute the local dry-run workflow")
    run_parser.add_argument("source")
    run_parser.add_argument("--job-id")
    run_parser.add_argument("--json", action="store_true", dest="as_json")
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

    if args.command == "run-job":
        result = run_job(RunJobRequest(source=args.source, job_id=args.job_id))
        payload = {
            key: str(value) if hasattr(value, "__fspath__") else value
            for key, value in asdict(result).items()
        }
        if args.as_json:
            print(json.dumps(payload))
        else:
            print(result.run_id)
            print(result.status)
            print(f"{result.state_backend}: {result.state_db_target}")
            print(result.trace_path)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
