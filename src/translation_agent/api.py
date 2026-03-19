"""Public Python API entrypoint for Phase 0."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from translation_agent.config import Settings, load_settings, validate_environment
from translation_agent.observability.events import (
    configure_structured_logging,
    get_structured_logger,
    log_structured_event,
)
from translation_agent.observability.tracing import JsonlTraceSink, TraceEvent
from translation_agent.storage.blobs import LocalBlobStore
from translation_agent.storage.runs import PostgresRunStore


@dataclass(slots=True)
class RunJobRequest:
    source: str
    job_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RunJobResult:
    run_id: str
    job_id: str
    status: str
    source: str
    blob_root: Path
    trace_path: Path
    state_backend: str = "postgres"
    state_db_target: str = ""


def run_job(request: RunJobRequest, settings: Settings | None = None) -> RunJobResult:
    """Bootstrap a run record and local runtime artifacts."""

    settings = settings or load_settings()
    validation = validate_environment(settings)
    if not validation.ok:
        message = validation.state_db_error or "invalid runtime configuration"
        raise RuntimeError(message)
    state_db_dsn = settings.state_db_dsn
    if state_db_dsn is None:
        raise RuntimeError("TA_STATE_DB_DSN is required")
    blob_dir = settings.blob_dir
    trace_dir = settings.trace_dir

    configure_structured_logging()
    logger = get_structured_logger("translation_agent.api")
    blob_store = LocalBlobStore(blob_dir)

    run_id = uuid4().hex
    job_id = request.job_id or run_id
    now = datetime.now(UTC)

    manifest_entry = blob_store.put_bytes(
        f"jobs/{run_id}-request.json",
        _serialize_request(request, now).encode("utf-8"),
    )
    with PostgresRunStore(state_db_dsn) as run_store:
        run_store.create_run(
            run_id=run_id,
            status="bootstrapped",
            input_data={
                "job_id": job_id,
                "source": request.source,
                "artifact_ref": manifest_entry.key,
            },
            metadata=request.metadata,
        )
    trace_path = trace_dir / f"{run_id}.jsonl"
    with JsonlTraceSink(trace_path) as trace_sink:
        trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="run.bootstrapped",
                attributes={
                    "job_id": job_id,
                    "source": request.source,
                    "artifact_ref": manifest_entry.key,
                },
            )
        )

    log_structured_event(
        logger,
        "run.bootstrapped",
        run_id=run_id,
        job_id=job_id,
        source=request.source,
        artifact_ref=manifest_entry.key,
        trace_path=str(trace_path),
    )
    return RunJobResult(
        run_id=run_id,
        job_id=job_id,
        status="bootstrapped",
        source=request.source,
        blob_root=blob_dir,
        state_db_target=validation.state_db_target,
        trace_path=trace_path,
    )


def _serialize_request(request: RunJobRequest, created_at: datetime) -> str:
    payload = {
        "source": request.source,
        "job_id": request.job_id,
        "created_at": created_at.isoformat(),
        "metadata": request.metadata,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
