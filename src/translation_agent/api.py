"""Public Python API entrypoint for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
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
from translation_agent.storage.runs import SQLiteRunStore


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
    state_db_path: Path
    trace_path: Path


def run_job(request: RunJobRequest, settings: Settings | None = None) -> RunJobResult:
    """Bootstrap a run record and local runtime artifacts."""

    settings = settings or load_settings()
    validate_environment(settings)

    configure_structured_logging()
    logger = get_structured_logger("translation_agent.api")
    blob_store = LocalBlobStore(settings.blob_dir)

    run_id = uuid4().hex
    job_id = request.job_id or run_id
    now = datetime.now(UTC)

    manifest_entry = blob_store.put_bytes(
        f"jobs/{run_id}-request.json",
        _serialize_request(request, now).encode("utf-8"),
    )
    with SQLiteRunStore(settings.state_db_path) as run_store:
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
    trace_path = settings.trace_dir / f"{run_id}.jsonl"
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
        blob_root=settings.blob_dir,
        state_db_path=settings.state_db_path,
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
