"""Public Python API entrypoint for the Phase 2 dry-run workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from translation_agent.config import Settings, load_settings, validate_environment
from translation_agent.graph import GraphState, build_runtime, run_workflow, sync_trace_artifact
from translation_agent.models import JobContext
from translation_agent.observability.events import (
    configure_structured_logging,
    get_structured_logger,
    log_structured_event,
)
from translation_agent.observability.tracing import JsonlTraceSink, TraceEvent
from translation_agent.storage import PostgresOperationalStore, SQLiteOperationalStore
from translation_agent.storage.blobs import LocalBlobStore


@dataclass(slots=True)
class RunJobRequest:
    source: str
    job_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    tenant_id: str = "tenant-local"
    project_id: str = "project-local"
    target_language: str = "fr"
    source_language: str = "en"
    requested_by: str = "system@local"
    profile_ref: str | None = "profiles/default"


@dataclass(slots=True)
class RunJobResult:
    run_id: str
    job_id: str
    status: str
    source: str
    blob_root: Path
    trace_path: Path
    state_backend: str = "sqlite"
    state_db_target: str = ""


def run_job(request: RunJobRequest, settings: Settings | None = None) -> RunJobResult:
    """Execute the deterministic dry-run workflow from the public API."""

    settings = settings or load_settings()
    validation = validate_environment(settings)
    if not validation.ok:
        message = validation.state_db_error or "invalid runtime configuration"
        raise RuntimeError(message)
    blob_dir = settings.blob_dir
    trace_dir = settings.trace_dir

    configure_structured_logging()
    logger = get_structured_logger("translation_agent.api")
    blob_store = LocalBlobStore(blob_dir)

    run_id = uuid4().hex
    job_id = request.job_id or run_id
    now = datetime.now(UTC)
    job = JobContext(
        job_id=job_id,
        tenant_id=request.tenant_id,
        project_id=request.project_id,
        source_video_ref=request.source,
        target_language=request.target_language,
        source_language=request.source_language,
        requested_by=request.requested_by,
        created_at=now,
        profile_ref=request.profile_ref,
    )

    manifest_entry = blob_store.put_bytes(
        f"jobs/{run_id}-request.json",
        _serialize_request(request, now).encode("utf-8"),
    )
    with _open_operational_store(settings) as run_store:
        run_store.create_run(
            run_id=run_id,
            tenant_id=job.tenant_id,
            project_id=job.project_id,
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
        runtime_run_store = _open_operational_store(settings)
        runtime = None
        try:
            runtime = build_runtime(
                settings=settings,
                blob_store=blob_store,
                run_store=runtime_run_store,
                decision_store=runtime_run_store,
                memory_batch_store=runtime_run_store,
                trace_sink=trace_sink,
                source_artifact_ref=manifest_entry.key,
                scenario=request.metadata.get("scenario", "happy"),
            )
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
            initial_state = GraphState(
                run_id=run_id,
                job=job,
                current_stage="ingest",
                source_video_ref=request.source,
                source_artifact_ref=manifest_entry.key,
            )
            runtime.run_store.update_run(run_id, status="running")
            trace_sink.record(
                TraceEvent(
                    run_id=run_id,
                    name="run.started",
                    attributes={"job_id": job_id, "scenario": runtime.scenario},
                )
            )
            final_state = run_workflow(initial_state, runtime)
        except Exception as exc:
            runtime_run_store.update_run(
                run_id,
                status="failed",
                output_data={"final_stage": "bootstrap" if runtime is None else None},
                error={"message": str(exc)},
            )
            trace_sink.record(
                TraceEvent(
                    run_id=run_id,
                    name="run.failed",
                    attributes={
                        "error": str(exc),
                        "phase": "bootstrap" if runtime is None else "run",
                    },
                )
            )
            log_structured_event(
                logger,
                "run.failed",
                level="error",
                run_id=run_id,
                job_id=job_id,
                source=request.source,
                artifact_ref=manifest_entry.key,
                error=str(exc),
                trace_path=str(trace_path),
            )
            runtime_run_store.close()
            raise

        final_status = _final_status(final_state)
        runtime_run_store.update_run(
            run_id,
            status=final_status,
            output_data={
                "final_stage": final_state.current_stage,
                "published_artifact_refs": list(final_state.published_artifact_refs),
                "human_review_required": final_state.human_review_required,
                "translation_failed": final_state.translation_failed,
                "memory_batch_ids": list(final_state.memory_batch_ids),
            },
        )
        trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name="run.completed",
                attributes={
                    "status": final_status,
                    "published_artifact_refs": list(final_state.published_artifact_refs),
                },
            )
        )
        sync_trace_artifact(final_state, runtime)
        runtime_run_store.close()

    log_structured_event(
        logger,
        "run.completed",
        run_id=run_id,
        job_id=job_id,
        source=request.source,
        artifact_ref=manifest_entry.key,
        status=final_status,
        trace_path=str(trace_path),
    )
    return RunJobResult(
        run_id=run_id,
        job_id=job_id,
        status=final_status,
        source=request.source,
        blob_root=blob_dir,
        state_backend=validation.state_backend,
        state_db_target=validation.state_db_target,
        trace_path=trace_path,
    )


def _open_operational_store(
    settings: Settings,
) -> PostgresOperationalStore | SQLiteOperationalStore:
    if settings.state_db_dsn:
        return PostgresOperationalStore(settings.state_db_dsn)
    return SQLiteOperationalStore(settings.state_db_path)


def _serialize_request(request: RunJobRequest, created_at: datetime) -> str:
    payload = {
        "source": request.source,
        "job_id": request.job_id,
        "created_at": created_at.isoformat(),
        "metadata": request.metadata,
        "tenant_id": request.tenant_id,
        "project_id": request.project_id,
        "target_language": request.target_language,
        "source_language": request.source_language,
        "requested_by": request.requested_by,
        "profile_ref": request.profile_ref,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _final_status(state: GraphState) -> str:
    if state.translation_failed:
        return "translation_failed"
    if state.human_review_required:
        return "human_review_required"
    failed_transcription_facts = [
        fact for fact in state.routing_facts if fact.fact_type == "transcription_provider_failed"
    ]
    if failed_transcription_facts:
        return "completed_with_degraded_transcription"
    return "completed"
