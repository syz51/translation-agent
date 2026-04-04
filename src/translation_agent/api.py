"""Public Python API entrypoint for the Phase 2 dry-run workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from translation_agent.config import (
    Settings,
    load_settings,
    sanitize_db_target,
    validate_environment,
)
from translation_agent.errors import exception_error_payload
from translation_agent.graph import (
    GraphState,
    RoutingFact,
    build_runtime,
    run_transcription_resume_workflow,
    run_translation_resume_workflow,
    run_workflow,
    sync_trace_artifact,
)
from translation_agent.language_codes import canonicalize_language_code
from translation_agent.media_identity import compute_media_fingerprint
from translation_agent.models import (
    AssetContext,
    AssetContextInput,
    AudioArtifact,
    FinalTranscriptDecision,
    HistoricalRunLink,
    JobContext,
    ReviewDraftResolution,
    ReviewedSpanDecision,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.nodes.common import (
    audio_artifact_key,
    raw_transcript_candidate_key,
    transcript_candidate_key,
    transcript_decision_key,
    transcript_investigation_key,
    transcript_review_key,
)
from translation_agent.observability.events import (
    configure_structured_logging,
    get_structured_logger,
    log_structured_event,
)
from translation_agent.observability.tracing import (
    CompositeTraceSink,
    JsonlTraceSink,
    TraceEvent,
    TraceSink,
)
from translation_agent.review_flow import (
    approve_translation_review,
    build_review_payload,
    resolve_translation_review,
    save_review_draft_resolution,
)
from translation_agent.run_status import (
    RunStatusSnapshot,
    derive_run_status_snapshot,
    tail_trace_events,
)
from translation_agent.storage import (
    PostgresOperationalStore,
    RunRecord,
    SQLiteOperationalStore,
    job_path,
    operational_job_key,
)
from translation_agent.storage.blobs import LocalBlobStore
from translation_agent.subtitles import render_translation_srt, subtitle_count


@dataclass(slots=True)
class RunJobRequest:
    source: str
    job_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    tenant_id: str = "tenant-local"
    project_id: str = "project-local"
    target_language: str | None = None
    source_language: str | None = None
    requested_by: str = "system@local"
    profile_ref: str | None = "profiles/default"
    asset_id: str | None = None
    asset_context: AssetContextInput | None = None
    reference_transcript_source: str | None = None
    reference_transcript_format: Literal["srt"] | None = None
    reference_mode: Literal["none", "evaluate_and_regenerate"] = "none"
    translation_variant_policy: Literal["single", "dual_experiment"] = "single"
    review_mode: Literal["auto", "always", "never"] = "auto"


@dataclass(slots=True)
class RunJobResult:
    run_id: str
    job_id: str
    status: str
    source: str
    source_language: str
    target_language: str
    blob_root: Path
    trace_path: Path
    default_output_path: Path | None = None
    state_backend: str = "sqlite"
    state_db_target: str = ""
    failure_ref: str | None = None
    failure_summary: str | None = None
    failure_reasons: tuple[str, ...] = ()
    review_required_stage: str | None = None
    resolution_ref: str | None = None
    resolution_kind: str | None = None
    failure_tags: tuple[str, ...] = ()
    residual_failure_tags: tuple[str, ...] = ()
    approval_ref: str | None = None
    approved_candidate_id: str | None = None
    approved_source_transcript_candidate_id: str | None = None
    resume_commands: tuple[str, ...] = ()


@dataclass(slots=True)
class ConvertTranslationJsonToSrtResult:
    source_path: Path
    output_path: Path
    job_id: str
    candidate_id: str
    language: str
    subtitle_count: int


@dataclass(slots=True)
class MemoryEmbeddingBackfillResult:
    updated_entries: int
    state_backend: str
    state_db_target: str


@dataclass(slots=True)
class ResumedTranscriptState:
    audio_artifact_ref: str | None
    transcript_candidate_ids: tuple[str, ...]
    final_transcript_candidate_id: str | None
    final_transcript_decision_ref: str | None
    routing_facts: tuple[RoutingFact, ...]


def list_runs(settings: Settings | None = None) -> list[RunRecord]:
    """List persisted workflow runs in reverse chronological order."""

    settings = settings or load_settings()
    with _open_operational_store(settings) as run_store:
        return run_store.list_runs()


def get_run_status(run_id: str, settings: Settings | None = None) -> RunStatusSnapshot:
    """Build a live status snapshot from persisted run state and the trace tail."""

    settings = settings or load_settings()
    trace_path = settings.trace_dir / f"{run_id}.jsonl"
    with _open_operational_store(settings) as run_store:
        record = run_store.get_run(run_id)
        if record is None:
            raise ValueError(f"unknown run_id: {run_id}")
        node_executions = run_store.list_node_executions(run_id)
    trace_events = tail_trace_events(trace_path)
    return derive_run_status_snapshot(
        record,
        node_executions,
        trace_events,
        trace_path=trace_path,
    )


def run_job(
    request: RunJobRequest,
    settings: Settings | None = None,
    *,
    live_trace_sink: TraceSink | None = None,
) -> RunJobResult:
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
    source_language, target_language = _resolved_job_languages(request, settings)
    reference_transcript_format = _resolved_reference_transcript_format(request)
    media_fingerprint = compute_media_fingerprint(request.source)
    normalized_asset_id = _normalized_optional_identifier(request.asset_id)
    normalized_reference_source = _normalized_optional_identifier(
        request.reference_transcript_source
    )
    job: JobContext
    manifest_entry = None
    with _open_operational_store(settings) as run_store:
        asset = run_store.resolve_asset(
            asset_id=normalized_asset_id,
            media_fingerprint=media_fingerprint,
            first_seen_run_id=run_id,
            source_language=source_language,
            target_language=target_language,
        )
        asset_context = _resolved_asset_context(
            asset.media_key,
            request.asset_context,
            existing=run_store.get_asset_context(asset.media_key),
        )
        if asset_context is not None:
            run_store.save_asset_context(asset_context)
        job = JobContext(
            job_id=job_id,
            tenant_id=request.tenant_id,
            project_id=request.project_id,
            source_video_ref=request.source,
            target_language=target_language,
            source_language=source_language,
            requested_by=request.requested_by,
            created_at=now,
            profile_ref=request.profile_ref,
            asset_id=asset.asset_id,
            media_fingerprint=asset.media_fingerprint,
            media_key=asset.media_key,
            asset_context=asset_context,
            reference_transcript_source=normalized_reference_source,
            reference_transcript_format=reference_transcript_format,
            reference_mode=request.reference_mode,
            translation_variant_policy=request.translation_variant_policy,
        )
        manifest_entry = blob_store.put_bytes(
            f"jobs/{run_id}-request.json",
            _serialize_request(
                request,
                now,
                source_language=source_language,
                target_language=target_language,
            ).encode("utf-8"),
        )
        run_store.create_run(
            run_id=run_id,
            tenant_id=job.tenant_id,
            project_id=job.project_id,
            status="bootstrapped",
            input_data={
                "job_id": job_id,
                "source": request.source,
                "artifact_ref": manifest_entry.key,
                "asset_id": job.asset_id,
                "media_fingerprint": job.media_fingerprint,
                "media_key": job.media_key,
                "reference_mode": job.reference_mode,
            },
            metadata=request.metadata,
        )
        run_store.upsert_historical_run_link(
            HistoricalRunLink(
                run_id=run_id,
                media_key=job.media_key,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                project_id=job.project_id,
                source_language=job.source_language,
                target_language=job.target_language,
                created_at=job.created_at,
            )
        )
    trace_path = trace_dir / f"{run_id}.jsonl"
    with _open_trace_sink(trace_path, live_trace_sink=live_trace_sink) as trace_sink:
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
            _upsert_current_run_link(runtime_run_store, final_state)
        except Exception as exc:
            error_payload = exception_error_payload(exc)
            runtime_run_store.update_run(
                run_id,
                status="failed",
                output_data={"final_stage": "bootstrap" if runtime is None else None},
                error=error_payload,
            )
            trace_sink.record(
                TraceEvent(
                    run_id=run_id,
                    name="run.failed",
                    attributes={
                        "error": error_payload["message"],
                        "error_payload": error_payload,
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
                error=error_payload["message"],
                error_payload=error_payload,
                trace_path=str(trace_path),
            )
            runtime_run_store.close()
            raise

        final_status = _final_status(final_state)
        failure_ref, failure_summary, failure_reasons = _failure_details(
            final_state=final_state,
            blob_store=blob_store,
        )
        terminal_error = _terminal_run_error(
            final_state=final_state,
            failure_ref=failure_ref,
            failure_summary=failure_summary,
            failure_reasons=failure_reasons,
            translation_provider_id=getattr(runtime.translation_adapter, "provider_id", None),
        )
        runtime_run_store.update_run(
            run_id,
            status=final_status,
            output_data={
                "final_stage": final_state.current_stage,
                "published_artifact_refs": list(final_state.published_artifact_refs),
                "human_review_required": final_state.human_review_required,
                "review_required_stage": final_state.review_required_stage,
                "translation_failed": final_state.translation_failed,
                "memory_batch_ids": list(final_state.memory_batch_ids),
                "media_key": final_state.job.media_key,
                "transcript_decision_ref": final_state.final_transcript_decision_ref,
                "translation_decision_ref": final_state.final_translation_decision_ref,
                "transcript_investigation_ref": _investigation_ref(
                    final_state,
                    stage="transcript",
                ),
                "translation_investigation_ref": _investigation_ref(
                    final_state,
                    stage="translation",
                ),
                "reference_transcript_ref": final_state.reference_transcript_ref,
                "evaluation_report_ref": final_state.evaluation_report_ref,
                "regenerated_translation_draft_ref": final_state.regenerated_translation_draft_ref,
                "improvement_proposal_refs": list(final_state.improvement_proposal_refs),
                "resolution_ref": final_state.resolution_ref,
                "resolution_kind": final_state.resolution_kind,
                "failure_tags": list(final_state.failure_tags),
                "residual_failure_tags": list(final_state.residual_failure_tags),
                "approval_ref": final_state.approval_ref,
                "approved_candidate_id": final_state.approved_candidate_id,
                "approved_source_transcript_candidate_id": (
                    final_state.approved_source_transcript_candidate_id
                ),
                "failure_ref": failure_ref,
                "failure_summary": failure_summary,
                "failure_reasons": list(failure_reasons),
            },
            error=terminal_error,
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
        source_language=final_state.job.source_language,
        target_language=final_state.job.target_language,
        blob_root=blob_dir,
        state_backend=validation.state_backend,
        state_db_target=validation.state_db_target,
        trace_path=trace_path,
        default_output_path=_default_output_path(blob_dir, final_state),
        failure_ref=failure_ref,
        failure_summary=failure_summary,
        failure_reasons=failure_reasons,
        review_required_stage=final_state.review_required_stage,
        resolution_ref=final_state.resolution_ref,
        resolution_kind=final_state.resolution_kind,
        failure_tags=final_state.failure_tags,
        residual_failure_tags=final_state.residual_failure_tags,
        approval_ref=final_state.approval_ref,
        approved_candidate_id=final_state.approved_candidate_id,
        approved_source_transcript_candidate_id=final_state.approved_source_transcript_candidate_id,
        resume_commands=_resume_commands(run_id, final_state),
    )


def resume_translation(
    source_run_id: str,
    *,
    review_mode: Literal["auto", "always", "never"] = "auto",
    settings: Settings | None = None,
    live_trace_sink: TraceSink | None = None,
) -> RunJobResult:
    """Resume translation from persisted transcript artifacts of a prior run."""

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

    now = datetime.now(UTC)
    new_run_id = uuid4().hex

    with _open_operational_store(settings) as store:
        source_run = store.get_run(source_run_id)
        if source_run is None:
            raise ValueError(f"unknown run_id: {source_run_id}")
        source_input = _dict_payload(source_run.input_data)
        source_artifact_ref = _required_artifact_ref(source_input, run_id=source_run_id)
        request_payload = json.loads(blob_store.read_bytes(source_artifact_ref).decode("utf-8"))
        source_job = _job_context_from_run(
            run_record=source_run,
            input_data=source_input,
            request_payload=request_payload,
        )
        source_metadata = _dict_payload(source_run.metadata)
        transcript_candidates = store.list_transcript_candidates(
            source_job.job_id,
            storage_job_id=operational_job_key(source_job),
        )
        if not transcript_candidates:
            raise ValueError(
                f"run {source_run_id} does not have persisted transcript candidates to resume from"
            )
        transcript_decision = store.get_transcript_decision(
            source_job.job_id,
            storage_job_id=operational_job_key(source_job),
        )
        resumed_job_id = _resume_job_id(source_job.job_id, new_run_id)
        resumed_job = source_job.model_copy(
            update={
                "job_id": resumed_job_id,
                "created_at": now,
            }
        )
        request_artifact = blob_store.put_bytes(
            f"jobs/{new_run_id}-request.json",
            _serialize_resume_request_payload(
                request_payload,
                source_job=source_job,
                resumed_job=resumed_job,
                created_at=now,
                resumed_from_run_id=source_run_id,
            ).encode("utf-8"),
        )
        resume_metadata = {
            **{key: value for key, value in source_metadata.items() if isinstance(value, str)},
            "resume_mode": "translation",
            "resumed_from_run_id": source_run_id,
        }
        store.create_run(
            run_id=new_run_id,
            tenant_id=resumed_job.tenant_id,
            project_id=resumed_job.project_id,
            status="bootstrapped",
            input_data={
                "job_id": resumed_job.job_id,
                "source": resumed_job.source_video_ref,
                "artifact_ref": request_artifact.key,
                "asset_id": resumed_job.asset_id,
                "media_fingerprint": resumed_job.media_fingerprint,
                "media_key": resumed_job.media_key,
                "reference_mode": resumed_job.reference_mode,
                "resumed_from_run_id": source_run_id,
            },
            metadata=resume_metadata,
        )
        store.upsert_historical_run_link(
            HistoricalRunLink(
                run_id=new_run_id,
                media_key=resumed_job.media_key,
                job_id=resumed_job.job_id,
                tenant_id=resumed_job.tenant_id,
                project_id=resumed_job.project_id,
                source_language=resumed_job.source_language,
                target_language=resumed_job.target_language,
                created_at=resumed_job.created_at,
            )
        )
        transcript_state = _seed_resumed_transcript_state(
            store=store,
            blob_store=blob_store,
            source_job=source_job,
            resumed_job=resumed_job,
            source_run_id=source_run_id,
            transcript_candidates=transcript_candidates,
            transcript_decision=transcript_decision,
            source_output_data=_dict_payload(source_run.output_data),
        )

    trace_path = trace_dir / f"{new_run_id}.jsonl"
    with _open_trace_sink(trace_path, live_trace_sink=live_trace_sink) as trace_sink:
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
                source_artifact_ref=request_artifact.key,
                scenario=resume_metadata.get("scenario", "happy"),
            )
            trace_sink.record(
                TraceEvent(
                    run_id=new_run_id,
                    name="run.bootstrapped",
                    attributes={
                        "job_id": resumed_job.job_id,
                        "source": resumed_job.source_video_ref,
                        "artifact_ref": request_artifact.key,
                        "resumed_from_run_id": source_run_id,
                        "resume_mode": "translation",
                    },
                )
            )
            initial_state = GraphState(
                run_id=new_run_id,
                job=resumed_job,
                current_stage="generate_translation_candidates",
                source_video_ref=resumed_job.source_video_ref,
                source_artifact_ref=request_artifact.key,
                transcript_candidate_ids=transcript_state.transcript_candidate_ids,
                final_transcript_candidate_id=transcript_state.final_transcript_candidate_id,
                final_transcript_decision_ref=transcript_state.final_transcript_decision_ref,
                routing_facts=transcript_state.routing_facts,
            )
            runtime.run_store.update_run(new_run_id, status="running")
            trace_sink.record(
                TraceEvent(
                    run_id=new_run_id,
                    name="run.started",
                    attributes={
                        "job_id": resumed_job.job_id,
                        "scenario": runtime.scenario,
                        "resumed_from_run_id": source_run_id,
                        "resume_mode": "translation",
                    },
                )
            )
            final_state = run_translation_resume_workflow(initial_state, runtime)
            _upsert_current_run_link(runtime_run_store, final_state)
        except Exception as exc:
            error_payload = exception_error_payload(exc)
            runtime_run_store.update_run(
                new_run_id,
                status="failed",
                output_data={"final_stage": "bootstrap" if runtime is None else None},
                error=error_payload,
            )
            trace_sink.record(
                TraceEvent(
                    run_id=new_run_id,
                    name="run.failed",
                    attributes={
                        "error": error_payload["message"],
                        "error_payload": error_payload,
                        "phase": "bootstrap" if runtime is None else "run",
                        "resumed_from_run_id": source_run_id,
                    },
                )
            )
            log_structured_event(
                logger,
                "run.failed",
                level="error",
                run_id=new_run_id,
                job_id=resumed_job.job_id,
                source=resumed_job.source_video_ref,
                artifact_ref=request_artifact.key,
                error=error_payload["message"],
                error_payload=error_payload,
                trace_path=str(trace_path),
            )
            runtime_run_store.close()
            raise

        final_status = _final_status(final_state)
        failure_ref, failure_summary, failure_reasons = _failure_details(
            final_state=final_state,
            blob_store=blob_store,
        )
        terminal_error = _terminal_run_error(
            final_state=final_state,
            failure_ref=failure_ref,
            failure_summary=failure_summary,
            failure_reasons=failure_reasons,
            translation_provider_id=getattr(runtime.translation_adapter, "provider_id", None),
        )
        runtime_run_store.update_run(
            new_run_id,
            status=final_status,
            output_data={
                "final_stage": final_state.current_stage,
                "published_artifact_refs": list(final_state.published_artifact_refs),
                "human_review_required": final_state.human_review_required,
                "review_required_stage": final_state.review_required_stage,
                "translation_failed": final_state.translation_failed,
                "memory_batch_ids": list(final_state.memory_batch_ids),
                "media_key": final_state.job.media_key,
                "transcript_decision_ref": final_state.final_transcript_decision_ref,
                "translation_decision_ref": final_state.final_translation_decision_ref,
                "transcript_investigation_ref": _investigation_ref(
                    final_state,
                    stage="transcript",
                ),
                "translation_investigation_ref": _investigation_ref(
                    final_state,
                    stage="translation",
                ),
                "reference_transcript_ref": final_state.reference_transcript_ref,
                "evaluation_report_ref": final_state.evaluation_report_ref,
                "regenerated_translation_draft_ref": final_state.regenerated_translation_draft_ref,
                "improvement_proposal_refs": list(final_state.improvement_proposal_refs),
                "resolution_ref": final_state.resolution_ref,
                "resolution_kind": final_state.resolution_kind,
                "failure_tags": list(final_state.failure_tags),
                "residual_failure_tags": list(final_state.residual_failure_tags),
                "approval_ref": final_state.approval_ref,
                "approved_candidate_id": final_state.approved_candidate_id,
                "approved_source_transcript_candidate_id": (
                    final_state.approved_source_transcript_candidate_id
                ),
                "failure_ref": failure_ref,
                "failure_summary": failure_summary,
                "failure_reasons": list(failure_reasons),
                "resumed_from_run_id": source_run_id,
            },
            error=terminal_error,
        )
        trace_sink.record(
            TraceEvent(
                run_id=new_run_id,
                name="run.completed",
                attributes={
                    "status": final_status,
                    "published_artifact_refs": list(final_state.published_artifact_refs),
                    "resumed_from_run_id": source_run_id,
                },
            )
        )
        sync_trace_artifact(final_state, runtime)
        runtime_run_store.close()

    log_structured_event(
        logger,
        "run.completed",
        run_id=new_run_id,
        job_id=resumed_job.job_id,
        source=resumed_job.source_video_ref,
        artifact_ref=request_artifact.key,
        status=final_status,
        trace_path=str(trace_path),
    )
    result = RunJobResult(
        run_id=new_run_id,
        job_id=resumed_job.job_id,
        status=final_status,
        source=resumed_job.source_video_ref,
        source_language=final_state.job.source_language,
        target_language=final_state.job.target_language,
        blob_root=blob_dir,
        state_backend=validation.state_backend,
        state_db_target=validation.state_db_target,
        trace_path=trace_path,
        default_output_path=_default_output_path(blob_dir, final_state),
        failure_ref=failure_ref,
        failure_summary=failure_summary,
        failure_reasons=failure_reasons,
        review_required_stage=final_state.review_required_stage,
        resolution_ref=final_state.resolution_ref,
        resolution_kind=final_state.resolution_kind,
        failure_tags=final_state.failure_tags,
        residual_failure_tags=final_state.residual_failure_tags,
        approval_ref=final_state.approval_ref,
        approved_candidate_id=final_state.approved_candidate_id,
        approved_source_transcript_candidate_id=final_state.approved_source_transcript_candidate_id,
        resume_commands=_resume_commands(new_run_id, final_state),
    )
    if review_mode == "never":
        return result
    return result


def resume_transcription(
    source_run_id: str,
    *,
    provider_ids: tuple[str, ...] | None = None,
    review_mode: Literal["auto", "always", "never"] = "auto",
    settings: Settings | None = None,
    live_trace_sink: TraceSink | None = None,
) -> RunJobResult:
    """Resume transcription from persisted audio while preserving other providers."""

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

    now = datetime.now(UTC)
    new_run_id = uuid4().hex
    requested_provider_ids = _normalized_provider_ids(provider_ids)

    with _open_operational_store(settings) as store:
        source_run = store.get_run(source_run_id)
        if source_run is None:
            raise ValueError(f"unknown run_id: {source_run_id}")
        source_input = _dict_payload(source_run.input_data)
        source_artifact_ref = _required_artifact_ref(source_input, run_id=source_run_id)
        request_payload = json.loads(blob_store.read_bytes(source_artifact_ref).decode("utf-8"))
        source_job = _job_context_from_run(
            run_record=source_run,
            input_data=source_input,
            request_payload=request_payload,
        )
        source_metadata = _dict_payload(source_run.metadata)
        transcript_candidates = store.list_transcript_candidates(
            source_job.job_id,
            storage_job_id=operational_job_key(source_job),
        )
        failed_provider_ids = _failed_transcription_provider_ids(
            store=store,
            source_run=source_run,
        )
        retry_provider_ids = requested_provider_ids or failed_provider_ids
        if not retry_provider_ids:
            raise ValueError(
                f"run {source_run_id} does not have failed transcription providers to resume from"
            )
        resumed_job_id = _resume_job_id(source_job.job_id, new_run_id)
        resumed_job = source_job.model_copy(
            update={
                "job_id": resumed_job_id,
                "created_at": now,
            }
        )
        request_artifact = blob_store.put_bytes(
            f"jobs/{new_run_id}-request.json",
            _serialize_resume_request_payload(
                request_payload,
                source_job=source_job,
                resumed_job=resumed_job,
                created_at=now,
                resumed_from_run_id=source_run_id,
            ).encode("utf-8"),
        )
        resume_metadata = {
            **{key: value for key, value in source_metadata.items() if isinstance(value, str)},
            "resume_mode": "transcription",
            "resumed_from_run_id": source_run_id,
        }
        store.create_run(
            run_id=new_run_id,
            tenant_id=resumed_job.tenant_id,
            project_id=resumed_job.project_id,
            status="bootstrapped",
            input_data={
                "job_id": resumed_job.job_id,
                "source": resumed_job.source_video_ref,
                "artifact_ref": request_artifact.key,
                "asset_id": resumed_job.asset_id,
                "media_fingerprint": resumed_job.media_fingerprint,
                "media_key": resumed_job.media_key,
                "reference_mode": resumed_job.reference_mode,
                "resumed_from_run_id": source_run_id,
                "resume_mode": "transcription",
                "retry_provider_ids": list(retry_provider_ids),
            },
            metadata=resume_metadata,
        )
        store.upsert_historical_run_link(
            HistoricalRunLink(
                run_id=new_run_id,
                media_key=resumed_job.media_key,
                job_id=resumed_job.job_id,
                tenant_id=resumed_job.tenant_id,
                project_id=resumed_job.project_id,
                source_language=resumed_job.source_language,
                target_language=resumed_job.target_language,
                created_at=resumed_job.created_at,
            )
        )
        transcript_state = _seed_resumed_transcription_retry_state(
            store=store,
            blob_store=blob_store,
            source_job=source_job,
            resumed_job=resumed_job,
            source_run_id=source_run_id,
            transcript_candidates=transcript_candidates,
            retry_provider_ids=retry_provider_ids,
        )

    trace_path = trace_dir / f"{new_run_id}.jsonl"
    with _open_trace_sink(trace_path, live_trace_sink=live_trace_sink) as trace_sink:
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
                source_artifact_ref=request_artifact.key,
                scenario=resume_metadata.get("scenario", "happy"),
            )
            runtime.transcription_adapters = _selected_transcription_adapters(
                runtime.transcription_adapters,
                retry_provider_ids=retry_provider_ids,
            )
            trace_sink.record(
                TraceEvent(
                    run_id=new_run_id,
                    name="run.bootstrapped",
                    attributes={
                        "job_id": resumed_job.job_id,
                        "source": resumed_job.source_video_ref,
                        "artifact_ref": request_artifact.key,
                        "resumed_from_run_id": source_run_id,
                        "resume_mode": "transcription",
                        "retry_provider_ids": list(retry_provider_ids),
                    },
                )
            )
            initial_state = GraphState(
                run_id=new_run_id,
                job=resumed_job,
                current_stage="fanout_transcription",
                source_video_ref=resumed_job.source_video_ref,
                source_artifact_ref=request_artifact.key,
                audio_artifact_ref=transcript_state.audio_artifact_ref,
                transcript_candidate_ids=transcript_state.transcript_candidate_ids,
                routing_facts=transcript_state.routing_facts,
            )
            runtime.run_store.update_run(new_run_id, status="running")
            trace_sink.record(
                TraceEvent(
                    run_id=new_run_id,
                    name="run.started",
                    attributes={
                        "job_id": resumed_job.job_id,
                        "scenario": runtime.scenario,
                        "resumed_from_run_id": source_run_id,
                        "resume_mode": "transcription",
                        "retry_provider_ids": list(retry_provider_ids),
                    },
                )
            )
            final_state = run_transcription_resume_workflow(initial_state, runtime)
            _upsert_current_run_link(runtime_run_store, final_state)
        except Exception as exc:
            error_payload = exception_error_payload(exc)
            runtime_run_store.update_run(
                new_run_id,
                status="failed",
                output_data={"final_stage": "bootstrap" if runtime is None else None},
                error=error_payload,
            )
            trace_sink.record(
                TraceEvent(
                    run_id=new_run_id,
                    name="run.failed",
                    attributes={
                        "error": error_payload["message"],
                        "error_payload": error_payload,
                        "phase": "bootstrap" if runtime is None else "run",
                        "resumed_from_run_id": source_run_id,
                    },
                )
            )
            log_structured_event(
                logger,
                "run.failed",
                level="error",
                run_id=new_run_id,
                job_id=resumed_job.job_id,
                source=resumed_job.source_video_ref,
                artifact_ref=request_artifact.key,
                error=error_payload["message"],
                error_payload=error_payload,
                trace_path=str(trace_path),
            )
            runtime_run_store.close()
            raise

        final_status = _final_status(final_state)
        failure_ref, failure_summary, failure_reasons = _failure_details(
            final_state=final_state,
            blob_store=blob_store,
        )
        terminal_error = _terminal_run_error(
            final_state=final_state,
            failure_ref=failure_ref,
            failure_summary=failure_summary,
            failure_reasons=failure_reasons,
            translation_provider_id=getattr(runtime.translation_adapter, "provider_id", None),
        )
        runtime_run_store.update_run(
            new_run_id,
            status=final_status,
            output_data={
                "final_stage": final_state.current_stage,
                "published_artifact_refs": list(final_state.published_artifact_refs),
                "human_review_required": final_state.human_review_required,
                "review_required_stage": final_state.review_required_stage,
                "translation_failed": final_state.translation_failed,
                "memory_batch_ids": list(final_state.memory_batch_ids),
                "media_key": final_state.job.media_key,
                "transcript_decision_ref": final_state.final_transcript_decision_ref,
                "translation_decision_ref": final_state.final_translation_decision_ref,
                "transcript_investigation_ref": _investigation_ref(
                    final_state,
                    stage="transcript",
                ),
                "translation_investigation_ref": _investigation_ref(
                    final_state,
                    stage="translation",
                ),
                "reference_transcript_ref": final_state.reference_transcript_ref,
                "evaluation_report_ref": final_state.evaluation_report_ref,
                "regenerated_translation_draft_ref": final_state.regenerated_translation_draft_ref,
                "improvement_proposal_refs": list(final_state.improvement_proposal_refs),
                "resolution_ref": final_state.resolution_ref,
                "resolution_kind": final_state.resolution_kind,
                "failure_tags": list(final_state.failure_tags),
                "residual_failure_tags": list(final_state.residual_failure_tags),
                "approval_ref": final_state.approval_ref,
                "approved_candidate_id": final_state.approved_candidate_id,
                "approved_source_transcript_candidate_id": (
                    final_state.approved_source_transcript_candidate_id
                ),
                "failure_ref": failure_ref,
                "failure_summary": failure_summary,
                "failure_reasons": list(failure_reasons),
                "resumed_from_run_id": source_run_id,
                "retry_provider_ids": list(retry_provider_ids),
            },
            error=terminal_error,
        )
        trace_sink.record(
            TraceEvent(
                run_id=new_run_id,
                name="run.completed",
                attributes={
                    "status": final_status,
                    "published_artifact_refs": list(final_state.published_artifact_refs),
                    "resumed_from_run_id": source_run_id,
                    "retry_provider_ids": list(retry_provider_ids),
                },
            )
        )
        sync_trace_artifact(final_state, runtime)
        runtime_run_store.close()

    log_structured_event(
        logger,
        "run.completed",
        run_id=new_run_id,
        job_id=resumed_job.job_id,
        source=resumed_job.source_video_ref,
        artifact_ref=request_artifact.key,
        status=final_status,
        trace_path=str(trace_path),
    )
    result = RunJobResult(
        run_id=new_run_id,
        job_id=resumed_job.job_id,
        status=final_status,
        source=resumed_job.source_video_ref,
        source_language=final_state.job.source_language,
        target_language=final_state.job.target_language,
        blob_root=blob_dir,
        state_backend=validation.state_backend,
        state_db_target=validation.state_db_target,
        trace_path=trace_path,
        default_output_path=_default_output_path(blob_dir, final_state),
        failure_ref=failure_ref,
        failure_summary=failure_summary,
        failure_reasons=failure_reasons,
        review_required_stage=final_state.review_required_stage,
        resolution_ref=final_state.resolution_ref,
        resolution_kind=final_state.resolution_kind,
        failure_tags=final_state.failure_tags,
        residual_failure_tags=final_state.residual_failure_tags,
        approval_ref=final_state.approval_ref,
        approved_candidate_id=final_state.approved_candidate_id,
        approved_source_transcript_candidate_id=final_state.approved_source_transcript_candidate_id,
        resume_commands=_resume_commands(new_run_id, final_state),
    )
    if review_mode == "never":
        return result
    return result


def review_job(
    run_id: str,
    *,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Load a machine-readable translation review payload for one run."""

    settings = settings or load_settings()
    blob_store = LocalBlobStore(settings.blob_dir)
    with _open_operational_store(settings) as store:
        return build_review_payload(run_id, store=store, blob_store=blob_store)


def approve_review(
    run_id: str,
    *,
    candidate_id: str,
    approved_by: str | None = None,
    note: str | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Approve a persisted translation review and republish canonical outputs."""

    settings = settings or load_settings()
    blob_store = LocalBlobStore(settings.blob_dir)
    with _open_operational_store(settings) as store:
        return approve_translation_review(
            run_id,
            candidate_id=candidate_id,
            approved_by=approved_by,
            note=note,
            store=store,
            blob_store=blob_store,
        )


def resolve_review(
    run_id: str,
    *,
    resolution: str,
    candidate_id: str | None = None,
    reviewed_span_decisions: tuple[ReviewedSpanDecision | dict[str, object], ...] = (),
    failure_tags: tuple[str, ...] = (),
    approved_by: str | None = None,
    note: str | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Resolve a persisted translation review with graded human supervision."""

    settings = settings or load_settings()
    blob_store = LocalBlobStore(settings.blob_dir)
    with _open_operational_store(settings) as store:
        return resolve_translation_review(
            run_id,
            resolution_kind=resolution,  # type: ignore[arg-type]
            candidate_id=candidate_id,
            reviewed_span_decisions=tuple(
                decision
                if isinstance(decision, ReviewedSpanDecision)
                else ReviewedSpanDecision.model_validate(decision)
                for decision in reviewed_span_decisions
            ),
            failure_tags=failure_tags,  # type: ignore[arg-type]
            approved_by=approved_by,
            note=note,
            store=store,
            blob_store=blob_store,
        )


def save_review_draft(
    run_id: str,
    *,
    draft_resolution: ReviewDraftResolution | dict[str, object],
    settings: Settings | None = None,
) -> dict[str, object]:
    """Persist an in-progress review draft for a later review-job resume."""

    settings = settings or load_settings()
    blob_store = LocalBlobStore(settings.blob_dir)
    with _open_operational_store(settings) as store:
        normalized = (
            draft_resolution
            if isinstance(draft_resolution, ReviewDraftResolution)
            else ReviewDraftResolution.model_validate(draft_resolution)
        )
        return save_review_draft_resolution(
            run_id,
            draft_resolution=normalized,
            store=store,
            blob_store=blob_store,
        )


def convert_translation_json_to_srt(
    source_path: str | Path,
    output_path: str | Path | None = None,
) -> ConvertTranslationJsonToSrtResult:
    source = Path(source_path).expanduser().resolve()
    destination = _resolved_srt_output_path(source, output_path)
    if destination == source:
        raise ValueError("output path must be different from source path")
    translation = _load_translation_candidate(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_translation_srt(translation), encoding="utf-8")
    return ConvertTranslationJsonToSrtResult(
        source_path=source,
        output_path=destination.resolve(),
        job_id=translation.job_id,
        candidate_id=translation.candidate_id,
        language=translation.language,
        subtitle_count=subtitle_count(translation),
    )


def backfill_memory_embeddings(
    *,
    settings: Settings | None = None,
    limit: int | None = None,
) -> MemoryEmbeddingBackfillResult:
    """Refresh deterministic memory embeddings and lexical search documents."""

    settings = settings or load_settings()
    validation = validate_environment(settings)
    if not validation.ok:
        message = validation.state_db_error or "invalid runtime configuration"
        raise RuntimeError(message)
    with _open_operational_store(settings) as run_store:
        updated_entries = run_store.backfill_memory_embeddings(limit=limit)
    return MemoryEmbeddingBackfillResult(
        updated_entries=updated_entries,
        state_backend="postgres" if settings.state_db_dsn else "sqlite",
        state_db_target=sanitize_db_target(
            settings.state_db_dsn if settings.state_db_dsn else settings.state_db_path
        ),
    )


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_artifact_ref(input_data: dict[str, Any], *, run_id: str) -> str:
    artifact_ref = input_data.get("artifact_ref")
    if isinstance(artifact_ref, str) and artifact_ref:
        return artifact_ref
    raise ValueError(f"run {run_id} is missing source artifact ref")


def _job_context_from_run(
    *,
    run_record,
    input_data: dict[str, Any],
    request_payload: dict[str, Any],
) -> JobContext:
    created_at_raw = request_payload.get("created_at") or run_record.created_at
    created_at = (
        created_at_raw
        if isinstance(created_at_raw, datetime)
        else datetime.fromisoformat(str(created_at_raw))
    )
    return JobContext(
        job_id=str(request_payload.get("job_id") or input_data.get("job_id") or run_record.run_id),
        tenant_id=str(request_payload.get("tenant_id") or "tenant-local"),
        project_id=str(request_payload.get("project_id") or "project-local"),
        source_video_ref=str(request_payload.get("source") or ""),
        target_language=str(request_payload.get("target_language") or "unknown"),
        source_language=str(request_payload.get("source_language") or "unknown"),
        requested_by=str(request_payload.get("requested_by") or "system@local"),
        created_at=created_at,
        profile_ref=_normalized_optional_identifier(request_payload.get("profile_ref")),
        asset_id=_normalized_optional_identifier(input_data.get("asset_id")),
        media_fingerprint=_normalized_optional_identifier(input_data.get("media_fingerprint")),
        media_key=str(input_data.get("media_key") or f"source-ref:{run_record.run_id}"),
        asset_context=_asset_context_from_payload(
            input_data.get("media_key"),
            request_payload.get("asset_context"),
        ),
        reference_transcript_source=_normalized_optional_identifier(
            request_payload.get("reference_transcript_source")
        ),
        reference_transcript_format=request_payload.get("reference_transcript_format"),
        reference_mode=request_payload.get("reference_mode") or "none",
        translation_variant_policy=request_payload.get("translation_variant_policy") or "single",
    )


def _resume_job_id(source_job_id: str, new_run_id: str) -> str:
    return f"{source_job_id}-resume-{new_run_id[:8]}"


def _normalized_provider_ids(provider_ids: tuple[str, ...] | None) -> tuple[str, ...]:
    if not provider_ids:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for provider_id in provider_ids:
        cleaned = str(provider_id).strip().lower()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return tuple(normalized)


def _serialize_resume_request_payload(
    request_payload: dict[str, Any],
    *,
    source_job: JobContext,
    resumed_job: JobContext,
    created_at: datetime,
    resumed_from_run_id: str,
) -> str:
    payload = {
        **request_payload,
        "job_id": resumed_job.job_id,
        "created_at": created_at.isoformat(),
        "source": source_job.source_video_ref,
        "tenant_id": source_job.tenant_id,
        "project_id": source_job.project_id,
        "target_language": source_job.target_language,
        "source_language": source_job.source_language,
        "requested_by": source_job.requested_by,
        "profile_ref": source_job.profile_ref,
        "asset_id": source_job.asset_id,
        "reference_transcript_source": source_job.reference_transcript_source,
        "reference_transcript_format": source_job.reference_transcript_format,
        "reference_mode": source_job.reference_mode,
        "resumed_from_run_id": resumed_from_run_id,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _seed_resumed_transcript_state(
    *,
    store,
    blob_store: LocalBlobStore,
    source_job: JobContext,
    resumed_job: JobContext,
    source_run_id: str,
    transcript_candidates: list[TranscriptCandidate],
    transcript_decision: FinalTranscriptDecision | None,
    source_output_data: dict[str, Any],
) -> ResumedTranscriptState:
    copied_candidate_ids: list[str] = []
    storage_job_id = operational_job_key(resumed_job)
    for candidate in transcript_candidates:
        copied_raw_payload_ref = candidate.raw_payload_ref
        if candidate.raw_payload_ref and blob_store.exists(candidate.raw_payload_ref):
            copied_raw_payload_ref = raw_transcript_candidate_key(
                resumed_job,
                candidate.provider_id,
            )
            blob_store.put_bytes(
                copied_raw_payload_ref,
                blob_store.read_bytes(candidate.raw_payload_ref),
            )
        copied_candidate = candidate.model_copy(
            update={
                "job_id": resumed_job.job_id,
                "raw_payload_ref": copied_raw_payload_ref,
                "metadata": {
                    **candidate.metadata,
                    "raw_payload_ref": copied_raw_payload_ref,
                    "resumed_from_run_id": source_run_id,
                },
            }
        )
        store.save_transcript_candidate(copied_candidate, storage_job_id=storage_job_id)
        _write_blob_model_artifact(
            blob_store,
            transcript_candidate_key(resumed_job, copied_candidate.candidate_id),
            copied_candidate,
        )
        copied_candidate_ids.append(copied_candidate.candidate_id)

    copied_investigation_ref = None
    source_investigation_ref = _normalized_optional_identifier(
        source_output_data.get("transcript_investigation_ref")
    )
    if source_investigation_ref is None and transcript_decision is not None:
        source_investigation_ref = _normalized_optional_identifier(
            transcript_decision.investigation_ref
        )
    if source_investigation_ref is None:
        fallback_ref = transcript_investigation_key(source_job)
        if blob_store.exists(fallback_ref):
            source_investigation_ref = fallback_ref
    if source_investigation_ref is not None and blob_store.exists(source_investigation_ref):
        copied_investigation_ref = transcript_investigation_key(resumed_job)
        blob_store.put_bytes(
            copied_investigation_ref,
            blob_store.read_bytes(source_investigation_ref),
        )

    copied_decision_ref = None
    final_transcript_candidate_id = (
        transcript_decision.winner_candidate_id
        if transcript_decision is not None
        else _fallback_final_transcript_candidate_id(transcript_candidates)
    )
    if final_transcript_candidate_id is None:
        raise ValueError("resume translation requires a resolved final transcript candidate")
    if transcript_decision is not None:
        for review_id in transcript_decision.review_refs:
            source_review_ref = transcript_review_key(source_job, review_id)
            target_review_ref = transcript_review_key(resumed_job, review_id)
            if blob_store.exists(source_review_ref):
                blob_store.put_bytes(target_review_ref, blob_store.read_bytes(source_review_ref))
        copied_decision = transcript_decision.model_copy(
            update={
                "job_id": resumed_job.job_id,
                "investigation_ref": copied_investigation_ref,
            }
        )
        store.save_transcript_decision(copied_decision, storage_job_id=storage_job_id)
        copied_decision_ref = transcript_decision_key(resumed_job)
        _write_blob_model_artifact(
            blob_store,
            copied_decision_ref,
            copied_decision,
        )

    routing_facts = [
        RoutingFact(
            stage="resume_translation",
            fact_type="resumed_from_run",
            value=source_run_id,
            source_ref=source_output_data.get("failure_ref") if source_output_data else None,
        )
    ]
    if copied_decision_ref is not None and transcript_decision is not None:
        routing_facts.extend(
            (
                RoutingFact(
                    stage="adjudicate_transcript",
                    fact_type="decision_mode",
                    value=transcript_decision.decision_mode,
                    source_ref=copied_decision_ref,
                ),
                RoutingFact(
                    stage="adjudicate_transcript",
                    fact_type="disagreement_bucket",
                    value=transcript_decision.disagreement_bucket,
                    source_ref=copied_decision_ref,
                ),
            )
        )
    if copied_investigation_ref is not None:
        routing_facts.append(
            RoutingFact(
                stage="adjudicate_transcript",
                fact_type="investigation_ref",
                value=copied_investigation_ref,
                source_ref=copied_investigation_ref,
            )
        )

    return ResumedTranscriptState(
        audio_artifact_ref=None,
        transcript_candidate_ids=tuple(copied_candidate_ids),
        final_transcript_candidate_id=final_transcript_candidate_id,
        final_transcript_decision_ref=copied_decision_ref,
        routing_facts=tuple(routing_facts),
    )


def _seed_resumed_transcription_retry_state(
    *,
    store,
    blob_store: LocalBlobStore,
    source_job: JobContext,
    resumed_job: JobContext,
    source_run_id: str,
    transcript_candidates: list[TranscriptCandidate],
    retry_provider_ids: tuple[str, ...],
) -> ResumedTranscriptState:
    copied_audio = _copy_audio_artifact(
        blob_store=blob_store,
        source_job=source_job,
        resumed_job=resumed_job,
        source_run_id=source_run_id,
    )
    copied_candidate_ids: list[str] = []
    storage_job_id = operational_job_key(resumed_job)
    retry_provider_id_set = set(retry_provider_ids)
    for candidate in transcript_candidates:
        if candidate.provider_id in retry_provider_id_set:
            continue
        copied_raw_payload_ref = candidate.raw_payload_ref
        if candidate.raw_payload_ref and blob_store.exists(candidate.raw_payload_ref):
            copied_raw_payload_ref = raw_transcript_candidate_key(
                resumed_job,
                candidate.provider_id,
            )
            blob_store.put_bytes(
                copied_raw_payload_ref,
                blob_store.read_bytes(candidate.raw_payload_ref),
            )
        copied_candidate = candidate.model_copy(
            update={
                "job_id": resumed_job.job_id,
                "raw_payload_ref": copied_raw_payload_ref,
                "metadata": {
                    **candidate.metadata,
                    "raw_payload_ref": copied_raw_payload_ref,
                    "resumed_from_run_id": source_run_id,
                },
            }
        )
        store.save_transcript_candidate(copied_candidate, storage_job_id=storage_job_id)
        _write_blob_model_artifact(
            blob_store,
            transcript_candidate_key(resumed_job, copied_candidate.candidate_id),
            copied_candidate,
        )
        copied_candidate_ids.append(copied_candidate.candidate_id)

    routing_facts = [  # preserve the provenance and the explicit retry target set
        RoutingFact(
            stage="resume_transcription",
            fact_type="resumed_from_run",
            value=source_run_id,
            source_ref=audio_artifact_key(resumed_job),
        ),
        *(
            RoutingFact(
                stage="resume_transcription",
                fact_type="transcription_provider_retry_requested",
                value=provider_id,
                source_ref=audio_artifact_key(resumed_job),
            )
            for provider_id in retry_provider_ids
        ),
    ]

    return ResumedTranscriptState(
        audio_artifact_ref=copied_audio.blob_ref,
        transcript_candidate_ids=tuple(copied_candidate_ids),
        final_transcript_candidate_id=None,
        final_transcript_decision_ref=None,
        routing_facts=tuple(routing_facts),
    )


def _copy_audio_artifact(
    *,
    blob_store: LocalBlobStore,
    source_job: JobContext,
    resumed_job: JobContext,
    source_run_id: str,
) -> AudioArtifact:
    source_audio_ref = audio_artifact_key(source_job)
    if not blob_store.exists(source_audio_ref):
        raise ValueError(f"run {source_run_id} does not have a persisted audio artifact to resume")
    source_audio = AudioArtifact.model_validate_json(blob_store.read_bytes(source_audio_ref))
    if not blob_store.exists(source_audio.blob_ref):
        raise ValueError(f"run {source_run_id} is missing audio blob {source_audio.blob_ref}")
    copied_blob_ref = job_path(resumed_job, "artifacts", Path(source_audio.blob_ref).name)
    blob_store.put_bytes(copied_blob_ref, blob_store.read_bytes(source_audio.blob_ref))
    copied_audio = source_audio.model_copy(
        update={
            "job_id": resumed_job.job_id,
            "blob_ref": copied_blob_ref,
            "extraction_metadata": {
                **source_audio.extraction_metadata,
                "resumed_from_run_id": source_run_id,
            },
        }
    )
    _write_blob_model_artifact(blob_store, audio_artifact_key(resumed_job), copied_audio)
    return copied_audio


def _failed_transcription_provider_ids(
    *,
    store,
    source_run: RunRecord,
) -> tuple[str, ...]:
    provider_ids: list[str] = []
    seen: set[str] = set()
    error_payload = _dict_payload(source_run.error)
    for provider_id in _provider_ids_from_error_payload(error_payload):
        if provider_id in seen:
            continue
        provider_ids.append(provider_id)
        seen.add(provider_id)
    for node_execution in store.list_node_executions(source_run.run_id):
        if node_execution.node_name != "fanout_transcription":
            continue
        payload = _dict_payload(node_execution.output_data)
        for fact in payload.get("routing_facts", []):
            if not isinstance(fact, dict):
                continue
            if fact.get("fact_type") != "transcription_provider_failed":
                continue
            provider_id = _normalized_optional_identifier(str(fact.get("value") or ""))
            if provider_id is None or provider_id in seen:
                continue
            provider_ids.append(provider_id)
            seen.add(provider_id)
    return tuple(provider_ids)


def _provider_ids_from_error_payload(error_payload: dict[str, Any]) -> tuple[str, ...]:
    entries = error_payload.get("provider_errors")
    if not isinstance(entries, list):
        return ()
    provider_ids: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider_id = _normalized_optional_identifier(str(entry.get("provider_id") or ""))
        if provider_id is None or provider_id in seen:
            continue
        provider_ids.append(provider_id)
        seen.add(provider_id)
    return tuple(provider_ids)


def _selected_transcription_adapters(
    adapters,
    *,
    retry_provider_ids: tuple[str, ...],
):
    retry_provider_id_set = set(retry_provider_ids)
    selected = tuple(
        adapter
        for adapter in adapters
        if getattr(adapter, "provider_id", None) in retry_provider_id_set
    )
    missing_provider_ids = sorted(
        retry_provider_id_set - {getattr(adapter, "provider_id", None) for adapter in selected}
    )
    if missing_provider_ids:
        raise ValueError(
            "requested transcription providers are not available in current runtime: "
            + ", ".join(missing_provider_ids)
        )
    return selected


def _fallback_final_transcript_candidate_id(
    transcript_candidates: list[TranscriptCandidate],
) -> str | None:
    if len(transcript_candidates) == 1:
        return transcript_candidates[0].candidate_id
    return None


def _write_blob_model_artifact(
    blob_store: LocalBlobStore,
    key: str,
    payload: Any,
) -> str:
    if hasattr(payload, "model_dump"):
        content = payload.model_dump(mode="json")
    else:
        content = payload
    blob_store.put_bytes(
        key,
        (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return key


def _open_operational_store(
    settings: Settings,
) -> PostgresOperationalStore | SQLiteOperationalStore:
    if settings.state_db_dsn:
        return PostgresOperationalStore(settings.state_db_dsn)
    return SQLiteOperationalStore(settings.state_db_path)


def _open_trace_sink(
    trace_path: Path,
    *,
    live_trace_sink: TraceSink | None = None,
) -> JsonlTraceSink | CompositeTraceSink:
    jsonl_trace_sink = JsonlTraceSink(trace_path)
    if live_trace_sink is None:
        return jsonl_trace_sink
    return CompositeTraceSink(jsonl_trace_sink, live_trace_sink)


def _serialize_request(
    request: RunJobRequest,
    created_at: datetime,
    *,
    source_language: str,
    target_language: str,
) -> str:
    payload = {
        "source": request.source,
        "job_id": request.job_id,
        "created_at": created_at.isoformat(),
        "metadata": request.metadata,
        "tenant_id": request.tenant_id,
        "project_id": request.project_id,
        "target_language": target_language,
        "source_language": source_language,
        "requested_by": request.requested_by,
        "profile_ref": request.profile_ref,
        "asset_id": request.asset_id,
        "asset_context": request.asset_context.model_dump(mode="json")
        if request.asset_context is not None
        else None,
        "reference_transcript_source": request.reference_transcript_source,
        "reference_transcript_format": _resolved_reference_transcript_format(request),
        "reference_mode": request.reference_mode,
        "translation_variant_policy": request.translation_variant_policy,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_translation_candidate(source_path: Path) -> TranslationCandidate:
    return TranslationCandidate.model_validate_json(source_path.read_bytes())


def _resolved_srt_output_path(source_path: Path, output_path: str | Path | None) -> Path:
    if output_path is not None:
        return Path(output_path).expanduser().resolve()
    if source_path.suffix:
        return source_path.with_suffix(".srt")
    return source_path.with_name(f"{source_path.name}.srt")


def _resolved_reference_transcript_format(request: RunJobRequest) -> Literal["srt"] | None:
    if request.reference_transcript_format not in {None, "srt"}:
        raise ValueError("unsupported reference transcript format")
    if request.reference_transcript_source is None:
        if request.reference_mode != "none":
            raise ValueError("reference transcript source is required for evaluate_and_regenerate")
        return None
    if request.reference_mode == "none":
        return request.reference_transcript_format or "srt"
    return request.reference_transcript_format or "srt"


def _resolved_job_languages(request: RunJobRequest, settings: Settings) -> tuple[str, str]:
    return (
        canonicalize_language_code(request.source_language or settings.default_source_language),
        canonicalize_language_code(request.target_language or settings.default_target_language),
    )


def _upsert_current_run_link(
    run_store: PostgresOperationalStore | SQLiteOperationalStore,
    state: GraphState,
) -> None:
    run_store.upsert_historical_run_link(
        HistoricalRunLink(
            run_id=state.run_id,
            media_key=state.job.media_key,
            job_id=state.job.job_id,
            tenant_id=state.job.tenant_id,
            project_id=state.job.project_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
            created_at=state.job.created_at,
            transcript_ref=(
                job_path(state.job, "published", "transcript.json")
                if state.final_transcript_candidate_id is not None
                else None
            ),
            translation_ref=(
                job_path(state.job, "published", "translation.json")
                if not state.translation_failed and not state.human_review_required
                else None
            ),
            transcript_decision_ref=state.final_transcript_decision_ref,
            translation_decision_ref=state.final_translation_decision_ref,
            evaluation_report_ref=state.evaluation_report_ref,
            regenerated_draft_ref=state.regenerated_translation_draft_ref,
        )
    )


def _normalized_optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _asset_context_from_payload(
    media_key: object,
    payload: object,
) -> AssetContext | None:
    if not isinstance(media_key, str) or not media_key:
        return None
    if not isinstance(payload, dict):
        return None
    return AssetContext.model_validate({"media_key": media_key, **payload})


def _resolved_asset_context(
    media_key: str,
    request_asset_context: AssetContextInput | None,
    *,
    existing: AssetContext | None,
) -> AssetContext | None:
    if request_asset_context is None:
        return existing
    update = request_asset_context.model_dump(mode="python")
    if existing is None:
        return AssetContext(media_key=media_key, **update)
    merged = existing.model_dump(mode="python")
    for key, value in update.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            merged[key] = tuple(dict.fromkeys((*merged.get(key, ()), *value)))
            continue
        if isinstance(value, dict):
            merged[key] = {**merged.get(key, {}), **value}
            continue
        merged[key] = value
    merged["updated_at"] = datetime.now(UTC)
    return AssetContext.model_validate(merged)


def _default_output_path(blob_root: Path, state: GraphState) -> Path | None:
    if state.translation_failed or (state.human_review_required and state.approval_ref is None):
        return None
    default_output_ref = job_path(state.job, "exports", "translation.srt")
    default_output_path = blob_root / default_output_ref
    if not default_output_path.exists():
        return None
    return default_output_path.resolve()


def _final_status(state: GraphState) -> str:
    if state.approval_ref is not None:
        return "completed_after_human_review"
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


def _failure_details(
    *,
    final_state: GraphState,
    blob_store: LocalBlobStore,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if not final_state.translation_failed:
        return None, None, ()

    failure_ref = job_path(final_state.job, "published", "translation-failed.json")
    if not blob_store.exists(failure_ref):
        return failure_ref, None, ()

    payload = json.loads(blob_store.read_bytes(failure_ref).decode("utf-8"))
    summary = payload.get("failure_summary")
    reasons = payload.get("failure_reasons")
    normalized_summary = summary if isinstance(summary, str) else None
    normalized_reasons = (
        tuple(reason for reason in reasons if isinstance(reason, str))
        if isinstance(reasons, list)
        else ()
    )
    return failure_ref, normalized_summary, normalized_reasons


def _terminal_run_error(
    *,
    final_state: GraphState,
    failure_ref: str | None,
    failure_summary: str | None,
    failure_reasons: tuple[str, ...],
    translation_provider_id: str | None = None,
) -> dict[str, object] | None:
    if not final_state.translation_failed:
        return None

    message = failure_summary or next(iter(failure_reasons), "All translation variants failed.")
    error: dict[str, object] = {
        "category": "translation_failed",
        "reason": "all_translation_variants_failed",
        "message": message,
        "failure_ref": failure_ref,
        "failure_summary": failure_summary,
        "failure_reasons": list(failure_reasons),
        "retryable": _translation_failure_retryable(failure_reasons),
    }
    if translation_provider_id:
        error["provider_id"] = translation_provider_id
    error_code = _translation_failure_code(failure_reasons)
    if error_code is not None:
        error["code"] = error_code
    return error


def _translation_failure_code(failure_reasons: tuple[str, ...]) -> str | None:
    lower_reasons = tuple(reason.lower() for reason in failure_reasons)
    if any("insufficient_quota" in reason for reason in lower_reasons):
        return "insufficient_quota"
    if any("current quota" in reason for reason in lower_reasons):
        return "insufficient_quota"
    if any("rate limit" in reason for reason in lower_reasons):
        return "rate_limit"
    return None


def _translation_failure_retryable(failure_reasons: tuple[str, ...]) -> bool:
    error_code = _translation_failure_code(failure_reasons)
    if error_code == "insufficient_quota":
        return False
    if error_code == "rate_limit":
        return True
    return False


def _resume_commands(run_id: str, state: GraphState) -> tuple[str, ...]:
    if state.review_required_stage != "translation":
        return ()
    return (
        f"uv run translation-agent review-job {run_id}",
        f"uv run translation-agent approve-review {run_id} --candidate-id <candidate-id>",
    )


def _investigation_ref(state: GraphState, *, stage: str) -> str | None:
    return next(
        (
            fact.source_ref
            for fact in state.routing_facts
            if fact.fact_type == "investigation_ref"
            and fact.stage == f"adjudicate_{stage}"
            and fact.source_ref is not None
        ),
        None,
    )
