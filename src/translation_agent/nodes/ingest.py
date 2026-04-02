"""Ingest node for the deterministic dry-run workflow."""

from __future__ import annotations

from typing import cast

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import HistoricalRunLink
from translation_agent.storage.operational import OperationalStore


def ingest_job(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Resolve the initial run facts into ref-only graph state."""

    job = state.job
    if hasattr(runtime.run_store, "resolve_asset"):
        run_store = cast("OperationalStore", runtime.run_store)
        asset = run_store.resolve_asset(
            asset_id=state.job.asset_id,
            media_fingerprint=state.job.media_fingerprint,
            first_seen_run_id=state.run_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
        )
        job = state.job.model_copy(
            update={
                "asset_id": asset.asset_id,
                "media_fingerprint": asset.media_fingerprint,
                "media_key": asset.media_key,
            }
        )
        run_store.upsert_historical_run_link(
            HistoricalRunLink(
                run_id=state.run_id,
                media_key=asset.media_key,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                project_id=job.project_id,
                source_language=job.source_language,
                target_language=job.target_language,
                created_at=job.created_at,
            )
        )

    return {
        "current_stage": "ingest",
        "job": job,
        "source_artifact_ref": runtime.source_artifact_ref,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="ingest",
                fact_type="job_initialized",
                value=state.job.job_id,
                source_ref=runtime.source_artifact_ref,
            ),
            RoutingFact(
                stage="ingest",
                fact_type="scenario",
                value=runtime.scenario,
                source_ref=runtime.source_artifact_ref,
            ),
            RoutingFact(
                stage="ingest",
                fact_type="media_key_resolved",
                value=job.media_key,
                source_ref=runtime.source_artifact_ref,
            ),
        ),
    }
