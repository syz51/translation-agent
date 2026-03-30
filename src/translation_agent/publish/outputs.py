"""Dry-run publishing helpers."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState
from translation_agent.models import PublishedArtifacts
from translation_agent.nodes.common import (
    published_artifacts_key,
    select_transcript_candidates,
    select_translation_candidates,
    write_model_artifact,
)


def publish_outputs(state: GraphState, runtime: WorkflowRuntime) -> tuple[PublishedArtifacts, str]:
    """Persist canonical dry-run artifacts and return the manifest ref."""

    transcript_ref: str | None = None
    translation_ref: str | None = None

    if state.final_transcript_candidate_id is not None:
        transcript_candidates = select_transcript_candidates(
            runtime,
            job_id=state.job.job_id,
            candidate_ids=(state.final_transcript_candidate_id,),
        )
        if transcript_candidates:
            transcript = transcript_candidates[0]
            transcript_ref = f"published/{state.job.job_id}/transcript.json"
            write_model_artifact(runtime, transcript_ref, transcript)

    if state.final_translation_candidate_id is not None and not state.human_review_required:
        translation_candidates = select_translation_candidates(
            runtime,
            job_id=state.job.job_id,
            candidate_ids=(state.final_translation_candidate_id,),
        )
        if translation_candidates:
            translation = translation_candidates[0]
            translation_ref = f"published/{state.job.job_id}/translation.json"
            write_model_artifact(runtime, translation_ref, translation)

    scorecard_ref = f"published/{state.job.job_id}/scorecard.json"
    write_model_artifact(
        runtime,
        scorecard_ref,
        {
            "run_id": state.run_id,
            "human_review_required": state.human_review_required,
            "translation_failed": state.translation_failed,
            "routing_fact_count": len(state.routing_facts),
        },
    )
    artifacts = PublishedArtifacts(
        final_transcript_ref=transcript_ref,
        final_translation_ref=translation_ref,
        scorecard_refs=(scorecard_ref,),
        trace_refs=(f"traces/{state.run_id}.jsonl",),
        export_refs=(
            f"exports/{state.job.job_id}.txt",
            f"exports/{state.job.job_id}.json",
        ),
        downstream_delivery_refs=(f"deliveries/{state.job.job_id}.json",),
    )
    manifest_key = published_artifacts_key(state.job.job_id)
    manifest_ref = write_model_artifact(runtime, manifest_key, artifacts)
    return artifacts, manifest_ref
