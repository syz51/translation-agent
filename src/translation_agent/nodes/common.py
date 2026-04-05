"""Shared helpers for deterministic workflow nodes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from translation_agent.graph.runtime import WorkflowRuntime, runtime_metadata
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    JobContext,
    MemoryQuery,
    RequestContext,
    ReviewBundle,
    SynthesizedTranscriptArtifact,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.normalization import (
    normalize_transcript_candidate,
    normalize_translation_candidate,
)
from translation_agent.storage import job_path, operational_job_key

TRANSCRIPT_REVIEW_STAGE = "transcript"
TRANSLATION_REVIEW_STAGE = "translation"


def build_request_context(state: GraphState, runtime: WorkflowRuntime) -> RequestContext:
    """Create the typed request context shared by adapter-backed nodes."""

    return RequestContext(
        run_id=state.run_id,
        job=state.job,
        source_artifact_ref=state.source_artifact_ref or runtime.source_artifact_ref,
        metadata=runtime_metadata({}, runtime),
    )


def build_memory_query(
    state: GraphState,
    *,
    stage: str,
    candidate_ids: tuple[str, ...],
    provider_ids: tuple[str, ...] = (),
    prompt_variant_ids: tuple[str, ...] = (),
    model_ids: tuple[str, ...] = (),
    failure_tags: tuple[str, ...] = (),
) -> MemoryQuery:
    """Create a deterministic recall request for review and adjudication nodes."""

    query_parts = [
        stage,
        f"{state.job.source_language}->{state.job.target_language}",
    ]
    if provider_ids:
        query_parts.append("providers:" + ",".join(sorted(dict.fromkeys(provider_ids))))
    if prompt_variant_ids:
        query_parts.append("variants:" + ",".join(sorted(dict.fromkeys(prompt_variant_ids))))
    if model_ids:
        query_parts.append("models:" + ",".join(sorted(dict.fromkeys(model_ids))))
    if failure_tags:
        query_parts.append("failure_tags:" + ",".join(sorted(dict.fromkeys(failure_tags))))
    return MemoryQuery(
        job=state.job,
        stage=stage,
        query_text=" | ".join(query_parts),
        asset_context=state.job.asset_context,
        candidate_ids=candidate_ids,
        provider_ids=provider_ids,
        prompt_variant_ids=prompt_variant_ids,
        model_ids=model_ids,
        series_id=state.job.asset_context.series_id if state.job.asset_context else None,
        franchise_id=state.job.asset_context.franchise_id if state.job.asset_context else None,
        speaker_ids=state.job.asset_context.speaker_ids if state.job.asset_context else (),
        content_type=state.job.asset_context.content_type if state.job.asset_context else None,
        topic_tags=state.job.asset_context.topic_tags if state.job.asset_context else (),
        style_profile_id=(
            state.job.asset_context.style_profile_id if state.job.asset_context else None
        ),
        failure_tags=failure_tags,
        media_key=state.job.media_key,
    )


def append_routing_fact(
    state: GraphState,
    *,
    fact_type: str,
    value: str,
    source_ref: str | None = None,
) -> tuple[RoutingFact, ...]:
    """Return a new routing-fact tuple with one appended fact."""

    return state.routing_facts + (
        RoutingFact(
            stage=state.current_stage,
            fact_type=fact_type,
            value=value,
            source_ref=source_ref,
        ),
    )


def append_routing_facts(
    state: GraphState,
    facts: Iterable[RoutingFact],
) -> tuple[RoutingFact, ...]:
    """Append multiple routing facts while preserving immutability."""

    return state.routing_facts + tuple(facts)


def write_model_artifact(
    runtime: WorkflowRuntime, key: str, payload: BaseModel | dict[str, Any]
) -> str:
    """Persist a model or mapping to the blob store and return the blob key."""

    content: dict[str, Any]
    if isinstance(payload, BaseModel):
        content = payload.model_dump(mode="json")
    else:
        content = payload
    runtime.blob_store.put_bytes(
        key,
        (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return key


def read_model_artifact[ModelT: BaseModel](
    runtime: WorkflowRuntime,
    key: str,
    model_type: type[ModelT],
) -> ModelT:
    """Load a persisted contract model from the blob store."""

    return model_type.model_validate_json(runtime.blob_store.read_bytes(key))


def transcript_candidate_key(job: JobContext, candidate_id: str) -> str:
    return job_path(job, "candidates", "transcripts", f"{candidate_id}.json")


def raw_transcript_candidate_key(job: JobContext, provider_id: str) -> str:
    return job_path(job, "raw", "transcript-candidates", f"{provider_id}.json")


def translation_candidate_key(job: JobContext, candidate_id: str) -> str:
    return job_path(job, "candidates", "translations", f"{candidate_id}.json")


def canonical_transcript_span_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "canonical-transcript-spans.json")


def transcript_synthesis_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "transcript-synthesis.json")


def transcript_selector_record_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "transcript-selector.json")


def transcript_span_review_record_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "transcript-span-review.json")


def final_transcript_artifact_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "final-transcript.json")


def raw_translation_candidate_key(
    job: JobContext,
    prompt_variant_id: str,
    source_transcript_token: str,
) -> str:
    return job_path(
        job,
        "raw",
        "translation-candidates",
        f"{prompt_variant_id}-{source_transcript_token}.json",
    )


def staged_transcript_candidate_key(job: JobContext, candidate_id: str) -> str:
    return job_path(job, "staging", "transcripts", f"{candidate_id}.json")


def staged_translation_candidate_key(job: JobContext, candidate_id: str) -> str:
    return job_path(job, "staging", "translations", f"{candidate_id}.json")


def transcript_review_key(job: JobContext, review_id: str) -> str:
    return job_path(job, "reviews", "transcript", f"{review_id}.json")


def translation_review_key(job: JobContext, review_id: str) -> str:
    return job_path(job, "reviews", "translation", f"{review_id}.json")


def transcript_decision_key(job: JobContext) -> str:
    return job_path(job, "decisions", "transcript.json")


def translation_decision_key(job: JobContext) -> str:
    return job_path(job, "decisions", "translation.json")


def transcript_investigation_key(job: JobContext) -> str:
    return job_path(job, "investigations", "transcript.json")


def translation_investigation_key(job: JobContext) -> str:
    return job_path(job, "investigations", "translation.json")


def memory_batch_key(job: JobContext, batch_id: str) -> str:
    return job_path(job, "memory", "batches", f"{batch_id}.json")


def memory_consolidation_key(job: JobContext, consolidation_id: str) -> str:
    return job_path(job, "memory", "consolidations", f"{consolidation_id}.json")


def prompt_evolution_key(job: JobContext, proposal_id: str) -> str:
    return job_path(job, "memory", "prompt-evolution", f"{proposal_id}.json")


def review_memory_bundle_key(job: JobContext, stage: str) -> str:
    return job_path(job, "memory", "recall", f"{stage}-review.json")


def adjudication_memory_bundle_key(job: JobContext, stage: str) -> str:
    return job_path(job, "memory", "recall", f"{stage}-adjudication.json")


def audio_artifact_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "audio.json")


def published_artifacts_key(job: JobContext) -> str:
    return job_path(job, "published", "artifacts.json")


def translation_failure_key(job: JobContext) -> str:
    return job_path(job, "published", "translation-failed.json")


def approval_record_key(job: JobContext) -> str:
    return job_path(job, "approvals", "translation.json")


def review_resolution_key(job: JobContext) -> str:
    return job_path(job, "review-resolutions", "translation.json")


def review_draft_key(job: JobContext) -> str:
    return job_path(job, "review", "draft-resolution.json")


def transcript_approval_learning_key(job: JobContext) -> str:
    return job_path(job, "learning", "transcript-approval.json")


def review_preview_key(job: JobContext, candidate_id: str, suffix: str) -> str:
    return job_path(job, "review", "previews", f"{candidate_id}{suffix}")


def select_transcript_candidates(
    runtime: WorkflowRuntime,
    *,
    job: JobContext,
    candidate_ids: tuple[str, ...],
) -> list[TranscriptCandidate]:
    """Read transcript candidates from the in-memory decision store."""

    candidates = runtime.decision_store.list_transcript_candidates(
        job.job_id,
        storage_job_id=operational_job_key(job),
    )
    selected_ids = set(candidate_ids)
    selected = [candidate for candidate in candidates if candidate.candidate_id in selected_ids]
    return sorted(selected, key=transcript_sort_key)


def select_translation_candidates(
    runtime: WorkflowRuntime,
    *,
    job: JobContext,
    candidate_ids: tuple[str, ...],
) -> list[TranslationCandidate]:
    """Read translation candidates from the in-memory decision store."""

    candidates = runtime.decision_store.list_translation_candidates(
        job.job_id,
        storage_job_id=operational_job_key(job),
    )
    selected_ids = set(candidate_ids)
    selected = [candidate for candidate in candidates if candidate.candidate_id in selected_ids]
    return sorted(selected, key=translation_sort_key)


def load_reviews(
    runtime: WorkflowRuntime,
    *,
    stage: str,
    job: JobContext,
    review_ids: tuple[str, ...],
) -> tuple[ReviewBundle, ...]:
    """Load review bundles from the blob store."""

    key_factory = (
        transcript_review_key if stage == TRANSCRIPT_REVIEW_STAGE else translation_review_key
    )
    return tuple(
        read_model_artifact(runtime, key_factory(job, review_id), ReviewBundle)
        for review_id in review_ids
    )


def transcript_sort_key(candidate: TranscriptCandidate) -> tuple[int, str]:
    return (int(candidate.metadata.get("provider_rank", 100)), candidate.candidate_id)


def translation_sort_key(candidate: TranslationCandidate) -> tuple[str, str, str]:
    return (
        candidate.source_transcript_ref
        or candidate.source_transcript_candidate_id
        or candidate.final_transcript_ref
        or "",
        candidate.prompt_variant_id,
        candidate.candidate_id,
    )


def synthesized_transcript_as_candidate(
    artifact: SynthesizedTranscriptArtifact,
) -> TranscriptCandidate:
    """Project the synthesized transcript artifact into adapter-friendly candidate shape."""

    provider_ids = artifact.quality_metrics.provider_support_summary
    return TranscriptCandidate(
        candidate_id=artifact.artifact_id,
        job_id=artifact.job_id,
        provider_id="synthesized-transcript",
        provider_request_id=None,
        language=artifact.language,
        segments=artifact.final_segments,
        full_text=artifact.full_text,
        speaker_map={},
        timing_resolution="segment",
        raw_payload_ref=None,
        normalization_version=str(
            artifact.transcript_metadata.get("normalization_version", "transcript-synthesis-v1")
        ),
        metadata={
            "artifact_id": artifact.artifact_id,
            "provider_support_summary": provider_ids,
            "synthesis_status": artifact.status,
        },
    )


def normalized_transcript(candidate: TranscriptCandidate) -> TranscriptCandidate:
    """Apply the canonical dry-run normalization tweaks."""

    return normalize_transcript_candidate(candidate)


def normalized_translation(candidate: TranslationCandidate) -> TranslationCandidate:
    """Apply the canonical dry-run normalization tweaks."""

    return normalize_translation_candidate(candidate)


def strip_private_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop transport-only metadata entries before persistence."""

    return {key: value for key, value in payload.items() if not key.startswith("_")}


def cleanup_local_path(path_value: str | None) -> None:
    """Best-effort cleanup for temporary local files created during adapter execution."""

    if not path_value:
        return
    path = Path(path_value)
    if path.exists():
        path.unlink()
    parent = path.parent
    if parent.exists():
        try:
            parent.rmdir()
        except OSError:
            return
