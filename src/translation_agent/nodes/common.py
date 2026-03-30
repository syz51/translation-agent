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
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.normalization import (
    normalize_transcript_candidate,
    normalize_translation_candidate,
)
from translation_agent.storage import job_path

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
) -> MemoryQuery:
    """Create a deterministic recall request for review and adjudication nodes."""

    return MemoryQuery(
        job=state.job,
        stage=stage,
        query_text=f"{stage} dry-run context for {state.job.project_id}",
        candidate_ids=candidate_ids,
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


def raw_translation_candidate_key(job: JobContext, prompt_variant_id: str) -> str:
    return job_path(job, "raw", "translation-candidates", f"{prompt_variant_id}.json")


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


def audio_artifact_key(job: JobContext) -> str:
    return job_path(job, "artifacts", "audio.json")


def published_artifacts_key(job: JobContext) -> str:
    return job_path(job, "published", "artifacts.json")


def translation_failure_key(job: JobContext) -> str:
    return job_path(job, "published", "translation-failed.json")


def select_transcript_candidates(
    runtime: WorkflowRuntime,
    *,
    job_id: str,
    candidate_ids: tuple[str, ...],
) -> list[TranscriptCandidate]:
    """Read transcript candidates from the in-memory decision store."""

    candidates = runtime.decision_store.list_transcript_candidates(job_id)
    selected_ids = set(candidate_ids)
    selected = [candidate for candidate in candidates if candidate.candidate_id in selected_ids]
    return sorted(selected, key=transcript_sort_key)


def select_translation_candidates(
    runtime: WorkflowRuntime,
    *,
    job_id: str,
    candidate_ids: tuple[str, ...],
) -> list[TranslationCandidate]:
    """Read translation candidates from the in-memory decision store."""

    candidates = runtime.decision_store.list_translation_candidates(job_id)
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


def translation_sort_key(candidate: TranslationCandidate) -> tuple[str, str]:
    return (candidate.prompt_variant_id, candidate.candidate_id)


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
