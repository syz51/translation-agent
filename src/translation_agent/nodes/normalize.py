"""Normalization nodes for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import TranscriptCandidate, TranslationCandidate
from translation_agent.nodes.common import (
    normalized_transcript,
    normalized_translation,
    read_model_artifact,
    transcript_candidate_key,
    translation_candidate_key,
    write_model_artifact,
)


def normalize_transcripts(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Normalize raw transcript payloads into canonical candidates."""

    candidates: list[TranscriptCandidate] = []
    for raw_ref in state.raw_transcript_candidate_refs:
        candidate = read_model_artifact(runtime, raw_ref, TranscriptCandidate)
        normalized = normalized_transcript(candidate)
        runtime.decision_store.save_transcript_candidate(normalized)
        write_model_artifact(runtime, transcript_candidate_key(normalized.candidate_id), normalized)
        candidates.append(normalized)

    if not candidates:
        raise RuntimeError("no transcript candidates survived normalization")

    return {
        "current_stage": "normalize_transcripts",
        "transcript_candidate_ids": tuple(candidate.candidate_id for candidate in candidates),
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="normalize_transcripts",
                fact_type="surviving_transcript_candidates",
                value=str(len(candidates)),
                source_ref=transcript_candidate_key(candidates[0].candidate_id),
            ),
        ),
    }


def normalize_translations(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Normalize raw translation payloads into canonical candidates."""

    candidates: list[TranslationCandidate] = []
    for raw_ref in state.raw_translation_candidate_refs:
        candidate = read_model_artifact(runtime, raw_ref, TranslationCandidate)
        normalized = normalized_translation(candidate)
        runtime.decision_store.save_translation_candidate(normalized)
        candidate_key = translation_candidate_key(normalized.candidate_id)
        write_model_artifact(runtime, candidate_key, normalized)
        candidates.append(normalized)

    return {
        "current_stage": "normalize_translations",
        "translation_candidate_ids": tuple(candidate.candidate_id for candidate in candidates),
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="normalize_translations",
                fact_type="surviving_translation_candidates",
                value=str(len(candidates)),
                source_ref=(
                    translation_candidate_key(candidates[0].candidate_id) if candidates else None
                ),
            ),
        ),
    }
