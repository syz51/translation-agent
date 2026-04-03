"""Review nodes for deterministic prose review and parser-backed bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    MemoryBundle,
    ReviewBundle,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.models.review import ReviewStage
from translation_agent.nodes.common import (
    TRANSCRIPT_REVIEW_STAGE,
    TRANSLATION_REVIEW_STAGE,
    build_memory_query,
    review_memory_bundle_key,
    select_transcript_candidates,
    select_translation_candidates,
    transcript_candidate_key,
    transcript_review_key,
    translation_candidate_key,
    translation_review_key,
    write_model_artifact,
)
from translation_agent.parallelism import ordered_parallel_map
from translation_agent.review import (
    PARSER_VERSION,
    build_review_context,
    build_review_prompt,
    build_structured_review,
    render_reviewer_output,
    review_bundle_from_draft,
    reviewer_roles_for_stage,
)


@dataclass(frozen=True, slots=True)
class _ReviewTask:
    stage: ReviewStage
    reviewer_role: str
    candidates: tuple[TranscriptCandidate, ...] | tuple[TranslationCandidate, ...]
    memory_bundle: MemoryBundle
    final_transcript: TranscriptCandidate | None


def review_transcripts(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Generate transcript reviewer prose and persist parsed bundles."""

    candidates = select_transcript_candidates(
        runtime,
        job=state.job,
        candidate_ids=state.transcript_candidate_ids,
    )
    memory_bundle = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="review_transcripts",
            candidate_ids=state.transcript_candidate_ids,
            provider_ids=tuple(candidate.provider_id for candidate in candidates),
        )
    )
    if not candidates:
        raise RuntimeError("review_transcripts requires at least one normalized candidate")

    memory_ref = write_model_artifact(
        runtime,
        review_memory_bundle_key(state.job, TRANSCRIPT_REVIEW_STAGE),
        memory_bundle,
    )
    review_ids = _generate_and_persist_reviews(
        state=state,
        runtime=runtime,
        tasks=tuple(
            _ReviewTask(
                stage=TRANSCRIPT_REVIEW_STAGE,
                reviewer_role=spec.reviewer_role,
                candidates=tuple(candidates),
                memory_bundle=memory_bundle,
                final_transcript=None,
            )
            for spec in reviewer_roles_for_stage(TRANSCRIPT_REVIEW_STAGE)
        ),
    )
    first_review_ref = transcript_review_key(state.job, review_ids[0])
    return {
        "current_stage": "review_transcripts",
        "transcript_review_ids": review_ids,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_transcripts",
                fact_type="review_count",
                value=str(len(review_ids)),
                source_ref=first_review_ref,
            ),
            RoutingFact(
                stage="review_transcripts",
                fact_type="review_memory_bundle",
                value=memory_ref,
                source_ref=memory_ref,
            ),
        ),
    }


def review_translations(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Generate translation reviewer prose and persist parsed bundles."""

    candidates = select_translation_candidates(
        runtime,
        job=state.job,
        candidate_ids=state.translation_candidate_ids,
    )
    transcript_provider_ids = tuple(
        transcript.provider_id
        for transcript in select_transcript_candidates(
            runtime,
            job=state.job,
            candidate_ids=tuple(
                candidate.source_transcript_candidate_id or "" for candidate in candidates
            ),
        )
    )
    memory_bundle = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="review_translations",
            candidate_ids=state.translation_candidate_ids,
            provider_ids=transcript_provider_ids,
            prompt_variant_ids=tuple(candidate.prompt_variant_id for candidate in candidates),
            model_ids=tuple(candidate.model_id for candidate in candidates),
        )
    )
    if not candidates:
        return {
            "current_stage": "review_translations",
            "translation_review_ids": (),
            "routing_facts": state.routing_facts
            + (
                RoutingFact(
                    stage="review_translations",
                    fact_type="review_skipped",
                    value="no_translation_candidates",
                    source_ref=None,
                ),
            ),
        }

    memory_ref = write_model_artifact(
        runtime,
        review_memory_bundle_key(state.job, TRANSLATION_REVIEW_STAGE),
        memory_bundle,
    )
    final_transcript = _load_final_transcript_candidate(state, runtime)
    review_ids = _generate_and_persist_reviews(
        state=state,
        runtime=runtime,
        tasks=tuple(
            _ReviewTask(
                stage=TRANSLATION_REVIEW_STAGE,
                reviewer_role=spec.reviewer_role,
                candidates=tuple(candidates),
                memory_bundle=memory_bundle,
                final_transcript=final_transcript,
            )
            for spec in reviewer_roles_for_stage(TRANSLATION_REVIEW_STAGE)
        ),
    )
    first_review_ref = translation_review_key(state.job, review_ids[0])
    return {
        "current_stage": "review_translations",
        "translation_review_ids": review_ids,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_translations",
                fact_type="review_count",
                value=str(len(review_ids)),
                source_ref=first_review_ref,
            ),
            RoutingFact(
                stage="review_translations",
                fact_type="review_memory_bundle",
                value=memory_ref,
                source_ref=memory_ref,
            ),
        ),
    }


def _generate_and_persist_reviews(
    *,
    state: GraphState,
    runtime: WorkflowRuntime,
    tasks: tuple[_ReviewTask, ...],
) -> tuple[str, ...]:
    gathered = ordered_parallel_map(
        tasks,
        max_workers=runtime.parallelism.review_max_workers,
        worker=lambda task: _build_review_bundle(state=state, task=task),
        sort_key=lambda input_index, _task: (input_index,),
    )
    review_ids: list[str] = []
    for task, result in zip(tasks, gathered, strict=True):
        if result.error is not None:
            raise result.error
        if result.value is None:  # pragma: no cover - defensive
            raise RuntimeError(f"missing {task.stage} review result for {task.reviewer_role}")
        review_ids.append(_persist_review(runtime, state.job, result.value))
    return tuple(review_ids)


def _build_review_bundle(
    *,
    state: GraphState,
    task: _ReviewTask,
) -> ReviewBundle:
    stage = task.stage
    reviewer_role = task.reviewer_role
    candidates = task.candidates
    final_transcript = task.final_transcript
    review_context = build_review_context(
        run_id=state.run_id,
        stage=stage,
        reviewer_role=reviewer_role,
        job=state.job,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        memory_bundle=task.memory_bundle,
    )
    candidate_refs = tuple(
        transcript_candidate_key(state.job, candidate.candidate_id)
        if stage == TRANSCRIPT_REVIEW_STAGE
        else translation_candidate_key(state.job, candidate.candidate_id)
        for candidate in candidates
    )
    raw_payload_refs = _raw_payload_refs(stage=stage, candidates=candidates)
    prompt_text = build_review_prompt(
        review_context,
        candidate_refs=candidate_refs,
        raw_payload_refs=raw_payload_refs,
        final_transcript_ref=(
            transcript_candidate_key(state.job, final_transcript.candidate_id)
            if final_transcript is not None
            else None
        ),
    )
    raw_review_text = render_reviewer_output(
        review_context,
        candidates=candidates,
        prompt_text=prompt_text,
        final_transcript=final_transcript,
    )
    draft = build_structured_review(
        review_context,
        candidates=candidates,
        final_transcript=final_transcript,
    )
    review = review_bundle_from_draft(
        review_id=_review_id(
            job=state.job,
            stage=stage,
            reviewer_role=reviewer_role,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        ),
        job_id=state.job.job_id,
        stage=stage,
        reviewer_role=reviewer_role,
        raw_review_text=raw_review_text,
        draft=draft,
    ).model_copy(update={"parser_version": PARSER_VERSION})
    return review


def _load_final_transcript_candidate(
    state: GraphState,
    runtime: WorkflowRuntime,
) -> TranscriptCandidate:
    if state.final_transcript_candidate_id is None:
        raise RuntimeError("translation review requires a final transcript candidate")
    candidates = select_transcript_candidates(
        runtime,
        job=state.job,
        candidate_ids=(state.final_transcript_candidate_id,),
    )
    if not candidates:
        raise RuntimeError("final transcript candidate not found for translation review")
    return candidates[0]


def _review_id(
    *,
    job,
    stage: ReviewStage,
    reviewer_role: str,
    candidate_ids: tuple[str, ...],
) -> str:
    prefix = "rev-tr" if stage == TRANSCRIPT_REVIEW_STAGE else "rev-tl"
    payload = "|".join(
        (
            job.tenant_id,
            job.project_id,
            job.job_id,
            stage,
            reviewer_role,
            *candidate_ids,
        )
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:12]
    role_slug = reviewer_role.replace("_", "-")
    return f"{prefix}-{role_slug}-{digest}"


def _persist_review(runtime: WorkflowRuntime, job, review: ReviewBundle) -> str:
    key = (
        transcript_review_key(job, review.review_id)
        if review.stage == TRANSCRIPT_REVIEW_STAGE
        else translation_review_key(job, review.review_id)
    )
    write_model_artifact(runtime, key, review)
    return review.review_id


def _raw_payload_refs(
    *,
    stage: ReviewStage,
    candidates: tuple[TranscriptCandidate, ...] | tuple[TranslationCandidate, ...],
) -> tuple[str, ...]:
    if stage == TRANSCRIPT_REVIEW_STAGE:
        return tuple(
            candidate.raw_payload_ref
            for candidate in candidates
            if isinstance(candidate, TranscriptCandidate) and candidate.raw_payload_ref is not None
        )
    return tuple(
        candidate.raw_response_ref
        for candidate in candidates
        if isinstance(candidate, TranslationCandidate) and candidate.raw_response_ref is not None
    )
