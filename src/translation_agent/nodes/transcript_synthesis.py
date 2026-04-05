"""Transcript span synthesis nodes."""

from __future__ import annotations

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    AdjudicationScorecard,
    FinalTranscriptDecision,
    TranscriptCanonicalSpanTable,
    TranscriptSynthesisRecord,
    TranscriptSynthesisReview,
)
from translation_agent.nodes.common import (
    canonical_transcript_span_key,
    final_transcript_artifact_key,
    read_model_artifact,
    select_transcript_candidates,
    transcript_decision_key,
    transcript_investigation_key,
    transcript_selector_record_key,
    transcript_span_review_record_key,
    transcript_synthesis_key,
    write_model_artifact,
)
from translation_agent.storage import operational_job_key
from translation_agent.transcript_synthesis import (
    blocking_failures_for_artifact,
    build_canonical_transcript_spans,
    build_span_candidates,
    run_global_adjudicator,
    run_reviewer_agent,
    run_selector_agent,
)
from translation_agent.transcript_synthesis import (
    materialize_synthesized_transcript as build_final_transcript_artifact,
)


def build_canonical_transcript_spans_node(
    state: GraphState, runtime: WorkflowRuntime
) -> dict[str, object]:
    """Build and persist the canonical transcript span table."""

    candidates = tuple(
        select_transcript_candidates(
            runtime,
            job=state.job,
            candidate_ids=state.transcript_candidate_ids,
        )
    )
    if not candidates:
        raise RuntimeError("canonical transcript span construction requires transcript candidates")
    canonical_spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(canonical_spans, candidates)
    payload = TranscriptCanonicalSpanTable(
        job_id=state.job.job_id,
        canonical_spans=canonical_spans,
        span_candidates=span_candidates,
    )
    canonical_ref = write_model_artifact(runtime, canonical_transcript_span_key(state.job), payload)
    return {
        "current_stage": "build_canonical_transcript_spans",
        "canonical_transcript_span_ref": canonical_ref,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="build_canonical_transcript_spans",
                fact_type="canonical_span_count",
                value=str(len(canonical_spans)),
                source_ref=canonical_ref,
            ),
        ),
    }


def select_transcript_spans(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Run the selector agent over canonical transcript spans."""

    span_table = _load_span_table(state, runtime)
    record = run_selector_agent(
        job=state.job,
        run_id=state.run_id,
        runtime=runtime,
        spans=span_table.canonical_spans,
        span_candidates=span_table.span_candidates,
    )
    synthesis_ref = write_model_artifact(runtime, transcript_selector_record_key(state.job), record)
    return {
        "current_stage": "select_transcript_spans",
        "transcript_selector_ref": synthesis_ref,
        "transcript_unresolved_span_count": len(record.unresolved_span_ids),
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="select_transcript_spans",
                fact_type="selector_unresolved_span_count",
                value=str(len(record.unresolved_span_ids)),
                source_ref=synthesis_ref,
            ),
        ),
    }


def review_transcript_spans(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Run the reviewer agent over the synthesized transcript draft."""

    span_table = _load_span_table(state, runtime)
    selector_record = _load_selector_record(state, runtime)
    review = run_reviewer_agent(
        job=state.job,
        run_id=state.run_id,
        runtime=runtime,
        spans=span_table.canonical_spans,
        span_candidates=span_table.span_candidates,
        selector_record=selector_record,
    )
    review_ref = write_model_artifact(runtime, transcript_span_review_record_key(state.job), review)
    return {
        "current_stage": "review_transcript_spans",
        "transcript_span_review_ref": review_ref,
        "transcript_unresolved_span_count": len(review.unresolved_span_ids),
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="review_transcript_spans",
                fact_type="review_unresolved_span_count",
                value=str(len(review.unresolved_span_ids)),
                source_ref=review_ref,
            ),
        ),
    }


def adjudicate_transcript_spans(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Run the global adjudicator over transcript spans still unresolved after review."""

    span_table = _load_span_table(state, runtime)
    selector_record = _load_selector_record(state, runtime)
    review = _load_review_record(state, runtime)
    adjudication = run_global_adjudicator(
        job=state.job,
        run_id=state.run_id,
        runtime=runtime,
        spans=span_table.canonical_spans,
        span_candidates=span_table.span_candidates,
        selector_record=selector_record,
        review=review,
    )
    adjudication_ref = write_model_artifact(
        runtime,
        transcript_synthesis_key(state.job),
        adjudication,
    )
    return {
        "current_stage": "adjudicate_transcript_spans",
        "final_transcript_synthesis_ref": adjudication_ref,
        "transcript_unresolved_span_count": len(adjudication.unresolved_span_ids),
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="adjudicate_transcript_spans",
                fact_type="global_unresolved_span_count",
                value=str(len(adjudication.unresolved_span_ids)),
                source_ref=adjudication_ref,
            ),
        ),
    }


def materialize_synthesized_transcript_node(
    state: GraphState, runtime: WorkflowRuntime
) -> dict[str, object]:
    """Persist the final synthesized transcript artifact and transcript decision."""

    span_table = _load_span_table(state, runtime)
    review = _load_review_record(state, runtime)
    selector_record = _load_selector_record(state, runtime)
    global_record = _load_global_record(state, runtime)
    artifact = build_final_transcript_artifact(
        job=state.job,
        run_id=state.run_id,
        language=state.job.source_language,
        spans=span_table.canonical_spans,
        span_candidates=span_table.span_candidates,
        selector_record=selector_record,
        review=review,
        global_record=global_record,
    )
    artifact_ref = write_model_artifact(runtime, final_transcript_artifact_key(state.job), artifact)
    blocking_failures = blocking_failures_for_artifact(artifact.quality_metrics)
    transcript_failed = bool(blocking_failures)
    primary_candidate_id = _primary_transcript_candidate_id(artifact)
    investigation_ref = write_model_artifact(
        runtime,
        transcript_investigation_key(state.job),
        {
            "job_id": state.job.job_id,
            "run_id": state.run_id,
            "stage": "transcript_synthesis",
            "canonical_span_ref": state.canonical_transcript_span_ref,
            "selector_record_ref": state.transcript_selector_ref,
            "synthesis_record_ref": state.final_transcript_synthesis_ref,
            "span_review_ref": state.transcript_span_review_ref,
            "transcript_artifact_ref": artifact_ref,
            "synthesis_status": "transcript_failed" if transcript_failed else "complete",
            "unresolved_span_count": artifact.quality_metrics.unresolved_span_count,
            "unresolved_span_ids": list(global_record.unresolved_span_ids),
            "blocking_failures": blocking_failures,
            "provider_support_summary": artifact.quality_metrics.provider_support_summary,
        },
    )
    decision = FinalTranscriptDecision(
        job_id=state.job.job_id,
        winner_candidate_id=primary_candidate_id,
        transcript_artifact_ref=artifact_ref,
        canonical_span_ref=state.canonical_transcript_span_ref,
        synthesis_record_ref=state.final_transcript_synthesis_ref,
        span_review_ref=state.transcript_span_review_ref,
        decision_mode="conflict_investigation" if transcript_failed else "automatic_finalize",
        decision_confidence=0.42 if transcript_failed else 0.88,
        rationale_summary=(
            "Synthesized transcript failed assembly invariants and translation was skipped."
            if transcript_failed
            else "Synthesized transcript passed base selection, review, and fill adjudication."
        ),
        review_refs=(
            (state.transcript_span_review_ref,)
            if state.transcript_span_review_ref is not None
            else ()
        ),
        investigation_ref=investigation_ref,
        disagreement_bucket="unresolved" if transcript_failed else "low",
        adjudication_scorecard=_transcript_scorecard(artifact),
        synthesis_status="blocked" if transcript_failed else "complete",
        canonical_span_count=artifact.quality_metrics.canonical_span_count,
        emitted_span_count=artifact.quality_metrics.emitted_span_count,
        unresolved_span_count=artifact.quality_metrics.unresolved_span_count,
        blocker_tags=blocking_failures,
        provider_support_summary=artifact.quality_metrics.provider_support_summary,
        provenance_refs=(artifact_ref,),
        escalated=transcript_failed,
        human_review_required=False,
    )
    runtime.decision_store.save_transcript_decision(
        decision,
        storage_job_id=operational_job_key(state.job),
    )
    decision_ref = write_model_artifact(runtime, transcript_decision_key(state.job), decision)
    return {
        "current_stage": "materialize_synthesized_transcript",
        "final_transcript_ref": artifact_ref,
        "final_transcript_candidate_id": primary_candidate_id,
        "final_transcript_decision_ref": decision_ref,
        "transcript_unresolved_span_count": artifact.quality_metrics.unresolved_span_count,
        "pending_memory_source_stage": "transcript_adjudication",
        "transcript_failed": transcript_failed,
        "human_review_required": False,
        "review_required_stage": None,
        "routing_facts": state.routing_facts
        + tuple(
            RoutingFact(
                stage="materialize_synthesized_transcript",
                fact_type="transcript_synthesis_blocker",
                value=failure,
                source_ref=artifact_ref,
            )
            for failure in blocking_failures
        )
        + (
            RoutingFact(
                stage="materialize_synthesized_transcript",
                fact_type="transcript_artifact_ref",
                value=artifact_ref,
                source_ref=artifact_ref,
            ),
            RoutingFact(
                stage="materialize_synthesized_transcript",
                fact_type="transcript_synthesis_status",
                value=artifact.status,
                source_ref=decision_ref,
            ),
            RoutingFact(
                stage="materialize_synthesized_transcript",
                fact_type="transcript_investigation_ref",
                value=investigation_ref,
                source_ref=investigation_ref,
            ),
        ),
    }


def _load_span_table(state: GraphState, runtime: WorkflowRuntime) -> TranscriptCanonicalSpanTable:
    if state.canonical_transcript_span_ref is None:
        raise RuntimeError("canonical transcript span table is required")
    return read_model_artifact(
        runtime,
        state.canonical_transcript_span_ref,
        TranscriptCanonicalSpanTable,
    )


def _load_selector_record(state: GraphState, runtime: WorkflowRuntime) -> TranscriptSynthesisRecord:
    if state.transcript_selector_ref is None:
        raise RuntimeError("transcript selector record is required")
    return read_model_artifact(
        runtime,
        state.transcript_selector_ref,
        TranscriptSynthesisRecord,
    )


def _load_global_record(state: GraphState, runtime: WorkflowRuntime) -> TranscriptSynthesisRecord:
    if state.final_transcript_synthesis_ref is None:
        raise RuntimeError("transcript synthesis record is required")
    return read_model_artifact(
        runtime,
        state.final_transcript_synthesis_ref,
        TranscriptSynthesisRecord,
    )


def _load_review_record(state: GraphState, runtime: WorkflowRuntime) -> TranscriptSynthesisReview:
    if state.transcript_span_review_ref is None:
        raise RuntimeError("transcript span review record is required")
    return read_model_artifact(runtime, state.transcript_span_review_ref, TranscriptSynthesisReview)


def _transcript_scorecard(artifact) -> AdjudicationScorecard:  # noqa: ANN001
    unresolved = artifact.quality_metrics.unresolved_span_count
    return AdjudicationScorecard(
        candidate_count=len(artifact.quality_metrics.provider_support_summary),
        preferred_candidate_id=None,
        average_confidence=0.0 if unresolved else 0.88,
        confidence_spread=0.0 if unresolved else 0.08,
        contradictory_evidence_count=unresolved,
        hard_contradiction_count=unresolved,
        blocking_hard_contradiction_count=unresolved,
        highest_issue_severity="critical" if unresolved else "minor",
        winner_mismatch=False,
        escalation_signal_count=unresolved,
        total_score=0.0 if unresolved else 0.88,
        content_risk_class="standard",
    )


def _primary_transcript_candidate_id(artifact) -> str | None:  # noqa: ANN001
    counts: dict[str, int] = {}
    for item in artifact.provenance:
        for candidate_id in item.candidate_ids:
            counts[candidate_id] = counts.get(candidate_id, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
