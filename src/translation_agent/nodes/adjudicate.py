"""Adjudication nodes for deterministic parser-backed review routing."""

from __future__ import annotations

from dataclasses import dataclass

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    AdjudicationContext,
    AdjudicationScorecard,
    FinalTranscriptDecision,
    FinalTranslationDecision,
)
from translation_agent.models.review import DecisionMode, DisagreementBucket
from translation_agent.nodes.common import (
    TRANSCRIPT_REVIEW_STAGE,
    TRANSLATION_REVIEW_STAGE,
    build_memory_query,
    load_reviews,
    select_transcript_candidates,
    select_translation_candidates,
    transcript_decision_key,
    transcript_investigation_key,
    translation_decision_key,
    translation_investigation_key,
    write_model_artifact,
)
from translation_agent.review import (
    adjudicate_reviews,
    adjudication_memory_bundle,
    content_risk_class_for_scenario,
)


def adjudicate_transcript(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Create a deterministic transcript decision from parsed reviewer bundles."""

    candidates = select_transcript_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.transcript_candidate_ids,
    )
    reviews = load_reviews(
        runtime,
        stage=TRANSCRIPT_REVIEW_STAGE,
        job=state.job,
        review_ids=state.transcript_review_ids,
    )
    adjudication_memory = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="adjudicate_transcript",
            candidate_ids=state.transcript_candidate_ids,
        )
    )
    context = AdjudicationContext(
        run_id=state.run_id,
        stage=TRANSCRIPT_REVIEW_STAGE,
        job=state.job,
        candidate_ids=state.transcript_candidate_ids,
        review_ids=state.transcript_review_ids,
        memory_bundle=adjudication_memory_bundle(
            stage=TRANSCRIPT_REVIEW_STAGE,
            memory_bundle=adjudication_memory,
        ),
        content_risk_class=content_risk_class_for_scenario(runtime.scenario),
    )
    outcome = adjudicate_reviews(candidates=candidates, reviews=reviews, context=context)
    investigation_ref = _persist_investigation(
        runtime,
        stage=TRANSCRIPT_REVIEW_STAGE,
        job=state.job,
        payload=outcome.investigation_payload,
    )
    decision = FinalTranscriptDecision(
        job_id=state.job.job_id,
        winner_candidate_id=outcome.winner_candidate_id,
        decision_mode=outcome.decision_mode,
        decision_confidence=outcome.decision_confidence,
        rationale_summary=outcome.rationale_summary,
        review_refs=state.transcript_review_ids,
        investigation_ref=investigation_ref,
        disagreement_bucket=outcome.disagreement_bucket,
        adjudication_scorecard=_scorecard(
            outcome=outcome,
            candidate_count=len(candidates),
            content_risk_class=context.content_risk_class,
        ),
        escalated=outcome.escalated,
        human_review_required=outcome.human_review_required,
    )
    runtime.decision_store.save_transcript_decision(decision)
    decision_ref = write_model_artifact(
        runtime,
        transcript_decision_key(state.job),
        decision,
    )
    return {
        "current_stage": "adjudicate_transcript",
        "final_transcript_candidate_id": outcome.winner_candidate_id,
        "final_transcript_decision_ref": decision_ref,
        "pending_memory_source_stage": "transcript_adjudication",
        "escalation_pending": decision.escalated,
        "human_review_required": decision.human_review_required,
        "routing_facts": state.routing_facts
        + _routing_facts(
            stage="adjudicate_transcript",
            decision_mode=decision.decision_mode,
            decision_ref=decision_ref,
            disagreement_bucket=outcome.disagreement_bucket,
            investigation_ref=investigation_ref,
        ),
    }


def adjudicate_translation(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Create a deterministic translation decision or recoverable failure state."""

    candidates = select_translation_candidates(
        runtime,
        job_id=state.job.job_id,
        candidate_ids=state.translation_candidate_ids,
    )
    reviews = load_reviews(
        runtime,
        stage=TRANSLATION_REVIEW_STAGE,
        job=state.job,
        review_ids=state.translation_review_ids,
    )
    if not candidates:
        decision = FinalTranslationDecision(
            job_id=state.job.job_id,
            winner_candidate_id=None,
            decision_mode="automatic_finalize",
            decision_confidence=0.0,
            rationale_summary=(
                "All translation variants failed; transcript preserved for recovery."
            ),
            review_refs=(),
            disagreement_bucket="unresolved",
            adjudication_scorecard=AdjudicationScorecard(
                candidate_count=0,
                preferred_candidate_id=None,
                average_confidence=0.0,
                confidence_spread=0.0,
                contradictory_evidence_count=0,
                highest_issue_severity="minor",
                winner_mismatch=False,
                escalation_signal_count=0,
                total_score=0.0,
                content_risk_class="standard",
            ),
            escalated=False,
            human_review_required=False,
            prompt_variant_winner=None,
            prompt_version_winner=None,
        )
        runtime.decision_store.save_translation_decision(decision)
        decision_ref = write_model_artifact(
            runtime,
            translation_decision_key(state.job),
            decision,
        )
        return {
            "current_stage": "adjudicate_translation",
            "final_translation_candidate_id": None,
            "final_translation_decision_ref": decision_ref,
            "pending_memory_source_stage": "translation_adjudication",
            "escalation_pending": False,
            "human_review_required": False,
            "translation_failed": True,
            "routing_facts": state.routing_facts
            + (
                RoutingFact(
                    stage="adjudicate_translation",
                    fact_type="decision_mode",
                    value=decision.decision_mode,
                    source_ref=decision_ref,
                ),
            ),
        }

    adjudication_memory = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="adjudicate_translation",
            candidate_ids=state.translation_candidate_ids,
        )
    )
    context = AdjudicationContext(
        run_id=state.run_id,
        stage=TRANSLATION_REVIEW_STAGE,
        job=state.job,
        candidate_ids=state.translation_candidate_ids,
        review_ids=state.translation_review_ids,
        memory_bundle=adjudication_memory_bundle(
            stage=TRANSLATION_REVIEW_STAGE,
            memory_bundle=adjudication_memory,
        ),
        content_risk_class=content_risk_class_for_scenario(runtime.scenario),
    )
    outcome = adjudicate_reviews(candidates=candidates, reviews=reviews, context=context)
    timeout_fallback = _translation_timeout_fallback(runtime=runtime, outcome=outcome)
    winner_candidate_id = (
        timeout_fallback.winner_candidate_id
        if timeout_fallback is not None
        else outcome.winner_candidate_id
    )
    winner = next(
        (candidate for candidate in candidates if candidate.candidate_id == winner_candidate_id),
        None,
    )
    investigation_ref = _persist_investigation(
        runtime,
        stage=TRANSLATION_REVIEW_STAGE,
        job=state.job,
        payload=(
            timeout_fallback.investigation_payload
            if timeout_fallback is not None
            else outcome.investigation_payload
        ),
    )
    decision = FinalTranslationDecision(
        job_id=state.job.job_id,
        winner_candidate_id=winner_candidate_id,
        decision_mode=(
            timeout_fallback.decision_mode
            if timeout_fallback is not None
            else outcome.decision_mode
        ),
        decision_confidence=(
            timeout_fallback.decision_confidence
            if timeout_fallback is not None
            else outcome.decision_confidence
        ),
        rationale_summary=(
            timeout_fallback.rationale_summary
            if timeout_fallback is not None
            else outcome.rationale_summary
        ),
        review_refs=state.translation_review_ids,
        investigation_ref=investigation_ref,
        disagreement_bucket=(
            timeout_fallback.disagreement_bucket
            if timeout_fallback is not None
            else outcome.disagreement_bucket
        ),
        adjudication_scorecard=_scorecard(
            outcome=outcome,
            candidate_count=len(candidates),
            content_risk_class=context.content_risk_class,
        ),
        escalated=timeout_fallback.escalated if timeout_fallback is not None else outcome.escalated,
        human_review_required=(
            timeout_fallback.human_review_required
            if timeout_fallback is not None
            else outcome.human_review_required
        ),
        winner_model_id=winner.model_id if winner is not None else None,
        prompt_variant_winner=winner.prompt_variant_id if winner is not None else None,
        prompt_version_winner=winner.prompt_version if winner is not None else None,
    )
    runtime.decision_store.save_translation_decision(decision)
    decision_ref = write_model_artifact(
        runtime,
        translation_decision_key(state.job),
        decision,
    )
    timeout_fact = ()
    if timeout_fallback is not None:
        timeout_fact = (
            RoutingFact(
                stage="adjudicate_translation",
                fact_type="investigation_timeout",
                value="conflict_investigator",
                source_ref=investigation_ref,
            ),
        )
    return {
        "current_stage": "adjudicate_translation",
        "final_translation_candidate_id": winner_candidate_id,
        "final_translation_decision_ref": decision_ref,
        "pending_memory_source_stage": "translation_adjudication",
        "escalation_pending": decision.escalated,
        "human_review_required": decision.human_review_required,
        "translation_failed": False,
        "routing_facts": state.routing_facts
        + timeout_fact
        + _routing_facts(
            stage="adjudicate_translation",
            decision_mode=decision.decision_mode,
            decision_ref=decision_ref,
            disagreement_bucket=decision.disagreement_bucket,
            investigation_ref=investigation_ref,
        ),
    }


def _persist_investigation(
    runtime: WorkflowRuntime,
    *,
    stage: str,
    job,
    payload: dict[str, object] | None,
) -> str | None:
    if payload is None:
        return None
    key = (
        transcript_investigation_key(job)
        if stage == TRANSCRIPT_REVIEW_STAGE
        else translation_investigation_key(job)
    )
    return write_model_artifact(runtime, key, payload)


@dataclass(frozen=True, slots=True)
class _TimeoutFallback:
    decision_mode: DecisionMode
    winner_candidate_id: str | None
    decision_confidence: float
    rationale_summary: str
    disagreement_bucket: DisagreementBucket
    escalated: bool
    human_review_required: bool
    investigation_payload: dict[str, object]


def _translation_timeout_fallback(*, runtime: WorkflowRuntime, outcome) -> _TimeoutFallback | None:
    if runtime.scenario != "translation_conflict_timeout":
        return None
    if outcome.decision_mode != "conflict_investigation":
        return None
    investigation_payload = dict(outcome.investigation_payload or {})
    investigation_payload.update(
        {
            "status": "timed_out",
            "timeout_seconds": 30.0,
            "fallback_decision_mode": "human_review",
        }
    )
    return _TimeoutFallback(
        decision_mode="human_review",
        winner_candidate_id=None,
        decision_confidence=0.0,
        rationale_summary=(
            "Translation conflict investigation timed out after medium disagreement, "
            "so the run escalated to human review."
        ),
        disagreement_bucket="unresolved",
        escalated=True,
        human_review_required=True,
        investigation_payload=investigation_payload,
    )


def _routing_facts(
    *,
    stage: str,
    decision_mode: str,
    decision_ref: str,
    disagreement_bucket: str,
    investigation_ref: str | None,
) -> tuple[RoutingFact, ...]:
    facts = [
        RoutingFact(
            stage=stage,
            fact_type="decision_mode",
            value=decision_mode,
            source_ref=decision_ref,
        ),
        RoutingFact(
            stage=stage,
            fact_type="disagreement_bucket",
            value=disagreement_bucket,
            source_ref=decision_ref,
        ),
    ]
    if investigation_ref is not None:
        facts.append(
            RoutingFact(
                stage=stage,
                fact_type="investigation_ref",
                value=investigation_ref,
                source_ref=investigation_ref,
            )
        )
    return tuple(facts)


def _scorecard(
    *,
    outcome,
    candidate_count: int,
    content_risk_class: str,
) -> AdjudicationScorecard:
    assessment = outcome.assessment
    return AdjudicationScorecard(
        candidate_count=candidate_count,
        preferred_candidate_id=assessment.preferred_candidate_id,
        average_confidence=assessment.average_confidence,
        confidence_spread=assessment.confidence_spread,
        contradictory_evidence_count=assessment.contradictory_evidence_count,
        highest_issue_severity=assessment.highest_issue_severity,
        winner_mismatch=assessment.winner_mismatch,
        escalation_signal_count=assessment.escalation_signal_count,
        total_score=assessment.total_score,
        content_risk_class=content_risk_class,
    )
