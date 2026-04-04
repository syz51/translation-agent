"""Adjudication nodes for deterministic parser-backed review routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    AdjudicationContext,
    AdjudicationScorecard,
    FinalTranscriptDecision,
    FinalTranslationDecision,
)
from translation_agent.models.review import DecisionMode, DisagreementBucket, ReviewStage
from translation_agent.nodes.common import (
    TRANSCRIPT_REVIEW_STAGE,
    TRANSLATION_REVIEW_STAGE,
    adjudication_memory_bundle_key,
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
from translation_agent.storage import operational_job_key


def adjudicate_transcript(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Create a deterministic transcript decision from parsed reviewer bundles."""

    candidates = select_transcript_candidates(
        runtime,
        job=state.job,
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
            provider_ids=tuple(candidate.provider_id for candidate in candidates),
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
        ranking_priors=_transcript_provider_ranking_priors(
            runtime=runtime,
            candidates=candidates,
            state=state,
        ),
    )
    memory_ref = write_model_artifact(
        runtime,
        adjudication_memory_bundle_key(state.job, TRANSCRIPT_REVIEW_STAGE),
        context.memory_bundle,
    )
    outcome = adjudicate_reviews(candidates=candidates, reviews=reviews, context=context)
    investigation_ref = _persist_investigation(
        runtime,
        stage=TRANSCRIPT_REVIEW_STAGE,
        job=state.job,
        payload=outcome.investigation_payload,
    )
    selected_transcript_candidate_id = (
        outcome.winner_candidate_id
        or outcome.assessment.preferred_candidate_id
        or (candidates[0].candidate_id if candidates else None)
    )
    deferred_review_fact = ()
    rationale_summary = outcome.rationale_summary
    if outcome.human_review_required:
        rationale_summary = (
            "Transcript disagreement remained unresolved after machine adjudication; "
            "translation generation will fall back to surviving transcript fan-out for "
            "translation review, and transcript provenance will stay visible until the "
            "unresolved transcript choice is settled."
        )
        deferred_review_fact = (
            RoutingFact(
                stage="adjudicate_transcript",
                fact_type="transcript_review_deferred_to_translation",
                value=selected_transcript_candidate_id or "none",
                source_ref=investigation_ref or memory_ref,
            ),
        )
    decision = FinalTranscriptDecision(
        job_id=state.job.job_id,
        winner_candidate_id=selected_transcript_candidate_id,
        decision_mode=outcome.decision_mode,
        decision_confidence=outcome.decision_confidence,
        rationale_summary=rationale_summary,
        review_refs=state.transcript_review_ids,
        investigation_ref=investigation_ref,
        disagreement_bucket=outcome.disagreement_bucket,
        adjudication_scorecard=_scorecard(
            outcome=outcome,
            candidate_count=len(candidates),
            content_risk_class=context.content_risk_class,
        ),
        escalated=outcome.escalated,
        human_review_required=False,
    )
    runtime.decision_store.save_transcript_decision(
        decision,
        storage_job_id=operational_job_key(state.job),
    )
    decision_ref = write_model_artifact(
        runtime,
        transcript_decision_key(state.job),
        decision,
    )
    return {
        "current_stage": "adjudicate_transcript",
        "final_transcript_candidate_id": selected_transcript_candidate_id,
        "final_transcript_decision_ref": decision_ref,
        "pending_memory_source_stage": "transcript_adjudication",
        "escalation_pending": decision.escalated,
        "human_review_required": False,
        "review_required_stage": None,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="adjudicate_transcript",
                fact_type="adjudication_memory_bundle",
                value=memory_ref,
                source_ref=memory_ref,
            ),
        )
        + deferred_review_fact
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
        job=state.job,
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
            decision_mode="human_review",
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
                hard_contradiction_count=0,
                blocking_hard_contradiction_count=0,
                highest_issue_severity="minor",
                winner_mismatch=False,
                escalation_signal_count=0,
                total_score=0.0,
                content_risk_class="standard",
            ),
            escalated=True,
            human_review_required=True,
            prompt_variant_winner=None,
            prompt_version_winner=None,
        )
        runtime.decision_store.save_translation_decision(
            decision,
            storage_job_id=operational_job_key(state.job),
        )
        decision_ref = write_model_artifact(
            runtime,
            translation_decision_key(state.job),
            decision,
        )
        return {
            "current_stage": "adjudicate_translation",
            "final_translation_candidate_id": None,
            "final_transcript_candidate_id": state.final_transcript_candidate_id,
            "final_translation_decision_ref": decision_ref,
            "pending_memory_source_stage": "translation_adjudication",
            "escalation_pending": True,
            "human_review_required": True,
            "review_required_stage": None,
            "translation_failed": True,
            "routing_facts": state.routing_facts
            + _routing_facts(
                stage="adjudicate_translation",
                decision_mode=decision.decision_mode,
                decision_ref=decision_ref,
                disagreement_bucket=decision.disagreement_bucket,
                investigation_ref=None,
            ),
        }

    adjudication_memory = runtime.memory_recall_backend.recall_memory(
        build_memory_query(
            state,
            stage="adjudicate_translation",
            candidate_ids=state.translation_candidate_ids,
            provider_ids=tuple(
                transcript.provider_id
                for transcript in select_transcript_candidates(
                    runtime,
                    job=state.job,
                    candidate_ids=tuple(
                        candidate.source_transcript_candidate_id or "" for candidate in candidates
                    ),
                )
            ),
            prompt_variant_ids=tuple(candidate.prompt_variant_id for candidate in candidates),
            model_ids=tuple(candidate.model_id for candidate in candidates),
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
        ranking_priors=_translation_candidate_ranking_priors(
            runtime=runtime,
            candidates=candidates,
            adjudication_memory=adjudication_memory,
            state=state,
        ),
    )
    memory_ref = write_model_artifact(
        runtime,
        adjudication_memory_bundle_key(state.job, TRANSLATION_REVIEW_STAGE),
        context.memory_bundle,
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
    runtime.decision_store.save_translation_decision(
        decision,
        storage_job_id=operational_job_key(state.job),
    )
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
        "final_transcript_candidate_id": (
            winner.source_transcript_candidate_id
            if winner is not None
            else state.final_transcript_candidate_id
        ),
        "final_translation_decision_ref": decision_ref,
        "pending_memory_source_stage": "translation_adjudication",
        "escalation_pending": decision.escalated,
        "human_review_required": decision.human_review_required,
        "review_required_stage": (
            TRANSLATION_REVIEW_STAGE
            if decision.human_review_required and not state.translation_failed and candidates
            else None
        ),
        "translation_failed": False,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="adjudicate_translation",
                fact_type="adjudication_memory_bundle",
                value=memory_ref,
                source_ref=memory_ref,
            ),
        )
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
    stage: ReviewStage,
    job,
    payload: dict[str, object] | None,
) -> str | None:
    if payload is None:
        return None
    runtime.decision_store.save_investigation(
        job_id=job.job_id,
        stage=stage,
        payload=payload,
        storage_job_id=operational_job_key(job),
    )
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
        hard_contradiction_count=assessment.hard_contradiction_count,
        blocking_hard_contradiction_count=assessment.blocking_hard_contradiction_count,
        highest_issue_severity=assessment.highest_issue_severity,
        winner_mismatch=assessment.winner_mismatch,
        escalation_signal_count=assessment.escalation_signal_count,
        total_score=assessment.total_score,
        content_risk_class=content_risk_class,
    )


def _transcript_provider_ranking_priors(
    *,
    runtime: WorkflowRuntime,
    candidates,
    state: GraphState,
) -> dict[str, float]:
    getter = getattr(runtime.decision_store, "get_transcript_provider_quality_stats", None)
    if not callable(getter):
        return {}

    priors: dict[str, float] = {}
    for candidate in candidates:
        stats = getter(
            provider_id=candidate.provider_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
        )
        if stats is None:
            continue
        priors[candidate.candidate_id] = _bounded_provider_prior(stats)
    return {candidate_id: boost for candidate_id, boost in priors.items() if boost > 0.0}


def _bounded_provider_prior(stats: Any) -> float:
    recent_boost = min(float(getattr(stats, "recent_approved_outcomes_30d", 0)) * 0.01, 0.05)
    volume_boost = min(float(getattr(stats, "total_approved_outcomes", 0)) * 0.01, 0.05)
    rate_boost = min(float(getattr(stats, "indirect_approval_rate", 0.0)) * 0.12, 0.15)
    hard_boost = min(float(getattr(stats, "hard_positive_score", 0.0)) * 0.02, 0.08)
    soft_boost = min(float(getattr(stats, "soft_positive_score", 0.0)) * 0.03, 0.06)
    negative_penalty = min(float(getattr(stats, "negative_feedback_score", 0.0)) * 0.03, 0.12)
    return round(
        min(0.25, recent_boost + volume_boost + rate_boost + hard_boost + soft_boost)
        - negative_penalty,
        4,
    )


def _translation_candidate_ranking_priors(
    *,
    runtime: WorkflowRuntime,
    candidates,
    adjudication_memory,
    state: GraphState,
) -> dict[str, float]:
    getter = getattr(runtime.decision_store, "get_translation_feedback_stats", None)
    if not callable(getter):
        return {}
    priors: dict[str, float] = {}
    transcript_by_id = {
        transcript.candidate_id: transcript
        for transcript in select_transcript_candidates(
            runtime,
            job=state.job,
            candidate_ids=tuple(
                candidate.source_transcript_candidate_id or "" for candidate in candidates
            ),
        )
    }
    for candidate in candidates:
        transcript = transcript_by_id.get(candidate.source_transcript_candidate_id or "")
        provider_id = transcript.provider_id if transcript is not None else "unknown-provider"
        combo_key = "::".join(
            (
                state.job.source_language,
                state.job.target_language,
                provider_id,
                candidate.model_id,
                candidate.prompt_variant_id,
                candidate.prompt_version,
            )
        )
        stats = getter(combo_key)
        prior = 0.0
        if stats is not None:
            prior = _bounded_translation_prior(stats)
        prior += _procedural_memory_prior(adjudication_memory.procedural_memory, combo_key)
        if prior != 0.0:
            priors[candidate.candidate_id] = max(-0.25, min(0.25, round(prior, 4)))
    return priors


def _bounded_translation_prior(stats: Any) -> float:
    pairwise_signal = min(
        max(
            (
                float(getattr(stats, "pairwise_win_score", 0.0))
                - float(getattr(stats, "pairwise_loss_score", 0.0))
            )
            * 0.05,
            -0.18,
        ),
        0.18,
    )
    approval_signal = min(float(getattr(stats, "approved_good_count", 0)) * 0.02, 0.08) + min(
        float(getattr(stats, "approved_best_available_count", 0)) * 0.01,
        0.04,
    )
    rejection_penalty = min(float(getattr(stats, "rejected_all_count", 0)) * 0.03, 0.12)
    failure_penalty = min(
        sum(float(value) for value in getattr(stats, "failure_tag_counts", {}).values()) * 0.01,
        0.08,
    )
    return round(
        max(
            -0.25,
            min(0.25, pairwise_signal + approval_signal - rejection_penalty - failure_penalty),
        ),
        4,
    )


def _procedural_memory_prior(procedural_memory, combo_key: str) -> float:
    prior = 0.0
    for entry in procedural_memory:
        if entry.metadata.get("combo_key") == combo_key:
            strength = str(entry.metadata.get("supervision_strength") or "")
            if strength == "strong":
                prior += 0.03
            elif strength == "soft":
                prior += 0.015
            elif strength == "negative":
                prior -= 0.02
        elif entry.metadata.get("category") == "anti_pattern":
            prior -= 0.005
    return round(prior, 4)
