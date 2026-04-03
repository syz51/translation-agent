"""Reference transcript evaluation and draft regeneration helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import cast

import pysubs2

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.memory.recall import build_scope_key
from translation_agent.models import (
    AssetRecord,
    EvaluatedRunReport,
    EvaluationFailure,
    EvaluationReport,
    HistoricalRunLink,
    PromotionGateOutcome,
    PromptChange,
    PromptCompatibilityTuple,
    PromptEvolutionProposal,
    ProposalAggregateMetrics,
    PublishedArtifacts,
    ReferenceSegment,
    ReferenceTranscript,
    RegeneratedTranslationDraft,
    Segment,
    StrongerGraderScore,
    TranscriptAlignmentReport,
    TranscriptCandidate,
    TranscriptMismatchSpan,
    TranslationCandidate,
    TranslationScore,
)
from translation_agent.nodes.common import (
    build_request_context,
    memory_batch_key,
    memory_consolidation_key,
    read_model_artifact,
    select_translation_candidates,
    write_model_artifact,
)
from translation_agent.parallelism import ordered_parallel_map
from translation_agent.storage import asset_path, operational_job_key

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def run_reference_evaluation(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Evaluate prior runs against a trusted transcript and regenerate a draft."""

    if state.job.reference_mode != "evaluate_and_regenerate":
        return {}
    if state.job.reference_transcript_source is None:
        raise RuntimeError("reference transcript source is required for reference evaluation")

    now = datetime.now(UTC)
    reference = _load_reference_transcript(state, runtime, now=now)
    reference_refs = _persist_reference_transcript(state, runtime, reference, now=now)
    _update_asset_latest_reference(runtime, state.job.media_key, reference_refs[0])

    failures: list[EvaluationFailure] = []
    evaluated_runs: list[EvaluatedRunReport] = []
    proposal_refs: tuple[str, ...] = ()
    regenerated_draft_ref: str | None = None
    recurring_patterns: list[str] = []

    try:
        evaluated_runs = _load_historical_runs(
            state,
            runtime,
            reference=reference,
            trusted_transcript_ref=reference_refs[0],
        )
        recurring_patterns = _recurring_failure_patterns(evaluated_runs)
    except Exception as exc:
        failures.append(
            EvaluationFailure(
                stage="evaluation",
                message=str(exc) or "historical evaluation failed",
            )
        )

    try:
        proposal_refs = _persist_improvement_proposals(
            state,
            runtime,
            evaluated_runs=evaluated_runs,
            recurring_patterns=recurring_patterns,
            evaluation_refs=tuple(
                report.translation.translation_ref
                for report in evaluated_runs
                if report.translation is not None and report.translation.translation_ref is not None
            ),
        )
    except Exception as exc:
        failures.append(
            EvaluationFailure(
                stage="proposal_generation",
                message=str(exc) or "proposal generation failed",
            )
        )

    try:
        regenerated_draft_ref = _generate_regenerated_draft(
            state,
            runtime,
            reference=reference,
            trusted_transcript_ref=reference_refs[0],
        )
    except Exception as exc:
        failures.append(
            EvaluationFailure(
                stage="regenerated_draft",
                message=str(exc) or "draft regeneration failed",
            )
        )

    current_compatibility = _current_run_compatibility(state, runtime)
    canary_metrics, control_metrics = _aggregate_outcomes_for_compatibility(
        evaluated_runs,
        current_compatibility,
    )
    canary_stronger_grader, control_stronger_grader = _stronger_grader_reports(
        canary_metrics,
        control_metrics,
    )
    gate_outcome = _evaluation_gate_outcome(
        canary_metrics=canary_metrics,
        control_metrics=control_metrics,
        canary_stronger_grader=canary_stronger_grader,
        control_stronger_grader=control_stronger_grader,
        proposal_refs=proposal_refs,
    )
    report = EvaluationReport(
        evaluation_id=f"evaluation-{state.run_id}",
        run_id=state.run_id,
        media_key=state.job.media_key,
        trusted_transcript_ref=reference_refs[0],
        evaluated_runs=tuple(evaluated_runs),
        recurring_failure_patterns=tuple(recurring_patterns),
        prior_official_translation_refs=tuple(
            report.translation.translation_ref
            for report in evaluated_runs
            if report.translation is not None and report.translation.translation_ref is not None
        ),
        proposal_refs=proposal_refs,
        regenerated_draft_ref=regenerated_draft_ref,
        failures=tuple(failures),
        canary_metrics=canary_metrics,
        control_metrics=control_metrics,
        canary_stronger_grader=canary_stronger_grader,
        control_stronger_grader=control_stronger_grader,
        gate_outcome=gate_outcome,
        proposal_compatibility=(current_compatibility,)
        if current_compatibility is not None
        else (),
        metadata={
            "asset_id": state.job.asset_id,
            "media_fingerprint": state.job.media_fingerprint,
            "reference_history_ref": reference_refs[1],
            "tenant_id": state.job.tenant_id,
            "project_id": state.job.project_id,
            "source_language": state.job.source_language,
            "target_language": state.job.target_language,
        },
    )
    evaluation_report_ref = asset_path(state.job.media_key, "evaluations", f"{state.run_id}.json")
    routing_facts = list(state.routing_facts)
    memory_batch_ids = state.memory_batch_ids
    for ref in reference_refs:
        routing_facts.append(
            RoutingFact(
                stage="reference_evaluation",
                fact_type="reference_transcript_published",
                value=ref,
                source_ref=ref,
            )
        )
    for proposal_ref in proposal_refs:
        routing_facts.append(
            RoutingFact(
                stage="reference_evaluation",
                fact_type="reference_improvement_proposal",
                value=proposal_ref.rsplit("/", 1)[-1],
                source_ref=proposal_ref,
            )
        )
    proposal_models = tuple(
        read_model_artifact(runtime, proposal_ref, PromptEvolutionProposal)
        for proposal_ref in proposal_refs
    )
    try:
        batch = runtime.memory_staging_backend.stage_evaluation_candidates(
            report,
            proposals=proposal_models,
        )
        if batch is not None:
            storage_job_id = operational_job_key(state.job)
            scoped_batch = batch.model_copy(
                update={
                    "batch_id": f"{batch.batch_id}-{state.run_id}",
                    "metadata": {
                        **batch.metadata,
                        "job_scope_key": storage_job_id,
                    },
                }
            )
            runtime.memory_batch_store.save_batch(scoped_batch, storage_job_id=storage_job_id)
            batch_ref = write_model_artifact(
                runtime,
                memory_batch_key(state.job, scoped_batch.batch_id),
                scoped_batch,
            )
            routing_facts.append(
                RoutingFact(
                    stage="reference_evaluation",
                    fact_type="memory_batch_staged",
                    value=scoped_batch.batch_id,
                    source_ref=batch_ref,
                )
            )
            memory_batch_ids = state.memory_batch_ids + (scoped_batch.batch_id,)
            consolidation = runtime.memory_consolidation_backend.consolidate_batch(scoped_batch)
            consolidated_batch = scoped_batch.model_copy(
                update={"consolidation_status": "consolidated"}
            )
            runtime.memory_batch_store.save_batch(consolidated_batch, storage_job_id=storage_job_id)
            write_model_artifact(
                runtime,
                memory_batch_key(state.job, consolidated_batch.batch_id),
                consolidated_batch,
            )
            consolidation_ref = write_model_artifact(
                runtime,
                memory_consolidation_key(state.job, consolidation.consolidation_id),
                consolidation,
            )
            routing_facts.append(
                RoutingFact(
                    stage="reference_evaluation",
                    fact_type="memory_batch_consolidated",
                    value=consolidation.consolidation_id,
                    source_ref=consolidation_ref,
                )
            )
    except Exception as exc:
        failures.append(
            EvaluationFailure(
                stage="memory_consolidation",
                message=str(exc) or "evaluation memory consolidation failed",
            )
        )
    if tuple(failures) != report.failures:
        report = report.model_copy(update={"failures": tuple(failures)})
    write_model_artifact(
        runtime,
        evaluation_report_ref,
        report,
    )
    routing_facts.extend(
        (
            RoutingFact(
                stage="reference_evaluation",
                fact_type="reference_evaluation_report",
                value=report.evaluation_id,
                source_ref=evaluation_report_ref,
            ),
            RoutingFact(
                stage="reference_evaluation",
                fact_type="reference_regenerated_draft",
                value=state.run_id,
                source_ref=regenerated_draft_ref,
            ),
        )
    )
    for failure in failures:
        routing_facts.append(
            RoutingFact(
                stage="reference_evaluation",
                fact_type="best_effort_failure",
                value=failure.stage,
                source_ref=failure.message,
            )
        )
    return {
        "current_stage": "reference_evaluation",
        "reference_transcript_ref": reference_refs[0],
        "evaluation_report_ref": evaluation_report_ref,
        "regenerated_translation_draft_ref": regenerated_draft_ref,
        "improvement_proposal_refs": proposal_refs,
        "memory_batch_ids": memory_batch_ids,
        "routing_facts": tuple(routing_facts),
    }


def update_historical_run_link(
    state: GraphState,
    runtime: WorkflowRuntime,
    artifacts: PublishedArtifacts,
) -> None:
    """Persist the current run's final artifact refs on the asset link."""

    upsert_link = getattr(runtime.run_store, "upsert_historical_run_link", None)
    if not callable(upsert_link):
        return
    prompt_updates = _translation_link_updates(state, runtime, artifacts.final_translation_ref)
    upsert_link(
        HistoricalRunLink(
            run_id=state.run_id,
            media_key=state.job.media_key,
            job_id=state.job.job_id,
            tenant_id=state.job.tenant_id,
            project_id=state.job.project_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
            created_at=state.job.created_at,
            transcript_ref=artifacts.final_transcript_ref,
            translation_ref=artifacts.final_translation_ref,
            transcript_decision_ref=state.final_transcript_decision_ref,
            translation_decision_ref=state.final_translation_decision_ref,
            evaluation_report_ref=state.evaluation_report_ref,
            regenerated_draft_ref=state.regenerated_translation_draft_ref,
            translation_model_id=prompt_updates["translation_model_id"],
            translation_prompt_variant_id=prompt_updates["translation_prompt_variant_id"],
            translation_base_prompt_version=prompt_updates["translation_base_prompt_version"],
            translation_effective_prompt_version=prompt_updates[
                "translation_effective_prompt_version"
            ],
            prompt_scope_kind=prompt_updates["prompt_scope_kind"],
            prompt_scope_key=prompt_updates["prompt_scope_key"],
            prompt_resolution_mode=prompt_updates["prompt_resolution_mode"],
            prompt_proposal_id=prompt_updates["prompt_proposal_id"],
        )
    )


def _translation_link_updates(
    state: GraphState,
    runtime: WorkflowRuntime,
    translation_ref: str | None,
) -> dict[str, str | None]:
    if translation_ref is None or not runtime.blob_store.exists(translation_ref):
        return {
            "translation_model_id": None,
            "translation_prompt_variant_id": None,
            "translation_base_prompt_version": None,
            "translation_effective_prompt_version": None,
            "prompt_scope_kind": "project_pair",
            "prompt_scope_key": build_scope_key(
                scope_kind="project_pair",
                tenant_id=state.job.tenant_id,
                project_id=state.job.project_id,
                source_language=state.job.source_language,
                target_language=state.job.target_language,
            ),
            "prompt_resolution_mode": None,
            "prompt_proposal_id": None,
        }
    translation = read_model_artifact(runtime, translation_ref, TranslationCandidate)
    resolver = translation.metadata.get("prompt_resolver")
    if not isinstance(resolver, dict):
        resolver = {}
    selected_proposal_id = resolver.get("selected_proposal_id")
    return {
        "translation_model_id": translation.model_id,
        "translation_prompt_variant_id": translation.prompt_variant_id,
        "translation_base_prompt_version": str(
            resolver.get("base_prompt_version") or translation.prompt_version
        ),
        "translation_effective_prompt_version": translation.prompt_version,
        "prompt_scope_kind": str(resolver.get("scope_kind"))
        if resolver.get("scope_kind")
        else "project_pair",
        "prompt_scope_key": str(resolver.get("scope_key"))
        if resolver.get("scope_key")
        else build_scope_key(
            scope_kind="project_pair",
            tenant_id=state.job.tenant_id,
            project_id=state.job.project_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
        ),
        "prompt_resolution_mode": str(resolver.get("resolution_mode"))
        if resolver.get("resolution_mode")
        else "control",
        "prompt_proposal_id": str(selected_proposal_id)
        if isinstance(selected_proposal_id, str)
        else None,
    }


def _load_reference_transcript(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    now: datetime,
) -> ReferenceTranscript:
    source = state.job.reference_transcript_source
    if source is None:
        raise RuntimeError("reference transcript source is required for reference evaluation")
    if Path(source).expanduser().exists():
        content = Path(source).expanduser().read_text(encoding="utf-8")
    elif runtime.blob_store.exists(source):
        content = runtime.blob_store.read_bytes(source).decode("utf-8")
    else:
        raise RuntimeError(f"reference transcript source was not found: {source}")
    segments = _parse_srt(content)
    return ReferenceTranscript(
        reference_id=f"reference-{state.run_id}",
        media_key=state.job.media_key,
        asset_id=state.job.asset_id,
        source=source,
        format=state.job.reference_transcript_format or "srt",
        segments=segments,
        full_text=" ".join(segment.text for segment in segments).strip(),
        created_at=now,
    )


def _persist_reference_transcript(
    state: GraphState,
    runtime: WorkflowRuntime,
    reference: ReferenceTranscript,
    *,
    now: datetime,
) -> tuple[str, str]:
    latest_ref = asset_path(state.job.media_key, "references", "transcript", "latest.json")
    history_ref = asset_path(
        state.job.media_key,
        "references",
        "transcript",
        "history",
        f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.json",
    )
    write_model_artifact(runtime, latest_ref, reference)
    write_model_artifact(runtime, history_ref, reference)
    return latest_ref, history_ref


def _load_historical_runs(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    reference: ReferenceTranscript,
    trusted_transcript_ref: str,
) -> list[EvaluatedRunReport]:
    list_links = cast(
        Callable[..., list[HistoricalRunLink]] | None,
        getattr(runtime.run_store, "list_historical_run_links", None),
    )
    if not callable(list_links):
        return []
    links = tuple(
        list_links(
            state.job.media_key,
            exclude_run_id=state.run_id,
        )
    )
    gathered = ordered_parallel_map(
        links,
        max_workers=runtime.parallelism.reference_evaluation_max_workers,
        worker=lambda link: _evaluate_historical_run(
            runtime,
            link=link,
            reference=reference,
            trusted_transcript_ref=trusted_transcript_ref,
        ),
        sort_key=lambda input_index, _link: (input_index,),
    )
    reports: list[EvaluatedRunReport] = []
    for result in gathered:
        if result.error is not None:
            raise result.error
        if result.value is None:  # pragma: no cover - defensive
            raise RuntimeError("missing historical evaluation result")
        reports.append(result.value)
    return reports


def _evaluate_historical_run(
    runtime: WorkflowRuntime,
    *,
    link: HistoricalRunLink,
    reference: ReferenceTranscript,
    trusted_transcript_ref: str,
) -> EvaluatedRunReport:
    transcript_report = None
    if link.transcript_ref and runtime.blob_store.exists(link.transcript_ref):
        transcript = read_model_artifact(runtime, link.transcript_ref, TranscriptCandidate)
        transcript_report = _evaluate_transcript(
            link=link,
            reference=reference,
            transcript=transcript,
            trusted_transcript_ref=trusted_transcript_ref,
        )
    translation_report = None
    if link.translation_ref and runtime.blob_store.exists(link.translation_ref):
        translation = read_model_artifact(runtime, link.translation_ref, TranslationCandidate)
        translation_report = _evaluate_translation(
            link=link,
            reference=reference,
            translation=translation,
            trusted_transcript_ref=trusted_transcript_ref,
        )
    return EvaluatedRunReport(
        run=link,
        transcript=transcript_report,
        translation=translation_report,
    )


def _evaluate_transcript(
    *,
    link: HistoricalRunLink,
    reference: ReferenceTranscript,
    transcript: TranscriptCandidate,
    trusted_transcript_ref: str,
) -> TranscriptAlignmentReport:
    reference_tokens = _normalized_tokens(reference.full_text)
    observed_tokens = _normalized_tokens(transcript.full_text)
    matcher = SequenceMatcher(a=reference_tokens, b=observed_tokens)
    mismatch_spans: list[TranscriptMismatchSpan] = []
    omission_count = 0
    insertion_count = 0
    high_divergence_count = 0
    for sequence, opcode in enumerate(matcher.get_opcodes()):
        tag, ref_start, ref_end, obs_start, obs_end = opcode
        if tag == "equal":
            continue
        omitted = tuple(reference_tokens[ref_start:ref_end])
        inserted = tuple(observed_tokens[obs_start:obs_end])
        if tag in {"delete", "replace"}:
            omission_count += len(omitted)
        if tag in {"insert", "replace"}:
            insertion_count += len(inserted)
        if tag == "replace":
            high_divergence_count += 1
        mismatch_spans.append(
            TranscriptMismatchSpan(
                segment_id=f"mismatch-{sequence}",
                start_ms=0,
                end_ms=0,
                reference_text=" ".join(omitted),
                candidate_text=" ".join(inserted),
                similarity=(
                    0.0 if not omitted and not inserted else _token_similarity(omitted, inserted)
                ),
                omitted_tokens=omitted,
                inserted_tokens=inserted,
                evidence_ref=link.transcript_ref,
            )
        )
    matched_count = sum(match.size for match in matcher.get_matching_blocks())
    compared_count = max(len(reference_tokens), 1)
    return TranscriptAlignmentReport(
        run_id=link.run_id,
        media_key=link.media_key,
        transcript_ref=link.transcript_ref,
        trusted_transcript_ref=trusted_transcript_ref,
        coverage=round(min(matched_count / compared_count, 1.0), 4),
        omission_count=omission_count,
        insertion_count=insertion_count,
        high_divergence_count=high_divergence_count,
        mismatch_spans=tuple(mismatch_spans),
    )


def _evaluate_translation(
    *,
    link: HistoricalRunLink,
    reference: ReferenceTranscript,
    translation: TranslationCandidate,
    trusted_transcript_ref: str,
) -> TranslationScore:
    source_segments = tuple(segment.text for segment in reference.segments)
    target_segments = tuple(segment.target_text or "" for segment in translation.segments)
    populated_segments = sum(1 for text in target_segments if text.strip())
    segment_coverage = round(populated_segments / max(len(source_segments), 1), 4)
    target_text = translation.full_text
    source_text = " ".join(source_segments)
    source_entities = _named_entities(source_text)
    target_entities = _named_entities(target_text)
    entity_consistency = round(
        _preservation_ratio(source_entities, target_entities),
        4,
    )
    source_numbers = _numeric_markers(source_text)
    target_numbers = _numeric_markers(target_text)
    numeric_consistency = round(
        _preservation_ratio(source_numbers, target_numbers),
        4,
    )
    glossary_compliance = 1.0
    target_token_count = len(_normalized_tokens(target_text))
    source_token_count = len(_normalized_tokens(source_text))
    length_ratio = target_token_count / max(source_token_count, 1)
    omission_risk = round(max(0.0, 1.0 - min(segment_coverage, 1.0)), 4)
    addition_risk = round(max(0.0, min(length_ratio - 1.4, 1.0)), 4) if length_ratio > 1.4 else 0.0
    faithfulness_judge = round(
        max(
            0.0,
            min(
                1.0,
                (segment_coverage * 0.55)
                + (entity_consistency * 0.20)
                + (numeric_consistency * 0.15)
                + ((1.0 - addition_risk) * 0.10),
            ),
        ),
        4,
    )
    fluency_judge = round(_fluency_score(target_segments or (target_text,)), 4)
    primary_quality_score = round(
        max(
            0.0,
            min(
                1.0,
                0.45 * faithfulness_judge
                + 0.20 * segment_coverage
                + 0.15 * entity_consistency
                + 0.10 * numeric_consistency
                + 0.10 * glossary_compliance,
            ),
        ),
        4,
    )
    notes: list[str] = []
    if entity_consistency < 0.8 and source_entities:
        notes.append("named entity preservation dropped")
    if omission_risk > 0.25:
        notes.append("segment-level omissions detected")
    if numeric_consistency < 1.0 and source_numbers:
        notes.append("numeric/date/unit mismatch detected")
    return TranslationScore(
        run_id=link.run_id,
        translation_ref=link.translation_ref,
        trusted_transcript_ref=trusted_transcript_ref,
        faithfulness_judge=faithfulness_judge,
        segment_coverage=segment_coverage,
        entity_consistency=entity_consistency,
        numeric_date_unit_consistency=numeric_consistency,
        glossary_compliance=glossary_compliance,
        omission_risk=omission_risk,
        addition_risk=round(min(addition_risk, 1.0), 4),
        fluency_judge=fluency_judge,
        primary_quality_score=primary_quality_score,
        severe_failure_bucket=(
            primary_quality_score < 0.55
            or omission_risk >= 0.35
            or entity_consistency < 0.5
            or numeric_consistency < 0.5
        ),
        notes=tuple(notes),
        faithfulness=faithfulness_judge,
        terminology_consistency=glossary_compliance,
        named_entity_preservation=entity_consistency,
        repeated_failure_terms=(),
    )


def _persist_improvement_proposals(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    evaluated_runs: list[EvaluatedRunReport],
    recurring_patterns: list[str],
    evaluation_refs: tuple[str, ...],
) -> tuple[str, ...]:
    current_compatibility = _current_run_compatibility(state, runtime)
    if current_compatibility is None or not recurring_patterns:
        return ()
    list_proposals = cast(
        Callable[..., list[PromptEvolutionProposal]] | None,
        getattr(runtime.run_store, "list_prompt_evolution_proposals", None),
    )
    save_proposal = cast(
        Callable[[PromptEvolutionProposal], None] | None,
        getattr(runtime.run_store, "save_prompt_evolution_proposal", None),
    )
    existing = (
        list_proposals(
            prompt_family=current_compatibility.prompt_family,
            target_model_id=current_compatibility.model_id,
            source_language=current_compatibility.source_language,
            target_language=current_compatibility.target_language,
            prompt_variant_id=current_compatibility.prompt_variant_id,
            base_prompt_version=current_compatibility.base_prompt_version,
            scope_kind=current_compatibility.scope_kind,
            scope_key=current_compatibility.scope_key,
            media_key=state.job.media_key,
        )
        if callable(list_proposals)
        else []
    )
    refs: list[str] = []
    current_patterns = _qualifying_patterns_for_compatibility(evaluated_runs, current_compatibility)
    if len(current_patterns) >= 3 and not existing:
        proposal = _build_prompt_proposal(
            state,
            runtime,
            compatibility=current_compatibility,
            patterns=tuple(sorted(set(current_patterns))),
            evidence_refs=evaluation_refs,
            status="proposed",
        )
        refs.append(_save_proposal(runtime, proposal, save_proposal))
        return tuple(refs)

    active = [proposal for proposal in existing if proposal.status == "active"]
    canaries = [proposal for proposal in existing if proposal.status == "canary"]
    proposed = [proposal for proposal in existing if proposal.status == "proposed"]
    if canaries:
        canary = canaries[0]
        updated = _update_canary_status(canary, evaluated_runs)
        if updated.status == "active":
            for proposal in active:
                superseded = proposal.model_copy(update={"status": "superseded"})
                refs.append(_save_proposal(runtime, superseded, save_proposal))
        refs.append(_save_proposal(runtime, updated, save_proposal))
        return tuple(refs)
    if proposed and not canaries:
        canary = proposed[0].model_copy(update={"status": "canary"})
        refs.append(_save_proposal(runtime, canary, save_proposal))
        return tuple(refs)
    return ()


def _generate_regenerated_draft(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    reference: ReferenceTranscript,
    trusted_transcript_ref: str,
) -> str:
    transcript = TranscriptCandidate(
        candidate_id=f"reference-transcript-{state.run_id}",
        job_id=state.job.job_id,
        provider_id="reference-transcript",
        language=state.job.source_language,
        segments=tuple(
            Segment(
                segment_id=segment.segment_id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                source_text=segment.text,
            )
            for segment in reference.segments
        ),
        full_text=reference.full_text,
        normalization_version="reference-transcript-v1",
        metadata={"trusted_transcript_ref": trusted_transcript_ref},
    )
    resolved_prompt = runtime.prompt_resolver.resolve_translation_prompt(
        base_prompt_version=getattr(
            runtime.translation_adapter,
            "_prompt_version",
            "unversioned",
        ),
        prompt_variant_id="variant-a",
        model_id=runtime.translation_adapter.model_id,
        source_language=state.job.source_language,
        target_language=state.job.target_language,
        tenant_id=state.job.tenant_id,
        project_id=state.job.project_id,
        media_key=state.job.media_key,
        run_id=state.run_id,
    )
    base_request_context = build_request_context(state, runtime)
    request_context = base_request_context.model_copy(
        update={
            "metadata": {
                **base_request_context.metadata,
                "resolved_translation_prompt": resolved_prompt.model_dump(mode="json"),
                "reference_transcript_ref": trusted_transcript_ref,
            }
        }
    )
    generated_candidate = runtime.translation_adapter.generate_translation(
        transcript,
        "variant-a",
        request_context,
    )
    candidate = generated_candidate.model_copy(
        update={
            "prompt_version": resolved_prompt.effective_prompt_version,
            "metadata": {
                **generated_candidate.metadata,
                "prompt_resolver": resolved_prompt.model_dump(mode="json"),
                "generated_from_reference_transcript": True,
                "replaces_canonical": False,
            },
        }
    )
    draft_ref = asset_path(state.job.media_key, "regenerated-drafts", f"{state.run_id}.json")
    draft = RegeneratedTranslationDraft(
        draft_id=f"regenerated-draft-{state.run_id}",
        run_id=state.run_id,
        media_key=state.job.media_key,
        trusted_transcript_ref=trusted_transcript_ref,
        translation_candidate_id=candidate.candidate_id,
        full_text=candidate.full_text,
        segment_count=len(candidate.segments),
        draft_ref=draft_ref,
        prompt_variant_id=candidate.prompt_variant_id,
        prompt_version=candidate.prompt_version,
        prompt_provenance_refs=resolved_prompt.applied_proposal_refs,
        generated_from_reference_transcript=True,
        replaces_canonical=False,
    )
    write_model_artifact(
        runtime,
        draft_ref,
        {
            **draft.model_dump(mode="json"),
            "translation_candidate": candidate.model_dump(mode="json"),
        },
    )
    return draft_ref


def _update_asset_latest_reference(
    runtime: WorkflowRuntime,
    media_key: str,
    latest_reference_transcript_ref: str,
) -> None:
    get_asset = cast(
        Callable[[str], AssetRecord | None] | None,
        getattr(runtime.run_store, "get_asset", None),
    )
    save_asset_record = cast(
        Callable[[AssetRecord], AssetRecord] | None,
        getattr(runtime.run_store, "save_asset_record", None),
    )
    if not callable(get_asset) or not callable(save_asset_record):
        return
    asset = get_asset(media_key)
    if asset is None:
        return
    save_asset_record(
        asset.model_copy(
            update={"latest_reference_transcript_ref": latest_reference_transcript_ref}
        )
    )


def _recurring_failure_patterns(evaluated_runs: list[EvaluatedRunReport]) -> list[str]:
    counter: Counter[str] = Counter()
    for report in evaluated_runs:
        if report.translation is None:
            continue
        if report.translation.omission_risk >= 0.25:
            counter["omission_risk_high"] += 1
        if report.translation.entity_consistency < 0.8:
            counter["named_entity_preservation_low"] += 1
        if report.translation.numeric_date_unit_consistency < 1.0:
            counter["numeric_date_unit_consistency_low"] += 1
        if report.translation.addition_risk >= 0.25:
            counter["addition_risk_high"] += 1
    return sorted(pattern for pattern, count in counter.items() if count >= 3)


def _proposal_instruction(pattern: str) -> str:
    mapping = {
        "omission_risk_high": (
            "Translate every trusted transcript segment and avoid dropping short or repeated lines."
        ),
        "named_entity_preservation_low": (
            "Preserve named entities exactly unless there is a clear localized form."
        ),
        "numeric_date_unit_consistency_low": (
            "Preserve numbers, dates, and units exactly unless localization "
            "requires formatting changes."
        ),
        "addition_risk_high": (
            "Do not add unsupported content beyond what is anchored in the source transcript."
        ),
    }
    return mapping.get(pattern, "Improve translation faithfulness against the trusted transcript.")


def _current_run_compatibility(
    state: GraphState,
    runtime: WorkflowRuntime,
) -> PromptCompatibilityTuple | None:
    if state.final_translation_candidate_id is None:
        return None
    candidates = select_translation_candidates(
        runtime,
        job=state.job,
        candidate_ids=(state.final_translation_candidate_id,),
    )
    if not candidates:
        return None
    candidate = candidates[0]
    resolver = candidate.metadata.get("prompt_resolver")
    base_prompt_version = (
        resolver.get("base_prompt_version") if isinstance(resolver, dict) else None
    ) or getattr(runtime.translation_adapter, "_prompt_version", candidate.prompt_version)
    return PromptCompatibilityTuple(
        prompt_family="translation",
        model_id=candidate.model_id,
        prompt_variant_id=candidate.prompt_variant_id,
        base_prompt_version=str(base_prompt_version),
        source_language=state.job.source_language,
        target_language=state.job.target_language,
        scope_kind="project_pair",
        scope_key=build_scope_key(
            scope_kind="project_pair",
            tenant_id=state.job.tenant_id,
            project_id=state.job.project_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
        ),
    )


def _qualifying_patterns_for_compatibility(
    evaluated_runs: list[EvaluatedRunReport],
    compatibility: PromptCompatibilityTuple,
) -> list[str]:
    counter: Counter[str] = Counter()
    for report in evaluated_runs:
        if report.translation is None:
            continue
        if _compatibility_from_link(report.run) != compatibility:
            continue
        if report.translation.omission_risk >= 0.25:
            counter["omission_risk_high"] += 1
        if report.translation.entity_consistency < 0.8:
            counter["named_entity_preservation_low"] += 1
        if report.translation.numeric_date_unit_consistency < 1.0:
            counter["numeric_date_unit_consistency_low"] += 1
        if report.translation.addition_risk >= 0.25:
            counter["addition_risk_high"] += 1
    return sorted(pattern for pattern, count in counter.items() if count >= 3)


def _build_prompt_proposal(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    compatibility: PromptCompatibilityTuple,
    patterns: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    status: str,
) -> PromptEvolutionProposal:
    proposal_id = f"reference-eval-{state.run_id}-{compatibility.prompt_variant_id}"
    return PromptEvolutionProposal(
        proposal_id=proposal_id,
        job_id=state.job.job_id,
        source_consolidation_id=f"reference-evaluation-{state.run_id}",
        prompt_family="translation",
        target_model_id=compatibility.model_id,
        target_prompt_version=compatibility.base_prompt_version,
        target_prompt_variant_id=compatibility.prompt_variant_id,
        base_prompt_version=compatibility.base_prompt_version,
        compatibility=compatibility,
        status=status,  # type: ignore[arg-type]
        rationale=(
            "Trusted transcript evaluation found the same translation failure pattern across "
            "at least three completed runs for one compatible prompt tuple."
        ),
        suggested_changes=tuple(
            PromptChange(section="system", instruction=_proposal_instruction(pattern))
            for pattern in patterns
        ),
        evidence_refs=evidence_refs,
        promotion_status="candidate",
        gate_outcome=PromotionGateOutcome(
            heuristic_gate_pass=False,
            stronger_grader_pass=False,
            human_support_pass=False,
            material_disagreement=False,
            eligible_for_pair_promotion=False,
            quality_gate_status="pending",
        ),
        metadata={
            "proposal_origin": "reference_evaluation",
            "source_language": state.job.source_language,
            "target_language": state.job.target_language,
            "media_key": state.job.media_key,
            "failure_patterns": list(patterns),
            "scope_kind": compatibility.scope_kind,
            "scope_key": compatibility.scope_key,
            "tenant_id": state.job.tenant_id,
            "project_id": state.job.project_id,
        },
    )


def _save_proposal(
    runtime: WorkflowRuntime,
    proposal: PromptEvolutionProposal,
    save_proposal: Callable[[PromptEvolutionProposal], None] | None,
) -> str:
    media_key = proposal.metadata.get("media_key")
    if not isinstance(media_key, str) or not media_key:
        media_key = "unknown-media"
    proposal_ref = asset_path(
        media_key,
        "improvement-proposals",
        f"{proposal.proposal_id}.json",
    )
    proposal = proposal.model_copy(
        update={
            "metadata": {
                **proposal.metadata,
                "proposal_ref": proposal_ref,
            }
        }
    )
    write_model_artifact(runtime, proposal_ref, proposal)
    if callable(save_proposal):
        save_proposal(proposal)
    return proposal_ref


def _compatibility_from_link(link: HistoricalRunLink) -> PromptCompatibilityTuple | None:
    if (
        link.translation_model_id is None
        or link.translation_prompt_variant_id is None
        or link.translation_base_prompt_version is None
        or link.prompt_scope_kind is None
        or link.prompt_scope_key is None
    ):
        return None
    return PromptCompatibilityTuple(
        prompt_family="translation",
        model_id=link.translation_model_id,
        prompt_variant_id=link.translation_prompt_variant_id,
        base_prompt_version=link.translation_base_prompt_version,
        source_language=link.source_language,
        target_language=link.target_language,
        scope_kind=link.prompt_scope_kind,  # type: ignore[arg-type]
        scope_key=link.prompt_scope_key,
    )


def _aggregate_outcomes_for_compatibility(
    evaluated_runs: list[EvaluatedRunReport],
    compatibility: PromptCompatibilityTuple | None,
) -> tuple[ProposalAggregateMetrics | None, ProposalAggregateMetrics | None]:
    if compatibility is None:
        return None, None
    canary_scores = [
        report.translation
        for report in evaluated_runs
        if report.translation is not None
        and _compatibility_from_link(report.run) == compatibility
        and report.run.prompt_resolution_mode == "canary"
    ]
    control_scores = [
        report.translation
        for report in evaluated_runs
        if report.translation is not None
        and _compatibility_from_link(report.run) == compatibility
        and report.run.prompt_resolution_mode in {"control", "active"}
    ]
    return _aggregate_scores(canary_scores), _aggregate_scores(control_scores)


def _aggregate_scores(scores: list[TranslationScore]) -> ProposalAggregateMetrics | None:
    if not scores:
        return None
    return ProposalAggregateMetrics(
        run_count=len(scores),
        primary_quality_score=round(mean(score.primary_quality_score for score in scores), 4),
        faithfulness_judge=round(mean(score.faithfulness_judge for score in scores), 4),
        segment_coverage=round(mean(score.segment_coverage for score in scores), 4),
        entity_consistency=round(mean(score.entity_consistency for score in scores), 4),
        numeric_date_unit_consistency=round(
            mean(score.numeric_date_unit_consistency for score in scores), 4
        ),
        glossary_compliance=round(mean(score.glossary_compliance for score in scores), 4),
        omission_risk=round(mean(score.omission_risk for score in scores), 4),
        addition_risk=round(mean(score.addition_risk for score in scores), 4),
        fluency_judge=round(mean(score.fluency_judge or 0.0 for score in scores), 4),
        severe_failure_bucket_rate=round(
            sum(1 for score in scores if score.severe_failure_bucket) / len(scores),
            4,
        ),
    )


def _update_canary_status(
    proposal: PromptEvolutionProposal,
    evaluated_runs: list[EvaluatedRunReport],
) -> PromptEvolutionProposal:
    compatibility = proposal.compatibility
    if compatibility is None:
        return proposal
    canary_metrics, control_metrics = _aggregate_outcomes_for_compatibility(
        evaluated_runs,
        compatibility,
    )
    if canary_metrics is None:
        return proposal
    canary_stronger_grader, control_stronger_grader = _stronger_grader_reports(
        canary_metrics,
        control_metrics,
    )
    gate_outcome = _evaluation_gate_outcome(
        canary_metrics=canary_metrics,
        control_metrics=control_metrics,
        canary_stronger_grader=canary_stronger_grader,
        control_stronger_grader=control_stronger_grader,
        proposal_refs=(proposal.proposal_id,),
    )
    status = proposal.status
    rollback_reason = proposal.rollback_reason
    if control_metrics is not None:
        last_ten_canary_scores = [
            report.translation.primary_quality_score
            for report in evaluated_runs[-10:]
            if report.translation is not None
            and _compatibility_from_link(report.run) == compatibility
            and report.run.prompt_resolution_mode == "canary"
        ]
        rolling_canary = (
            mean(last_ten_canary_scores)
            if last_ten_canary_scores
            else canary_metrics.primary_quality_score
        )
        if (
            control_metrics.severe_failure_bucket_rate > 0
            and canary_metrics.severe_failure_bucket_rate
            >= control_metrics.severe_failure_bucket_rate * 2
        ) or rolling_canary <= control_metrics.primary_quality_score - 0.02:
            status = "rolled_back"
            rollback_reason = "canary underperformed control on quality or severe failures"
        elif (
            gate_outcome.material_disagreement is False
            and canary_metrics.run_count >= 30
            and canary_metrics.primary_quality_score >= control_metrics.primary_quality_score + 0.02
            and canary_metrics.segment_coverage >= control_metrics.segment_coverage - 0.01
            and canary_metrics.entity_consistency >= control_metrics.entity_consistency - 0.01
            and canary_metrics.numeric_date_unit_consistency
            >= control_metrics.numeric_date_unit_consistency - 0.01
            and canary_metrics.glossary_compliance >= control_metrics.glossary_compliance - 0.01
            and canary_metrics.omission_risk <= control_metrics.omission_risk + 0.01
            and canary_metrics.addition_risk <= control_metrics.addition_risk + 0.01
        ):
            status = "active"
            rollback_reason = None
    return proposal.model_copy(
        update={
            "status": status,
            "rollback_reason": rollback_reason,
            "canary_run_count": canary_metrics.run_count,
            "control_run_count": control_metrics.run_count if control_metrics is not None else 0,
            "canary_metrics": canary_metrics,
            "control_metrics": control_metrics or ProposalAggregateMetrics(),
            "canary_stronger_grader": canary_stronger_grader,
            "control_stronger_grader": control_stronger_grader,
            "gate_outcome": gate_outcome,
        }
    )


def _stronger_grader_reports(
    canary_metrics: ProposalAggregateMetrics | None,
    control_metrics: ProposalAggregateMetrics | None,
) -> tuple[StrongerGraderScore | None, StrongerGraderScore | None]:
    return _stronger_grader_score(canary_metrics), _stronger_grader_score(control_metrics)


def _stronger_grader_score(
    metrics: ProposalAggregateMetrics | None,
) -> StrongerGraderScore | None:
    if metrics is None:
        return None
    confidence = round(min(max(metrics.run_count / 10.0, 0.1), 0.99), 4)
    score = round(
        max(
            min(
                0.5 * metrics.primary_quality_score
                + 0.2 * metrics.faithfulness_judge
                + 0.15 * metrics.entity_consistency
                + 0.15 * (1.0 - max(metrics.omission_risk, metrics.addition_risk)),
                1.0,
            ),
            0.0,
        ),
        4,
    )
    return StrongerGraderScore(
        grader_id="fixed-rubric-judge",
        grader_version="v1",
        score=score,
        confidence=confidence,
        notes=("fallback_stronger_grader",),
        metadata={"preferred_backend": "COMET/XCOMET", "backend_used": "fixed-rubric-judge"},
    )


def _evaluation_gate_outcome(
    *,
    canary_metrics: ProposalAggregateMetrics | None,
    control_metrics: ProposalAggregateMetrics | None,
    canary_stronger_grader: StrongerGraderScore | None,
    control_stronger_grader: StrongerGraderScore | None,
    proposal_refs: tuple[str, ...],
) -> PromotionGateOutcome:
    heuristic_gate_pass = bool(
        canary_metrics is not None
        and (
            control_metrics is None
            or canary_metrics.primary_quality_score >= control_metrics.primary_quality_score
        )
        and (
            control_metrics is None
            or canary_metrics.severe_failure_bucket_rate
            <= control_metrics.severe_failure_bucket_rate
        )
    )
    stronger_grader_pass = bool(
        canary_stronger_grader is not None
        and (
            control_stronger_grader is None
            or canary_stronger_grader.score >= control_stronger_grader.score
        )
    )
    material_disagreement = bool(heuristic_gate_pass) != bool(stronger_grader_pass)
    notes = []
    if proposal_refs:
        notes.append("proposal_artifacts_attached")
    if material_disagreement:
        notes.append("heuristic_stronger_grader_disagreement")
    quality_gate_status = (
        "disagreed"
        if material_disagreement
        else "passed"
        if heuristic_gate_pass and stronger_grader_pass
        else "failed"
    )
    return PromotionGateOutcome(
        heuristic_gate_pass=heuristic_gate_pass,
        stronger_grader_pass=stronger_grader_pass,
        human_support_pass=False,
        rollback_signal_present=False,
        material_disagreement=material_disagreement,
        eligible_for_pair_promotion=False,
        quality_gate_status=quality_gate_status,  # type: ignore[arg-type]
        notes=tuple(notes),
    )


def _parse_srt(payload: str) -> tuple[ReferenceSegment, ...]:
    subtitles = pysubs2.SSAFile.from_string(payload, format_="srt")
    if payload.strip() and not subtitles.events:
        raise ValueError("malformed SRT timing line")

    segments = [
        ReferenceSegment(
            segment_id=f"ref-seg-{index + 1}",
            sequence=index,
            start_ms=event.start,
            end_ms=event.end,
            text=" ".join(event.plaintext.splitlines()).strip(),
        )
        for index, event in enumerate(subtitles.events)
    ]
    if any(not segment.text for segment in segments):
        raise ValueError("malformed SRT block")
    return tuple(segments)


def _normalized_tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(value)]


def _named_entities(value: str) -> tuple[str, ...]:
    entities = {
        token
        for token in re.findall(r"\b[\w.-]+\b", value)
        if len(token) > 1
        and (
            any(character.isupper() for character in token[1:])
            or token.isupper()
            or any(character.isdigit() for character in token)
        )
    }
    return tuple(sorted(entities))


def _numeric_markers(value: str) -> tuple[str, ...]:
    return tuple(
        sorted(match.group(0).replace(",", ".") for match in re.finditer(r"\d+(?:[.,]\d+)?", value))
    )


def _token_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 1.0
    return round(SequenceMatcher(a=left, b=right).ratio(), 4)


def _preservation_ratio(source_values: tuple[str, ...], target_values: tuple[str, ...]) -> float:
    if not source_values:
        return 1.0
    return len(set(source_values) & set(target_values)) / max(len(set(source_values)), 1)


def _fluency_score(values: tuple[str, ...]) -> float:
    populated = [value for value in values if value.strip()]
    if not populated:
        return 0.0
    punctuation_ratio = sum(
        1 for value in populated if value.rstrip().endswith((".", "!", "?"))
    ) / len(populated)
    repetition_penalty = 0.0
    for value in populated:
        tokens = _normalized_tokens(value)
        if len(tokens) != len(set(tokens)):
            repetition_penalty += 0.05
    return max(0.0, min(0.75 + punctuation_ratio * 0.2 - repetition_penalty, 1.0))
