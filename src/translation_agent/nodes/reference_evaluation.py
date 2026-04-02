"""Reference transcript evaluation and draft regeneration helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

import pysubs2

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import (
    AssetRecord,
    EvaluatedRunReport,
    EvaluationReport,
    HistoricalRunLink,
    PromptChange,
    PromptEvolutionProposal,
    PublishedArtifacts,
    ReferenceSegment,
    ReferenceTranscript,
    RegeneratedTranslationDraft,
    Segment,
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
    write_model_artifact,
)
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

    evaluated_runs = _load_historical_runs(
        state,
        runtime,
        reference=reference,
        trusted_transcript_ref=reference_refs[0],
    )
    recurring_patterns = _recurring_failure_patterns(evaluated_runs)
    proposal_refs = _persist_improvement_proposals(
        state,
        runtime,
        recurring_patterns=recurring_patterns,
        evaluation_refs=tuple(
            report.translation.translation_ref
            for report in evaluated_runs
            if report.translation is not None and report.translation.translation_ref is not None
        ),
    )
    regenerated_draft_ref = _generate_regenerated_draft(
        state,
        runtime,
        reference=reference,
        trusted_transcript_ref=reference_refs[0],
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
        metadata={
            "asset_id": state.job.asset_id,
            "media_fingerprint": state.job.media_fingerprint,
            "reference_history_ref": reference_refs[1],
        },
    )
    evaluation_report_ref = write_model_artifact(
        runtime,
        asset_path(state.job.media_key, "evaluations", f"{state.run_id}.json"),
        report,
    )
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
        )
    )


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
    reports: list[EvaluatedRunReport] = []
    for link in list_links(
        state.job.media_key,
        exclude_run_id=state.run_id,
    ):
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
        reports.append(
            EvaluatedRunReport(
                run=link,
                transcript=transcript_report,
                translation=translation_report,
            )
        )
    return reports


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
    segment_coverage = populated_segments / max(len(source_segments), 1)
    source_entities = _named_entities(" ".join(source_segments))
    target_text = translation.full_text
    preserved_entities = sum(1 for entity in source_entities if entity in target_text)
    named_entity_preservation = preserved_entities / max(len(source_entities), 1)
    repeated_terms = _repeated_terms(" ".join(source_segments))
    repeated_term_hits = sum(1 for term in repeated_terms if term in target_text.lower())
    terminology_consistency = repeated_term_hits / max(len(repeated_terms), 1)
    target_token_count = len(_normalized_tokens(target_text))
    source_token_count = len(_normalized_tokens(" ".join(source_segments)))
    length_ratio = target_token_count / max(source_token_count, 1)
    omission_risk = round(max(0.0, 1.0 - min(segment_coverage, 1.0)), 4)
    addition_risk = round(max(0.0, min(length_ratio - 1.5, 1.0)), 4) if length_ratio > 1.5 else 0.0
    faithfulness = round(
        max(
            0.0,
            min(
                1.0,
                (segment_coverage * 0.5)
                + (named_entity_preservation * 0.3)
                + (terminology_consistency * 0.2)
                - (addition_risk * 0.2),
            ),
        ),
        4,
    )
    notes: list[str] = []
    if terminology_consistency < 0.6 and repeated_terms:
        notes.append("repeated terminology drift")
    if named_entity_preservation < 0.8 and source_entities:
        notes.append("named entity preservation dropped")
    if omission_risk > 0.25:
        notes.append("segment-level omissions detected")
    return TranslationScore(
        run_id=link.run_id,
        translation_ref=link.translation_ref,
        trusted_transcript_ref=trusted_transcript_ref,
        faithfulness=faithfulness,
        omission_risk=omission_risk,
        addition_risk=round(min(addition_risk, 1.0), 4),
        terminology_consistency=round(min(terminology_consistency, 1.0), 4),
        named_entity_preservation=round(min(named_entity_preservation, 1.0), 4),
        repeated_failure_terms=tuple(repeated_terms if terminology_consistency < 0.6 else ()),
        notes=tuple(notes),
    )


def _persist_improvement_proposals(
    state: GraphState,
    runtime: WorkflowRuntime,
    *,
    recurring_patterns: list[str],
    evaluation_refs: tuple[str, ...],
) -> tuple[str, ...]:
    if not recurring_patterns:
        return ()
    proposal_id = f"reference-eval-{state.run_id}"
    proposal = PromptEvolutionProposal(
        proposal_id=proposal_id,
        job_id=state.job.job_id,
        source_consolidation_id=f"reference-evaluation-{state.run_id}",
        prompt_family="translation",
        target_model_id=runtime.translation_adapter.model_id,
        target_prompt_version=getattr(
            runtime.translation_adapter,
            "_prompt_version",
            "unversioned",
        ),
        target_prompt_variant_id=None,
        status="proposed",
        activation_mode="approval_required",
        auto_activate=False,
        rationale=(
            "Trusted transcript evidence shows repeated translation failures across prior runs."
        ),
        suggested_changes=tuple(
            PromptChange(section="system", instruction=_proposal_instruction(pattern))
            for pattern in recurring_patterns
        ),
        evidence_refs=evaluation_refs,
        metadata={
            "proposal_origin": "reference_evaluation",
            "source_language": state.job.source_language,
            "target_language": state.job.target_language,
            "media_key": state.job.media_key,
            "failure_patterns": recurring_patterns,
        },
    )
    proposal_ref = asset_path(
        state.job.media_key,
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
    save_proposal = getattr(runtime.run_store, "save_prompt_evolution_proposal", None)
    if callable(save_proposal):
        save_proposal(proposal)
    return (proposal_ref,)


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
        media_key=state.job.media_key,
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
        if report.translation.named_entity_preservation < 0.8:
            counter["named_entity_preservation_low"] += 1
        if report.translation.terminology_consistency < 0.6:
            counter["terminology_consistency_low"] += 1
    return sorted(pattern for pattern, count in counter.items() if count >= 2)


def _proposal_instruction(pattern: str) -> str:
    mapping = {
        "omission_risk_high": (
            "Translate every trusted transcript segment and avoid dropping short or repeated lines."
        ),
        "named_entity_preservation_low": (
            "Preserve named entities exactly unless there is a clear localized form."
        ),
        "terminology_consistency_low": (
            "Keep repeated source terminology stable across the full transcript."
        ),
    }
    return mapping.get(pattern, "Improve translation faithfulness against the trusted transcript.")


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
    entities = {token for token in re.findall(r"\b[A-Z][A-Za-z0-9]+\b", value) if len(token) > 1}
    return tuple(sorted(entities))


def _repeated_terms(value: str) -> tuple[str, ...]:
    counts = Counter(token.lower() for token in _WORD_RE.findall(value) if len(token) >= 4)
    return tuple(sorted(token for token, count in counts.items() if count >= 2))


def _token_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 1.0
    return round(SequenceMatcher(a=left, b=right).ratio(), 4)
