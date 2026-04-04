"""Span-level transcript synthesis pipeline with optional live reasoning."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from hashlib import sha256
from typing import TYPE_CHECKING

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

from translation_agent.models import (
    CanonicalTranscriptSpan,
    Segment,
    SynthesizedTranscriptArtifact,
    TranscriptCandidate,
    TranscriptQualityMetrics,
    TranscriptReviewIssue,
    TranscriptSpanCandidate,
    TranscriptSpanDecision,
    TranscriptSpanProvenance,
    TranscriptSynthesisRecord,
    TranscriptSynthesisReview,
    TranscriptUnresolvedSpan,
)
from translation_agent.observability import TraceEvent
from translation_agent.parallelism import (
    ParallelTaskClass,
    concurrency_trace_attributes,
)

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from translation_agent.graph.runtime import WorkflowRuntime
    from translation_agent.models.jobs import JobContext

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_CANONICAL_SPAN_GAP_MS = 320


def build_canonical_transcript_spans(
    candidates: tuple[TranscriptCandidate, ...],
) -> tuple[CanonicalTranscriptSpan, ...]:
    """Build utterance-like canonical spans from the union of provider segments."""

    flattened: list[tuple[TranscriptCandidate, Segment]] = []
    for candidate in candidates:
        for segment in candidate.segments:
            if not (segment.source_text or "").strip():
                continue
            if segment.end_ms <= segment.start_ms:
                continue
            flattened.append((candidate, segment))
    if not flattened:
        return ()

    flattened.sort(
        key=lambda item: (
            item[1].start_ms,
            item[1].end_ms,
            item[0].candidate_id,
            item[1].segment_id,
        )
    )

    groups: list[list[tuple[TranscriptCandidate, Segment]]] = []
    current: list[tuple[TranscriptCandidate, Segment]] = []
    current_end = -1
    for item in flattened:
        _candidate, segment = item
        if not current:
            current = [item]
            current_end = segment.end_ms
            continue
        if segment.start_ms <= current_end + _CANONICAL_SPAN_GAP_MS:
            current.append(item)
            current_end = max(current_end, segment.end_ms)
            continue
        groups.append(current)
        current = [item]
        current_end = segment.end_ms
    if current:
        groups.append(current)

    spans: list[CanonicalTranscriptSpan] = []
    for index, group in enumerate(groups, start=1):
        supporting_candidate_ids = tuple(
            dict.fromkeys(candidate.candidate_id for candidate, _segment in group)
        )
        supporting_provider_ids = tuple(
            dict.fromkeys(candidate.provider_id for candidate, _segment in group)
        )
        speakers = tuple(
            dict.fromkeys(
                segment.speaker for _candidate, segment in group if segment.speaker is not None
            )
        )
        span_id = f"canonical-span-{index:04d}"
        spans.append(
            CanonicalTranscriptSpan(
                canonical_span_id=span_id,
                start_ms=min(segment.start_ms for _candidate, segment in group),
                end_ms=max(segment.end_ms for _candidate, segment in group),
                speaker=speakers[0] if len(speakers) == 1 else None,
                supporting_candidate_ids=supporting_candidate_ids,
                supporting_provider_ids=supporting_provider_ids,
                metadata={
                    "supporting_segment_ids": [
                        f"{candidate.candidate_id}:{segment.segment_id}"
                        for candidate, segment in group
                    ]
                },
            )
        )
    return tuple(spans)


def build_span_candidates(
    spans: tuple[CanonicalTranscriptSpan, ...],
    candidates: tuple[TranscriptCandidate, ...],
) -> tuple[TranscriptSpanCandidate, ...]:
    """Materialize provider evidence for each canonical transcript span."""

    by_candidate = {candidate.candidate_id: candidate for candidate in candidates}
    span_candidates: list[TranscriptSpanCandidate] = []
    for index, span in enumerate(spans):
        previous_text = _neighbor_text(spans, by_candidate, index - 1)
        next_text = _neighbor_text(spans, by_candidate, index + 1)
        for candidate in candidates:
            overlapping = tuple(
                segment
                for segment in candidate.segments
                if _overlap_ms(span.start_ms, span.end_ms, segment.start_ms, segment.end_ms) > 0
            )
            if not overlapping:
                continue
            overlap_ms = sum(
                _overlap_ms(span.start_ms, span.end_ms, segment.start_ms, segment.end_ms)
                for segment in overlapping
            )
            text = " ".join(
                (segment.source_text or "").strip()
                for segment in overlapping
                if (segment.source_text or "").strip()
            ).strip()
            if not text:
                continue
            speaker = next(
                (segment.speaker for segment in overlapping if segment.speaker is not None),
                None,
            )
            span_candidates.append(
                TranscriptSpanCandidate(
                    span_candidate_id=f"{span.canonical_span_id}--{candidate.candidate_id}",
                    canonical_span_id=span.canonical_span_id,
                    provider_id=candidate.provider_id,
                    candidate_id=candidate.candidate_id,
                    source_segment_ids=tuple(segment.segment_id for segment in overlapping),
                    start_ms=min(segment.start_ms for segment in overlapping),
                    end_ms=max(segment.end_ms for segment in overlapping),
                    timing_overlap_ms=overlap_ms,
                    timing_overlap_ratio=round(
                        overlap_ms / max(span.end_ms - span.start_ms, 1),
                        4,
                    ),
                    normalized_text=text,
                    speaker_label=speaker,
                    previous_span_text=previous_text,
                    next_span_text=next_text,
                    metadata={
                        "support_segment_count": len(overlapping),
                        "provider_rank": int(candidate.metadata.get("provider_rank", 100)),
                    },
                )
            )
    return tuple(span_candidates)


def run_selector_agent(
    *,
    job: JobContext,
    run_id: str,
    runtime: WorkflowRuntime,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
) -> TranscriptSynthesisRecord:
    """Emit structured per-span synthesis decisions."""

    if _should_use_live_reasoning(runtime):
        decisions = _live_selector_decisions(
            run_id=run_id,
            runtime=runtime,
            spans=spans,
            span_candidates=span_candidates,
        )
    else:
        decisions = tuple(
            _selector_decision_for_span(
                span,
                _span_candidates_for(span.canonical_span_id, span_candidates),
            )
            for span in spans
        )
    unresolved_span_ids = tuple(
        decision.canonical_span_id
        for decision in decisions
        if decision.decision_type == "mark_unresolved"
    )
    payload = {
        "record_id": _record_id(job.job_id, "selector", run_id),
        "job_id": job.job_id,
        "run_id": run_id,
        "agent_role": "selector",
        "reasoning_provider": runtime.reasoning_profile.provider_id,
        "reasoning_model_id": runtime.reasoning_profile.model_id,
        "canonical_span_count": len(spans),
        "decisions": decisions,
        "unresolved_span_ids": unresolved_span_ids,
        "provider_support_summary": _provider_support_summary(span_candidates),
        "metadata": {"base_url_source": runtime.reasoning_profile.base_url_source},
    }
    return TranscriptSynthesisRecord.model_validate(payload)


def run_reviewer_agent(
    *,
    job: JobContext,
    run_id: str,
    runtime: WorkflowRuntime,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
    selector_record: TranscriptSynthesisRecord,
) -> TranscriptSynthesisReview:
    """Audit synthesized span decisions for grounding, coverage, timing, and provenance."""

    if _should_use_live_reasoning(runtime):
        return _live_reviewer_review(
            job=job,
            run_id=run_id,
            runtime=runtime,
            spans=spans,
            span_candidates=span_candidates,
            selector_record=selector_record,
        )

    decision_by_span = {
        decision.canonical_span_id: decision for decision in selector_record.decisions
    }
    accepted_span_ids: list[str] = []
    corrected_decisions: list[TranscriptSpanDecision] = []
    unresolved_span_ids: list[str] = []
    dropped_supported_span_ids: list[str] = []
    issues: list[TranscriptReviewIssue] = []
    for span in spans:
        supported = _span_candidates_for(span.canonical_span_id, span_candidates)
        decision = decision_by_span.get(span.canonical_span_id)
        if not supported:
            continue
        if decision is None:
            dropped_supported_span_ids.append(span.canonical_span_id)
            unresolved_span_ids.append(span.canonical_span_id)
            issues.append(
                TranscriptReviewIssue(
                    canonical_span_id=span.canonical_span_id,
                    issue_type="coverage",
                    severity="critical",
                    description="Supported speech span was not synthesized.",
                )
            )
            continue
        if decision.decision_type == "mark_unresolved":
            unresolved_span_ids.append(span.canonical_span_id)
            issues.append(
                TranscriptReviewIssue(
                    canonical_span_id=span.canonical_span_id,
                    issue_type="coverage",
                    severity="major",
                    description="Selector left a supported span unresolved.",
                )
            )
            continue
        if not decision.source_fragment_refs:
            unresolved_span_ids.append(span.canonical_span_id)
            issues.append(
                TranscriptReviewIssue(
                    canonical_span_id=span.canonical_span_id,
                    issue_type="provenance",
                    severity="critical",
                    description="Resolved span is missing provenance fragment refs.",
                )
            )
            continue
        if not _decision_grounded(decision, supported):
            candidate_merge = _bounded_merge_decision(
                span,
                supported,
                rationale="reviewer_correction",
            )
            if candidate_merge is not None:
                corrected_decisions.append(candidate_merge)
                accepted_span_ids.append(span.canonical_span_id)
            else:
                unresolved_span_ids.append(span.canonical_span_id)
            issues.append(
                TranscriptReviewIssue(
                    canonical_span_id=span.canonical_span_id,
                    issue_type="grounding",
                    severity="critical",
                    description="Resolved span included text not grounded in provider evidence.",
                )
            )
            continue
        if decision.start_ms < span.start_ms or decision.end_ms > span.end_ms:
            corrected_decisions.append(
                decision.model_copy(
                    update={
                        "start_ms": max(decision.start_ms, span.start_ms),
                        "end_ms": min(decision.end_ms, span.end_ms),
                    }
                )
            )
            issues.append(
                TranscriptReviewIssue(
                    canonical_span_id=span.canonical_span_id,
                    issue_type="timing",
                    severity="major",
                    description="Resolved span timing exceeded canonical span bounds.",
                )
            )
        accepted_span_ids.append(span.canonical_span_id)

    for left, right in zip(spans, spans[1:], strict=False):
        left_decision = decision_by_span.get(left.canonical_span_id)
        right_decision = decision_by_span.get(right.canonical_span_id)
        if left_decision is None or right_decision is None:
            continue
        if left_decision.decision_type == "mark_unresolved":
            continue
        if right_decision.decision_type == "mark_unresolved":
            continue
        if right_decision.start_ms < left_decision.end_ms:
            unresolved_span_ids.extend([left.canonical_span_id, right.canonical_span_id])
            issues.append(
                TranscriptReviewIssue(
                    canonical_span_id=right.canonical_span_id,
                    issue_type="timing",
                    severity="critical",
                    description="Resolved transcript spans overlap and must be re-adjudicated.",
                )
            )

    payload = {
        "review_id": _record_id(job.job_id, "reviewer", run_id),
        "job_id": job.job_id,
        "run_id": run_id,
        "reasoning_provider": runtime.reasoning_profile.provider_id,
        "reasoning_model_id": runtime.reasoning_profile.model_id,
        "accepted_span_ids": tuple(dict.fromkeys(accepted_span_ids)),
        "corrected_decisions": tuple(corrected_decisions),
        "unresolved_span_ids": tuple(dict.fromkeys(unresolved_span_ids)),
        "dropped_supported_span_ids": tuple(dict.fromkeys(dropped_supported_span_ids)),
        "issues": tuple(issues),
        "metadata": {"base_url_source": runtime.reasoning_profile.base_url_source},
    }
    return TranscriptSynthesisReview.model_validate(payload)


def run_global_adjudicator(
    *,
    job: JobContext,
    run_id: str,
    runtime: WorkflowRuntime,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
    selector_record: TranscriptSynthesisRecord,
    review: TranscriptSynthesisReview,
) -> TranscriptSynthesisRecord:
    """Resolve remaining unresolved transcript spans or leave them blocked."""

    if _should_use_live_reasoning(runtime):
        decisions = _live_global_decisions(
            run_id=run_id,
            runtime=runtime,
            spans=spans,
            span_candidates=span_candidates,
            selector_record=selector_record,
            review=review,
        )
        payload = {
            "record_id": _record_id(job.job_id, "global_adjudicator", run_id),
            "job_id": job.job_id,
            "run_id": run_id,
            "agent_role": "global_adjudicator",
            "reasoning_provider": runtime.reasoning_profile.provider_id,
            "reasoning_model_id": runtime.reasoning_profile.model_id,
            "canonical_span_count": len(spans),
            "decisions": decisions,
            "unresolved_span_ids": tuple(
                decision.canonical_span_id
                for decision in decisions
                if decision.decision_type == "mark_unresolved"
            ),
            "provider_support_summary": _provider_support_summary(span_candidates),
            "metadata": {"base_url_source": runtime.reasoning_profile.base_url_source},
        }
        return TranscriptSynthesisRecord.model_validate(payload)

    candidate_decisions = {
        decision.canonical_span_id: decision for decision in selector_record.decisions
    }
    candidate_decisions.update(
        {decision.canonical_span_id: decision for decision in review.corrected_decisions}
    )
    unresolved_set = set(selector_record.unresolved_span_ids) | set(review.unresolved_span_ids)
    for span in spans:
        if span.canonical_span_id not in candidate_decisions:
            candidate_decisions[span.canonical_span_id] = TranscriptSpanDecision(
                canonical_span_id=span.canonical_span_id,
                decision_type="mark_unresolved",
                rationale="No selector decision was available for the canonical span.",
            )
    for span in spans:
        if span.canonical_span_id not in unresolved_set:
            continue
        supported = _span_candidates_for(span.canonical_span_id, span_candidates)
        candidate_decisions[span.canonical_span_id] = _global_decision_for_span(
            span,
            supported,
            existing=candidate_decisions.get(span.canonical_span_id),
        )
    decisions = tuple(candidate_decisions[span.canonical_span_id] for span in spans)

    payload = {
        "record_id": _record_id(job.job_id, "global_adjudicator", run_id),
        "job_id": job.job_id,
        "run_id": run_id,
        "agent_role": "global_adjudicator",
        "reasoning_provider": runtime.reasoning_profile.provider_id,
        "reasoning_model_id": runtime.reasoning_profile.model_id,
        "canonical_span_count": len(spans),
        "decisions": decisions,
        "unresolved_span_ids": tuple(
            decision.canonical_span_id
            for decision in decisions
            if decision.decision_type == "mark_unresolved"
        ),
        "provider_support_summary": _provider_support_summary(span_candidates),
        "metadata": {"base_url_source": runtime.reasoning_profile.base_url_source},
    }
    return TranscriptSynthesisRecord.model_validate(payload)


def materialize_synthesized_transcript(
    *,
    job: JobContext,
    run_id: str,
    language: str,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
    selector_record: TranscriptSynthesisRecord,
    review: TranscriptSynthesisReview,
    global_record: TranscriptSynthesisRecord,
) -> SynthesizedTranscriptArtifact:
    """Create the synthesized transcript artifact and deterministic quality metrics."""

    final_decisions = {
        decision.canonical_span_id: decision for decision in selector_record.decisions
    }
    final_decisions.update(
        {decision.canonical_span_id: decision for decision in review.corrected_decisions}
    )
    final_decisions.update(
        {decision.canonical_span_id: decision for decision in global_record.decisions}
    )

    final_segments: list[Segment] = []
    provenance: list[TranscriptSpanProvenance] = []
    unresolved_spans: list[TranscriptUnresolvedSpan] = []

    for span in spans:
        supported = _span_candidates_for(span.canonical_span_id, span_candidates)
        decision = final_decisions.get(span.canonical_span_id)
        if decision is None or decision.decision_type == "mark_unresolved":
            if supported:
                unresolved_spans.append(
                    TranscriptUnresolvedSpan(
                        canonical_span_id=span.canonical_span_id,
                        start_ms=span.start_ms,
                        end_ms=span.end_ms,
                        provider_ids=tuple(dict.fromkeys(item.provider_id for item in supported)),
                        candidate_ids=tuple(dict.fromkeys(item.candidate_id for item in supported)),
                        reason=(
                            decision.rationale
                            if decision is not None and decision.rationale
                            else "No grounded span synthesis decision survived adjudication."
                        ),
                    )
                )
            continue

        final_segments.append(
            Segment(
                segment_id=_segment_id_for_decision(span.canonical_span_id, decision),
                start_ms=decision.start_ms,
                end_ms=decision.end_ms,
                speaker=decision.speaker_label,
                source_text=decision.output_text,
                annotations={
                    "source_span_id": span.canonical_span_id,
                    "synthesis_mode": decision.decision_type,
                    "source_fragment_refs": list(decision.source_fragment_refs),
                    "source_candidate_ids": list(decision.selected_candidate_ids),
                },
            )
        )
        evidence = [
            candidate
            for candidate in supported
            if candidate.span_candidate_id in set(decision.selected_span_candidate_ids)
        ]
        provenance.append(
            TranscriptSpanProvenance(
                canonical_span_id=span.canonical_span_id,
                synthesis_mode=decision.decision_type,
                source_fragment_refs=decision.source_fragment_refs,
                provider_ids=tuple(dict.fromkeys(item.provider_id for item in evidence)),
                candidate_ids=tuple(dict.fromkeys(item.candidate_id for item in evidence)),
                reasoning_refs=tuple(
                    ref
                    for ref in (
                        selector_record.record_id,
                        review.review_id,
                        global_record.record_id,
                    )
                    if ref
                ),
            )
        )

    quality_metrics = _quality_metrics(spans, span_candidates, final_segments, unresolved_spans)
    blocker_tags = blocking_failures_for_artifact(quality_metrics)
    artifact = SynthesizedTranscriptArtifact(
        artifact_id=f"synth-{job.job_id}",
        job_id=job.job_id,
        run_id=run_id,
        language=language,
        transcript_metadata={
            "selector_record_id": selector_record.record_id,
            "review_id": review.review_id,
            "global_adjudicator_record_id": global_record.record_id,
            "reasoning_provider": selector_record.reasoning_provider,
            "reasoning_model_id": selector_record.reasoning_model_id,
            "normalization_version": "transcript-synthesis-v1",
            "blocker_tags": list(blocker_tags),
        },
        canonical_spans=spans,
        span_candidates=span_candidates,
        final_segments=tuple(final_segments),
        provenance=tuple(provenance),
        unresolved_spans=tuple(unresolved_spans),
        quality_metrics=quality_metrics,
        full_text=" ".join(
            (segment.source_text or "").strip()
            for segment in final_segments
            if (segment.source_text or "").strip()
        ),
        status=("ready" if not blocker_tags else "blocked"),
    )
    return SynthesizedTranscriptArtifact.model_validate(artifact.model_dump(mode="json"))


def blocking_failures_for_artifact(metrics: TranscriptQualityMetrics) -> tuple[str, ...]:
    """Return deterministic blocking failure tags for publish gating."""

    failures: list[str] = []
    if metrics.unresolved_span_count > 0:
        failures.append("unresolved_supported_spans")
    if metrics.overlap_count > 0:
        failures.append("transcript_overlaps")
    if metrics.non_monotonic_count > 0:
        failures.append("transcript_non_monotonic")
    if metrics.zero_length_count > 0:
        failures.append("transcript_zero_length_segments")
    if metrics.dropped_supported_span_count > 0:
        failures.append("transcript_coverage_drop")
    return tuple(failures)


def _selector_decision_for_span(
    span: CanonicalTranscriptSpan,
    candidates: tuple[TranscriptSpanCandidate, ...],
) -> TranscriptSpanDecision:
    if not candidates:
        return TranscriptSpanDecision(
            canonical_span_id=span.canonical_span_id,
            decision_type="mark_unresolved",
            rationale="No provider evidence overlapped the canonical speech span.",
        )

    ranked = sorted(candidates, key=_span_candidate_sort_key, reverse=True)
    top = ranked[0]
    if len(ranked) == 1:
        return _select_decision(span, top, rationale="Only one provider supported the span.")

    runner_up = ranked[1]
    if _texts_equivalent(top.normalized_text, runner_up.normalized_text):
        return _select_decision(
            span,
            top,
            rationale="Providers aligned on meaning and timing; selected the stronger overlap.",
        )

    if _dominates(top, runner_up):
        return _select_decision(
            span,
            top,
            rationale="One provider dominated coverage, meaning stability, and timing.",
        )

    merge = _bounded_merge_decision(
        span,
        ranked[:2],
        rationale="Complementary provider evidence covered distinct grounded parts of the span.",
    )
    if merge is not None:
        return merge

    return TranscriptSpanDecision(
        canonical_span_id=span.canonical_span_id,
        decision_type="mark_unresolved",
        rationale="Provider evidence conflicted and no bounded grounded merge was available.",
        conflict_reasons=("conflicting_provider_evidence",),
    )


def _global_decision_for_span(
    span: CanonicalTranscriptSpan,
    candidates: tuple[TranscriptSpanCandidate, ...],
    *,
    existing: TranscriptSpanDecision | None,
) -> TranscriptSpanDecision:
    if len(candidates) == 1:
        return _select_decision(
            span,
            candidates[0],
            rationale="Global adjudicator accepted the only grounded provider span.",
        )
    ranked = sorted(candidates, key=_span_candidate_sort_key, reverse=True)
    if ranked and len(ranked) > 1 and _dominates(ranked[0], ranked[1]):
        return _select_decision(
            span,
            ranked[0],
            rationale="Global adjudicator resolved the span in favor of the stronger provider.",
        )
    merge = _bounded_merge_decision(
        span,
        ranked[:2],
        rationale="Global adjudicator merged complementary grounded fragments.",
    )
    if merge is not None:
        return merge
    reason = (
        existing.rationale
        if existing is not None and existing.rationale
        else "Global adjudicator could not resolve the span without ungrounded rewriting."
    )
    return TranscriptSpanDecision(
        canonical_span_id=span.canonical_span_id,
        decision_type="mark_unresolved",
        rationale=reason,
        conflict_reasons=("transcript_blocked",),
    )


def _select_decision(
    span: CanonicalTranscriptSpan,
    candidate: TranscriptSpanCandidate,
    *,
    rationale: str,
) -> TranscriptSpanDecision:
    return TranscriptSpanDecision(
        canonical_span_id=span.canonical_span_id,
        decision_type="select_provider_span",
        selected_candidate_ids=(candidate.candidate_id,),
        selected_span_candidate_ids=(candidate.span_candidate_id,),
        source_fragment_refs=tuple(
            f"{candidate.candidate_id}:{segment_id}" for segment_id in candidate.source_segment_ids
        ),
        output_text=candidate.normalized_text,
        speaker_label=candidate.speaker_label or span.speaker,
        start_ms=max(span.start_ms, candidate.start_ms),
        end_ms=min(span.end_ms, candidate.end_ms),
        rationale=rationale,
    )


def _bounded_merge_decision(
    span: CanonicalTranscriptSpan,
    candidates: tuple[TranscriptSpanCandidate, ...] | list[TranscriptSpanCandidate],
    *,
    rationale: str,
) -> TranscriptSpanDecision | None:
    deduped: dict[str, TranscriptSpanCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.span_candidate_id, candidate)
    unique_candidates = tuple(deduped.values())
    if len(unique_candidates) < 2:
        return None
    if not _merge_is_complementary(unique_candidates):
        return None
    ordered = tuple(
        sorted(
            unique_candidates,
            key=lambda candidate: (
                candidate.start_ms,
                -candidate.timing_overlap_ratio,
                candidate.provider_id,
            ),
        )
    )
    fragments: list[str] = []
    fragment_refs: list[str] = []
    selected_candidate_ids: list[str] = []
    selected_span_candidate_ids: list[str] = []
    for candidate in ordered:
        fragment = candidate.normalized_text.strip()
        if not fragment:
            continue
        if any(
            _texts_equivalent(fragment, existing) or fragment in existing for existing in fragments
        ):
            continue
        fragments.append(fragment)
        selected_candidate_ids.append(candidate.candidate_id)
        selected_span_candidate_ids.append(candidate.span_candidate_id)
        fragment_refs.extend(
            f"{candidate.candidate_id}:{segment_id}" for segment_id in candidate.source_segment_ids
        )
    merged_text = " ".join(fragments).strip()
    if not merged_text:
        return None
    if len(_tokenize(merged_text)) <= max(
        len(_tokenize(candidate.normalized_text)) for candidate in ordered
    ):
        return None
    if not _grounded_in_any(merged_text, ordered):
        return None
    return TranscriptSpanDecision(
        canonical_span_id=span.canonical_span_id,
        decision_type="merge_provider_spans",
        selected_candidate_ids=tuple(dict.fromkeys(selected_candidate_ids)),
        selected_span_candidate_ids=tuple(dict.fromkeys(selected_span_candidate_ids)),
        source_fragment_refs=tuple(dict.fromkeys(fragment_refs)),
        output_text=merged_text,
        speaker_label=next(
            (candidate.speaker_label for candidate in ordered if candidate.speaker_label),
            span.speaker,
        ),
        start_ms=max(span.start_ms, min(candidate.start_ms for candidate in ordered)),
        end_ms=min(span.end_ms, max(candidate.end_ms for candidate in ordered)),
        rationale=rationale,
    )


def _merge_is_complementary(candidates: tuple[TranscriptSpanCandidate, ...]) -> bool:
    first = candidates[0]
    first_tokens = set(_tokenize(first.normalized_text))
    for candidate in candidates[1:]:
        candidate_tokens = set(_tokenize(candidate.normalized_text))
        if first_tokens & candidate_tokens:
            return True
        if candidate.start_ms != first.start_ms or candidate.end_ms != first.end_ms:
            return True
    return False


def _decision_grounded(
    decision: TranscriptSpanDecision,
    supported: tuple[TranscriptSpanCandidate, ...],
) -> bool:
    selected = tuple(
        candidate
        for candidate in supported
        if candidate.span_candidate_id in set(decision.selected_span_candidate_ids)
    )
    if not selected:
        return False
    return _grounded_in_any(decision.output_text, selected)


def _grounded_in_any(text: str, candidates: tuple[TranscriptSpanCandidate, ...]) -> bool:
    output_tokens = set(_tokenize(text))
    if not output_tokens:
        return False
    source_tokens = set()
    for candidate in candidates:
        source_tokens.update(_tokenize(candidate.normalized_text))
    return output_tokens.issubset(source_tokens)


def _quality_metrics(
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
    final_segments: list[Segment],
    unresolved_spans: list[TranscriptUnresolvedSpan],
) -> TranscriptQualityMetrics:
    overlap_count = 0
    non_monotonic_count = 0
    zero_length_count = sum(1 for segment in final_segments if segment.end_ms <= segment.start_ms)
    for left, right in zip(final_segments, final_segments[1:], strict=False):
        if right.start_ms < left.end_ms:
            overlap_count += 1
        if right.start_ms < left.start_ms or right.end_ms < left.end_ms:
            non_monotonic_count += 1
    supported_span_ids = {
        candidate.canonical_span_id for candidate in span_candidates if candidate.normalized_text
    }
    unresolved_span_ids = {span.canonical_span_id for span in unresolved_spans}
    emitted_span_ids = {
        str(segment.annotations.get("source_span_id") or segment.segment_id)
        for segment in final_segments
    }
    dropped_supported_span_count = len(supported_span_ids - unresolved_span_ids - emitted_span_ids)
    return TranscriptQualityMetrics(
        canonical_span_count=len(spans),
        supported_span_count=len(supported_span_ids),
        emitted_span_count=len(final_segments),
        unresolved_span_count=len(unresolved_spans),
        overlap_count=overlap_count,
        non_monotonic_count=non_monotonic_count,
        zero_length_count=zero_length_count,
        dropped_supported_span_count=dropped_supported_span_count,
        provider_support_summary=_provider_support_summary(span_candidates),
    )


def _provider_support_summary(
    span_candidates: tuple[TranscriptSpanCandidate, ...],
) -> dict[str, int]:
    summary: dict[str, int] = defaultdict(int)
    for item in span_candidates:
        summary[item.provider_id] += 1
    return dict(sorted(summary.items()))


def _span_candidates_for(
    canonical_span_id: str,
    span_candidates: tuple[TranscriptSpanCandidate, ...],
) -> tuple[TranscriptSpanCandidate, ...]:
    return tuple(item for item in span_candidates if item.canonical_span_id == canonical_span_id)


def _neighbor_text(
    spans: tuple[CanonicalTranscriptSpan, ...],
    by_candidate: dict[str, TranscriptCandidate],
    index: int,
) -> str | None:
    if index < 0 or index >= len(spans):
        return None
    span = spans[index]
    for candidate_id in span.supporting_candidate_ids:
        candidate = by_candidate.get(candidate_id)
        if candidate is None:
            continue
        fragments = [
            (segment.source_text or "").strip()
            for segment in candidate.segments
            if _overlap_ms(span.start_ms, span.end_ms, segment.start_ms, segment.end_ms) > 0
            and (segment.source_text or "").strip()
        ]
        if fragments:
            return " ".join(fragments)
    return None


def _span_candidate_sort_key(candidate: TranscriptSpanCandidate) -> tuple[float, int, int]:
    token_count = len(_tokenize(candidate.normalized_text))
    provider_rank = int(candidate.metadata.get("provider_rank", 100))
    return (
        round(candidate.timing_overlap_ratio, 4),
        token_count,
        -provider_rank,
    )


def _dominates(left: TranscriptSpanCandidate, right: TranscriptSpanCandidate) -> bool:
    left_tokens = len(_tokenize(left.normalized_text))
    right_tokens = len(_tokenize(right.normalized_text))
    if (
        left.timing_overlap_ratio >= right.timing_overlap_ratio + 0.15
        and left_tokens >= right_tokens
    ):
        return True
    return (
        left_tokens >= right_tokens * 1.2
        and left.timing_overlap_ratio >= right.timing_overlap_ratio
    )


def _texts_equivalent(left: str, right: str) -> bool:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return overlap >= 0.82


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(text))


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _record_id(job_id: str, role: str, run_id: str) -> str:
    digest = sha256(f"{job_id}:{role}:{run_id}".encode()).hexdigest()[:12]
    return f"ts-{role}-{digest}"


def _segment_id_for_decision(canonical_span_id: str, decision: TranscriptSpanDecision) -> str:
    if decision.source_fragment_refs:
        first = decision.source_fragment_refs[0]
        if ":" in first:
            return first.split(":", 1)[1]
    return canonical_span_id


def _should_use_live_reasoning(runtime: WorkflowRuntime) -> bool:
    profile = runtime.reasoning_profile
    return bool(profile.live_adapter_enabled and profile.api_key and profile.model_id)


def _reasoning_client(runtime: WorkflowRuntime) -> OpenAI:
    profile = runtime.reasoning_profile
    return OpenAI(
        api_key=profile.api_key,
        base_url=profile.base_url,
        timeout=60.0,
        max_retries=0,
    )


def _live_selector_decisions(
    *,
    run_id: str,
    runtime: WorkflowRuntime,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
) -> tuple[TranscriptSpanDecision, ...]:
    response = _invoke_reasoning_json(
        runtime=runtime,
        run_id=run_id,
        schema_name="transcript_selector",
        schema=_selector_output_schema(),
        system_prompt=(
            "You are the transcript selector. For each canonical transcript span, choose the best "
            "grounded provider span, merge grounded fragments, or mark the span unresolved. "
            "Never invent words not present in the evidence. You may reconcile slight timing drift "
            "only within the canonical span and the supported evidence timing."
        ),
        user_prompt=json.dumps(
            {
                "spans": [_span_payload(span) for span in spans],
                "span_candidates": [_candidate_payload(item) for item in span_candidates],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        trace_name_prefix="transcript_synthesis.selector",
        trace_attributes={
            "canonical_span_count": len(spans),
            "span_candidate_count": len(span_candidates),
        },
    )
    fallback_map = {
        span.canonical_span_id: _selector_decision_for_span(
            span,
            _span_candidates_for(span.canonical_span_id, span_candidates),
        )
        for span in spans
    }
    return _normalized_decisions(
        spans=spans,
        raw_decisions=response.get("decisions"),
        fallback_map=fallback_map,
    )


def _live_reviewer_review(
    *,
    job: JobContext,
    run_id: str,
    runtime: WorkflowRuntime,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
    selector_record: TranscriptSynthesisRecord,
) -> TranscriptSynthesisReview:
    response = _invoke_reasoning_json(
        runtime=runtime,
        run_id=run_id,
        schema_name="transcript_reviewer",
        schema=_reviewer_output_schema(),
        system_prompt=(
            "You are the transcript reviewer. Audit selector decisions for grounding, timing, "
            "coverage, and provenance. Correct only when fully grounded in the provided evidence. "
            "Otherwise leave the span unresolved."
        ),
        user_prompt=json.dumps(
            {
                "spans": [_span_payload(span) for span in spans],
                "span_candidates": [_candidate_payload(item) for item in span_candidates],
                "selector_decisions": [
                    decision.model_dump(mode="json") for decision in selector_record.decisions
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        trace_name_prefix="transcript_synthesis.reviewer",
        trace_attributes={
            "canonical_span_count": len(spans),
            "span_candidate_count": len(span_candidates),
            "selector_decision_count": len(selector_record.decisions),
            "selector_unresolved_span_count": len(selector_record.unresolved_span_ids),
        },
    )
    corrected_decisions = _normalized_decisions(
        spans=spans,
        raw_decisions=response.get("corrected_decisions"),
        fallback_map={},
        allow_missing=True,
    )
    raw_issues = response.get("issues")
    issue_items = raw_issues if isinstance(raw_issues, list) else []
    issues = tuple(
        TranscriptReviewIssue.model_validate(item) for item in issue_items if isinstance(item, dict)
    )
    return TranscriptSynthesisReview.model_validate(
        {
            "review_id": _record_id(job.job_id, "reviewer", run_id),
            "job_id": job.job_id,
            "run_id": run_id,
            "reasoning_provider": runtime.reasoning_profile.provider_id,
            "reasoning_model_id": runtime.reasoning_profile.model_id,
            "accepted_span_ids": tuple(_string_items(response.get("accepted_span_ids"))),
            "corrected_decisions": corrected_decisions,
            "unresolved_span_ids": tuple(_string_items(response.get("unresolved_span_ids"))),
            "dropped_supported_span_ids": tuple(
                _string_items(response.get("dropped_supported_span_ids"))
            ),
            "issues": issues,
            "metadata": {"base_url_source": runtime.reasoning_profile.base_url_source},
        }
    )


def _live_global_decisions(
    *,
    run_id: str,
    runtime: WorkflowRuntime,
    spans: tuple[CanonicalTranscriptSpan, ...],
    span_candidates: tuple[TranscriptSpanCandidate, ...],
    selector_record: TranscriptSynthesisRecord,
    review: TranscriptSynthesisReview,
) -> tuple[TranscriptSpanDecision, ...]:
    response = _invoke_reasoning_json(
        runtime=runtime,
        run_id=run_id,
        schema_name="transcript_global_adjudicator",
        schema=_selector_output_schema(),
        system_prompt=(
            "You are the transcript global adjudicator. Resolve only the remaining risky or "
            "unresolved transcript spans. Prefer grounded resolutions, slight timing "
            "reconciliation, or bounded grounded merges. If evidence is still insufficient, "
            "mark the span unresolved."
        ),
        user_prompt=json.dumps(
            {
                "spans": [_span_payload(span) for span in spans],
                "span_candidates": [_candidate_payload(item) for item in span_candidates],
                "selector_decisions": [
                    decision.model_dump(mode="json") for decision in selector_record.decisions
                ],
                "review": {
                    "accepted_span_ids": list(review.accepted_span_ids),
                    "corrected_decisions": [
                        decision.model_dump(mode="json") for decision in review.corrected_decisions
                    ],
                    "unresolved_span_ids": list(review.unresolved_span_ids),
                    "issues": [issue.model_dump(mode="json") for issue in review.issues],
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        trace_name_prefix="transcript_synthesis.global_adjudicator",
        trace_attributes={
            "canonical_span_count": len(spans),
            "span_candidate_count": len(span_candidates),
            "selector_decision_count": len(selector_record.decisions),
            "selector_unresolved_span_count": len(selector_record.unresolved_span_ids),
            "review_corrected_decision_count": len(review.corrected_decisions),
            "review_unresolved_span_count": len(review.unresolved_span_ids),
        },
    )
    fallback_map = {}
    for span in spans:
        supported = _span_candidates_for(span.canonical_span_id, span_candidates)
        existing = next(
            (
                decision
                for decision in selector_record.decisions
                if decision.canonical_span_id == span.canonical_span_id
            ),
            None,
        )
        fallback_map[span.canonical_span_id] = _global_decision_for_span(
            span,
            supported,
            existing=existing,
        )
    return _normalized_decisions(
        spans=spans,
        raw_decisions=response.get("decisions"),
        fallback_map=fallback_map,
    )


def _invoke_reasoning_json(
    *,
    run_id: str,
    runtime: WorkflowRuntime,
    schema_name: str,
    schema: dict[str, object],
    system_prompt: str,
    user_prompt: str,
    trace_name_prefix: str,
    trace_attributes: dict[str, object],
) -> dict[str, object]:
    request_trace_attributes = {
        "schema_name": schema_name,
        "reasoning_provider": runtime.reasoning_profile.provider_id,
        "reasoning_model_id": runtime.reasoning_profile.model_id,
        **trace_attributes,
    }
    _record_reasoning_trace_event(
        runtime,
        run_id=run_id,
        event_name="transcript.reasoning.started",
        schema_name=schema_name,
        system_prompt_chars=len(system_prompt),
        user_prompt_chars=len(user_prompt),
    )
    try:
        with runtime.global_concurrency_limiter.acquire(
            runtime.parallelism.token_cost(ParallelTaskClass.PROVIDER_IO),
            task_class=ParallelTaskClass.PROVIDER_IO,
        ) as acquisition:
            request_trace_attributes = {
                **request_trace_attributes,
                **concurrency_trace_attributes(acquisition, effective_stage_workers=1),
            }
            runtime.trace_sink.record(
                TraceEvent(
                    run_id=run_id,
                    name=f"{trace_name_prefix}.started",
                    attributes=request_trace_attributes,
                )
            )
            response = _reasoning_client(runtime).chat.completions.create(
                model=runtime.reasoning_profile.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                timeout=60.0,
            )
    except (APIConnectionError, APITimeoutError, APIStatusError, ValueError) as exc:
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name=f"{trace_name_prefix}.failed",
                attributes={
                    **request_trace_attributes,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        )
        _record_reasoning_trace_event(
            runtime,
            run_id=run_id,
            event_name="transcript.reasoning.failed",
            schema_name=schema_name,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return {}

    payload = response.model_dump(mode="json")
    if not isinstance(payload, dict):
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name=f"{trace_name_prefix}.failed",
                attributes={
                    **request_trace_attributes,
                    "error": "response payload was not a JSON object",
                },
            )
        )
        _record_reasoning_trace_event(
            runtime,
            run_id=run_id,
            event_name="transcript.reasoning.parse_failed",
            schema_name=schema_name,
            failure_stage="payload_not_dict",
        )
        return {}
    try:
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            runtime.trace_sink.record(
                TraceEvent(
                    run_id=run_id,
                    name=f"{trace_name_prefix}.failed",
                    attributes={
                        **request_trace_attributes,
                        "error": "response content was not a string",
                    },
                )
            )
            _record_reasoning_trace_event(
                runtime,
                run_id=run_id,
                event_name="transcript.reasoning.parse_failed",
                schema_name=schema_name,
                failure_stage="content_not_string",
            )
            return {}
        parsed = json.loads(content)
    except Exception as exc:
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name=f"{trace_name_prefix}.failed",
                attributes={
                    **request_trace_attributes,
                    "error": "response content was not valid JSON",
                    "error_type": type(exc).__name__,
                },
            )
        )
        _record_reasoning_trace_event(
            runtime,
            run_id=run_id,
            event_name="transcript.reasoning.parse_failed",
            schema_name=schema_name,
            failure_stage="content_parse_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return {}
    if not isinstance(parsed, dict):
        runtime.trace_sink.record(
            TraceEvent(
                run_id=run_id,
                name=f"{trace_name_prefix}.failed",
                attributes={
                    **request_trace_attributes,
                    "error": "parsed response was not a JSON object",
                },
            )
        )
        _record_reasoning_trace_event(
            runtime,
            run_id=run_id,
            event_name="transcript.reasoning.parse_failed",
            schema_name=schema_name,
            failure_stage="parsed_not_dict",
        )
        return {}

    choice_count = payload.get("choices")
    response_choice_count = len(choice_count) if isinstance(choice_count, list) else 0
    runtime.trace_sink.record(
        TraceEvent(
            run_id=run_id,
            name=f"{trace_name_prefix}.completed",
            attributes={
                **request_trace_attributes,
                "content_chars": len(content),
                "response_choice_count": response_choice_count,
            },
        )
    )
    _record_reasoning_trace_event(
        runtime,
        run_id=run_id,
        event_name="transcript.reasoning.completed",
        schema_name=schema_name,
        content_chars=len(content),
        response_choice_count=response_choice_count,
    )
    return parsed


def _record_reasoning_trace_event(
    runtime: WorkflowRuntime,
    *,
    run_id: str,
    event_name: str,
    schema_name: str,
    **attributes: object,
) -> None:
    runtime.trace_sink.record(
        TraceEvent(
            run_id=run_id,
            name=event_name,
            attributes={
                "schema_name": schema_name,
                "provider_id": runtime.reasoning_profile.provider_id,
                "model_id": runtime.reasoning_profile.model_id,
                **attributes,
            },
        )
    )


def _normalized_decisions(
    *,
    spans: tuple[CanonicalTranscriptSpan, ...],
    raw_decisions: object,
    fallback_map: dict[str, TranscriptSpanDecision],
    allow_missing: bool = False,
) -> tuple[TranscriptSpanDecision, ...]:
    decision_map: dict[str, TranscriptSpanDecision] = {}
    if isinstance(raw_decisions, list):
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            try:
                decision = TranscriptSpanDecision.model_validate(item)
            except ValidationError:
                continue
            decision_map[decision.canonical_span_id] = decision

    normalized: list[TranscriptSpanDecision] = []
    for span in spans:
        decision = decision_map.get(span.canonical_span_id)
        if decision is not None:
            normalized.append(decision)
            continue
        fallback = fallback_map.get(span.canonical_span_id)
        if fallback is not None:
            normalized.append(fallback)
            continue
        if not allow_missing:
            normalized.append(
                TranscriptSpanDecision(
                    canonical_span_id=span.canonical_span_id,
                    decision_type="mark_unresolved",
                    rationale="Model output omitted a required span decision.",
                )
            )
    return tuple(normalized)


def _selector_output_schema() -> dict[str, object]:
    decision_schema = TranscriptSpanDecision.model_json_schema()
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": decision_schema,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _reviewer_output_schema() -> dict[str, object]:
    decision_schema = TranscriptSpanDecision.model_json_schema()
    issue_schema = TranscriptReviewIssue.model_json_schema()
    return {
        "type": "object",
        "properties": {
            "accepted_span_ids": {"type": "array", "items": {"type": "string"}},
            "corrected_decisions": {"type": "array", "items": decision_schema},
            "unresolved_span_ids": {"type": "array", "items": {"type": "string"}},
            "dropped_supported_span_ids": {"type": "array", "items": {"type": "string"}},
            "issues": {"type": "array", "items": issue_schema},
        },
        "required": [
            "accepted_span_ids",
            "corrected_decisions",
            "unresolved_span_ids",
            "dropped_supported_span_ids",
            "issues",
        ],
        "additionalProperties": False,
    }


def _span_payload(span: CanonicalTranscriptSpan) -> dict[str, object]:
    return span.model_dump(mode="json")


def _candidate_payload(candidate: TranscriptSpanCandidate) -> dict[str, object]:
    return candidate.model_dump(mode="json")


def _string_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())
