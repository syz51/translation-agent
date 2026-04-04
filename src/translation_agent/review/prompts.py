"""Structured reviewer prompt helpers and deterministic review synthesis."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from translation_agent.models import (
    CandidatePreference,
    JobContext,
    MemoryBundle,
    ReviewBundle,
    ReviewContext,
    ReviewIssue,
    Segment,
    StructuredEvidence,
    SuggestedFix,
    TranscriptCandidate,
    TranslationCandidate,
)

ReviewStage = Literal["transcript", "translation"]
ReviewCandidate = TranscriptCandidate | TranslationCandidate

REQUIRED_REVIEW_SECTIONS = (
    "Winner",
    "Confidence",
    "Why",
    "Key Errors By Candidate",
    "Quoted Evidence",
    "Suggested Fixes",
    "Escalate?",
)
PARSER_VERSION = "structured-review-v1"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_ENTITY_RE = re.compile(r"\b[\w.-]+\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True, slots=True)
class ReviewerRoleSpec:
    """Static prompt metadata for a reviewer role."""

    stage: ReviewStage
    reviewer_role: str
    focus: str
    policy_ref: str


@dataclass(frozen=True, slots=True)
class ReviewDraftIssue:
    candidate_id: str
    category: str
    severity: str
    description: str


@dataclass(frozen=True, slots=True)
class ReviewDraftFix:
    category: str
    candidate_id: str
    description: str


@dataclass(frozen=True, slots=True)
class ReviewDraftEvidence:
    candidate_id: str
    segment_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class StructuredReviewDraft:
    candidate_preferences: tuple[CandidatePreference, ...]
    confidence: float
    evidence: tuple[StructuredEvidence, ...]
    issues: tuple[ReviewIssue, ...]
    suggested_fixes: tuple[SuggestedFix, ...]
    escalation_signal: bool
    why_lines: tuple[str, ...]


TRANSCRIPT_REVIEWER_SPECS = (
    ReviewerRoleSpec(
        stage="transcript",
        reviewer_role="accuracy_reviewer",
        focus="Consensus, speaker fidelity, entities, numbers, and timestamp integrity.",
        policy_ref="policy/transcript-review/accuracy-v2",
    ),
    ReviewerRoleSpec(
        stage="transcript",
        reviewer_role="coherence_reviewer",
        focus="Coverage, normalization integrity, formatting, and segmentation coherence.",
        policy_ref="policy/transcript-review/coherence-v2",
    ),
)
TRANSLATION_REVIEWER_SPECS = (
    ReviewerRoleSpec(
        stage="translation",
        reviewer_role="faithfulness_reviewer",
        focus="Coverage, entity preservation, numeric fidelity, glossary compliance, and meaning.",
        policy_ref="policy/translation-review/faithfulness-v2",
    ),
    ReviewerRoleSpec(
        stage="translation",
        reviewer_role="style_reviewer",
        focus="Fluency, formatting, tone, and whether style changes remain source-grounded.",
        policy_ref="policy/translation-review/style-v2",
    ),
)


def reviewer_roles_for_stage(stage: ReviewStage) -> tuple[ReviewerRoleSpec, ...]:
    if stage == "transcript":
        return TRANSCRIPT_REVIEWER_SPECS
    return TRANSLATION_REVIEWER_SPECS


def build_review_context(
    *,
    run_id: str,
    stage: ReviewStage,
    reviewer_role: str,
    job: JobContext,
    candidate_ids: tuple[str, ...],
    memory_bundle: MemoryBundle,
) -> ReviewContext:
    spec = _role_spec(stage, reviewer_role)
    return ReviewContext(
        run_id=run_id,
        stage=stage,
        reviewer_role=reviewer_role,
        job=job,
        candidate_ids=candidate_ids,
        memory_bundle=scoped_memory_bundle(stage=stage, memory_bundle=memory_bundle),
        policy_ref=spec.policy_ref,
    )


def scoped_memory_bundle(
    *,
    stage: ReviewStage,
    memory_bundle: MemoryBundle,
) -> MemoryBundle:
    return MemoryBundle(
        semantic_memory=memory_bundle.semantic_memory[:4],
        glossary=memory_bundle.glossary[:2],
        rules=memory_bundle.rules[:2],
        episodic_memory=memory_bundle.episodic_memory[:1],
        procedural_memory=memory_bundle.procedural_memory[:2],
        provider_caveats=memory_bundle.provider_caveats if stage == "transcript" else (),
    )


def adjudication_memory_bundle(
    *,
    stage: ReviewStage,
    memory_bundle: MemoryBundle,
) -> MemoryBundle:
    return MemoryBundle(
        semantic_memory=memory_bundle.semantic_memory[:2],
        glossary=memory_bundle.glossary[:1],
        rules=memory_bundle.rules[:1],
        episodic_memory=memory_bundle.episodic_memory[:1],
        procedural_memory=memory_bundle.procedural_memory[:1],
        provider_caveats=memory_bundle.provider_caveats if stage == "transcript" else (),
    )


def build_review_prompt(
    context: ReviewContext,
    *,
    candidate_refs: tuple[str, ...],
    raw_payload_refs: tuple[str, ...],
    final_transcript_ref: str | None = None,
    transcript_context: dict[str, object] | None = None,
) -> str:
    spec = _role_spec(context.stage, context.reviewer_role)
    memory_notes = [
        f"glossary={len(context.memory_bundle.glossary)}",
        f"rules={len(context.memory_bundle.rules)}",
        f"semantic={len(context.memory_bundle.semantic_memory)}",
        f"episodic={len(context.memory_bundle.episodic_memory)}",
        f"procedural={len(context.memory_bundle.procedural_memory)}",
    ]
    lines = [
        f"Stage: {context.stage}",
        f"Role: {context.reviewer_role}",
        f"Focus: {spec.focus}",
        f"Job ID: {context.job.job_id}",
        f"Candidate Refs: {', '.join(candidate_refs)}",
        ("Raw Payload Refs: " + (", ".join(raw_payload_refs) if raw_payload_refs else "none")),
        f"Memory Slice: {', '.join(memory_notes)}",
        "Emit structured evidence anchored to canonical source spans.",
    ]
    if final_transcript_ref is not None:
        lines.append(f"Final Transcript Ref: {final_transcript_ref}")
    if transcript_context:
        blockers = transcript_context.get("transcript_blockers")
        if isinstance(blockers, list) and blockers:
            blocker_summary = ", ".join(str(item) for item in blockers)
            lines.append(f"Transcript synthesis blockers: {blocker_summary}")
        synthesis_status = transcript_context.get("transcript_synthesis_status")
        if isinstance(synthesis_status, str) and synthesis_status:
            lines.append(f"Transcript Synthesis Status: {synthesis_status}")
        decision_ref = transcript_context.get("transcript_decision_ref")
        if isinstance(decision_ref, str) and decision_ref:
            lines.append(f"Transcript Decision Ref: {decision_ref}")
        investigation_ref = transcript_context.get("transcript_investigation_ref")
        if isinstance(investigation_ref, str) and investigation_ref:
            lines.append(f"Transcript Investigation Ref: {investigation_ref}")
    anti_patterns = [
        entry.content for entry in context.memory_bundle.procedural_memory if entry.content.strip()
    ]
    if anti_patterns:
        lines.append("Recurring anti-patterns to check:")
        lines.extend(f"- {entry}" for entry in anti_patterns[:3])
    return "\n".join(lines)


def build_structured_review(
    context: ReviewContext,
    *,
    candidates: Sequence[ReviewCandidate],
    final_transcript: TranscriptCandidate | None = None,
) -> StructuredReviewDraft:
    if context.stage == "transcript":
        transcript_candidates = tuple(
            candidate for candidate in candidates if isinstance(candidate, TranscriptCandidate)
        )
        return _build_transcript_review(context, transcript_candidates)
    translation_candidates = tuple(
        candidate for candidate in candidates if isinstance(candidate, TranslationCandidate)
    )
    if final_transcript is None:
        raise ValueError("translation review requires the approved transcript")
    return _build_translation_review(context, translation_candidates, final_transcript)


def render_reviewer_output(
    context: ReviewContext,
    *,
    candidates: Sequence[ReviewCandidate],
    prompt_text: str,
    final_transcript: TranscriptCandidate | None = None,
) -> str:
    draft = build_structured_review(
        context,
        candidates=candidates,
        final_transcript=final_transcript,
    )
    winner = draft.candidate_preferences[0].candidate_id if draft.candidate_preferences else "none"
    issue_lines = [
        (
            f"{issue.candidate_id or 'general'} | {issue.dimension} | "
            f"{issue.severity} | {issue.description}"
        )
        for issue in draft.issues
    ] or ["none | meaning | minor | No material issues."]
    evidence_lines = [
        (
            f"{item.candidate_id} | {item.source_span_id or 'none'} | "
            f"{item.dimension}:{item.polarity}:{item.normalized_value or 'null'} | "
            f"{item.evidence_text}"
        )
        for item in draft.evidence
    ] or ["none | none | meaning:supports:null | No evidence."]
    fix_lines = [
        f"{fix.issue_category} | {fix.candidate_id or 'general'} | {fix.description}"
        for fix in draft.suggested_fixes
    ] or ["general | none | Keep the selected candidate."]
    why_lines = draft.why_lines + (f"Prompt contract preserved: {prompt_text.splitlines()[-1]}",)
    return "\n".join(
        [
            f"Winner: {winner}",
            f"Confidence: {draft.confidence:.2f}",
            "Why:",
            *[f"- {line}" for line in why_lines],
            "Key Errors By Candidate:",
            *[f"- {line}" for line in issue_lines],
            "Quoted Evidence:",
            *[f"- {line}" for line in evidence_lines],
            "Suggested Fixes:",
            *[f"- {line}" for line in fix_lines],
            f"Escalate?: {'yes' if draft.escalation_signal else 'no'}",
        ]
    )


def review_bundle_from_draft(
    *,
    review_id: str,
    job_id: str,
    stage: ReviewStage,
    reviewer_role: str,
    raw_review_text: str,
    draft: StructuredReviewDraft,
) -> ReviewBundle:
    return ReviewBundle(
        review_id=review_id,
        job_id=job_id,
        stage=stage,
        reviewer_role=reviewer_role,
        candidate_preferences=draft.candidate_preferences,
        confidence=draft.confidence,
        raw_review_text=raw_review_text,
        structured_evidence=draft.evidence,
        review_issues=draft.issues,
        issue_categories=tuple(dict.fromkeys(issue.dimension for issue in draft.issues)),
        suggested_fixes=draft.suggested_fixes,
        escalation_signal=draft.escalation_signal,
        parser_version=PARSER_VERSION,
        output_version=PARSER_VERSION,
    )


def _build_transcript_review(
    context: ReviewContext,
    candidates: tuple[TranscriptCandidate, ...],
) -> StructuredReviewDraft:
    if not candidates:
        raise ValueError("transcript review requires transcript candidates")
    span_map = _canonicalize_transcript_spans(candidates)
    consensus_by_span = _consensus_text_by_span(span_map)
    speaker_by_span = _consensus_speaker_by_span(span_map)
    candidate_scores: dict[str, float] = {}
    evidence: list[StructuredEvidence] = []
    issues: list[ReviewIssue] = []
    fixes: list[SuggestedFix] = []
    for candidate in candidates:
        metrics = _transcript_candidate_metrics(
            candidate, span_map, consensus_by_span, speaker_by_span
        )
        candidate_scores[candidate.candidate_id] = metrics["score"]
        for span_id, span_segments in span_map.items():
            candidate_segment = _candidate_transcript_segment_for_span(candidate, span_segments)
            if candidate_segment is None:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="coverage",
                        polarity="refutes",
                        normalized_value="omitted",
                        severity="major",
                        evidence_text="Candidate omits a canonical transcript span.",
                    )
                )
                issues.append(
                    ReviewIssue(
                        candidate_id=candidate.candidate_id,
                        dimension="coverage",
                        severity="major",
                        description="Missing a transcript span that survived normalization.",
                        source_span_id=span_id,
                    )
                )
                fixes.append(
                    SuggestedFix(
                        issue_category="coverage",
                        candidate_id=candidate.candidate_id,
                        description="Restore the missing transcript span before publication.",
                    )
                )
                continue
            consensus_text = consensus_by_span[span_id]
            similarity = _transcript_meaning_score(_segment_text(candidate_segment), consensus_text)
            if similarity < 0.55:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="meaning",
                        polarity="refutes",
                        normalized_value=_segment_text(candidate_segment),
                        severity="critical",
                        evidence_text="Segment meaning diverges from the transcript consensus.",
                    )
                )
                issues.append(
                    ReviewIssue(
                        candidate_id=candidate.candidate_id,
                        dimension="meaning",
                        severity="critical",
                        description="Segment meaning conflicts with the consensus transcript span.",
                        source_span_id=span_id,
                    )
                )
            else:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="coverage",
                        polarity="supports",
                        normalized_value="preserved",
                        severity="minor",
                        evidence_text="Candidate preserves the canonical transcript span.",
                    )
                )
            if _extract_entities(_segment_text(candidate_segment)) != _extract_entities(
                consensus_text
            ):
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="entity",
                        polarity="refutes",
                        normalized_value="|".join(
                            sorted(_extract_entities(_segment_text(candidate_segment)))
                        )
                        or None,
                        severity="major",
                        evidence_text="Named entities differ from the span consensus.",
                    )
                )
            if _extract_numbers(_segment_text(candidate_segment)) != _extract_numbers(
                consensus_text
            ):
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="number_date_unit",
                        polarity="refutes",
                        normalized_value="|".join(
                            sorted(_extract_numbers(_segment_text(candidate_segment)))
                        )
                        or None,
                        severity="major",
                        evidence_text="Numbers or units differ from the span consensus.",
                    )
                )
            expected_speaker = speaker_by_span.get(span_id)
            if (
                expected_speaker
                and candidate_segment.speaker
                and candidate_segment.speaker != expected_speaker
            ):
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="formatting",
                        polarity="refutes",
                        normalized_value=candidate_segment.speaker,
                        severity="minor",
                        evidence_text="Speaker labeling differs from the span majority.",
                    )
                )

    ranked = _rank_candidates(candidate_scores)
    winner = ranked[0][0]
    escalation_signal = any(
        item.dimension in {"meaning", "coverage", "entity", "number_date_unit"}
        and item.severity in {"major", "critical"}
        and item.polarity == "refutes"
        for item in evidence
    )
    return StructuredReviewDraft(
        candidate_preferences=_candidate_preferences(ranked),
        confidence=round(max(0.35, min(candidate_scores[winner], 0.98)), 2),
        evidence=_filter_evidence_for_role(context.reviewer_role, tuple(evidence)),
        issues=_filter_issues_for_role(context.reviewer_role, tuple(issues)),
        suggested_fixes=tuple(dict.fromkeys(fixes)),
        escalation_signal=escalation_signal,
        why_lines=(
            f"{context.reviewer_role} ranked candidates from canonical span evidence.",
            "Transcript review used provider consensus, coverage, timestamps, "
            "and speaker consistency.",
        ),
    )


def _build_translation_review(
    context: ReviewContext,
    candidates: tuple[TranslationCandidate, ...],
    final_transcript: TranscriptCandidate,
) -> StructuredReviewDraft:
    if not candidates:
        raise ValueError("translation review requires translation candidates")
    scenario = next(
        (
            scenario_value
            for candidate in candidates
            if isinstance((scenario_value := candidate.metadata.get("scenario")), str)
        ),
        None,
    )
    if scenario in {
        "translation_conflict",
        "translation_conflict_timeout",
        "translation_high_risk",
        "translation_escalation",
        "translation_human_review",
    }:
        return _scenario_translation_review(context, candidates, scenario)
    source_spans = _source_span_segments(final_transcript)
    glossary_pairs = _glossary_pairs(context.memory_bundle)
    candidate_scores: dict[str, float] = {}
    evidence: list[StructuredEvidence] = []
    issues: list[ReviewIssue] = []
    fixes: list[SuggestedFix] = []
    peer_text_by_span = _translation_peer_text_by_span(candidates)
    for candidate in candidates:
        coverage_scores: list[float] = []
        entity_scores: list[float] = []
        numeric_scores: list[float] = []
        glossary_scores: list[float] = []
        addition_penalties: list[float] = []
        fluency_scores: list[float] = []
        candidate_segments = _candidate_segments_by_span(candidate, source_spans)
        for span_id, source_segment in source_spans.items():
            translation_segment = candidate_segments.get(span_id)
            if translation_segment is None or not _segment_target_text(translation_segment):
                coverage_scores.append(0.0)
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="coverage",
                        polarity="refutes",
                        normalized_value="omitted",
                        severity="major",
                        evidence_text="Translation candidate omits a source span.",
                    )
                )
                issues.append(
                    ReviewIssue(
                        candidate_id=candidate.candidate_id,
                        dimension="coverage",
                        severity="major",
                        description="Source span is missing in the translation candidate.",
                        source_span_id=span_id,
                    )
                )
                fixes.append(
                    SuggestedFix(
                        issue_category="coverage",
                        candidate_id=candidate.candidate_id,
                        description="Translate the missing source span.",
                    )
                )
                continue
            target_text = _segment_target_text(translation_segment)
            source_text = _segment_text(source_segment)
            coverage_scores.append(1.0)
            evidence.append(
                StructuredEvidence(
                    source_span_id=span_id,
                    candidate_id=candidate.candidate_id,
                    dimension="coverage",
                    polarity="supports",
                    normalized_value="preserved",
                    severity="minor",
                    evidence_text="Candidate preserves the source span in the translation.",
                )
            )
            source_entities = _extract_entities(source_text)
            target_entities = _extract_entities(target_text)
            entity_score = _preservation_score(source_entities, target_entities)
            entity_scores.append(entity_score)
            if source_entities and entity_score < 1.0:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="entity",
                        polarity="refutes",
                        normalized_value="|".join(sorted(target_entities)) or None,
                        severity="major",
                        evidence_text="Named entities are not preserved across the span.",
                    )
                )
            elif source_entities:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="entity",
                        polarity="supports",
                        normalized_value="|".join(sorted(target_entities)) or None,
                        severity="minor",
                        evidence_text="Named entities are preserved across the span.",
                    )
                )
            source_numbers = _extract_numbers(source_text)
            target_numbers = _extract_numbers(target_text)
            numeric_score = _preservation_score(source_numbers, target_numbers)
            numeric_scores.append(numeric_score)
            if source_numbers and numeric_score < 1.0:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="number_date_unit",
                        polarity="refutes",
                        normalized_value="|".join(sorted(target_numbers)) or None,
                        severity="critical",
                        evidence_text="Numbers, dates, or units changed across the span.",
                    )
                )
            glossary_score = _glossary_score(source_text, target_text, glossary_pairs)
            glossary_scores.append(glossary_score)
            if glossary_score < 1.0 and glossary_pairs:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="terminology",
                        polarity="refutes",
                        normalized_value=_first_glossary_target(source_text, glossary_pairs),
                        severity="major",
                        evidence_text=(
                            "Glossary term usage diverges from the preferred target term."
                        ),
                    )
                )
            addition_penalty = _addition_penalty(
                source_text,
                target_text,
                tuple(text for text in peer_text_by_span.get(span_id, ()) if text != target_text),
            )
            addition_penalties.append(addition_penalty)
            if addition_penalty >= 0.2:
                evidence.append(
                    StructuredEvidence(
                        source_span_id=span_id,
                        candidate_id=candidate.candidate_id,
                        dimension="meaning",
                        polarity="refutes",
                        normalized_value="added_content",
                        severity="major",
                        evidence_text=(
                            "Candidate adds ungrounded content relative to the source span."
                        ),
                    )
                )
            fluency_scores.append(_fluency_score(target_text))

        segment_coverage = mean(coverage_scores) if coverage_scores else 0.0
        entity_consistency = mean(entity_scores) if entity_scores else 1.0
        numeric_consistency = mean(numeric_scores) if numeric_scores else 1.0
        glossary_compliance = mean(glossary_scores) if glossary_scores else 1.0
        addition_risk = mean(addition_penalties) if addition_penalties else 0.0
        omission_risk = 1.0 - segment_coverage
        fluency = mean(fluency_scores) if fluency_scores else 0.5
        faithfulness = max(
            0.0,
            min(
                1.0,
                0.45 * segment_coverage
                + 0.20 * entity_consistency
                + 0.20 * numeric_consistency
                + 0.15 * glossary_compliance
                - 0.20 * addition_risk,
            ),
        )
        role_bonus = 0.1 * fluency if context.reviewer_role == "style_reviewer" else 0.0
        candidate_scores[candidate.candidate_id] = round(
            max(0.0, min(faithfulness + role_bonus - omission_risk * 0.05, 0.99)), 4
        )

    ranked = _rank_candidates(candidate_scores)
    winner = ranked[0][0]
    escalation_signal = any(
        item.polarity == "refutes"
        and item.severity in {"major", "critical"}
        and item.dimension in {"meaning", "coverage", "entity", "number_date_unit", "terminology"}
        for item in evidence
    )
    return StructuredReviewDraft(
        candidate_preferences=_candidate_preferences(ranked),
        confidence=round(max(0.35, min(candidate_scores[winner], 0.98)), 2),
        evidence=_filter_evidence_for_role(context.reviewer_role, tuple(evidence)),
        issues=_filter_issues_for_role(context.reviewer_role, tuple(issues)),
        suggested_fixes=tuple(dict.fromkeys(fixes)),
        escalation_signal=escalation_signal,
        why_lines=(
            f"{context.reviewer_role} ranked translations from source-span evidence.",
            "Translation review used coverage, entities, numbers, glossary "
            "terms, and addition risk.",
        ),
    )


def _scenario_translation_review(
    context: ReviewContext,
    candidates: tuple[TranslationCandidate, ...],
    scenario: str,
) -> StructuredReviewDraft:
    if len(candidates) == 1:
        candidate = candidates[0]
        source_span_id = (
            _segment_source_span_id(candidate.segments[0]) if candidate.segments else "span:0:1250"
        )
        return StructuredReviewDraft(
            candidate_preferences=_candidate_preferences([(candidate.candidate_id, 0.82)]),
            confidence=0.82,
            evidence=(
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate.candidate_id,
                    dimension="coverage",
                    polarity="supports",
                    normalized_value="single_candidate",
                    severity="minor",
                    evidence_text=(
                        "Only one translation candidate survived, so review stays on "
                        "source-grounded single-candidate checks."
                    ),
                ),
            ),
            issues=(),
            suggested_fixes=(),
            escalation_signal=False,
            why_lines=(
                f"{context.reviewer_role} reviewed the only surviving translation candidate.",
                "No synthetic variant comparison was introduced for this scenario.",
            ),
        )
    candidate_by_variant = {candidate.prompt_variant_id: candidate for candidate in candidates}
    candidate_a = candidate_by_variant.get("variant-a", candidates[0])
    candidate_b = candidate_by_variant.get("variant-b", candidates[-1])
    source_span_id = (
        _segment_source_span_id(candidate_a.segments[0]) if candidate_a.segments else "span:0:1250"
    )
    if scenario in {"translation_conflict", "translation_high_risk"}:
        if context.reviewer_role == "faithfulness_reviewer":
            ranked = [
                (candidate_a.candidate_id, 0.86),
                (candidate_b.candidate_id, 0.58),
            ]
            evidence = (
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_a.candidate_id,
                    dimension="coverage",
                    polarity="supports",
                    normalized_value="preserved",
                    severity="minor",
                    evidence_text="Candidate preserves the source span with stable coverage.",
                ),
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_b.candidate_id,
                    dimension="coverage",
                    polarity="supports",
                    normalized_value="preserved",
                    severity="minor",
                    evidence_text="Candidate also preserves the source span coverage.",
                ),
            )
            issues = ()
        else:
            ranked = [
                (candidate_b.candidate_id, 0.46),
                (candidate_a.candidate_id, 0.43),
            ]
            evidence = (
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_b.candidate_id,
                    dimension="style",
                    polarity="supports",
                    normalized_value="fluent",
                    severity="minor",
                    evidence_text="Candidate reads more naturally for the target locale.",
                ),
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_a.candidate_id,
                    dimension="style",
                    polarity="supports",
                    normalized_value="literal",
                    severity="minor",
                    evidence_text="Candidate stays more literal and less idiomatic.",
                ),
            )
            issues = ()
        return StructuredReviewDraft(
            candidate_preferences=_candidate_preferences(ranked),
            confidence=ranked[0][1],
            evidence=evidence,
            issues=issues,
            suggested_fixes=(),
            escalation_signal=True,
            why_lines=(
                f"{context.reviewer_role} used the deterministic conflict fixture.",
                "The fake runtime drives escalation from structured preference divergence.",
            ),
        )
    if scenario == "translation_escalation":
        if context.reviewer_role == "faithfulness_reviewer":
            ranked = [
                (candidate_a.candidate_id, 0.88),
                (candidate_b.candidate_id, 0.62),
            ]
            evidence = (
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_a.candidate_id,
                    dimension="coverage",
                    polarity="supports",
                    normalized_value="preserved",
                    severity="minor",
                    evidence_text=(
                        "Candidate preserves the source span without unsupported additions."
                    ),
                ),
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_b.candidate_id,
                    dimension="meaning",
                    polarity="refutes",
                    normalized_value="added_content",
                    severity="major",
                    evidence_text="Candidate adds unsupported content for the source span.",
                ),
            )
            issues = (
                ReviewIssue(
                    candidate_id=candidate_b.candidate_id,
                    dimension="meaning",
                    severity="major",
                    description="Adds meaning that is not anchored to the source span.",
                    source_span_id=source_span_id,
                ),
            )
        else:
            ranked = [
                (candidate_b.candidate_id, 0.82),
                (candidate_a.candidate_id, 0.76),
            ]
            evidence = (
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_b.candidate_id,
                    dimension="style",
                    polarity="supports",
                    normalized_value="fluent",
                    severity="minor",
                    evidence_text="Candidate reads more naturally for the target locale.",
                ),
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_a.candidate_id,
                    dimension="style",
                    polarity="refutes",
                    normalized_value="stiff",
                    severity="major",
                    evidence_text=(
                        "Candidate style is materially less natural for the target locale."
                    ),
                ),
            )
            issues = (
                ReviewIssue(
                    candidate_id=candidate_a.candidate_id,
                    dimension="style",
                    severity="major",
                    description="Style is noticeably less natural for the target audience.",
                    source_span_id=source_span_id,
                ),
            )
        return StructuredReviewDraft(
            candidate_preferences=_candidate_preferences(ranked),
            confidence=ranked[0][1],
            evidence=evidence,
            issues=issues,
            suggested_fixes=(
                SuggestedFix(
                    issue_category="meaning",
                    candidate_id=candidate_b.candidate_id,
                    description="Remove unsupported additions while preserving the source meaning.",
                ),
            ),
            escalation_signal=True,
            why_lines=(
                f"{context.reviewer_role} used the deterministic scenario fixture for escalation.",
                "Scenario fixtures drive structured evidence in the fake "
                "runtime without lexical parsing.",
            ),
        )
    if scenario == "translation_conflict_timeout":
        ranked = (
            [
                (candidate_a.candidate_id, 0.82),
                (candidate_b.candidate_id, 0.71),
            ]
            if context.reviewer_role == "faithfulness_reviewer"
            else [
                (candidate_a.candidate_id, 0.8),
                (candidate_b.candidate_id, 0.74),
            ]
        )
        return StructuredReviewDraft(
            candidate_preferences=_candidate_preferences(ranked),
            confidence=0.82,
            evidence=(
                StructuredEvidence(
                    source_span_id=source_span_id,
                    candidate_id=candidate_b.candidate_id,
                    dimension="meaning",
                    polarity="refutes",
                    normalized_value="added_content",
                    severity="major",
                    evidence_text="Candidate introduces a competing meaning for the span.",
                ),
            ),
            issues=(),
            suggested_fixes=(),
            escalation_signal=True,
            why_lines=(
                f"{context.reviewer_role} used the deterministic timeout fixture.",
                "The fake runtime emits a medium-disagreement scenario for timeout handling.",
            ),
        )
    return StructuredReviewDraft(
        candidate_preferences=_candidate_preferences(
            [
                (candidate_a.candidate_id, 0.55),
                (candidate_b.candidate_id, 0.54),
            ]
        ),
        confidence=0.55,
        evidence=(
            StructuredEvidence(
                source_span_id=source_span_id,
                candidate_id=candidate_a.candidate_id,
                dimension="meaning",
                polarity="refutes",
                normalized_value="conflict",
                severity="critical",
                evidence_text="Candidate meaning remains unsafe for automatic publication.",
            ),
            StructuredEvidence(
                source_span_id=source_span_id,
                candidate_id=candidate_b.candidate_id,
                dimension="meaning",
                polarity="refutes",
                normalized_value="conflict",
                severity="critical",
                evidence_text="Candidate meaning remains unsafe for automatic publication.",
            ),
        ),
        issues=(
            ReviewIssue(
                candidate_id=None,
                dimension="meaning",
                severity="critical",
                description="No candidate is safe enough for automatic publication.",
                source_span_id=source_span_id,
            ),
        ),
        suggested_fixes=(),
        escalation_signal=True,
        why_lines=(
            f"{context.reviewer_role} used the deterministic human-review fixture.",
            "The fake runtime emits a safety-boundary scenario for human escalation.",
        ),
    )


def _transcript_candidate_metrics(
    candidate: TranscriptCandidate,
    span_map: dict[str, list[tuple[str, Segment]]],
    consensus_by_span: dict[str, str],
    speaker_by_span: dict[str, str],
) -> dict[str, float]:
    coverage: list[float] = []
    meaning: list[float] = []
    timestamps: list[float] = []
    speakers: list[float] = []
    for span_id, entries in span_map.items():
        candidate_segment = _candidate_transcript_segment_for_span(candidate, entries)
        if candidate_segment is None:
            coverage.append(0.0)
            continue
        coverage.append(1.0)
        meaning.append(
            _transcript_meaning_score(
                _segment_text(candidate_segment),
                consensus_by_span[span_id],
            )
        )
        timestamps.append(1.0 if candidate_segment.end_ms >= candidate_segment.start_ms else 0.0)
        expected_speaker = speaker_by_span.get(span_id)
        speakers.append(
            1.0
            if not expected_speaker or expected_speaker == (candidate_segment.speaker or "")
            else 0.0
        )
    return {
        "score": round(
            0.40 * (mean(coverage) if coverage else 0.0)
            + 0.35 * (mean(meaning) if meaning else 0.0)
            + 0.15 * (mean(timestamps) if timestamps else 1.0)
            + 0.10 * (mean(speakers) if speakers else 1.0),
            4,
        )
    }


def _canonicalize_transcript_spans(
    candidates: tuple[TranscriptCandidate, ...],
) -> dict[str, list[tuple[str, Segment]]]:
    grouped: dict[str, list[tuple[str, Segment]]] = {}
    span_counter = 0
    for candidate in candidates:
        for segment in candidate.segments:
            matched_key = None
            for span_id, entries in grouped.items():
                if any(
                    _segments_overlap(segment, other_segment) >= 0.5 for _, other_segment in entries
                ):
                    matched_key = span_id
                    break
            if matched_key is None:
                matched_key = _segment_source_span_id(segment) or f"source-span-{span_counter}"
                span_counter += 1
                grouped[matched_key] = []
            grouped[matched_key].append((candidate.candidate_id, segment))
    return grouped


def _candidate_transcript_segment_for_span(
    candidate: TranscriptCandidate,
    span_segments: list[tuple[str, Segment]],
) -> Segment | None:
    exact = next(
        (segment for owner, segment in span_segments if owner == candidate.candidate_id),
        None,
    )
    if exact is not None:
        return exact
    reference_segment = max(
        (segment for _, segment in span_segments),
        key=lambda segment: max(segment.end_ms - segment.start_ms, 0),
    )
    best_segment: Segment | None = None
    best_score = 0.0
    for segment in candidate.segments:
        score = _source_span_coverage_ratio(segment, reference_segment)
        if score > best_score:
            best_segment = segment
            best_score = score
    if best_segment is not None and best_score >= 0.5:
        return best_segment
    return None


def _source_span_segments(final_transcript: TranscriptCandidate) -> dict[str, Segment]:
    return {
        _segment_source_span_id(segment) or segment.segment_id: segment
        for segment in final_transcript.segments
    }


def _candidate_segments_by_span(
    candidate: TranslationCandidate,
    source_spans: dict[str, Segment],
) -> dict[str, Segment]:
    mapped: dict[str, Segment] = {}
    for segment in candidate.segments:
        span_id = _segment_source_span_id(segment)
        if span_id in source_spans:
            mapped[span_id] = segment
            continue
        best_match = None
        best_overlap = 0.0
        for source_span_id, source_segment in source_spans.items():
            overlap = _segments_overlap(segment, source_segment)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = source_span_id
        if best_match is not None and best_overlap >= 0.5:
            mapped[best_match] = segment
    return mapped


def _translation_peer_text_by_span(
    candidates: tuple[TranslationCandidate, ...],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        for segment in candidate.segments:
            span_id = _segment_source_span_id(segment)
            if span_id is None:
                continue
            target_text = _segment_target_text(segment)
            if target_text:
                values[span_id].append(target_text)
    return {key: tuple(items) for key, items in values.items()}


def _source_span_coverage_ratio(candidate_segment: Segment, source_segment: Segment) -> float:
    if (
        candidate_segment.end_ms == candidate_segment.start_ms
        and source_segment.end_ms == source_segment.start_ms
    ):
        return (
            1.0
            if _normalize_text(_segment_text(candidate_segment))
            == _normalize_text(_segment_text(source_segment))
            else 0.0
        )
    source_duration = max(source_segment.end_ms - source_segment.start_ms, 0)
    if source_duration <= 0:
        return 0.0
    overlap_start = max(candidate_segment.start_ms, source_segment.start_ms)
    overlap_end = min(candidate_segment.end_ms, source_segment.end_ms)
    overlap = max(overlap_end - overlap_start, 0)
    return overlap / source_duration


def _consensus_text_by_span(
    span_map: dict[str, list[tuple[str, Segment]]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for span_id, entries in span_map.items():
        texts = [_segment_text(segment) for _, segment in entries if _segment_text(segment)]
        counts = Counter(_normalize_text(text) for text in texts)
        normalized = counts.most_common(1)[0][0] if counts else ""
        result[span_id] = next((text for text in texts if _normalize_text(text) == normalized), "")
    return result


def _consensus_speaker_by_span(
    span_map: dict[str, list[tuple[str, Segment]]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for span_id, entries in span_map.items():
        speakers = [segment.speaker or "" for _, segment in entries if segment.speaker]
        if not speakers:
            continue
        result[span_id] = Counter(speakers).most_common(1)[0][0]
    return result


def _transcript_meaning_score(candidate_text: str, consensus_text: str) -> float:
    similarity = _text_similarity(candidate_text, consensus_text)
    candidate_tokens = set(_tokens(candidate_text))
    consensus_tokens = set(_tokens(consensus_text))
    if candidate_tokens and consensus_tokens and consensus_tokens <= candidate_tokens:
        return max(similarity, 1.0)
    return similarity


def _filter_evidence_for_role(
    reviewer_role: str,
    evidence: tuple[StructuredEvidence, ...],
) -> tuple[StructuredEvidence, ...]:
    if reviewer_role in {"accuracy_reviewer", "faithfulness_reviewer"}:
        allowed = {"meaning", "entity", "number_date_unit", "terminology", "coverage"}
    else:
        allowed = {"coverage", "formatting", "style", "meaning"}
    return tuple(item for item in evidence if item.dimension in allowed)


def _filter_issues_for_role(
    reviewer_role: str,
    issues: tuple[ReviewIssue, ...],
) -> tuple[ReviewIssue, ...]:
    if reviewer_role in {"accuracy_reviewer", "faithfulness_reviewer"}:
        allowed = {"meaning", "entity", "number_date_unit", "terminology", "coverage"}
    else:
        allowed = {"coverage", "formatting", "style", "meaning"}
    return tuple(issue for issue in issues if issue.dimension in allowed)


def _candidate_preferences(ranked: list[tuple[str, float]]) -> tuple[CandidatePreference, ...]:
    return tuple(
        CandidatePreference(
            candidate_id=candidate_id, rank=index + 1, rationale=f"score={score:.4f}"
        )
        for index, (candidate_id, score) in enumerate(ranked)
    )


def _rank_candidates(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _role_spec(stage: ReviewStage, reviewer_role: str) -> ReviewerRoleSpec:
    for spec in reviewer_roles_for_stage(stage):
        if spec.reviewer_role == reviewer_role:
            return spec
    raise ValueError(f"unknown reviewer role {reviewer_role!r} for stage {stage!r}")


def _segment_text(segment: Segment) -> str:
    return segment.source_text or segment.target_text or ""


def _segment_target_text(segment: Segment) -> str:
    return segment.target_text or ""


def _segment_source_span_id(segment: Segment) -> str | None:
    source_span_id = segment.annotations.get("source_span_id")
    if isinstance(source_span_id, str) and source_span_id.strip():
        return source_span_id
    return None


def _segments_overlap(left: Segment, right: Segment) -> float:
    if left.end_ms == left.start_ms and right.end_ms == right.start_ms:
        return (
            1.0
            if _normalize_text(_segment_text(left)) == _normalize_text(_segment_text(right))
            else 0.0
        )
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    if union <= 0:
        return 0.0
    return intersection / union


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return [match.group(0) for match in _TOKEN_RE.finditer(normalized)]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _extract_entities(value: str) -> set[str]:
    entities: set[str] = set()
    for match in _ENTITY_RE.finditer(value):
        token = match.group(0)
        if len(token) <= 1:
            continue
        has_internal_upper = any(character.isupper() for character in token[1:])
        is_acronym = token.isupper()
        has_digits = any(character.isdigit() for character in token)
        if has_internal_upper or is_acronym or has_digits:
            entities.add(token)
    return entities


def _extract_numbers(value: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(value)}


def _preservation_score(source_values: set[str], target_values: set[str]) -> float:
    if not source_values:
        return 1.0
    return len(source_values & target_values) / max(len(source_values), 1)


def _glossary_pairs(memory_bundle: MemoryBundle) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for entry in memory_bundle.glossary:
        content = entry.content
        if "->" not in content:
            continue
        source_term, target_term = (part.strip() for part in content.split("->", 1))
        if source_term and target_term:
            pairs.append((source_term, target_term))
    return tuple(pairs)


def _glossary_score(
    source_text: str, target_text: str, glossary_pairs: tuple[tuple[str, str], ...]
) -> float:
    matched = 0
    total = 0
    source_fold = _normalize_text(source_text)
    target_fold = _normalize_text(target_text)
    for source_term, target_term in glossary_pairs:
        if _normalize_text(source_term) not in source_fold:
            continue
        total += 1
        if _normalize_text(target_term) in target_fold:
            matched += 1
    if total == 0:
        return 1.0
    return matched / total


def _first_glossary_target(
    source_text: str,
    glossary_pairs: tuple[tuple[str, str], ...],
) -> str | None:
    source_fold = _normalize_text(source_text)
    for source_term, target_term in glossary_pairs:
        if _normalize_text(source_term) in source_fold:
            return target_term
    return None


def _addition_penalty(source_text: str, target_text: str, peer_texts: tuple[str, ...]) -> float:
    source_length = len(_tokens(source_text))
    target_length = len(_tokens(target_text))
    ratio_penalty = 0.0
    if source_length:
        ratio_penalty = max(0.0, min((target_length / source_length) - 1.8, 1.0))
    peer_lengths = [len(_tokens(text)) for text in peer_texts if text.strip()]
    peer_median = mean(peer_lengths) if peer_lengths else target_length
    peer_penalty = 0.0
    if peer_median:
        peer_penalty = max(0.0, min((target_length / peer_median) - 1.5, 1.0))
    peer_outlier_penalty = 0.0
    if peer_lengths:
        peer_outlier_penalty = max(0.0, min((target_length - min(peer_lengths) - 2) / 4.0, 1.0))
    return round(max(ratio_penalty, peer_penalty, peer_outlier_penalty), 4)


def _fluency_score(target_text: str) -> float:
    tokens = _tokens(target_text)
    if not tokens:
        return 0.0
    repeated_penalty = 0.1 if len(tokens) != len(set(tokens)) else 0.0
    punctuation_bonus = 0.05 if target_text.rstrip().endswith((".", "!", "?")) else 0.0
    return max(0.0, min(0.75 + punctuation_bonus - repeated_penalty, 1.0))
