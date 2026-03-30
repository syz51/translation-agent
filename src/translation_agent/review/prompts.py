"""Deterministic reviewer prompt templates and dry-run prose generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from translation_agent.models import (
    JobContext,
    MemoryBundle,
    ReviewContext,
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
PARSER_VERSION = "phase-4-v1"


@dataclass(frozen=True, slots=True)
class ReviewerRoleSpec:
    """Static prompt metadata for a reviewer role."""

    stage: ReviewStage
    reviewer_role: str
    focus: str
    policy_ref: str


@dataclass(frozen=True, slots=True)
class ReviewDraftIssue:
    """Intermediate issue used to render deterministic reviewer prose."""

    candidate_id: str
    category: str
    severity: str
    description: str


@dataclass(frozen=True, slots=True)
class ReviewDraftFix:
    """Intermediate fix used to render deterministic reviewer prose."""

    category: str
    candidate_id: str
    description: str


@dataclass(frozen=True, slots=True)
class ReviewDraftEvidence:
    """Intermediate evidence item used to render deterministic reviewer prose."""

    candidate_id: str
    segment_id: str
    quote: str


class ReviewDraft(TypedDict):
    """Rendered review payload before it is turned into fixed-section prose."""

    winner: str
    confidence: float
    why: tuple[str, ...]
    issues: tuple[str, ...]
    evidence: tuple[str, ...]
    fixes: tuple[str, ...]
    escalate: bool


TRANSCRIPT_REVIEWER_SPECS = (
    ReviewerRoleSpec(
        stage="transcript",
        reviewer_role="accuracy_reviewer",
        focus="Literal accuracy, names, speaker fidelity, timestamps, and terminology.",
        policy_ref="policy/transcript-review/accuracy-v1",
    ),
    ReviewerRoleSpec(
        stage="transcript",
        reviewer_role="coherence_reviewer",
        focus="Omissions, additions, formatting, coherence, and plausibility.",
        policy_ref="policy/transcript-review/coherence-v1",
    ),
)
TRANSLATION_REVIEWER_SPECS = (
    ReviewerRoleSpec(
        stage="translation",
        reviewer_role="faithfulness_reviewer",
        focus="Faithfulness, terminology, and constraint preservation.",
        policy_ref="policy/translation-review/faithfulness-v1",
    ),
    ReviewerRoleSpec(
        stage="translation",
        reviewer_role="style_reviewer",
        focus="Fluency, tone, style fit, and readability.",
        policy_ref="policy/translation-review/style-v1",
    ),
)


def reviewer_roles_for_stage(stage: ReviewStage) -> tuple[ReviewerRoleSpec, ...]:
    """Return the two deterministic reviewer-role specs for a stage."""

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
    """Create the typed reviewer context with stage-scoped memory slices."""

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
    """Project memory down to the documented review-time slices."""

    episodic_slice = memory_bundle.episodic_memory[:2]
    if stage == "transcript":
        return MemoryBundle(
            glossary=memory_bundle.glossary,
            rules=memory_bundle.rules,
            episodic_memory=episodic_slice,
            provider_caveats=memory_bundle.provider_caveats,
        )

    return MemoryBundle(
        glossary=memory_bundle.glossary,
        rules=memory_bundle.rules,
        episodic_memory=episodic_slice,
    )


def adjudication_memory_bundle(
    *,
    stage: ReviewStage,
    memory_bundle: MemoryBundle,
) -> MemoryBundle:
    """Keep adjudication-time memory narrow and deterministic."""

    return MemoryBundle(
        semantic_memory=memory_bundle.semantic_memory[:2],
        glossary=memory_bundle.glossary,
        rules=memory_bundle.rules,
        episodic_memory=memory_bundle.episodic_memory[:1],
        provider_caveats=memory_bundle.provider_caveats if stage == "transcript" else (),
    )


def build_review_prompt(
    context: ReviewContext,
    *,
    candidate_refs: tuple[str, ...],
    raw_payload_refs: tuple[str, ...],
    final_transcript_ref: str | None = None,
) -> str:
    """Build the fixed reviewer prompt contract for deterministic review roles."""

    spec = _role_spec(context.stage, context.reviewer_role)
    memory_notes = [
        f"glossary={len(context.memory_bundle.glossary)}",
        f"rules={len(context.memory_bundle.rules)}",
        f"episodic={len(context.memory_bundle.episodic_memory)}",
        f"provider_caveats={len(context.memory_bundle.provider_caveats)}",
    ]
    lines = [
        f"Stage: {context.stage}",
        f"Role: {context.reviewer_role}",
        f"Focus: {spec.focus}",
        f"Job ID: {context.job.job_id}",
        f"Project Profile Ref: {context.job.profile_ref}",
        f"Candidate Refs: {', '.join(candidate_refs)}",
        ("Raw Payload Refs: " + (", ".join(raw_payload_refs) if raw_payload_refs else "none")),
        f"Memory Slice: {', '.join(memory_notes)}",
    ]
    if final_transcript_ref is not None:
        lines.append(f"Final Transcript Ref: {final_transcript_ref}")
    lines.append(
        "Return prose with exactly these sections: " + ", ".join(REQUIRED_REVIEW_SECTIONS) + "."
    )
    return "\n".join(lines)


def render_reviewer_output(
    context: ReviewContext,
    *,
    candidates: Sequence[ReviewCandidate],
    prompt_text: str,
    final_transcript: TranscriptCandidate | None = None,
) -> str:
    """Render deterministic reviewer prose matching the fixed parser contract."""

    if not candidates:
        raise ValueError("render_reviewer_output requires at least one candidate")

    draft: ReviewDraft
    if context.stage == "transcript":
        transcript_candidates = tuple(
            candidate for candidate in candidates if isinstance(candidate, TranscriptCandidate)
        )
        draft = _draft_transcript_review(
            context=context,
            candidates=transcript_candidates,
            prompt_text=prompt_text,
        )
    else:
        translation_candidates = tuple(
            candidate for candidate in candidates if isinstance(candidate, TranslationCandidate)
        )
        draft = _draft_translation_review(
            context=context,
            candidates=translation_candidates,
            prompt_text=prompt_text,
            final_transcript=final_transcript,
        )

    why_lines = draft["why"]
    issue_lines = draft["issues"]
    evidence_lines = draft["evidence"]
    fix_lines = draft["fixes"]
    return "\n".join(
        [
            f"Winner: {draft['winner']}",
            f"Confidence: {draft['confidence']:.2f}",
            "Why:",
            *[f"- {line}" for line in why_lines],
            "Key Errors By Candidate:",
            *[f"- {line}" for line in issue_lines],
            "Quoted Evidence:",
            *[f"- {line}" for line in evidence_lines],
            "Suggested Fixes:",
            *[f"- {line}" for line in fix_lines],
            f"Escalate?: {'yes' if draft['escalate'] else 'no'}",
        ]
    )


def _draft_transcript_review(
    *,
    context: ReviewContext,
    candidates: tuple[TranscriptCandidate, ...],
    prompt_text: str,
) -> ReviewDraft:
    if not candidates:
        raise ValueError("transcript review requires transcript candidates")

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -_transcript_score(candidate, candidates, context.reviewer_role),
            candidate.candidate_id,
        ),
    )
    winner = ordered[0]
    evidence = [
        ReviewDraftEvidence(
            candidate_id=winner.candidate_id,
            segment_id=_segment_id(winner),
            quote=winner.full_text,
        )
    ]
    issues: list[ReviewDraftIssue] = []
    fixes: list[ReviewDraftFix] = []
    divergence_detected = False
    for candidate in ordered[1:]:
        similarity = _text_similarity(winner.full_text, candidate.full_text)
        if similarity < 0.62:
            divergence_detected = True
            category = "accuracy" if context.reviewer_role == "accuracy_reviewer" else "coherence"
            issues.append(
                ReviewDraftIssue(
                    candidate_id=candidate.candidate_id,
                    category=category,
                    severity="critical",
                    description=(
                        "Conflicts with the winning transcript on the same spoken content."
                    ),
                )
            )
            evidence.append(
                ReviewDraftEvidence(
                    candidate_id=candidate.candidate_id,
                    segment_id=_segment_id(candidate),
                    quote=candidate.full_text,
                )
            )
            fixes.append(
                ReviewDraftFix(
                    category=category,
                    candidate_id=candidate.candidate_id,
                    description=(
                        "Re-run transcript alignment for the disputed span before publishing."
                    ),
                )
            )
        elif similarity < 0.92:
            divergence_detected = True
            category = (
                "terminology" if context.reviewer_role == "accuracy_reviewer" else "formatting"
            )
            issues.append(
                ReviewDraftIssue(
                    candidate_id=candidate.candidate_id,
                    category=category,
                    severity="major",
                    description=(
                        "Differs from the winning transcript enough to require manual comparison."
                    ),
                )
            )
            evidence.append(
                ReviewDraftEvidence(
                    candidate_id=candidate.candidate_id,
                    segment_id=_segment_id(candidate),
                    quote=candidate.full_text,
                )
            )
            fixes.append(
                ReviewDraftFix(
                    category=category,
                    candidate_id=candidate.candidate_id,
                    description=(
                        "Check named entities, punctuation, and speaker labeling "
                        "against the raw payload."
                    ),
                )
            )

    if len(candidates) == 1:
        divergence_detected = True
        issues.append(
            ReviewDraftIssue(
                candidate_id=winner.candidate_id,
                category="coverage",
                severity="major",
                description=(
                    "Only one surviving transcript candidate remained after normalization."
                ),
            )
        )
        fixes.append(
            ReviewDraftFix(
                category="coverage",
                candidate_id=winner.candidate_id,
                description=(
                    "Keep the surviving transcript but flag the missing comparison "
                    "path in scorecards."
                ),
            )
        )

    confidence = 0.93
    if divergence_detected:
        confidence -= 0.18
    if len(candidates) == 1:
        confidence -= 0.12
    confidence = max(0.35, min(confidence, 0.98))
    why_lines = [
        (
            f"{context.reviewer_role} selected {winner.candidate_id} after applying "
            f"{context.policy_ref}."
        ),
        (
            "Candidate refs and raw payload refs were compared through the "
            "deterministic dry-run review path."
        ),
        f"Prompt contract preserved: {prompt_text.splitlines()[-1]}",
    ]
    if divergence_detected:
        why_lines.append(
            "Competing transcript spans still contain material disagreements worth escalation."
        )

    return {
        "winner": winner.candidate_id,
        "confidence": confidence,
        "why": tuple(why_lines),
        "issues": tuple(_render_issue(issue) for issue in issues)
        or ("none | general | minor | No blocking issues beyond the selected winner.",),
        "evidence": tuple(_render_evidence(item) for item in evidence),
        "fixes": tuple(_render_fix(fix) for fix in fixes)
        or ("general | none | Preserve deterministic export formatting.",),
        "escalate": divergence_detected,
    }


def _draft_translation_review(
    *,
    context: ReviewContext,
    candidates: tuple[TranslationCandidate, ...],
    prompt_text: str,
    final_transcript: TranscriptCandidate | None,
) -> ReviewDraft:
    if not candidates:
        raise ValueError("translation review requires translation candidates")
    if final_transcript is None:
        raise ValueError("translation review requires the approved transcript")

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -_translation_score(candidate, final_transcript, context.reviewer_role),
            candidate.candidate_id,
        ),
    )
    winner = ordered[0]
    source_text = final_transcript.full_text.lower()
    evidence = [
        ReviewDraftEvidence(
            candidate_id=winner.candidate_id,
            segment_id=_segment_id(winner),
            quote=winner.full_text,
        )
    ]
    issues: list[ReviewDraftIssue] = []
    fixes: list[ReviewDraftFix] = []
    escalate = False
    for candidate in ordered[1:]:
        candidate_text = candidate.full_text.lower()
        if context.reviewer_role == "faithfulness_reviewer" and (
            "annule le sens source" in candidate_text or "improvise" in candidate_text
        ):
            severity = "critical"
            category = "faithfulness"
            description = "Introduces meaning drift severe enough to require human review."
        elif "workflow" in source_text and all(
            token not in candidate_text for token in ("workflow", "flux de travail")
        ):
            severity = "major"
            category = "faithfulness"
            description = "Drops the explicit workflow reference from the source transcript."
        elif "pipeline" in candidate_text and "pipeline" not in source_text:
            severity = "major"
            category = "terminology"
            description = "Introduces pipeline wording that is not grounded in the transcript."
        elif "salut" in candidate_text and context.reviewer_role == "style_reviewer":
            severity = "minor"
            category = "style"
            description = "Uses a more casual tone than the default project style."
        else:
            severity = "minor"
            category = "readability" if context.reviewer_role == "style_reviewer" else "constraint"
            description = "Requires only light copy edits."

        if severity != "minor" or context.reviewer_role == "style_reviewer":
            evidence.append(
                ReviewDraftEvidence(
                    candidate_id=candidate.candidate_id,
                    segment_id=_segment_id(candidate),
                    quote=candidate.full_text,
                )
            )
        issues.append(
            ReviewDraftIssue(
                candidate_id=candidate.candidate_id,
                category=category,
                severity=severity,
                description=description,
            )
        )
        fixes.append(
            ReviewDraftFix(
                category=category,
                candidate_id=candidate.candidate_id,
                description=(
                    "Adjust the disputed span and keep terminology aligned with the "
                    "approved transcript."
                ),
            )
        )
        if severity != "minor":
            escalate = True

    if len(candidates) == 1:
        confidence = 0.68
        issues.append(
            ReviewDraftIssue(
                candidate_id=winner.candidate_id,
                category="coverage",
                severity="minor",
                description=(
                    "Only one translation variant survived, so review coverage is reduced."
                ),
            )
        )
        fixes.append(
            ReviewDraftFix(
                category="coverage",
                candidate_id=winner.candidate_id,
                description="Publish the surviving variant with a lower confidence score.",
            )
        )
    else:
        confidence = 0.88 if not escalate else 0.74

    why_lines = [
        (f"{context.reviewer_role} selected {winner.candidate_id} using {context.policy_ref}."),
        (
            "The approved transcript and candidate variants were checked against the "
            "fixed prose-review contract."
        ),
        f"Prompt contract preserved: {prompt_text.splitlines()[-1]}",
    ]
    if escalate:
        why_lines.append(
            "At least one competing variant has a material issue that should feed adjudication."
        )

    return {
        "winner": winner.candidate_id,
        "confidence": confidence,
        "why": tuple(why_lines),
        "issues": tuple(_render_issue(issue) for issue in issues)
        or ("none | general | minor | No blocking issues beyond the selected winner.",),
        "evidence": tuple(_render_evidence(item) for item in evidence),
        "fixes": tuple(_render_fix(fix) for fix in fixes)
        or ("general | none | Keep terminology and tone aligned with the source transcript.",),
        "escalate": escalate,
    }


def _transcript_score(
    candidate: TranscriptCandidate,
    candidates: tuple[TranscriptCandidate, ...],
    reviewer_role: str,
) -> float:
    similarity = 1.0
    if len(candidates) > 1:
        similarity = sum(
            _text_similarity(candidate.full_text, other.full_text)
            for other in candidates
            if other.candidate_id != candidate.candidate_id
        ) / (len(candidates) - 1)
    provider_rank = int(candidate.metadata.get("provider_rank", 100))
    punctuation_bonus = 0.05 if candidate.full_text.endswith(".") else 0.0
    role_bonus = 0.05 if reviewer_role == "coherence_reviewer" and punctuation_bonus else 0.0
    if reviewer_role == "coherence_reviewer" and "escalation path" in candidate.full_text.lower():
        role_bonus += 0.7
    return (1.0 - provider_rank * 0.05) + similarity + punctuation_bonus + role_bonus


def _translation_score(
    candidate: TranslationCandidate,
    final_transcript: TranscriptCandidate,
    reviewer_role: str,
) -> float:
    text = candidate.full_text.lower()
    source_text = final_transcript.full_text.lower()
    score = 0.0
    if reviewer_role == "faithfulness_reviewer":
        if "workflow" in source_text and ("workflow" in text or "flux de travail" in text):
            score += 1.0
        if "pipeline" in text and "pipeline" not in source_text:
            score -= 0.6
        if "openai" in source_text and "openai" in text:
            score += 0.2
    else:
        if text.startswith("bonjour"):
            score += 0.5
        if text.startswith("salut"):
            score += 0.2
        if "workflow" in text:
            score += 0.1
        if "flux de travail" in text:
            score -= 0.7
        if "pipeline" in text:
            score -= 0.2
        if "improvise" in text:
            score -= 0.4
    if candidate.prompt_variant_id == "variant-a":
        score += 0.05
    return score


def _role_spec(stage: ReviewStage, reviewer_role: str) -> ReviewerRoleSpec:
    for spec in reviewer_roles_for_stage(stage):
        if spec.reviewer_role == reviewer_role:
            return spec
    raise ValueError(f"unknown reviewer role {reviewer_role!r} for stage {stage!r}")


def _segment_id(candidate: ReviewCandidate) -> str:
    if candidate.segments:
        return candidate.segments[0].segment_id
    return f"{candidate.candidate_id}-seg-1"


def _render_issue(issue: ReviewDraftIssue) -> str:
    return f"{issue.candidate_id} | {issue.category} | {issue.severity} | {issue.description}"


def _render_evidence(item: ReviewDraftEvidence) -> str:
    return f"{item.candidate_id} | {item.segment_id} | {item.quote}"


def _render_fix(fix: ReviewDraftFix) -> str:
    return f"{fix.category} | {fix.candidate_id} | {fix.description}"


def _text_similarity(left: str, right: str) -> float:
    left_tokens = {token.strip(".,!?;:").lower() for token in left.split() if token}
    right_tokens = {token.strip(".,!?;:").lower() for token in right.split() if token}
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)
