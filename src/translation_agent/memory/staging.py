"""Memory staging interfaces and deterministic reference implementation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from translation_agent.memory.recall import build_scope_key
from translation_agent.models import (
    EvaluationReport,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    JobContext,
    MemoryWrite,
    MemoryWriteBatch,
    PromptEvolutionProposal,
)
from translation_agent.models.memory import MemorySubtype


@runtime_checkable
class MemoryStagingBackend(Protocol):
    """Adjudication-boundary staging contract for candidate memory writes."""

    def stage_memory_candidates(
        self,
        decision: FinalTranscriptDecision | FinalTranslationDecision,
        *,
        source_stage: str,
    ) -> MemoryWriteBatch: ...

    def stage_evaluation_candidates(
        self,
        report: EvaluationReport,
        *,
        proposals: tuple[PromptEvolutionProposal, ...],
    ) -> MemoryWriteBatch | None: ...


class DeterministicMemoryStagingBackend:
    """Reference staging backend that keeps long-term memory free of raw artifacts."""

    def stage_memory_candidates(
        self,
        decision: FinalTranscriptDecision | FinalTranslationDecision,
        *,
        source_stage: str,
    ) -> MemoryWriteBatch:
        now = datetime.now(UTC)
        dedupe_keys: list[str] = []
        semantic_writes: tuple[MemoryWrite, ...] = ()
        primary_ref = _decision_primary_ref(decision)
        if primary_ref is not None:
            semantic_key = f"semantic:{source_stage}:{decision.job_id}:{primary_ref}"
            dedupe_keys.append(semantic_key)
            semantic_writes = (
                MemoryWrite(
                    kind="semantic",
                    memory_subtype="project_fact",
                    content=_semantic_summary(decision, source_stage=source_stage),
                    updated_at=now,
                    score=_score_for_decision(decision),
                    source_ref=decision.investigation_ref or decision.review_refs[0]
                    if decision.review_refs
                    else None,
                    promotion_status="candidate",
                    evidence_count=1,
                    supporting_run_count=1,
                    supporting_asset_count=1,
                    metadata={"dedupe_key": semantic_key},
                ),
            )

        episodic_key = f"episodic:{source_stage}:{decision.job_id}:{decision.decision_mode}"
        dedupe_keys.append(episodic_key)
        procedural_writes: tuple[MemoryWrite, ...] = ()
        if (
            isinstance(decision, FinalTranslationDecision)
            and decision.winner_candidate_id is not None
            and decision.prompt_variant_winner is not None
        ):
            procedural_key = (
                "procedural:"
                f"{source_stage}:{decision.job_id}:{decision.prompt_variant_winner}:"
                f"{decision.prompt_version_winner or 'unknown'}"
            )
            dedupe_keys.append(procedural_key)
            procedural_writes = (
                MemoryWrite(
                    kind="procedural",
                    memory_subtype="prompt_guidance",
                    content=(
                        "Strengthen terminology preservation and named-entity stability for "
                        "approved translation outcomes."
                    ),
                    updated_at=now,
                    score=_score_for_decision(decision),
                    source_ref=decision.investigation_ref or decision.review_refs[0]
                    if decision.review_refs
                    else None,
                    promotion_status="candidate",
                    evidence_count=1,
                    supporting_run_count=1,
                    supporting_asset_count=1,
                    metadata={
                        "dedupe_key": procedural_key,
                        "prompt_family": "translation",
                    },
                ),
            )

        episodic_subtype = _decision_memory_subtype(decision)
        episodic_metadata: dict[str, object] = {
            "dedupe_key": episodic_key,
            "event_id": batch_event_id(decision),
        }
        if episodic_subtype == "failure_pattern":
            episodic_metadata["failure_tags"] = ["translation_failed"]
        if episodic_subtype == "escalation_pattern":
            episodic_metadata["escalation_reasons"] = [decision.decision_mode]

        return MemoryWriteBatch(
            batch_id=f"batch-{source_stage}-{decision.job_id}",
            job_id=decision.job_id,
            source_stage=source_stage,
            investigation_ref=decision.investigation_ref,
            winner_candidate_id=primary_ref,
            decision_mode=decision.decision_mode,
            decision_confidence=decision.decision_confidence,
            disagreement_bucket=decision.disagreement_bucket,
            translation_model_winner=(
                decision.winner_model_id if isinstance(decision, FinalTranslationDecision) else None
            ),
            prompt_variant_winner=(
                decision.prompt_variant_winner
                if isinstance(decision, FinalTranslationDecision)
                else None
            ),
            prompt_version_winner=(
                decision.prompt_version_winner
                if isinstance(decision, FinalTranslationDecision)
                else None
            ),
            semantic_writes=semantic_writes,
            episodic_writes=(
                MemoryWrite(
                    kind="episodic",
                    memory_subtype=episodic_subtype,
                    content=_decision_episodic_summary(decision),
                    updated_at=now,
                    score=_score_for_decision(decision),
                    source_ref=decision.investigation_ref or decision.review_refs[0]
                    if decision.review_refs
                    else None,
                    promotion_status="candidate",
                    evidence_count=1,
                    supporting_run_count=1,
                    supporting_asset_count=1,
                    metadata=episodic_metadata,
                ),
            ),
            procedural_writes=procedural_writes,
            dedupe_keys=tuple(dedupe_keys),
        )

    def stage_evaluation_candidates(
        self,
        report: EvaluationReport,
        *,
        proposals: tuple[PromptEvolutionProposal, ...],
    ) -> MemoryWriteBatch | None:
        now = datetime.now(UTC)
        if not report.recurring_failure_patterns and not proposals:
            return None
        source_language = _report_metadata_string(report.metadata, "source_language") or "unknown"
        target_language = _report_metadata_string(report.metadata, "target_language") or "unknown"
        tenant_id = _report_metadata_string(report.metadata, "tenant_id") or "unknown"
        project_id = _report_metadata_string(report.metadata, "project_id") or "unknown"
        project_scope_key = build_scope_key(
            scope_kind="project_pair",
            tenant_id=tenant_id,
            project_id=project_id,
            source_language=source_language,
            target_language=target_language,
        )
        semantic_writes = tuple(
            MemoryWrite(
                kind="semantic",
                memory_subtype="failure_pattern",
                content=f"Reference evaluation observed recurring issue: {pattern}.",
                scope_kind="asset",
                scope_key=report.media_key,
                updated_at=now,
                score=0.7,
                source_ref=report.trusted_transcript_ref,
                promotion_status="candidate",
                evidence_count=len(report.evaluated_runs),
                supporting_run_count=len(report.evaluated_runs),
                supporting_asset_count=1,
                metadata={
                    "dedupe_key": f"evaluation:{report.media_key}:{pattern}",
                    "failure_tags": [pattern],
                    "media_key": report.media_key,
                },
            )
            for pattern in report.recurring_failure_patterns
        )
        procedural_writes = tuple(
            MemoryWrite(
                kind="procedural",
                memory_subtype="prompt_guidance",
                content=proposal.suggested_changes[0].instruction
                if proposal.suggested_changes
                else proposal.rationale,
                scope_kind="project_pair",
                scope_key=project_scope_key,
                updated_at=now,
                score=max(proposal.canary_metrics.primary_quality_score, 0.5),
                source_ref=proposal.evidence_refs[0] if proposal.evidence_refs else None,
                promotion_status=proposal.promotion_status,
                evidence_count=max(proposal.canary_run_count, len(report.evaluated_runs)),
                supporting_run_count=max(proposal.canary_run_count, len(report.evaluated_runs)),
                supporting_asset_count=1,
                supporting_project_count=1,
                quality_gate_status=(
                    proposal.gate_outcome.quality_gate_status
                    if proposal.gate_outcome is not None
                    else "pending"
                ),
                metadata={
                    "dedupe_key": f"evaluation:{report.media_key}:{proposal.proposal_id}",
                    "prompt_family": proposal.prompt_family,
                    "proposal_id": proposal.proposal_id,
                    "media_key": report.media_key,
                },
            )
            for proposal in proposals
        )
        dedupe_keys = tuple(write.metadata["dedupe_key"] for write in semantic_writes) + tuple(
            write.metadata["dedupe_key"] for write in procedural_writes
        )
        episodic_key = f"evaluation:{report.media_key}:episodic"
        return MemoryWriteBatch(
            batch_id=f"batch-reference-evaluation-{report.run_id}",
            job_id=report.run_id,
            source_stage="reference_evaluation",
            decision_ref=report.trusted_transcript_ref,
            investigation_ref=None,
            winner_candidate_id=None,
            decision_mode="reference_evaluation",
            decision_confidence=1.0 if report.evaluated_runs else 0.0,
            disagreement_bucket="low" if proposals else "medium",
            semantic_writes=semantic_writes,
            episodic_writes=(
                MemoryWrite(
                    kind="episodic",
                    memory_subtype="project_fact",
                    content="Reference evaluation produced non-blocking learning artifacts.",
                    scope_kind="asset",
                    scope_key=report.media_key,
                    updated_at=now,
                    score=0.65,
                    source_ref=report.trusted_transcript_ref,
                    promotion_status="candidate",
                    evidence_count=len(report.evaluated_runs),
                    supporting_run_count=len(report.evaluated_runs),
                    supporting_asset_count=1,
                    metadata={
                        "dedupe_key": episodic_key,
                        "event_id": report.run_id,
                        "media_key": report.media_key,
                    },
                ),
            ),
            procedural_writes=procedural_writes,
            dedupe_keys=dedupe_keys + (episodic_key,),
            metadata={
                "media_key": report.media_key,
                "proposal_count": len(proposals),
                "source_language": source_language,
                "target_language": target_language,
                "tenant_id": tenant_id,
                "project_id": project_id,
            },
        )


def batch_metadata_for_job(job: JobContext, **extra: object) -> dict[str, object]:
    return {
        "tenant_id": job.tenant_id,
        "project_id": job.project_id,
        "source_language": job.source_language,
        "target_language": job.target_language,
        "media_key": job.media_key,
        **extra,
    }


def project_pair_scope_key(job: JobContext) -> str:
    return build_scope_key(
        scope_kind="project_pair",
        tenant_id=job.tenant_id,
        project_id=job.project_id,
        source_language=job.source_language,
        target_language=job.target_language,
    )


def pair_scope_key(job: JobContext) -> str:
    return build_scope_key(
        scope_kind="pair",
        tenant_id=job.tenant_id,
        project_id=job.project_id,
        source_language=job.source_language,
        target_language=job.target_language,
    )


def scope_for_memory_subtype(
    memory_subtype: str | None,
    *,
    job: JobContext,
) -> tuple[str, str]:
    if memory_subtype in {"failure_pattern", "escalation_pattern"}:
        return ("asset", job.media_key)
    return ("project_pair", project_pair_scope_key(job))


def apply_scope_defaults(write: MemoryWrite, *, job: JobContext) -> MemoryWrite:
    if write.scope_kind != "global" or write.scope_key != "global":
        return write
    scope_kind, scope_key = scope_for_memory_subtype(write.memory_subtype, job=job)
    return write.model_copy(
        update={
            "scope_kind": scope_kind,
            "scope_key": scope_key,
            "origin_scope_kind": write.origin_scope_kind or scope_kind,
            "origin_scope_key": write.origin_scope_key or scope_key,
        }
    )


def _semantic_summary(
    decision: FinalTranscriptDecision | FinalTranslationDecision,
    *,
    source_stage: str,
) -> str:
    stage_label = source_stage.replace("_", " ")
    primary_ref = _decision_primary_ref(decision) or "no-primary-ref"
    return (
        f"{stage_label} trusted {primary_ref} after "
        f"{decision.decision_mode} with {decision.disagreement_bucket} disagreement."
    )


def _score_for_decision(
    decision: FinalTranscriptDecision | FinalTranslationDecision,
) -> float:
    return round(max(0.0, min(decision.decision_confidence or 0.5, 0.99)), 4)


def batch_event_id(decision: FinalTranscriptDecision | FinalTranslationDecision) -> str:
    return f"{decision.job_id}:{decision.decision_mode}"


def _decision_memory_subtype(
    decision: FinalTranscriptDecision | FinalTranslationDecision,
) -> MemorySubtype:
    if isinstance(decision, FinalTranslationDecision):
        if decision.winner_candidate_id is None:
            return "failure_pattern"
        if decision.human_review_required or decision.escalated:
            return "escalation_pattern"
    return "project_fact"


def _decision_episodic_summary(
    decision: FinalTranscriptDecision | FinalTranslationDecision,
) -> str:
    if isinstance(decision, FinalTranslationDecision) and decision.winner_candidate_id is None:
        return "Translation generation failed before adjudication could select a viable candidate."
    if isinstance(decision, FinalTranslationDecision) and decision.human_review_required:
        return "Translation adjudication required human review before behavior could widen."
    if isinstance(decision, FinalTranslationDecision) and decision.escalated:
        return "Translation adjudication escalated after unstable disagreement signals."
    return decision.rationale_summary


def _decision_primary_ref(
    decision: FinalTranscriptDecision | FinalTranslationDecision,
) -> str | None:
    if isinstance(decision, FinalTranslationDecision):
        return decision.winner_candidate_id
    return decision.transcript_artifact_ref


def _report_metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
