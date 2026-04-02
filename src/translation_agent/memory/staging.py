"""Memory staging interfaces and deterministic reference implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import (
    EvaluationReport,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    MemoryWrite,
    MemoryWriteBatch,
    PromptEvolutionProposal,
)


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
        dedupe_keys: list[str] = []
        semantic_writes: tuple[MemoryWrite, ...] = ()
        if decision.winner_candidate_id is not None:
            semantic_key = (
                f"semantic:{source_stage}:{decision.job_id}:{decision.winner_candidate_id}"
            )
            dedupe_keys.append(semantic_key)
            semantic_writes = (
                MemoryWrite(
                    kind="semantic",
                    content=_semantic_summary(decision, source_stage=source_stage),
                    source_ref=decision.investigation_ref or decision.review_refs[0]
                    if decision.review_refs
                    else None,
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
                    content=(
                        "Strengthen terminology preservation and named-entity stability for "
                        "approved translation outcomes."
                    ),
                    source_ref=decision.investigation_ref or decision.review_refs[0]
                    if decision.review_refs
                    else None,
                    metadata={
                        "dedupe_key": procedural_key,
                        "prompt_family": "translation",
                    },
                ),
            )

        return MemoryWriteBatch(
            batch_id=f"batch-{source_stage}-{decision.job_id}",
            job_id=decision.job_id,
            source_stage=source_stage,
            investigation_ref=decision.investigation_ref,
            winner_candidate_id=decision.winner_candidate_id,
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
                    content=decision.rationale_summary,
                    source_ref=decision.investigation_ref or decision.review_refs[0]
                    if decision.review_refs
                    else None,
                    metadata={"dedupe_key": episodic_key},
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
        if not report.recurring_failure_patterns and not proposals:
            return None
        semantic_writes = tuple(
            MemoryWrite(
                kind="semantic",
                content=f"Reference evaluation observed recurring issue: {pattern}.",
                source_ref=report.trusted_transcript_ref,
                metadata={"dedupe_key": f"evaluation:{report.media_key}:{pattern}"},
            )
            for pattern in report.recurring_failure_patterns
        )
        procedural_writes = tuple(
            MemoryWrite(
                kind="procedural",
                content=proposal.suggested_changes[0].instruction
                if proposal.suggested_changes
                else proposal.rationale,
                source_ref=proposal.evidence_refs[0] if proposal.evidence_refs else None,
                metadata={
                    "dedupe_key": f"evaluation:{report.media_key}:{proposal.proposal_id}",
                    "prompt_family": proposal.prompt_family,
                    "proposal_id": proposal.proposal_id,
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
                    content="Reference evaluation produced approval-gated learning proposals.",
                    source_ref=report.trusted_transcript_ref,
                    metadata={"dedupe_key": episodic_key},
                ),
            ),
            procedural_writes=procedural_writes,
            dedupe_keys=dedupe_keys + (episodic_key,),
            metadata={"media_key": report.media_key, "proposal_count": len(proposals)},
        )


def _semantic_summary(
    decision: FinalTranscriptDecision | FinalTranslationDecision,
    *,
    source_stage: str,
) -> str:
    stage_label = source_stage.replace("_", " ")
    return (
        f"{stage_label} trusted {decision.winner_candidate_id} after "
        f"{decision.decision_mode} with {decision.disagreement_bucket} disagreement."
    )
