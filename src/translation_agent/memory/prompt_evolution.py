"""Prompt evolution interfaces and deterministic reference implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import MemoryConsolidation, PromptChange, PromptEvolutionProposal


@runtime_checkable
class PromptEvolutionBackend(Protocol):
    """Create gated prompt updates from consolidated outcomes only."""

    def propose_prompt_evolution(
        self,
        consolidation: MemoryConsolidation,
        *,
        translation_model_id: str | None,
        evidence_ref: str,
    ) -> PromptEvolutionProposal | None: ...


class DeterministicPromptEvolutionBackend:
    """Reference prompt evolution logic for translation-only improvements."""

    def propose_prompt_evolution(
        self,
        consolidation: MemoryConsolidation,
        *,
        translation_model_id: str | None,
        evidence_ref: str,
    ) -> PromptEvolutionProposal | None:
        if consolidation.source_stage != "translation_adjudication":
            return None
        if translation_model_id is None or consolidation.source_prompt_variant_id is None:
            return None
        activation_mode = (
            "auto_activate_eligible"
            if consolidation.source_disagreement_bucket == "low"
            else "approval_required"
        )
        target_prompt_version = f"{consolidation.source_prompt_version or 'unversioned'}-phase5"
        return PromptEvolutionProposal(
            proposal_id=f"prompt-evolution-{consolidation.consolidation_id}",
            job_id=consolidation.job_id,
            source_consolidation_id=consolidation.consolidation_id,
            prompt_family="translation",
            target_model_id=translation_model_id,
            target_prompt_version=target_prompt_version,
            target_prompt_variant_id=consolidation.source_prompt_variant_id,
            activation_mode=activation_mode,
            auto_activate=activation_mode == "auto_activate_eligible",
            rationale=(
                "Consolidated translation outcomes favored the winning prompt variant without "
                "relying on raw reviewer prose."
            ),
            suggested_changes=(
                PromptChange(
                    section="system",
                    instruction=(
                        "Bias the prompt toward terminology preservation and stable named "
                        "entity handling."
                    ),
                ),
                PromptChange(
                    section="guardrails",
                    instruction=(
                        "Keep the successful variant's style boundary while avoiding ad-lib "
                        "wording changes."
                    ),
                ),
            ),
            evidence_refs=tuple(
                ref for ref in (evidence_ref, consolidation.source_decision_ref) if ref is not None
            ),
            metadata={"procedural_write_count": consolidation.procedural_write_count},
        )
