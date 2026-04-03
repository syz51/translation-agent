"""Prompt evolution interfaces for automatic, evaluation-driven proposals."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.memory.recall import build_scope_key
from translation_agent.models import (
    MemoryConsolidation,
    PromotionGateOutcome,
    PromptChange,
    PromptCompatibilityTuple,
    PromptEvolutionProposal,
)


@runtime_checkable
class PromptEvolutionBackend(Protocol):
    """Create prompt updates from evaluated outcomes only."""

    def propose_prompt_evolution(
        self,
        consolidation: MemoryConsolidation,
        *,
        translation_model_id: str | None,
        evidence_ref: str,
    ) -> PromptEvolutionProposal | None: ...


class DeterministicPromptEvolutionBackend:
    """Emit project-pair prompt proposal candidates from low-risk translation wins."""

    supports_parallel_compute_only = True

    def propose_prompt_evolution(
        self,
        consolidation: MemoryConsolidation,
        *,
        translation_model_id: str | None,
        evidence_ref: str,
    ) -> PromptEvolutionProposal | None:
        if consolidation.source_stage != "translation_adjudication":
            return None
        if consolidation.source_disagreement_bucket != "low":
            return None
        if consolidation.procedural_write_count <= 0:
            return None
        if (
            consolidation.source_prompt_variant_id is None
            or consolidation.source_prompt_version is None
        ):
            return None
        model_id = translation_model_id or consolidation.source_translation_model_id
        if model_id is None:
            return None
        if (
            consolidation.source_tenant_id is None
            or consolidation.source_project_id is None
            or consolidation.source_language is None
            or consolidation.target_language is None
        ):
            return None

        compatibility = PromptCompatibilityTuple(
            prompt_family="translation",
            model_id=model_id,
            prompt_variant_id=consolidation.source_prompt_variant_id,
            base_prompt_version=consolidation.source_prompt_version,
            source_language=consolidation.source_language,
            target_language=consolidation.target_language,
            scope_kind="project_pair",
            scope_key=build_scope_key(
                scope_kind="project_pair",
                tenant_id=consolidation.source_tenant_id,
                project_id=consolidation.source_project_id,
                source_language=consolidation.source_language,
                target_language=consolidation.target_language,
            ),
        )
        return PromptEvolutionProposal(
            proposal_id=f"project-pair-{consolidation.consolidation_id}",
            job_id=consolidation.job_id,
            source_consolidation_id=consolidation.consolidation_id,
            prompt_family="translation",
            target_model_id=model_id,
            target_prompt_version=consolidation.source_prompt_version,
            target_prompt_variant_id=consolidation.source_prompt_variant_id,
            base_prompt_version=consolidation.source_prompt_version,
            compatibility=compatibility,
            status="proposed",
            rationale=(
                "Low-disagreement translation adjudication produced stable prompt guidance for "
                "this exact project/language pair."
            ),
            suggested_changes=(
                PromptChange(
                    section="system",
                    instruction=(
                        "Prefer the project-pair guidance supported by recent approved "
                        "translation outcomes."
                    ),
                ),
            ),
            evidence_refs=(evidence_ref,),
            promotion_status="candidate",
            gate_outcome=PromotionGateOutcome(
                quality_gate_status="pending",
                notes=("project_pair_candidate_only",),
            ),
            metadata={
                "proposal_origin": "mainline_adjudication",
                "source_language": consolidation.source_language,
                "target_language": consolidation.target_language,
                "scope_kind": "project_pair",
                "scope_key": compatibility.scope_key,
                "tenant_id": consolidation.source_tenant_id,
                "project_id": consolidation.source_project_id,
            },
        )
