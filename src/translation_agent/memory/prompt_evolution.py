"""Prompt evolution interfaces for automatic, evaluation-driven proposals."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from translation_agent.models import MemoryConsolidation, PromptEvolutionProposal


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
    """Mainline adjudication no longer emits prompt proposals directly."""

    supports_parallel_compute_only = True

    def propose_prompt_evolution(
        self,
        consolidation: MemoryConsolidation,
        *,
        translation_model_id: str | None,
        evidence_ref: str,
    ) -> PromptEvolutionProposal | None:
        del consolidation, translation_model_id, evidence_ref
        return None
