"""Canonical memory models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .base import ContractModel
from .jobs import JobContext

NonEmptyStr = Annotated[str, Field(min_length=1)]
MemoryKind = Literal["semantic", "episodic", "glossary", "rule", "procedural"]
ConsolidationStatus = Literal["pending", "consolidated", "skipped", "failed"]
PromptFamily = Literal["translation", "reviewer", "adjudicator"]
PromptEvolutionStatus = Literal["proposed", "approved", "rejected"]
PromptActivationMode = Literal["auto_activate_eligible", "approval_required"]


class MemoryEntry(ContractModel):
    memory_id: NonEmptyStr
    kind: MemoryKind
    content: NonEmptyStr
    source_ref: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCaveat(ContractModel):
    provider_id: NonEmptyStr
    note: NonEmptyStr


class MemoryQuery(ContractModel):
    """Typed memory recall request used by reviewer and adjudicator contexts."""

    job: JobContext
    stage: NonEmptyStr
    query_text: NonEmptyStr
    candidate_ids: tuple[str, ...] = ()
    max_items: int = Field(default=10, ge=1, le=100)


class MemoryBundle(ContractModel):
    """Read-only memory slices passed into review and adjudication."""

    semantic_memory: tuple[MemoryEntry, ...] = ()
    episodic_memory: tuple[MemoryEntry, ...] = ()
    glossary: tuple[MemoryEntry, ...] = ()
    rules: tuple[MemoryEntry, ...] = ()
    provider_caveats: tuple[ProviderCaveat, ...] = ()


class MemoryWrite(ContractModel):
    kind: MemoryKind
    content: NonEmptyStr
    source_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteBatch(ContractModel):
    """Candidate memory writes emitted at adjudication boundaries."""

    batch_id: NonEmptyStr
    job_id: NonEmptyStr
    source_stage: NonEmptyStr
    decision_ref: str | None = None
    investigation_ref: str | None = None
    winner_candidate_id: str | None = None
    decision_mode: str | None = None
    decision_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreement_bucket: str | None = None
    translation_model_winner: str | None = None
    prompt_variant_winner: str | None = None
    prompt_version_winner: str | None = None
    semantic_writes: tuple[MemoryWrite, ...] = ()
    episodic_writes: tuple[MemoryWrite, ...] = ()
    procedural_writes: tuple[MemoryWrite, ...] = ()
    dedupe_keys: tuple[str, ...] = ()
    consolidation_status: ConsolidationStatus = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryConsolidation(ContractModel):
    """Result of background consolidation for one staged memory batch."""

    consolidation_id: NonEmptyStr
    batch_id: NonEmptyStr
    job_id: NonEmptyStr
    source_stage: NonEmptyStr
    source_decision_ref: str | None = None
    source_decision_mode: str | None = None
    source_disagreement_bucket: str | None = None
    source_translation_model_id: str | None = None
    source_prompt_variant_id: str | None = None
    source_prompt_version: str | None = None
    semantic_memory_ids: tuple[str, ...] = ()
    episodic_memory_ids: tuple[str, ...] = ()
    skipped_dedupe_keys: tuple[str, ...] = ()
    procedural_write_count: int = Field(default=0, ge=0)


class PromptChange(ContractModel):
    section: NonEmptyStr
    instruction: NonEmptyStr


class PromptEvolutionProposal(ContractModel):
    """A gated prompt-improvement proposal derived from consolidated outcomes."""

    proposal_id: NonEmptyStr
    job_id: NonEmptyStr
    source_consolidation_id: NonEmptyStr
    prompt_family: PromptFamily
    target_model_id: NonEmptyStr
    target_prompt_version: str | None = None
    target_prompt_variant_id: str | None = None
    status: PromptEvolutionStatus = "proposed"
    activation_mode: PromptActivationMode = "approval_required"
    auto_activate: bool = False
    rationale: NonEmptyStr
    suggested_changes: tuple[PromptChange, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_activation(self) -> PromptEvolutionProposal:
        if self.auto_activate and self.activation_mode != "auto_activate_eligible":
            raise ValueError("auto_activate proposals must be auto_activate_eligible")
        if self.auto_activate and self.prompt_family != "translation":
            raise ValueError("only translation prompt proposals can auto-activate")
        return self
