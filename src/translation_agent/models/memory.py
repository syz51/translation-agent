"""Canonical memory and prompt-evolution models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .base import ContractModel
from .jobs import JobContext

NonEmptyStr = Annotated[str, Field(min_length=1)]
MemoryKind = Literal["semantic", "episodic", "glossary", "rule", "procedural"]
MemorySubtype = Literal[
    "glossary",
    "style_rule",
    "provider_caveat",
    "language_convention",
    "project_fact",
    "failure_pattern",
    "escalation_pattern",
    "prompt_guidance",
]
MemoryScopeKind = Literal[
    "asset",
    "project_pair",
    "pair",
    "source_language",
    "target_language",
    "global",
]
MemoryLifecycleStatus = Literal["active", "stale", "rolled_back", "superseded", "expired"]
ConsolidationStatus = Literal["pending", "consolidated", "skipped", "failed"]
PromptFamily = Literal["translation", "reviewer", "adjudicator"]
PromotionStatus = Literal[
    "not_promoted",
    "candidate",
    "promoted",
    "superseded",
    "blocked",
]
QualityGateStatus = Literal[
    "not_evaluated",
    "pending",
    "passed",
    "failed",
    "disagreed",
]
PromptEvolutionStatus = Literal[
    "proposed",
    "canary",
    "active",
    "rolled_back",
    "superseded",
    "approved",
    "rejected",
]
PromptResolutionMode = Literal["control", "canary", "active"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryEntry(ContractModel):
    memory_id: NonEmptyStr
    kind: MemoryKind
    memory_subtype: MemorySubtype | None = None
    content: NonEmptyStr
    source_ref: str | None = None
    scope_kind: MemoryScopeKind | None = None
    scope_key: NonEmptyStr | None = None
    updated_at: datetime | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    lifecycle_status: MemoryLifecycleStatus = "active"
    expires_at: datetime | None = None
    superseded_by: str | None = None
    origin_scope_kind: MemoryScopeKind | None = None
    origin_scope_key: NonEmptyStr | None = None
    promotion_status: PromotionStatus = "not_promoted"
    evidence_count: int = Field(default=0, ge=0)
    supporting_run_count: int = Field(default=0, ge=0)
    supporting_asset_count: int = Field(default=0, ge=0)
    supporting_project_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    promotion_ref: str | None = None
    quality_gate_status: QualityGateStatus = "not_evaluated"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope_contract(self) -> MemoryEntry:
        legacy_shape = (
            self.scope_kind is None and self.scope_key is None and self.updated_at is None
        )
        if legacy_shape:
            return self
        if self.scope_kind is None or self.scope_key is None:
            raise ValueError("scoped memory entries require scope_kind and scope_key")
        if self.updated_at is None:
            raise ValueError("scoped memory entries require updated_at")
        if self.score is None:
            raise ValueError("scoped memory entries require score")
        if self.scope_kind == "global" and self.scope_key != "global":
            raise ValueError("global scope entries must use scope_key='global'")
        if self.scope_kind == "asset" and self.scope_key == "global":
            raise ValueError("asset scope entries require a concrete scope_key")
        return self


class ProviderCaveat(ContractModel):
    provider_id: NonEmptyStr
    note: NonEmptyStr


class MemoryQuery(ContractModel):
    """Typed memory recall request used by reviewer and adjudicator contexts."""

    job: JobContext
    stage: NonEmptyStr
    query_text: NonEmptyStr
    candidate_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    prompt_variant_ids: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    disagreement_bucket: str | None = None
    glossary_misses: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    numbers_dates: tuple[str, ...] = ()
    failure_tags: tuple[str, ...] = ()
    escalation_reasons: tuple[str, ...] = ()
    media_key: str | None = None
    max_items: int = Field(default=10, ge=1, le=100)


class MemoryBundle(ContractModel):
    """Read-only memory slices passed into review and adjudication."""

    semantic_memory: tuple[MemoryEntry, ...] = ()
    episodic_memory: tuple[MemoryEntry, ...] = ()
    glossary: tuple[MemoryEntry, ...] = ()
    rules: tuple[MemoryEntry, ...] = ()
    procedural_memory: tuple[MemoryEntry, ...] = ()
    provider_caveats: tuple[ProviderCaveat, ...] = ()


class MemoryWrite(ContractModel):
    kind: MemoryKind
    memory_subtype: MemorySubtype | None = None
    content: NonEmptyStr
    scope_kind: MemoryScopeKind = "global"
    scope_key: NonEmptyStr = "global"
    updated_at: datetime = Field(default_factory=utc_now)
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_ref: str | None = None
    origin_scope_kind: MemoryScopeKind | None = None
    origin_scope_key: NonEmptyStr | None = None
    promotion_status: PromotionStatus = "not_promoted"
    evidence_count: int = Field(default=0, ge=0)
    supporting_run_count: int = Field(default=0, ge=0)
    supporting_asset_count: int = Field(default=0, ge=0)
    supporting_project_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    promotion_ref: str | None = None
    quality_gate_status: QualityGateStatus = "not_evaluated"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope_key(self) -> MemoryWrite:
        if self.scope_kind == "global" and self.scope_key != "global":
            raise ValueError("global scope writes must use scope_key='global'")
        if self.scope_kind == "asset" and self.scope_key == "global":
            raise ValueError("asset scope writes require a concrete scope_key")
        return self


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
    source_tenant_id: str | None = None
    source_project_id: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    scope_kind: MemoryScopeKind | None = None
    scope_key: str | None = None
    semantic_memory_ids: tuple[str, ...] = ()
    episodic_memory_ids: tuple[str, ...] = ()
    procedural_memory_ids: tuple[str, ...] = ()
    skipped_dedupe_keys: tuple[str, ...] = ()
    procedural_write_count: int = Field(default=0, ge=0)


class PromptChange(ContractModel):
    section: NonEmptyStr
    instruction: NonEmptyStr


class PromptCompatibilityTuple(ContractModel):
    prompt_family: PromptFamily
    model_id: NonEmptyStr
    prompt_variant_id: NonEmptyStr
    base_prompt_version: NonEmptyStr
    source_language: NonEmptyStr
    target_language: NonEmptyStr
    scope_kind: MemoryScopeKind
    scope_key: NonEmptyStr


class ProposalAggregateMetrics(ContractModel):
    run_count: int = Field(default=0, ge=0)
    primary_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness_judge: float = Field(default=0.0, ge=0.0, le=1.0)
    segment_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    numeric_date_unit_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    glossary_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    omission_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    addition_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    fluency_judge: float | None = Field(default=None, ge=0.0, le=1.0)
    severe_failure_bucket_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class StrongerGraderScore(ContractModel):
    grader_id: NonEmptyStr
    grader_version: NonEmptyStr
    metric_family: NonEmptyStr = "stronger_grader"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionGateOutcome(ContractModel):
    heuristic_gate_pass: bool = False
    stronger_grader_pass: bool = False
    human_support_pass: bool = False
    rollback_signal_present: bool = False
    material_disagreement: bool = False
    eligible_for_pair_promotion: bool = False
    quality_gate_status: QualityGateStatus = "not_evaluated"
    notes: tuple[str, ...] = ()


class ResolvedTranslationPrompt(ContractModel):
    """Prompt resolver output used by translation-generation boundaries."""

    prompt_variant_id: NonEmptyStr
    base_prompt_version: NonEmptyStr
    effective_prompt_version: NonEmptyStr
    resolution_mode: PromptResolutionMode = "control"
    selected_proposal_id: str | None = None
    scope_kind: MemoryScopeKind = "project_pair"
    scope_key: NonEmptyStr
    instructions: tuple[str, ...] = ()
    applied_proposal_refs: tuple[str, ...] = ()


class PromptEvolutionProposal(ContractModel):
    """Prompt-improvement proposal derived from evaluated outcomes."""

    proposal_id: NonEmptyStr
    job_id: NonEmptyStr
    source_consolidation_id: NonEmptyStr
    prompt_family: PromptFamily
    target_model_id: NonEmptyStr
    target_prompt_version: str | None = None
    target_prompt_variant_id: str | None = None
    base_prompt_version: str | None = None
    compatibility: PromptCompatibilityTuple | None = None
    status: PromptEvolutionStatus = "proposed"
    rationale: NonEmptyStr
    suggested_changes: tuple[PromptChange, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    canary_run_count: int = Field(default=0, ge=0)
    control_run_count: int = Field(default=0, ge=0)
    canary_metrics: ProposalAggregateMetrics = Field(default_factory=ProposalAggregateMetrics)
    control_metrics: ProposalAggregateMetrics = Field(default_factory=ProposalAggregateMetrics)
    canary_stronger_grader: StrongerGraderScore | None = None
    control_stronger_grader: StrongerGraderScore | None = None
    promotion_status: PromotionStatus = "candidate"
    gate_outcome: PromotionGateOutcome | None = None
    rollback_reason: str | None = None
    activation_mode: str | None = None
    auto_activate: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, payload: object) -> object:
        if not isinstance(payload, dict):
            return payload
        data = dict(payload)
        status = data.get("status")
        rollback_reason = data.get("rollback_reason")
        if status == "approved":
            data["status"] = "active"
        elif status == "rejected":
            data["status"] = "rolled_back"
            data["rollback_reason"] = rollback_reason or "legacy rejected proposal"
        compatibility = data.get("compatibility")
        metadata = data.get("metadata")
        if compatibility is None and isinstance(metadata, dict):
            prompt_variant_id = data.get("target_prompt_variant_id")
            base_prompt_version = data.get("base_prompt_version") or data.get(
                "target_prompt_version"
            )
            source_language = _metadata_string(metadata, "source_language")
            target_language = _metadata_string(metadata, "target_language")
            scope_kind = _metadata_scope_kind(metadata.get("scope_kind")) or "project_pair"
            scope_key = _metadata_string(metadata, "scope_key") or (
                f"{source_language}::{target_language}"
                if source_language is not None and target_language is not None
                else None
            )
            if all(
                value is not None
                for value in (
                    prompt_variant_id,
                    base_prompt_version,
                    source_language,
                    target_language,
                    scope_kind,
                    scope_key,
                )
            ):
                data["compatibility"] = PromptCompatibilityTuple(
                    prompt_family=data["prompt_family"],
                    model_id=data["target_model_id"],
                    prompt_variant_id=prompt_variant_id or "",
                    base_prompt_version=base_prompt_version or "",
                    source_language=source_language or "",
                    target_language=target_language or "",
                    scope_kind=scope_kind or "pair",
                    scope_key=scope_key or "",
                )
        return data

    @model_validator(mode="after")
    def validate_compatibility(self) -> PromptEvolutionProposal:
        if self.prompt_family != "translation" and (
            self.auto_activate is True
            or self.activation_mode == "auto_activate_eligible"
            or self.status in {"canary", "active"}
        ):
            raise ValueError("only translation prompt proposals can auto-activate")
        if self.status == "rolled_back" and not self.rollback_reason:
            raise ValueError("rolled_back prompt proposals require rollback_reason")
        if self.compatibility is not None:
            if self.compatibility.prompt_family != self.prompt_family:
                raise ValueError("proposal compatibility must match prompt_family")
            if self.compatibility.model_id != self.target_model_id:
                raise ValueError("proposal compatibility must match target_model_id")
        return self


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _metadata_scope_kind(value: object) -> MemoryScopeKind | None:
    if value in {
        "asset",
        "project_pair",
        "pair",
        "source_language",
        "target_language",
        "global",
    }:
        return value  # type: ignore[return-value]
    return None
