"""Canonical memory models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from .base import ContractModel
from .jobs import JobContext

NonEmptyStr = Annotated[str, Field(min_length=1)]
MemoryKind = Literal["semantic", "episodic", "glossary", "rule", "procedural"]
ConsolidationStatus = Literal["pending", "consolidated", "skipped", "failed"]


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
    semantic_writes: tuple[MemoryWrite, ...] = ()
    episodic_writes: tuple[MemoryWrite, ...] = ()
    procedural_writes: tuple[MemoryWrite, ...] = ()
    dedupe_keys: tuple[str, ...] = ()
    consolidation_status: ConsolidationStatus = "pending"
