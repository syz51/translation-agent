"""Asset metadata, context, and relation models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from .base import ContractModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
AssetRelationKind = Literal[
    "same_series",
    "same_franchise",
    "same_speaker_cluster",
    "shares_style_profile",
    "derived_from_reference",
    "shares_glossary_with",
]
MetadataConfidence = Literal["low", "medium", "high"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class AssetContextInput(ContractModel):
    """Input metadata used to enrich a resolved asset identity."""

    canonical_title: str | None = None
    content_type: str | None = None
    series_id: str | None = None
    season_number: int | None = Field(default=None, ge=0)
    episode_number: int | None = Field(default=None, ge=0)
    franchise_id: str | None = None
    channel_id: str | None = None
    speaker_ids: tuple[str, ...] = ()
    topic_tags: tuple[str, ...] = ()
    style_profile_id: str | None = None
    metadata_confidence: MetadataConfidence = "medium"
    metadata_sources: tuple[str, ...] = ()
    inferred_from: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssetContext(AssetContextInput):
    """Persisted asset-context record keyed by durable media identity."""

    media_key: NonEmptyStr
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AssetRelation(ContractModel):
    """Typed cross-asset linkage used for widening and recall."""

    relation_id: NonEmptyStr
    src_media_key: NonEmptyStr
    dst_media_key: NonEmptyStr
    relation_kind: AssetRelationKind
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
