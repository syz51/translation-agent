"""Ingest node for the deterministic dry-run workflow."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path
from typing import cast

from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.models import AssetContext, AssetRelation, HistoricalRunLink
from translation_agent.storage.operational import OperationalStore

_EPISODE_RE = re.compile(
    r"^(?P<series>.+?)[\s._-]+s(?P<season>\d{1,2})e(?P<episode>\d{1,3})(?:[\s._-]+(?P<title>.+))?$",
    re.IGNORECASE,
)
_TAG_STOPWORDS = frozenset({"1080p", "2160p", "4k", "h264", "hevc", "web", "x264"})


def ingest_job(state: GraphState, runtime: WorkflowRuntime) -> dict[str, object]:
    """Resolve the initial run facts into ref-only graph state."""

    job = state.job
    derived_relations: tuple[AssetRelation, ...] = ()
    if hasattr(runtime.run_store, "resolve_asset"):
        run_store = cast("OperationalStore", runtime.run_store)
        asset = run_store.resolve_asset(
            asset_id=state.job.asset_id,
            media_fingerprint=state.job.media_fingerprint,
            first_seen_run_id=state.run_id,
            source_language=state.job.source_language,
            target_language=state.job.target_language,
        )
        job = state.job.model_copy(
            update={
                "asset_id": asset.asset_id,
                "media_fingerprint": asset.media_fingerprint,
                "media_key": asset.media_key,
            }
        )
        existing_context = run_store.get_asset_context(asset.media_key)
        resolved_context = _derive_asset_context(
            job=job,
            source_path=job.source_video_ref,
            existing=existing_context,
        )
        if resolved_context is not None:
            saved_context = run_store.save_asset_context(resolved_context)
            derived_relations = _upsert_related_asset_contexts(run_store, saved_context)
            job = job.model_copy(update={"asset_context": saved_context})
        run_store.upsert_historical_run_link(
            HistoricalRunLink(
                run_id=state.run_id,
                media_key=asset.media_key,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                project_id=job.project_id,
                source_language=job.source_language,
                target_language=job.target_language,
                created_at=job.created_at,
            )
        )

    return {
        "current_stage": "ingest",
        "job": job,
        "source_artifact_ref": runtime.source_artifact_ref,
        "routing_facts": state.routing_facts
        + (
            RoutingFact(
                stage="ingest",
                fact_type="job_initialized",
                value=state.job.job_id,
                source_ref=runtime.source_artifact_ref,
            ),
            RoutingFact(
                stage="ingest",
                fact_type="scenario",
                value=runtime.scenario,
                source_ref=runtime.source_artifact_ref,
            ),
            RoutingFact(
                stage="ingest",
                fact_type="media_key_resolved",
                value=job.media_key,
                source_ref=runtime.source_artifact_ref,
            ),
        )
        + (
            (
                RoutingFact(
                    stage="ingest",
                    fact_type="asset_context_enriched",
                    value=job.asset_context.media_key,
                    source_ref=runtime.source_artifact_ref,
                ),
            )
            if job.asset_context is not None
            else ()
        )
        + tuple(
            RoutingFact(
                stage="ingest",
                fact_type="asset_relation_derived",
                value=relation.relation_id,
                source_ref=runtime.source_artifact_ref,
            )
            for relation in derived_relations
        ),
    }


def _derive_asset_context(
    *,
    job,
    source_path: str,
    existing: AssetContext | None,
) -> AssetContext:
    current = job.asset_context
    filename = Path(source_path).stem
    episode_match = _EPISODE_RE.match(filename)
    metadata_sources = _merge_unique(
        getattr(existing, "metadata_sources", ()),
        getattr(current, "metadata_sources", ()),
    )
    inferred_from = _merge_unique(
        getattr(existing, "inferred_from", ()),
        getattr(current, "inferred_from", ()),
    )
    metadata = {
        **(existing.metadata if existing is not None else {}),
        **(current.metadata if current is not None else {}),
    }
    base: dict[str, object] = (
        existing.model_dump(mode="python") if existing is not None else {"media_key": job.media_key}
    )
    if current is not None:
        base.update(_asset_context_update(current))
    derived_title = (
        _clean_title(episode_match.group("title"))
        if episode_match and episode_match.group("title")
        else _clean_title(filename)
    )
    if not base.get("canonical_title"):
        base["canonical_title"] = derived_title
        inferred_from = _merge_unique(inferred_from, ("canonical_title:source_filename",))
    if episode_match:
        if not base.get("content_type"):
            base["content_type"] = "episode"
        if not base.get("series_id"):
            base["series_id"] = _slugify(episode_match.group("series"))
        if base.get("season_number") is None:
            base["season_number"] = int(episode_match.group("season"))
        if base.get("episode_number") is None:
            base["episode_number"] = int(episode_match.group("episode"))
        metadata_sources = _merge_unique(metadata_sources, ("ingest_filename",))
    elif not base.get("content_type"):
        base["content_type"] = _content_type_from_filename(filename)
    if not base.get("style_profile_id") and isinstance(base.get("content_type"), str):
        base["style_profile_id"] = _slugify(
            f"{job.project_id}-{base['content_type']}-{job.target_language}"
        )
        inferred_from = _merge_unique(inferred_from, ("style_profile_id:project-content",))
    if not base.get("topic_tags"):
        derived_tags = _topic_tags_from_filename(filename)
        if derived_tags:
            base["topic_tags"] = derived_tags
            inferred_from = _merge_unique(inferred_from, ("topic_tags:source_filename",))
    metadata_sources = _merge_unique(metadata_sources, ("ingest",))
    metadata["derived_source_name"] = filename
    if base.get("metadata_confidence") in {None, ""}:
        base["metadata_confidence"] = "medium" if episode_match else "low"
    base["metadata_sources"] = metadata_sources
    base["inferred_from"] = inferred_from
    base["metadata"] = metadata
    return AssetContext.model_validate(base)


def _asset_context_update(asset_context: AssetContext) -> dict[str, object]:
    return {
        key: value
        for key, value in asset_context.model_dump(mode="python").items()
        if key not in {"media_key", "created_at", "updated_at"}
    }


def _upsert_related_asset_contexts(
    run_store: OperationalStore,
    asset_context: AssetContext,
) -> tuple[AssetRelation, ...]:
    relations: list[AssetRelation] = []
    for other in run_store.list_asset_contexts():
        if other.media_key == asset_context.media_key:
            continue
        relation_specs: list[tuple[str, float]] = []
        if asset_context.series_id and asset_context.series_id == other.series_id:
            relation_specs.append(("same_series", 0.93))
        if asset_context.franchise_id and asset_context.franchise_id == other.franchise_id:
            relation_specs.append(("same_franchise", 0.9))
        if (
            asset_context.style_profile_id
            and asset_context.style_profile_id == other.style_profile_id
        ):
            relation_specs.append(("shares_style_profile", 0.82))
        shared_speakers = set(asset_context.speaker_ids) & set(other.speaker_ids)
        if shared_speakers:
            relation_specs.append(
                ("same_speaker_cluster", min(0.96, 0.74 + len(shared_speakers) * 0.05))
            )
        for relation_kind, confidence in relation_specs:
            relation = AssetRelation(
                relation_id=_relation_id(
                    relation_kind=relation_kind,
                    left_media_key=asset_context.media_key,
                    right_media_key=other.media_key,
                ),
                src_media_key=asset_context.media_key,
                dst_media_key=other.media_key,
                relation_kind=relation_kind,  # type: ignore[arg-type]
                confidence=round(confidence, 4),
                metadata={"derived_by": "ingest_enrichment"},
            )
            relations.append(run_store.save_asset_relation(relation))
    return tuple(relations)


def _relation_id(*, relation_kind: str, left_media_key: str, right_media_key: str) -> str:
    ordered = "::".join(sorted((left_media_key, right_media_key)))
    digest = sha256(f"{relation_kind}:{ordered}".encode()).hexdigest()[:12]
    return f"{relation_kind}:{digest}"


def _content_type_from_filename(filename: str) -> str:
    lowered = filename.casefold()
    if "trailer" in lowered:
        return "trailer"
    if "clip" in lowered:
        return "clip"
    if "podcast" in lowered:
        return "podcast"
    return "video"


def _topic_tags_from_filename(filename: str) -> tuple[str, ...]:
    tags = []
    for token in _slugify(filename).split("-"):
        if len(token) < 4 or token in _TAG_STOPWORDS or token.startswith("s0"):
            continue
        if token.isdigit():
            continue
        tags.append(token)
    return tuple(dict.fromkeys(tags[:4]))


def _clean_title(value: str) -> str:
    cleaned = re.sub(r"[\s._-]+", " ", value).strip()
    return cleaned or "untitled"


def _slugify(value: str) -> str:
    return "-".join(part for part in re.split(r"[^A-Za-z0-9]+", value.casefold()) if part)


def _merge_unique(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*left, *right)))
