"""Reopenable translation review and approval helpers."""

from __future__ import annotations

import getpass
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from translation_agent.graph import GraphState, build_phase_two_runtime
from translation_agent.graph.state import RoutingFact
from translation_agent.memory.staging import (
    apply_scope_defaults,
    batch_metadata_for_job,
    project_pair_scope_key,
)
from translation_agent.models import (
    FailureTag,
    FinalTranslationDecision,
    HumanApprovalRecord,
    HumanReviewedCandidateContext,
    HumanReviewResolutionRecord,
    HumanSupervisionKind,
    JobContext,
    MemoryWrite,
    MemoryWriteBatch,
    PromptChange,
    PromptCompatibilityTuple,
    PromptEvolutionProposal,
    ReviewBundle,
    Segment,
    StructuredEvidence,
    TranscriptApprovalLearningEvent,
    TranscriptCandidate,
    TranscriptProviderQualityStats,
    TranslationCandidate,
    TranslationFeedbackStats,
)
from translation_agent.models.jobs import ReferenceMode
from translation_agent.nodes.common import (
    approval_record_key,
    memory_batch_key,
    memory_consolidation_key,
    operational_job_key,
    read_model_artifact,
    review_preview_key,
    review_resolution_key,
    select_transcript_candidates,
    select_translation_candidates,
    transcript_approval_learning_key,
    transcript_decision_key,
    transcript_investigation_key,
    translation_decision_key,
    translation_investigation_key,
    translation_review_key,
    write_model_artifact,
)
from translation_agent.nodes.reference_evaluation import update_historical_run_link
from translation_agent.observability import NoOpTraceSink
from translation_agent.publish.outputs import publish_outputs
from translation_agent.storage import LocalBlobStore, OperationalStore, job_path
from translation_agent.subtitles import render_translation_srt

_HARD_CONTRADICTION_DIMENSIONS = {"meaning", "entity", "number_date_unit", "coverage"}
_NO_OVERLAPPING_EXCERPT = "[no overlapping excerpt]"
_FAILURE_TAG_INSTRUCTIONS: dict[str, str] = {
    "honorific_leak": "Forbid carrying Korean honorific suffixes into Chinese output.",
    "romanization_leak": (
        "Prefer natural Chinese renderings over raw romanization unless corroborated."
    ),
    "ungrounded_addition": "Prefer conservative translation under transcript uncertainty.",
    "late_run_degeneration": "Forbid improvisational expansion in late cues.",
    "subtitle_gibberish": "Reject mixed-script junk and unresolved transliterations.",
}


@dataclass(frozen=True, slots=True)
class ReviewSessionContext:
    store: OperationalStore
    blob_store: LocalBlobStore
    runtime: Any
    run_record: Any
    job: JobContext
    source_artifact_ref: str


def build_review_payload(
    run_id: str,
    *,
    store: OperationalStore,
    blob_store: LocalBlobStore,
) -> dict[str, Any]:
    """Return a machine-readable review payload for one run."""

    context = _load_review_session(run_id, store=store, blob_store=blob_store)
    runtime = context.runtime
    storage_job_id = operational_job_key(context.job)
    translation_candidates = select_translation_candidates(
        runtime,
        job=context.job,
        candidate_ids=tuple(
            candidate.candidate_id
            for candidate in runtime.decision_store.list_translation_candidates(
                context.job.job_id,
                storage_job_id=storage_job_id,
            )
        ),
    )
    transcript_candidates = {
        candidate.candidate_id: candidate
        for candidate in select_transcript_candidates(
            runtime,
            job=context.job,
            candidate_ids=tuple(
                candidate.candidate_id
                for candidate in runtime.decision_store.list_transcript_candidates(
                    context.job.job_id,
                    storage_job_id=storage_job_id,
                )
            ),
        )
    }
    output_data = _dict_payload(context.run_record.output_data)
    translation_decision = _optional_model(
        runtime,
        output_data.get("translation_decision_ref"),
        model_type=FinalTranslationDecision,
    )
    if translation_decision is None and runtime.blob_store.exists(
        translation_decision_key(context.job)
    ):
        translation_decision = read_model_artifact(
            runtime,
            translation_decision_key(context.job),
            FinalTranslationDecision,
        )
    transcript_decision = _optional_json(
        runtime,
        output_data.get("transcript_decision_ref") or transcript_decision_key(context.job),
    )
    transcript_investigation = _optional_json(
        runtime,
        output_data.get("transcript_investigation_ref")
        or transcript_investigation_key(context.job),
    )
    translation_investigation = _optional_json(
        runtime,
        output_data.get("translation_investigation_ref")
        or translation_investigation_key(context.job),
    )
    approval = _optional_json(runtime, approval_record_key(context.job))
    resolution = _load_resolution_payload(context)

    preferred_candidate_id = None
    if translation_decision is not None:
        preferred_candidate_id = (
            translation_decision.adjudication_scorecard.preferred_candidate_id
            or translation_decision.winner_candidate_id
        )
    ranked_candidates = sorted(
        translation_candidates,
        key=lambda candidate: (
            0 if candidate.candidate_id == preferred_candidate_id else 1,
            candidate.source_transcript_candidate_id or "",
            candidate.prompt_variant_id,
            candidate.candidate_id,
        ),
    )
    candidate_payloads: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked_candidates, start=1):
        transcript = transcript_candidates.get(candidate.source_transcript_candidate_id or "")
        previews = _render_candidate_previews(
            blob_store=context.blob_store,
            job=context.job,
            translation_candidate=candidate,
            transcript_candidate=transcript,
        )
        candidate_payloads.append(
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "source_transcript_candidate_id": candidate.source_transcript_candidate_id,
                "model_id": candidate.model_id,
                "prompt_variant_id": candidate.prompt_variant_id,
                "prompt_version": candidate.prompt_version,
                "language": candidate.language,
                "combo_key": _candidate_combo_key(
                    context.job,
                    candidate,
                    transcript,
                ),
                "translation_text_preview": candidate.full_text[:400],
                "translation_ref": job_path(
                    context.job, "candidates", "translations", f"{candidate.candidate_id}.json"
                ),
                "translation_preview_json_path": str(previews["translation_json_path"]),
                "translation_preview_srt_path": str(previews["translation_srt_path"]),
                "source_transcript": {
                    "candidate_id": transcript.candidate_id if transcript is not None else None,
                    "provider_id": transcript.provider_id if transcript is not None else None,
                    "text_preview": transcript.full_text[:400] if transcript is not None else None,
                    "transcript_preview_json_path": str(previews["transcript_json_path"])
                    if previews["transcript_json_path"] is not None
                    else None,
                    "transcript_preview_txt_path": str(previews["transcript_txt_path"])
                    if previews["transcript_txt_path"] is not None
                    else None,
                },
            }
        )

    machine_review_refs = []
    translation_reviews: tuple[ReviewBundle, ...] = ()
    if translation_decision is not None:
        machine_review_refs = [
            translation_review_key(context.job, review_id)
            for review_id in translation_decision.review_refs
        ]
        translation_reviews = tuple(
            read_model_artifact(
                runtime,
                translation_review_key(context.job, review_id),
                ReviewBundle,
            )
            for review_id in translation_decision.review_refs
            if runtime.blob_store.exists(translation_review_key(context.job, review_id))
        )

    contradiction_summary = _build_contradiction_summary(
        translation_candidates=translation_candidates,
        reviews=translation_reviews,
        preferred_candidate_id=preferred_candidate_id,
        candidate_rank_map={
            candidate.candidate_id: rank
            for rank, candidate in enumerate(ranked_candidates, start=1)
        },
    )
    candidate_summaries = contradiction_summary["candidate_summaries"]
    for payload in candidate_payloads:
        candidate_summary = candidate_summaries.get(str(payload["candidate_id"]), {})
        payload["contradiction_count"] = candidate_summary.get("contradiction_count", 0)
        payload["blocking_hard_contradiction_count"] = candidate_summary.get(
            "blocking_hard_contradiction_count",
            0,
        )
        payload["reviewer_preferences"] = candidate_summary.get("reviewer_preferences", [])
        payload["contradictions"] = candidate_summary.get("contradictions", [])

    recommended_failure_tags = _recommended_failure_tags(
        candidate_payloads=candidate_payloads,
        contradiction_summary=contradiction_summary,
        reviews=translation_reviews,
    )
    return {
        "run_id": run_id,
        "job_id": context.job.job_id,
        "status": context.run_record.status,
        "review_required_stage": output_data.get("review_required_stage"),
        "review_available": output_data.get("review_required_stage") == "translation",
        "resolution_ref": resolution.get("resolution_ref") if resolution else None,
        "resolution_kind": resolution.get("resolution_kind") if resolution else None,
        "failure_tags": list(resolution.get("failure_tags", [])) if resolution else [],
        "residual_failure_tags": (
            list(resolution.get("residual_failure_tags", [])) if resolution else []
        ),
        "recommended_failure_tags": list(recommended_failure_tags),
        "summary": (
            translation_decision.rationale_summary
            if translation_decision is not None
            else output_data.get("failure_summary")
        ),
        "candidate_count": len(candidate_payloads),
        "candidates": candidate_payloads,
        "transcript_review_summary": {
            "decision_ref": output_data.get("transcript_decision_ref")
            or transcript_decision_key(context.job),
            "decision": transcript_decision,
            "investigation_ref": output_data.get("transcript_investigation_ref")
            or transcript_investigation_key(context.job),
            "investigation": transcript_investigation,
        },
        "translation_review_summary": {
            "decision_ref": output_data.get("translation_decision_ref")
            or translation_decision_key(context.job),
            "decision": (
                translation_decision.model_dump(mode="json")
                if translation_decision is not None
                else None
            ),
            "investigation_ref": output_data.get("translation_investigation_ref")
            or translation_investigation_key(context.job),
            "investigation": translation_investigation,
            "machine_review_refs": machine_review_refs,
        },
        "human_review_summary": contradiction_summary["summary"],
        "review_diffs": contradiction_summary["review_diffs"],
        "approval": approval,
        "resume_commands": [
            f"uv run translation-agent review-job {run_id}",
            f"uv run translation-agent approve-review {run_id} --candidate-id <candidate-id>",
            (
                "uv run translation-agent resolve-review "
                f"{run_id} --resolution approved_best_available --candidate-id <candidate-id>"
            ),
        ],
    }


def resolve_translation_review(
    run_id: str,
    *,
    resolution_kind: HumanSupervisionKind,
    candidate_id: str | None,
    failure_tags: tuple[FailureTag, ...] = (),
    approved_by: str | None,
    note: str | None,
    store: OperationalStore,
    blob_store: LocalBlobStore,
) -> dict[str, Any]:
    """Resolve one pending translation review and republish canonical artifacts in place."""

    context = _load_review_session(run_id, store=store, blob_store=blob_store)
    output_data = _dict_payload(context.run_record.output_data)
    if output_data.get("review_required_stage") != "translation":
        raise ValueError("run does not have a pending translation review")
    resolution_ref = review_resolution_key(context.job)
    if context.blob_store.exists(resolution_ref):
        existing = _optional_json(context.runtime, resolution_ref)
        raise ValueError(f"run already has a resolution record: {existing}")
    approval_ref = approval_record_key(context.job)
    if resolution_kind in {
        "approved_good",
        "approved_best_available",
    } and context.blob_store.exists(approval_ref):
        existing = _optional_json(context.runtime, approval_ref)
        raise ValueError(f"run already has an approval record: {existing}")

    runtime = context.runtime
    storage_job_id = operational_job_key(context.job)
    translation_candidates = {
        candidate.candidate_id: candidate
        for candidate in runtime.decision_store.list_translation_candidates(
            context.job.job_id,
            storage_job_id=storage_job_id,
        )
    }
    transcript_candidates = {
        candidate.candidate_id: candidate
        for candidate in runtime.decision_store.list_transcript_candidates(
            context.job.job_id,
            storage_job_id=storage_job_id,
        )
    }
    reviewed_candidates = _reviewed_candidate_contexts(
        job=context.job,
        translation_candidates=translation_candidates,
        transcript_candidates=transcript_candidates,
    )
    _validate_resolution_inputs(
        resolution_kind=resolution_kind,
        candidate_id=candidate_id,
        failure_tags=failure_tags,
        translation_candidates=translation_candidates,
    )
    approved_candidate = (
        translation_candidates[candidate_id]
        if candidate_id is not None and candidate_id in translation_candidates
        else None
    )
    approved_transcript = (
        transcript_candidates.get(approved_candidate.source_transcript_candidate_id or "")
        if approved_candidate is not None
        else None
    )
    if approved_candidate is not None and approved_transcript is None:
        raise ValueError("approved source transcript candidate was not found")

    from translation_agent.models import FinalTranslationDecision

    machine_translation_decision = read_model_artifact(
        runtime,
        output_data.get("translation_decision_ref") or translation_decision_key(context.job),
        FinalTranslationDecision,
    )
    resolved_at = datetime.now(UTC)
    approved_by = _normalized_approved_by(approved_by)
    normalized_failure_tags = tuple(dict.fromkeys(failure_tags))
    residual_failure_tags = normalized_failure_tags if resolution_kind != "approved_good" else ()
    resolution_record = HumanReviewResolutionRecord(
        run_id=run_id,
        job_id=context.job.job_id,
        resolution_kind=resolution_kind,
        candidate_id=approved_candidate.candidate_id if approved_candidate is not None else None,
        approved_by=approved_by,
        note=note or "",
        failure_tags=normalized_failure_tags,
        residual_failure_tags=residual_failure_tags,
        transcript_provider_id=approved_transcript.provider_id if approved_transcript else None,
        source_transcript_candidate_id=(
            approved_transcript.candidate_id if approved_transcript is not None else None
        ),
        model_id=approved_candidate.model_id if approved_candidate is not None else None,
        prompt_variant_id=(
            approved_candidate.prompt_variant_id if approved_candidate is not None else None
        ),
        prompt_version=approved_candidate.prompt_version
        if approved_candidate is not None
        else None,
        base_prompt_version=(
            _candidate_base_prompt_version(approved_candidate) if approved_candidate else None
        ),
        combo_key=(
            _candidate_combo_key(context.job, approved_candidate, approved_transcript)
            if approved_candidate is not None and approved_transcript is not None
            else None
        ),
        source_language=context.job.source_language,
        target_language=context.job.target_language,
        tenant_id=context.job.tenant_id,
        project_id=context.job.project_id,
        media_key=context.job.media_key,
        machine_translation_decision_ref=output_data.get("translation_decision_ref")
        or translation_decision_key(context.job),
        machine_investigation_ref=output_data.get("translation_investigation_ref")
        or (
            translation_investigation_key(context.job)
            if runtime.blob_store.exists(translation_investigation_key(context.job))
            else None
        ),
        machine_review_refs=tuple(
            translation_review_key(context.job, review_id)
            for review_id in machine_translation_decision.review_refs
        ),
        reviewed_candidates=reviewed_candidates,
        resolved_at=resolved_at,
    )
    write_model_artifact(runtime, resolution_ref, resolution_record)
    store.save_human_review_resolution(resolution_record)

    approval_record = None
    learning_ref = None
    if approved_candidate is not None and approved_transcript is not None:
        approval_record = HumanApprovalRecord(
            run_id=run_id,
            job_id=context.job.job_id,
            approved_candidate_id=approved_candidate.candidate_id,
            approved_source_transcript_candidate_id=approved_transcript.candidate_id,
            approved_by=approved_by,
            note=note or "",
            approved_at=resolved_at,
            machine_translation_decision_ref=resolution_record.machine_translation_decision_ref,
            machine_investigation_ref=resolution_record.machine_investigation_ref,
            machine_review_refs=resolution_record.machine_review_refs,
        )
        write_model_artifact(runtime, approval_ref, approval_record)
        learning_ref = transcript_approval_learning_key(context.job)
        learning_event = TranscriptApprovalLearningEvent(
            run_id=run_id,
            job_id=context.job.job_id,
            approved_translation_candidate_id=approved_candidate.candidate_id,
            approved_source_transcript_candidate_id=approved_transcript.candidate_id,
            transcript_provider_id=approved_transcript.provider_id,
            source_language=context.job.source_language,
            target_language=context.job.target_language,
            tenant_id=context.job.tenant_id,
            project_id=context.job.project_id,
            media_key=context.job.media_key,
            linked_translation_approval_ref=approval_ref,
            supervision_kind=resolution_kind,
            supervision_strength=resolution_record.supervision_strength,
            created_at=resolved_at,
        )
        write_model_artifact(runtime, learning_ref, learning_event)

    _update_provider_quality_stats(
        store=store,
        reviewed_candidates=reviewed_candidates,
        selected_provider_id=approved_transcript.provider_id if approved_transcript else None,
        job=context.job,
        resolution_kind=resolution_kind,
        resolved_at=resolved_at,
    )
    _update_translation_feedback_stats(
        store=store,
        resolution=resolution_record,
        resolved_at=resolved_at,
    )
    proposal_refs = _update_feedback_prompt_proposals(
        context=context,
        resolution=resolution_record,
        resolution_ref=resolution_ref,
    )
    memory_batch, consolidation_ref, routing_facts = _write_resolution_learning_memory(
        context=context,
        resolution=resolution_record,
        resolution_ref=resolution_ref,
        learning_ref=learning_ref,
    )

    state = _resolution_publish_state(
        context=context,
        output_data=output_data,
        approved_candidate=approved_candidate,
        approved_transcript=approved_transcript,
        resolution_ref=resolution_ref,
        resolution_kind=resolution_kind,
        failure_tags=normalized_failure_tags,
        residual_failure_tags=residual_failure_tags,
        approval_ref=approval_ref,
        learning_ref=learning_ref,
        memory_batch=memory_batch,
        routing_facts=routing_facts,
        improvement_proposal_refs=proposal_refs,
    )
    artifacts, manifest_ref = publish_outputs(state, runtime)
    update_historical_run_link(state, runtime, artifacts)
    published_refs = tuple(
        ref
        for ref in (
            manifest_ref,
            artifacts.final_transcript_ref,
            artifacts.final_translation_ref,
            artifacts.recoverable_translation_failure_ref,
            *artifacts.resolution_refs,
            *artifacts.approval_refs,
            *artifacts.learning_refs,
            *artifacts.reference_transcript_refs,
            *artifacts.evaluation_report_refs,
            *artifacts.regenerated_draft_refs,
            *artifacts.improvement_proposal_refs,
            *artifacts.scorecard_refs,
            *artifacts.trace_refs,
            *artifacts.export_refs,
            *artifacts.downstream_delivery_refs,
            *artifacts.memory_batch_refs,
            *artifacts.memory_consolidation_refs,
            *artifacts.prompt_evolution_refs,
        )
        if ref is not None
    )
    updated_output_data = {
        **output_data,
        "resolution_ref": resolution_ref,
        "resolution_kind": resolution_kind,
        "failure_tags": list(normalized_failure_tags),
        "residual_failure_tags": list(residual_failure_tags),
        "approval_ref": approval_ref if approval_record is not None else None,
        "approved_candidate_id": approved_candidate.candidate_id if approved_candidate else None,
        "approved_source_transcript_candidate_id": (
            approved_transcript.candidate_id if approved_transcript is not None else None
        ),
        "review_required_stage": "translation",
        "human_review_required": False,
        "translation_failed": resolution_kind == "rejected_all",
        "final_stage": "resolve_review",
        "published_artifact_refs": list(published_refs),
        "final_transcript_ref": artifacts.final_transcript_ref,
        "final_translation_ref": artifacts.final_translation_ref,
        "learning_ref": learning_ref,
        "improvement_proposal_refs": sorted(
            {
                *(str(ref) for ref in output_data.get("improvement_proposal_refs", [])),
                *proposal_refs,
            }
        ),
        "memory_batch_ids": sorted(
            {
                *(str(batch_id) for batch_id in output_data.get("memory_batch_ids", [])),
                memory_batch.batch_id,
            }
        ),
        "memory_consolidation_refs": sorted(
            {
                *(str(ref) for ref in output_data.get("memory_consolidation_refs", [])),
                consolidation_ref,
            }
        ),
    }
    store.update_run(
        run_id,
        status=(
            "completed_after_human_review"
            if resolution_kind in {"approved_good", "approved_best_available"}
            else "rejected_after_human_review"
        ),
        output_data=updated_output_data,
        error=None,
    )
    return {
        "run_id": run_id,
        "job_id": context.job.job_id,
        "status": (
            "completed_after_human_review"
            if resolution_kind in {"approved_good", "approved_best_available"}
            else "rejected_after_human_review"
        ),
        "resolution_ref": resolution_ref,
        "resolution_kind": resolution_kind,
        "failure_tags": list(normalized_failure_tags),
        "residual_failure_tags": list(residual_failure_tags),
        "approval_ref": approval_ref if approval_record is not None else None,
        "approved_candidate_id": approved_candidate.candidate_id if approved_candidate else None,
        "approved_source_transcript_candidate_id": (
            approved_transcript.candidate_id if approved_transcript is not None else None
        ),
        "final_transcript_ref": artifacts.final_transcript_ref,
        "final_translation_ref": artifacts.final_translation_ref,
        "default_output_path": (
            str(
                (
                    context.blob_store.root / job_path(context.job, "exports", "translation.srt")
                ).resolve()
            )
            if resolution_kind in {"approved_good", "approved_best_available"}
            else None
        ),
    }


def approve_translation_review(
    run_id: str,
    *,
    candidate_id: str,
    approved_by: str | None,
    note: str | None,
    store: OperationalStore,
    blob_store: LocalBlobStore,
) -> dict[str, Any]:
    """Backward-compatible alias for resolving a review as approved_good."""

    return resolve_translation_review(
        run_id,
        resolution_kind="approved_good",
        candidate_id=candidate_id,
        failure_tags=(),
        approved_by=approved_by,
        note=note,
        store=store,
        blob_store=blob_store,
    )


def _load_review_session(
    run_id: str,
    *,
    store: OperationalStore,
    blob_store: LocalBlobStore,
) -> ReviewSessionContext:
    run_record = store.get_run(run_id)
    if run_record is None:
        raise ValueError(f"unknown run_id: {run_id}")
    input_data = _dict_payload(run_record.input_data)
    source_artifact_ref = str(input_data.get("artifact_ref") or "")
    if not source_artifact_ref:
        raise ValueError("run is missing source artifact ref")
    job = _job_context_from_run(
        run_record=run_record,
        input_data=input_data,
        request_payload=json.loads(blob_store.read_bytes(source_artifact_ref).decode("utf-8")),
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=store,
        decision_store=store,
        memory_batch_store=store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_artifact_ref,
    )
    return ReviewSessionContext(
        store=store,
        blob_store=blob_store,
        runtime=runtime,
        run_record=run_record,
        job=job,
        source_artifact_ref=source_artifact_ref,
    )


def _job_context_from_run(
    *,
    run_record: Any,
    input_data: dict[str, Any],
    request_payload: dict[str, Any],
) -> JobContext:
    created_at_raw = request_payload.get("created_at") or run_record.created_at
    created_at = (
        created_at_raw
        if isinstance(created_at_raw, datetime)
        else datetime.fromisoformat(str(created_at_raw))
    )
    return JobContext(
        job_id=str(request_payload.get("job_id") or input_data.get("job_id") or run_record.run_id),
        tenant_id=str(request_payload.get("tenant_id") or "tenant-local"),
        project_id=str(request_payload.get("project_id") or "project-local"),
        source_video_ref=str(request_payload.get("source") or ""),
        target_language=str(request_payload.get("target_language") or "unknown"),
        source_language=str(request_payload.get("source_language") or "unknown"),
        requested_by=str(request_payload.get("requested_by") or "system@local"),
        created_at=created_at,
        profile_ref=request_payload.get("profile_ref"),
        asset_id=_optional_string(input_data.get("asset_id")),
        media_fingerprint=_optional_string(input_data.get("media_fingerprint")),
        media_key=str(input_data.get("media_key") or f"source-ref:{run_record.run_id}"),
        reference_transcript_source=_optional_string(
            request_payload.get("reference_transcript_source")
        ),
        reference_transcript_format=request_payload.get("reference_transcript_format"),
        reference_mode=cast(
            ReferenceMode,
            request_payload.get("reference_mode") or "none",
        ),
    )


def _render_candidate_previews(
    *,
    blob_store: LocalBlobStore,
    job: JobContext,
    translation_candidate: TranslationCandidate,
    transcript_candidate: TranscriptCandidate | None,
) -> dict[str, Path | None]:
    translation_json_ref = review_preview_key(job, translation_candidate.candidate_id, ".json")
    translation_srt_ref = review_preview_key(job, translation_candidate.candidate_id, ".srt")
    blob_store.put_bytes(
        translation_json_ref,
        (
            json.dumps(
                translation_candidate.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    blob_store.put_bytes(
        translation_srt_ref,
        render_translation_srt(translation_candidate).encode("utf-8"),
    )
    transcript_json_path: Path | None = None
    transcript_txt_path: Path | None = None
    if transcript_candidate is not None:
        transcript_json_ref = review_preview_key(
            job,
            f"{translation_candidate.candidate_id}-transcript",
            ".json",
        )
        transcript_txt_ref = review_preview_key(
            job,
            f"{translation_candidate.candidate_id}-transcript",
            ".txt",
        )
        transcript_json_path = blob_store.put_bytes(
            transcript_json_ref,
            (
                json.dumps(
                    transcript_candidate.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        ).path
        transcript_txt_path = blob_store.put_bytes(
            transcript_txt_ref,
            (transcript_candidate.full_text + "\n").encode("utf-8"),
        ).path
    return {
        "translation_json_path": (blob_store.root / translation_json_ref).resolve(),
        "translation_srt_path": (blob_store.root / translation_srt_ref).resolve(),
        "transcript_json_path": transcript_json_path.resolve()
        if transcript_json_path is not None
        else None,
        "transcript_txt_path": transcript_txt_path.resolve()
        if transcript_txt_path is not None
        else None,
    }


def _build_contradiction_summary(
    *,
    translation_candidates: list[TranslationCandidate],
    reviews: tuple[ReviewBundle, ...],
    preferred_candidate_id: str | None = None,
    candidate_rank_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    candidate_map = {candidate.candidate_id: candidate for candidate in translation_candidates}
    effective_rank_map = candidate_rank_map or {
        candidate.candidate_id: index
        for index, candidate in enumerate(translation_candidates, start=1)
    }
    candidate_summaries: dict[str, dict[str, Any]] = {
        candidate.candidate_id: {
            "candidate_id": candidate.candidate_id,
            "contradiction_count": 0,
            "blocking_hard_contradiction_count": 0,
            "reviewer_preferences": [],
            "contradictions": [],
        }
        for candidate in translation_candidates
    }
    contradiction_index: dict[
        tuple[str, str | None, str, str, str | None],
        dict[str, Any],
    ] = {}
    review_diff_groups: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    reviewer_preferences: list[dict[str, Any]] = []

    for review in reviews:
        top_choice = review.candidate_preferences[0] if review.candidate_preferences else None
        reviewer_preferences.append(
            {
                "reviewer_role": review.reviewer_role,
                "preferred_candidate_id": (
                    top_choice.candidate_id if top_choice is not None else None
                ),
                "confidence": review.confidence,
                "escalation_signal": review.escalation_signal,
            }
        )
        for preference in review.candidate_preferences:
            candidate_summary = candidate_summaries.get(preference.candidate_id)
            if candidate_summary is None:
                continue
            cast(list[dict[str, Any]], candidate_summary["reviewer_preferences"]).append(
                {
                    "reviewer_role": review.reviewer_role,
                    "rank": preference.rank,
                    "rationale": preference.rationale,
                    "confidence": review.confidence,
                }
            )
        for evidence in review.structured_evidence:
            if evidence.polarity != "refutes":
                continue
            candidate = candidate_map.get(evidence.candidate_id)
            if candidate is None:
                continue
            candidate_summary = candidate_summaries[evidence.candidate_id]
            _record_contradiction(
                contradiction_index=contradiction_index,
                candidate_summary=candidate_summary,
                candidate=candidate,
                review=review,
                evidence=evidence,
            )
            _record_review_diff_group(
                review_diff_groups=review_diff_groups,
                candidate=candidate,
                candidate_rank=effective_rank_map.get(candidate.candidate_id, 10**9),
                review=review,
                evidence=evidence,
            )

    for candidate_summary in candidate_summaries.values():
        contradictions = cast(list[dict[str, Any]], candidate_summary["contradictions"])
        contradictions.sort(
            key=lambda item: (
                _span_sort_key(cast(str | None, item["source_span_id"])),
                _severity_sort_key(cast(str, item["severity"])),
                cast(str, item["dimension"]),
            )
        )
        candidate_summary["contradiction_count"] = len(contradictions)
        candidate_summary["blocking_hard_contradiction_count"] = sum(
            1 for item in contradictions if cast(bool, item["blocking_hard_contradiction"])
        )

    review_diffs = _build_review_diffs(
        review_diff_groups=review_diff_groups,
        preferred_candidate_id=preferred_candidate_id,
        translation_candidates=translation_candidates,
        candidate_rank_map=effective_rank_map,
    )
    contradiction_count = sum(
        cast(int, candidate_summary["contradiction_count"])
        for candidate_summary in candidate_summaries.values()
    )
    blocking_hard_count = sum(
        cast(int, candidate_summary["blocking_hard_contradiction_count"])
        for candidate_summary in candidate_summaries.values()
    )
    return {
        "summary": {
            "contradiction_count": contradiction_count,
            "blocking_hard_contradiction_count": blocking_hard_count,
            "reviewer_preferences": reviewer_preferences,
        },
        "candidate_summaries": candidate_summaries,
        "review_diffs": review_diffs,
    }


def _record_contradiction(
    *,
    contradiction_index: dict[tuple[str, str | None, str, str, str | None], dict[str, Any]],
    candidate_summary: dict[str, Any],
    candidate: TranslationCandidate,
    review: ReviewBundle,
    evidence: StructuredEvidence,
) -> None:
    key = (
        evidence.candidate_id,
        evidence.source_span_id,
        evidence.dimension,
        evidence.evidence_text,
        evidence.normalized_value,
    )
    existing = contradiction_index.get(key)
    if existing is None:
        source_excerpt, target_excerpt = _segment_excerpts_for_span(
            candidate.segments,
            evidence.source_span_id,
        )
        existing = {
            "source_span_id": evidence.source_span_id,
            "time_range": _format_span_range(evidence.source_span_id),
            "dimension": evidence.dimension,
            "severity": evidence.severity,
            "evidence_text": evidence.evidence_text,
            "normalized_value": evidence.normalized_value,
            "reviewer_roles": [review.reviewer_role],
            "source_excerpt": source_excerpt,
            "target_excerpt": target_excerpt,
            "blocking_hard_contradiction": evidence.dimension in _HARD_CONTRADICTION_DIMENSIONS,
        }
        contradiction_index[key] = existing
        cast(list[dict[str, Any]], candidate_summary["contradictions"]).append(existing)
        return

    roles = cast(list[str], existing["reviewer_roles"])
    if review.reviewer_role not in roles:
        roles.append(review.reviewer_role)
    existing["severity"] = _max_severity(cast(str, existing["severity"]), evidence.severity)


def _record_review_diff_group(
    *,
    review_diff_groups: dict[tuple[str | None, str, str], dict[str, Any]],
    candidate: TranslationCandidate,
    candidate_rank: int,
    review: ReviewBundle,
    evidence: StructuredEvidence,
) -> None:
    group_value = evidence.normalized_value or evidence.evidence_text
    key = (evidence.source_span_id, evidence.dimension, group_value)
    source_excerpt, target_excerpt = _segment_excerpts_for_span(
        candidate.segments,
        evidence.source_span_id,
    )
    existing = review_diff_groups.get(key)
    if existing is None:
        existing = {
            "source_span_id": evidence.source_span_id,
            "time_range": _format_span_range(evidence.source_span_id),
            "dimension": evidence.dimension,
            "severity": evidence.severity,
            "blocking_hard_contradiction": evidence.dimension in _HARD_CONTRADICTION_DIMENSIONS,
            "evidence_text": evidence.evidence_text,
            "_evidence_sort_key": (candidate_rank, review.reviewer_role, evidence.evidence_text),
            "normalized_value": evidence.normalized_value,
            "reviewer_roles": [review.reviewer_role],
            "candidates": {},
        }
        review_diff_groups[key] = existing
    else:
        existing["severity"] = _max_severity(cast(str, existing["severity"]), evidence.severity)
        sort_key = (candidate_rank, review.reviewer_role, evidence.evidence_text)
        if cast(tuple[int, str, str], existing["_evidence_sort_key"]) > sort_key:
            existing["evidence_text"] = evidence.evidence_text
            existing["_evidence_sort_key"] = sort_key
        if existing.get("normalized_value") is None and evidence.normalized_value is not None:
            existing["normalized_value"] = evidence.normalized_value

    reviewer_roles = cast(list[str], existing["reviewer_roles"])
    if review.reviewer_role not in reviewer_roles:
        reviewer_roles.append(review.reviewer_role)

    candidate_entries = cast(dict[str, dict[str, Any]], existing["candidates"])
    candidate_entry = candidate_entries.get(candidate.candidate_id)
    if candidate_entry is None:
        candidate_entries[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "rank": candidate_rank,
            "prompt_variant_id": candidate.prompt_variant_id,
            "model_id": candidate.model_id,
            "source_transcript_candidate_id": candidate.source_transcript_candidate_id,
            "source_excerpt": source_excerpt,
            "target_excerpt": target_excerpt,
        }
        return

    if candidate_entry.get("source_excerpt") is None and source_excerpt is not None:
        candidate_entry["source_excerpt"] = source_excerpt
    if candidate_entry.get("target_excerpt") is None and target_excerpt is not None:
        candidate_entry["target_excerpt"] = target_excerpt


def _build_review_diffs(
    *,
    review_diff_groups: dict[tuple[str | None, str, str], dict[str, Any]],
    preferred_candidate_id: str | None,
    translation_candidates: list[TranslationCandidate],
    candidate_rank_map: dict[str, int],
) -> list[dict[str, Any]]:
    review_diffs: list[dict[str, Any]] = []
    for group in review_diff_groups.values():
        group_candidates = sorted(
            (
                _review_diff_candidate_payload(
                    candidate,
                    source_span_id=cast(str | None, group["source_span_id"]),
                    candidate_rank=candidate_rank_map.get(candidate.candidate_id, 10**9),
                )
                for candidate in translation_candidates
            ),
            key=lambda item: (cast(int, item["rank"]), cast(str, item["candidate_id"])),
        )
        if len(group_candidates) < 2:
            continue
        anchor = next(
            (
                item
                for item in group_candidates
                if cast(str, item["candidate_id"]) == preferred_candidate_id
            ),
            group_candidates[0],
        )
        alternates = [
            item
            for item in group_candidates
            if cast(str, item["candidate_id"]) != cast(str, anchor["candidate_id"])
        ]
        anchor_excerpt = _comparison_excerpt(cast(str | None, anchor.get("target_excerpt")))
        emitted = False
        for other in alternates:
            other_excerpt = _comparison_excerpt(cast(str | None, other.get("target_excerpt")))
            if other_excerpt is None or other_excerpt == anchor_excerpt:
                continue
            review_diffs.append(
                _review_diff_card(
                    group=group,
                    left_candidate=anchor,
                    right_candidate=other,
                )
            )
            emitted = True
        if emitted or not alternates:
            continue
        review_diffs.append(
            _review_diff_card(
                group=group,
                left_candidate=anchor,
                right_candidate=alternates[0],
                force_right_placeholder=True,
            )
        )

    review_diffs.sort(
        key=lambda item: (
            0 if cast(bool, item["blocking_hard_contradiction"]) else 1,
            _span_sort_key(cast(str | None, item["source_span_id"])),
            _severity_sort_key(cast(str, item["severity"])),
            cast(int, cast(dict[str, Any], item["right_candidate"])["rank"]),
            cast(str, item["diff_id"]),
        )
    )
    return review_diffs


def _review_diff_candidate_payload(
    candidate: TranslationCandidate,
    *,
    source_span_id: str | None,
    candidate_rank: int,
) -> dict[str, Any]:
    source_excerpt, target_excerpt = _segment_excerpts_for_span(candidate.segments, source_span_id)
    return {
        "candidate_id": candidate.candidate_id,
        "rank": candidate_rank,
        "prompt_variant_id": candidate.prompt_variant_id,
        "model_id": candidate.model_id,
        "source_transcript_candidate_id": candidate.source_transcript_candidate_id,
        "source_excerpt": source_excerpt,
        "target_excerpt": target_excerpt,
    }


def _review_diff_card(
    *,
    group: dict[str, Any],
    left_candidate: dict[str, Any],
    right_candidate: dict[str, Any],
    force_right_placeholder: bool = False,
) -> dict[str, Any]:
    right_excerpt = (
        _NO_OVERLAPPING_EXCERPT
        if force_right_placeholder
        else _display_excerpt(cast(str | None, right_candidate.get("target_excerpt")))
    )
    source_excerpt = (
        cast(str | None, left_candidate.get("source_excerpt"))
        or cast(str | None, right_candidate.get("source_excerpt"))
        or _NO_OVERLAPPING_EXCERPT
    )
    return {
        "diff_id": _review_diff_id(
            source_span_id=cast(str | None, group["source_span_id"]),
            dimension=cast(str, group["dimension"]),
            left_candidate_id=cast(str, left_candidate["candidate_id"]),
            right_candidate_id=cast(str, right_candidate["candidate_id"]),
        ),
        "source_span_id": group["source_span_id"],
        "time_range": group["time_range"],
        "dimension": group["dimension"],
        "severity": group["severity"],
        "blocking_hard_contradiction": group["blocking_hard_contradiction"],
        "evidence_text": group["evidence_text"],
        "normalized_value": group["normalized_value"],
        "reviewer_roles": sorted(cast(list[str], group["reviewer_roles"])),
        "source_excerpt": source_excerpt,
        "left_candidate": {
            "candidate_id": left_candidate["candidate_id"],
            "rank": left_candidate["rank"],
            "prompt_variant_id": left_candidate["prompt_variant_id"],
            "model_id": left_candidate["model_id"],
            "source_transcript_candidate_id": left_candidate["source_transcript_candidate_id"],
            "target_excerpt": _display_excerpt(
                cast(str | None, left_candidate.get("target_excerpt"))
            ),
        },
        "right_candidate": {
            "candidate_id": right_candidate["candidate_id"],
            "rank": right_candidate["rank"],
            "prompt_variant_id": right_candidate["prompt_variant_id"],
            "model_id": right_candidate["model_id"],
            "source_transcript_candidate_id": right_candidate["source_transcript_candidate_id"],
            "target_excerpt": right_excerpt,
        },
    }


def _review_diff_id(
    *,
    source_span_id: str | None,
    dimension: str,
    left_candidate_id: str,
    right_candidate_id: str,
) -> str:
    return "::".join(
        (
            source_span_id or "no-span",
            dimension,
            left_candidate_id,
            right_candidate_id,
        )
    )


def _segment_excerpts_for_span(
    segments: tuple[Segment, ...],
    source_span_id: str | None,
) -> tuple[str | None, str | None]:
    start_ms, end_ms = _parse_span_id(source_span_id)
    if start_ms is None or end_ms is None:
        return None, None
    overlapping_segments = [
        segment for segment in segments if segment.end_ms > start_ms and segment.start_ms < end_ms
    ]
    if not overlapping_segments:
        return None, None
    return (
        _truncate_excerpt(_join_segment_text(overlapping_segments, "source_text")),
        _truncate_excerpt(_join_segment_text(overlapping_segments, "target_text")),
    )


def _join_segment_text(segments: list[Segment], field_name: str) -> str | None:
    joined = " ".join(
        text.strip()
        for segment in segments
        if isinstance((text := getattr(segment, field_name)), str) and text.strip()
    ).strip()
    return joined or None


def _truncate_excerpt(text: str | None, limit: int = 220) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _comparison_excerpt(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.split()).strip()
    return normalized or None


def _display_excerpt(text: str | None) -> str:
    return text or _NO_OVERLAPPING_EXCERPT


def _parse_span_id(source_span_id: str | None) -> tuple[int | None, int | None]:
    if source_span_id is None:
        return None, None
    _, separator, remainder = source_span_id.partition(":")
    if separator == "":
        return None, None
    start_text, separator, end_text = remainder.partition(":")
    if separator == "":
        return None, None
    try:
        return int(start_text), int(end_text)
    except ValueError:
        return None, None


def _format_span_range(source_span_id: str | None) -> str | None:
    start_ms, end_ms = _parse_span_id(source_span_id)
    if start_ms is None or end_ms is None:
        return source_span_id
    return f"{_format_ms(start_ms)}-{_format_ms(end_ms)}"


def _format_ms(value: int) -> str:
    minutes, remainder = divmod(value, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _span_sort_key(source_span_id: str | None) -> tuple[int, int]:
    start_ms, end_ms = _parse_span_id(source_span_id)
    if start_ms is None or end_ms is None:
        return (10**9, 10**9)
    return (start_ms, end_ms)


def _severity_sort_key(severity: str) -> int:
    return {"critical": 0, "major": 1, "minor": 2}.get(severity, 3)


def _max_severity(left: str, right: str) -> str:
    return left if _severity_sort_key(left) <= _severity_sort_key(right) else right


def _load_resolution_payload(context: ReviewSessionContext) -> dict[str, Any] | None:
    payload = _optional_json(context.runtime, review_resolution_key(context.job))
    if payload is not None:
        payload["resolution_ref"] = review_resolution_key(context.job)
        return payload
    approval = _optional_json(context.runtime, approval_record_key(context.job))
    if approval is None:
        return None
    return {
        "resolution_ref": approval_record_key(context.job),
        "resolution_kind": "approved_good",
        "failure_tags": [],
        "residual_failure_tags": [],
        "candidate_id": approval.get("approved_candidate_id"),
    }


def _recommended_failure_tags(
    *,
    candidate_payloads: list[dict[str, Any]],
    contradiction_summary: dict[str, Any],
    reviews: tuple[ReviewBundle, ...],
) -> tuple[FailureTag, ...]:
    scores: dict[str, int] = {}
    source_transcript_ids = {
        str(payload.get("source_transcript_candidate_id"))
        for payload in candidate_payloads
        if payload.get("source_transcript_candidate_id") is not None
    }
    if len(source_transcript_ids) > 1:
        scores["source_transcript_instability"] = 1
    for review in reviews:
        for evidence in review.structured_evidence:
            text = f"{evidence.dimension} {evidence.evidence_text}".casefold()
            if evidence.dimension == "entity":
                scores["name_entity_drift"] = scores.get("name_entity_drift", 0) + 1
            if "honorific" in text:
                scores["honorific_leak"] = scores.get("honorific_leak", 0) + 1
            if "romaniz" in text or "transliter" in text:
                scores["romanization_leak"] = scores.get("romanization_leak", 0) + 1
            if "gibberish" in text or "mixed-script" in text or "junk" in text:
                scores["subtitle_gibberish"] = scores.get("subtitle_gibberish", 0) + 1
            if "late" in text or "degeneration" in text:
                scores["late_run_degeneration"] = scores.get("late_run_degeneration", 0) + 1
            if evidence.dimension == "coverage" and (
                "unsupported" in text or "addition" in text or "added" in text
            ):
                scores["ungrounded_addition"] = scores.get("ungrounded_addition", 0) + 1
            if evidence.dimension == "meaning" and "literal" in text:
                scores["literal_but_wrong_semantics"] = (
                    scores.get("literal_but_wrong_semantics", 0) + 1
                )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(tag for tag, _ in ordered[:4])  # type: ignore[return-value]


def _validate_resolution_inputs(
    *,
    resolution_kind: HumanSupervisionKind,
    candidate_id: str | None,
    failure_tags: tuple[FailureTag, ...],
    translation_candidates: dict[str, TranslationCandidate],
) -> None:
    if resolution_kind in {"approved_good", "approved_best_available"}:
        if candidate_id is None:
            raise ValueError("candidate_id is required for approved resolutions")
        if candidate_id not in translation_candidates:
            raise ValueError(f"unknown candidate_id: {candidate_id}")
    if resolution_kind == "rejected_all":
        if candidate_id is not None:
            raise ValueError("candidate_id is forbidden for rejected_all")
        if not failure_tags:
            raise ValueError("at least one failure_tag is required for rejected_all")


def _reviewed_candidate_contexts(
    *,
    job: JobContext,
    translation_candidates: dict[str, TranslationCandidate],
    transcript_candidates: dict[str, TranscriptCandidate],
) -> tuple[HumanReviewedCandidateContext, ...]:
    contexts: list[HumanReviewedCandidateContext] = []
    for candidate in sorted(translation_candidates.values(), key=lambda item: item.candidate_id):
        transcript = transcript_candidates.get(candidate.source_transcript_candidate_id or "")
        contexts.append(
            HumanReviewedCandidateContext(
                candidate_id=candidate.candidate_id,
                source_transcript_candidate_id=candidate.source_transcript_candidate_id,
                transcript_provider_id=transcript.provider_id if transcript is not None else None,
                model_id=candidate.model_id,
                prompt_variant_id=candidate.prompt_variant_id,
                prompt_version=candidate.prompt_version,
                base_prompt_version=_candidate_base_prompt_version(candidate),
                combo_key=_candidate_combo_key(job, candidate, transcript),
                prompt_resolution_mode=_candidate_prompt_resolution_mode(candidate),
                selected_proposal_id=_candidate_selected_proposal_id(candidate),
            )
        )
    return tuple(contexts)


def _candidate_base_prompt_version(candidate: TranslationCandidate) -> str:
    resolver = candidate.metadata.get("prompt_resolver")
    if isinstance(resolver, dict):
        base = resolver.get("base_prompt_version")
        if isinstance(base, str) and base.strip():
            return base
    return candidate.prompt_version


def _candidate_prompt_resolution_mode(candidate: TranslationCandidate) -> str | None:
    resolver = candidate.metadata.get("prompt_resolver")
    if not isinstance(resolver, dict):
        return None
    mode = resolver.get("resolution_mode")
    if isinstance(mode, str) and mode.strip():
        return mode
    return None


def _candidate_selected_proposal_id(candidate: TranslationCandidate) -> str | None:
    resolver = candidate.metadata.get("prompt_resolver")
    if not isinstance(resolver, dict):
        return None
    proposal_id = resolver.get("selected_proposal_id")
    if isinstance(proposal_id, str) and proposal_id.strip():
        return proposal_id
    return None


def _candidate_combo_key(
    job: JobContext,
    candidate: TranslationCandidate,
    transcript: TranscriptCandidate | None,
) -> str:
    provider_id = transcript.provider_id if transcript is not None else "unknown-provider"
    return "::".join(
        (
            job.source_language,
            job.target_language,
            provider_id,
            candidate.model_id,
            candidate.prompt_variant_id,
            candidate.prompt_version,
        )
    )


def _update_provider_quality_stats(
    *,
    store: OperationalStore,
    reviewed_candidates: tuple[HumanReviewedCandidateContext, ...],
    selected_provider_id: str | None,
    job: JobContext,
    resolution_kind: HumanSupervisionKind,
    resolved_at: datetime,
) -> None:
    provider_ids = {
        context.transcript_provider_id
        for context in reviewed_candidates
        if context.transcript_provider_id is not None
    }
    recent_cutoff = resolved_at - timedelta(days=30)
    for provider_id in provider_ids:
        current = store.get_transcript_provider_quality_stats(
            provider_id=provider_id,
            source_language=job.source_language,
            target_language=job.target_language,
        ) or TranscriptProviderQualityStats(
            provider_id=provider_id,
            source_language=job.source_language,
            target_language=job.target_language,
            updated_at=resolved_at,
        )
        review_timestamps = list(current.review_escalation_timestamps) + [resolved_at]
        approval_timestamps = list(current.approval_timestamps)
        soft_timestamps = list(current.approved_best_available_timestamps)
        rejected_timestamps = list(current.rejected_all_timestamps)
        total_approved_outcomes = current.total_approved_outcomes
        approved_good_count = current.approved_good_count
        approved_best_available_count = current.approved_best_available_count
        rejected_all_count = current.rejected_all_count
        selected_count = current.selected_via_approved_translation_count
        hard_positive_score = current.hard_positive_score
        soft_positive_score = current.soft_positive_score
        negative_feedback_score = current.negative_feedback_score
        last_approved_at = current.last_approved_at
        if resolution_kind == "approved_good" and provider_id == selected_provider_id:
            approval_timestamps.append(resolved_at)
            total_approved_outcomes += 1
            approved_good_count += 1
            selected_count += 1
            hard_positive_score += 1.0
            last_approved_at = resolved_at
        elif resolution_kind == "approved_best_available" and provider_id == selected_provider_id:
            approval_timestamps.append(resolved_at)
            soft_timestamps.append(resolved_at)
            total_approved_outcomes += 1
            approved_best_available_count += 1
            selected_count += 1
            soft_positive_score += 0.35
            last_approved_at = resolved_at
        elif resolution_kind == "rejected_all":
            rejected_timestamps.append(resolved_at)
            rejected_all_count += 1
            negative_feedback_score += 1.0
        recent_review_timestamps = [ts for ts in review_timestamps if ts >= recent_cutoff]
        recent_approval_timestamps = [ts for ts in approval_timestamps if ts >= recent_cutoff]
        recent_soft_timestamps = [ts for ts in soft_timestamps if ts >= recent_cutoff]
        recent_rejected_timestamps = [ts for ts in rejected_timestamps if ts >= recent_cutoff]
        total_review_escalations = len(review_timestamps)
        indirect_approval_rate = round(
            selected_count / total_review_escalations if total_review_escalations else 0.0,
            4,
        )
        store.save_transcript_provider_quality_stats(
            current.model_copy(
                update={
                    "total_review_escalations": total_review_escalations,
                    "total_approved_outcomes": total_approved_outcomes,
                    "selected_via_approved_translation_count": selected_count,
                    "recent_review_escalations_30d": len(recent_review_timestamps),
                    "recent_approved_outcomes_30d": len(recent_approval_timestamps),
                    "approved_good_count": approved_good_count,
                    "approved_best_available_count": approved_best_available_count,
                    "rejected_all_count": rejected_all_count,
                    "recent_approved_best_available_count_30d": len(recent_soft_timestamps),
                    "recent_rejected_all_count_30d": len(recent_rejected_timestamps),
                    "hard_positive_score": round(hard_positive_score, 4),
                    "soft_positive_score": round(soft_positive_score, 4),
                    "negative_feedback_score": round(negative_feedback_score, 4),
                    "indirect_approval_rate": indirect_approval_rate,
                    "review_escalation_timestamps": tuple(review_timestamps),
                    "approval_timestamps": tuple(approval_timestamps),
                    "approved_best_available_timestamps": tuple(soft_timestamps),
                    "rejected_all_timestamps": tuple(rejected_timestamps),
                    "last_approved_at": last_approved_at,
                    "updated_at": resolved_at,
                }
            )
        )


def _update_translation_feedback_stats(
    *,
    store: OperationalStore,
    resolution: HumanReviewResolutionRecord,
    resolved_at: datetime,
) -> None:
    current_stats: dict[str, TranslationFeedbackStats] = {}

    def _stats_for(context: HumanReviewedCandidateContext) -> TranslationFeedbackStats:
        existing = current_stats.get(context.combo_key)
        if existing is not None:
            return existing
        loaded = store.get_translation_feedback_stats(context.combo_key)
        if loaded is None:
            loaded = TranslationFeedbackStats(
                combo_key=context.combo_key,
                source_language=resolution.source_language,
                target_language=resolution.target_language,
                transcript_provider_id=context.transcript_provider_id or "unknown-provider",
                model_id=context.model_id,
                prompt_variant_id=context.prompt_variant_id,
                prompt_version=context.prompt_version,
                last_seen_at=resolved_at,
                updated_at=resolved_at,
            )
        current_stats[context.combo_key] = loaded
        return loaded

    selected = next(
        (
            context
            for context in resolution.reviewed_candidates
            if context.candidate_id == resolution.candidate_id
        ),
        None,
    )
    weight = 1.0 if resolution.resolution_kind == "approved_good" else 0.35
    for context in resolution.reviewed_candidates:
        stats = _stats_for(context)
        failure_tag_counts = dict(stats.failure_tag_counts)
        if (
            context.candidate_id == resolution.candidate_id
            and resolution.resolution_kind == "approved_good"
        ):
            updated = stats.model_copy(
                update={
                    "approved_good_count": stats.approved_good_count + 1,
                    "last_seen_at": resolved_at,
                    "updated_at": resolved_at,
                }
            )
        elif (
            context.candidate_id == resolution.candidate_id
            and resolution.resolution_kind == "approved_best_available"
        ):
            for tag in resolution.failure_tags:
                failure_tag_counts[tag] = failure_tag_counts.get(tag, 0) + 1
            updated = stats.model_copy(
                update={
                    "approved_best_available_count": stats.approved_best_available_count + 1,
                    "failure_tag_counts": failure_tag_counts,
                    "last_seen_at": resolved_at,
                    "updated_at": resolved_at,
                }
            )
        elif resolution.resolution_kind == "rejected_all":
            for tag in resolution.failure_tags:
                failure_tag_counts[tag] = failure_tag_counts.get(tag, 0) + 1
            updated = stats.model_copy(
                update={
                    "rejected_all_count": stats.rejected_all_count + 1,
                    "failure_tag_counts": failure_tag_counts,
                    "last_seen_at": resolved_at,
                    "updated_at": resolved_at,
                }
            )
        elif selected is not None and context.combo_key != selected.combo_key:
            updated = stats.model_copy(
                update={
                    "pairwise_loss_score": round(stats.pairwise_loss_score + weight, 4),
                    "last_seen_at": resolved_at,
                    "updated_at": resolved_at,
                }
            )
        else:
            updated = stats.model_copy(
                update={"last_seen_at": resolved_at, "updated_at": resolved_at}
            )
        current_stats[context.combo_key] = updated

    if selected is not None and resolution.resolution_kind in {
        "approved_good",
        "approved_best_available",
    }:
        selected_stats = _stats_for(selected)
        current_stats[selected.combo_key] = selected_stats.model_copy(
            update={
                "pairwise_win_score": round(selected_stats.pairwise_win_score + weight, 4),
                "last_seen_at": resolved_at,
                "updated_at": resolved_at,
            }
        )

    for stats in current_stats.values():
        store.save_translation_feedback_stats(stats)


def _write_resolution_learning_memory(
    *,
    context: ReviewSessionContext,
    resolution: HumanReviewResolutionRecord,
    resolution_ref: str,
    learning_ref: str | None,
) -> tuple[MemoryWriteBatch, str, tuple[RoutingFact, ...]]:
    now = resolution.resolved_at
    project_scope_key = project_pair_scope_key(context.job)
    semantic_writes: list[MemoryWrite] = []
    episodic_writes: list[MemoryWrite] = [
        MemoryWrite(
            kind="episodic",
            memory_subtype="project_fact",
            content=(
                f"Human review resolved {resolution.resolution_kind} "
                f"for run {context.run_record.run_id}."
            ),
            updated_at=now,
            score=0.95 if resolution.resolution_kind == "approved_good" else 0.7,
            source_ref=resolution_ref,
            promotion_status="candidate",
            evidence_count=1,
            supporting_run_count=1,
            supporting_asset_count=1,
            metadata={
                "dedupe_key": f"resolution:event:{context.run_record.run_id}",
                "event_id": context.run_record.run_id,
                "supervision_kind": resolution.resolution_kind,
                "supervision_strength": resolution.supervision_strength,
                "failure_tags": list(resolution.failure_tags),
            },
        )
    ]
    procedural_writes: list[MemoryWrite] = []
    if resolution.candidate_id is not None and resolution.transcript_provider_id is not None:
        semantic_writes.append(
            MemoryWrite(
                kind="semantic",
                memory_subtype="project_fact",
                content=(
                    f"Operator review for {context.job.media_key} preferred transcript provider "
                    f"{resolution.transcript_provider_id} under {resolution.resolution_kind}."
                ),
                updated_at=now,
                score=0.95 if resolution.resolution_kind == "approved_good" else 0.7,
                source_ref=learning_ref or resolution_ref,
                promotion_status="candidate",
                evidence_count=1,
                supporting_run_count=1,
                supporting_asset_count=1,
                supporting_project_count=1,
                metadata={
                    "dedupe_key": (
                        f"resolution:provider:{resolution.transcript_provider_id}:"
                        f"{context.job.media_key}:{resolution.resolution_kind}"
                    ),
                    "category": "transcript_source_learning",
                    "supervision_kind": resolution.resolution_kind,
                    "supervision_strength": resolution.supervision_strength,
                    "transcript_provider_id": resolution.transcript_provider_id,
                    "combo_key": resolution.combo_key,
                    "media_key": context.job.media_key,
                },
            )
        )
        procedural_writes.append(
            MemoryWrite(
                kind="procedural",
                memory_subtype="prompt_guidance",
                content=(
                    "Favor translation combo "
                    f"{resolution.combo_key} when review context is similar, "
                    f"but keep contradiction checks blocking."
                ),
                updated_at=now,
                score=0.92 if resolution.resolution_kind == "approved_good" else 0.65,
                source_ref=learning_ref or resolution_ref,
                promotion_status="candidate",
                evidence_count=1,
                supporting_run_count=1,
                supporting_asset_count=1,
                supporting_project_count=1,
                metadata={
                    "dedupe_key": f"resolution:combo:{resolution.combo_key}",
                    "supervision_kind": resolution.resolution_kind,
                    "supervision_strength": resolution.supervision_strength,
                    "failure_tags": list(resolution.failure_tags),
                    "transcript_provider_id": resolution.transcript_provider_id,
                    "prompt_variant_id": resolution.prompt_variant_id,
                    "prompt_version": resolution.prompt_version,
                    "model_id": resolution.model_id,
                    "combo_key": resolution.combo_key,
                    "category": "translation_combo_guidance",
                    "prompt_family": "translation",
                },
            )
        )
    if resolution.resolution_kind == "rejected_all":
        for tag in resolution.failure_tags:
            procedural_writes.append(
                MemoryWrite(
                    kind="procedural",
                    memory_subtype="prompt_guidance",
                    content=_failure_tag_instruction(tag),
                    updated_at=now,
                    score=0.78,
                    source_ref=resolution_ref,
                    promotion_status="candidate",
                    evidence_count=1,
                    supporting_run_count=1,
                    supporting_asset_count=1,
                    supporting_project_count=1,
                    metadata={
                        "dedupe_key": f"resolution:anti-pattern:{tag}:{project_scope_key}",
                        "supervision_kind": resolution.resolution_kind,
                        "supervision_strength": resolution.supervision_strength,
                        "failure_tags": [tag],
                        "category": "anti_pattern",
                        "prompt_family": "translation",
                    },
                )
            )
    batch = MemoryWriteBatch(
        batch_id=f"batch-translation_human_resolution-{context.run_record.run_id}",
        job_id=context.job.job_id,
        source_stage="translation_human_resolution",
        decision_ref=resolution.machine_translation_decision_ref,
        investigation_ref=resolution.machine_investigation_ref,
        winner_candidate_id=resolution.candidate_id,
        decision_mode=resolution.resolution_kind,
        decision_confidence=1.0,
        disagreement_bucket="unresolved",
        translation_model_winner=resolution.model_id,
        prompt_variant_winner=resolution.prompt_variant_id,
        prompt_version_winner=resolution.prompt_version,
        semantic_writes=tuple(
            apply_scope_defaults(write, job=context.job) for write in semantic_writes
        ),
        episodic_writes=tuple(
            apply_scope_defaults(write, job=context.job) for write in episodic_writes
        ),
        procedural_writes=tuple(
            apply_scope_defaults(write, job=context.job) for write in procedural_writes
        ),
        dedupe_keys=tuple(
            write.metadata["dedupe_key"]
            for write in (*semantic_writes, *episodic_writes, *procedural_writes)
        ),
        metadata=batch_metadata_for_job(
            context.job,
            resolution_kind=resolution.resolution_kind,
            approved_by=resolution.approved_by,
            failure_tags=list(resolution.failure_tags),
            supervision_kind=resolution.resolution_kind,
            supervision_strength=resolution.supervision_strength,
            transcript_provider_id=resolution.transcript_provider_id,
            prompt_variant_id=resolution.prompt_variant_id,
            prompt_version=resolution.prompt_version,
            model_id=resolution.model_id,
            combo_key=resolution.combo_key,
        ),
    )
    context.store.save_batch(batch, storage_job_id=operational_job_key(context.job))
    batch_ref = write_model_artifact(
        context.runtime,
        memory_batch_key(context.job, batch.batch_id),
        batch,
    )
    consolidation = context.runtime.memory_consolidation_backend.consolidate_batch(batch)
    updated_batch = batch.model_copy(update={"consolidation_status": "consolidated"})
    context.store.save_batch(updated_batch, storage_job_id=operational_job_key(context.job))
    consolidation_ref = write_model_artifact(
        context.runtime,
        memory_consolidation_key(context.job, consolidation.consolidation_id),
        consolidation,
    )
    routing_facts = (
        RoutingFact(
            stage="resolve_review",
            fact_type="memory_batch_staged",
            value=batch.batch_id,
            source_ref=batch_ref,
        ),
        RoutingFact(
            stage="resolve_review",
            fact_type="memory_batch_consolidated",
            value=consolidation.consolidation_id,
            source_ref=consolidation_ref,
        ),
    )
    return updated_batch, consolidation_ref, routing_facts


def _resolution_publish_state(
    *,
    context: ReviewSessionContext,
    output_data: dict[str, Any],
    approved_candidate: TranslationCandidate | None,
    approved_transcript: TranscriptCandidate | None,
    resolution_ref: str,
    resolution_kind: HumanSupervisionKind,
    failure_tags: tuple[FailureTag, ...],
    residual_failure_tags: tuple[FailureTag, ...],
    approval_ref: str,
    learning_ref: str | None,
    memory_batch: MemoryWriteBatch,
    routing_facts: tuple[RoutingFact, ...],
    improvement_proposal_refs: tuple[str, ...],
) -> GraphState:
    existing_facts = _existing_routing_facts(context)
    transcript_decision_ref = output_data.get("transcript_decision_ref") or transcript_decision_key(
        context.job
    )
    final_transcript_candidate_id = output_data.get("final_transcript_candidate_id")
    if not isinstance(final_transcript_candidate_id, str) or not final_transcript_candidate_id:
        transcript_decision = _optional_json(context.runtime, transcript_decision_ref)
        final_transcript_candidate_id = (
            str(transcript_decision.get("winner_candidate_id"))
            if isinstance(transcript_decision, dict)
            and isinstance(transcript_decision.get("winner_candidate_id"), str)
            else None
        )
    if approved_transcript is not None:
        final_transcript_candidate_id = approved_transcript.candidate_id
    return GraphState(
        run_id=context.run_record.run_id,
        job=context.job,
        current_stage="resolve_review",
        source_video_ref=context.job.source_video_ref,
        source_artifact_ref=context.source_artifact_ref,
        transcript_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in context.runtime.decision_store.list_transcript_candidates(
                context.job.job_id,
                storage_job_id=operational_job_key(context.job),
            )
        ),
        final_transcript_candidate_id=final_transcript_candidate_id,
        final_transcript_decision_ref=transcript_decision_ref,
        translation_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in context.runtime.decision_store.list_translation_candidates(
                context.job.job_id,
                storage_job_id=operational_job_key(context.job),
            )
        ),
        final_translation_candidate_id=(
            approved_candidate.candidate_id if approved_candidate is not None else None
        ),
        final_translation_decision_ref=output_data.get("translation_decision_ref")
        or translation_decision_key(context.job),
        reference_transcript_ref=_optional_string(output_data.get("reference_transcript_ref")),
        evaluation_report_ref=_optional_string(output_data.get("evaluation_report_ref")),
        regenerated_translation_draft_ref=_optional_string(
            output_data.get("regenerated_translation_draft_ref")
        ),
        improvement_proposal_refs=tuple(
            sorted(
                {
                    *(str(ref) for ref in output_data.get("improvement_proposal_refs", [])),
                    *improvement_proposal_refs,
                }
            )
        ),
        memory_batch_ids=tuple(
            sorted(
                {
                    *(str(batch_id) for batch_id in output_data.get("memory_batch_ids", [])),
                    memory_batch.batch_id,
                }
            )
        ),
        routing_facts=existing_facts
        + (
            RoutingFact(
                stage="resolve_review",
                fact_type="review_resolution_artifact",
                value=resolution_ref,
                source_ref=resolution_ref,
            ),
        )
        + (
            (
                RoutingFact(
                    stage="resolve_review",
                    fact_type="approval_artifact",
                    value=approval_ref,
                    source_ref=approval_ref,
                ),
            )
            if approved_candidate is not None
            else ()
        )
        + (
            (
                RoutingFact(
                    stage="resolve_review",
                    fact_type="learning_artifact",
                    value=learning_ref,
                    source_ref=learning_ref,
                ),
            )
            if learning_ref is not None
            else ()
        )
        + routing_facts,
        human_review_required=False,
        review_required_stage="translation",
        resolution_ref=resolution_ref,
        resolution_kind=resolution_kind,
        failure_tags=failure_tags,
        residual_failure_tags=residual_failure_tags,
        approval_ref=approval_ref if approved_candidate is not None else None,
        approved_candidate_id=approved_candidate.candidate_id
        if approved_candidate is not None
        else None,
        approved_source_transcript_candidate_id=(
            approved_transcript.candidate_id if approved_transcript is not None else None
        ),
        translation_failed=resolution_kind == "rejected_all",
    )


def _failure_tag_instruction(tag: FailureTag) -> str:
    return _FAILURE_TAG_INSTRUCTIONS.get(
        tag,
        "Prefer conservative, source-grounded translation when prior runs failed this way.",
    )


def _update_feedback_prompt_proposals(
    *,
    context: ReviewSessionContext,
    resolution: HumanReviewResolutionRecord,
    resolution_ref: str,
) -> tuple[str, ...]:
    list_proposals = cast(
        Callable[..., list[PromptEvolutionProposal]] | None,
        getattr(context.store, "list_prompt_evolution_proposals", None),
    )
    save_proposal = cast(
        Callable[[PromptEvolutionProposal], None] | None,
        getattr(context.store, "save_prompt_evolution_proposal", None),
    )
    if not callable(list_proposals) or not callable(save_proposal):
        return ()
    compatibilities = {
        (
            compatibility.model_id,
            compatibility.prompt_variant_id,
            compatibility.base_prompt_version,
        ): compatibility
        for compatibility in (
            _compatibility_for_candidate_context(context.job, candidate)
            for candidate in resolution.reviewed_candidates
        )
        if compatibility is not None
    }
    if not compatibilities:
        return ()
    refs: list[str] = []
    for compatibility in compatibilities.values():
        relevant_resolutions = [
            record
            for record in context.store.list_human_review_resolutions(
                source_language=context.job.source_language,
                target_language=context.job.target_language,
                model_id=compatibility.model_id,
                prompt_variant_id=compatibility.prompt_variant_id,
            )
            if _resolution_supports_compatibility(record, compatibility)
            and record.resolved_at >= resolution.resolved_at - timedelta(days=30)
        ]
        dominant_tag, tag_ratio = _dominant_failure_tag(relevant_resolutions)
        failure_run_count = sum(
            1
            for record in relevant_resolutions
            if (
                (
                    record.resolution_kind == "approved_best_available"
                    and _resolution_failure_weight(record, compatibility) > 0.0
                )
                or (
                    record.resolution_kind == "rejected_all"
                    and _resolution_failure_weight(record, compatibility) < 0.0
                )
            )
        )
        if (
            len(relevant_resolutions) < 3
            or failure_run_count < 2
            or dominant_tag is None
            or tag_ratio < 0.6
        ):
            continue
        existing = [
            proposal
            for proposal in list_proposals(
                prompt_family="translation",
                target_model_id=compatibility.model_id,
                source_language=compatibility.source_language,
                target_language=compatibility.target_language,
                prompt_variant_id=compatibility.prompt_variant_id,
                base_prompt_version=compatibility.base_prompt_version,
                scope_kind=compatibility.scope_kind,
                scope_key=compatibility.scope_key,
                media_key=None,
            )
            if proposal.compatibility == compatibility
            and proposal.metadata.get("proposal_origin") == "human_review_feedback"
        ]
        proposal = (
            existing[0]
            if existing
            else _build_feedback_prompt_proposal(
                context=context,
                compatibility=compatibility,
                dominant_tag=dominant_tag,
                resolution_ref=resolution_ref,
            )
        )
        proposal = _updated_feedback_proposal_status(proposal, relevant_resolutions, compatibility)
        proposal_ref = _save_feedback_prompt_proposal(context, proposal, save_proposal)
        refs.append(proposal_ref)
    return tuple(refs)


def _compatibility_for_candidate_context(
    job: JobContext,
    candidate: HumanReviewedCandidateContext,
) -> PromptCompatibilityTuple | None:
    if candidate.base_prompt_version is None:
        return None
    return PromptCompatibilityTuple(
        prompt_family="translation",
        model_id=candidate.model_id,
        prompt_variant_id=candidate.prompt_variant_id,
        base_prompt_version=candidate.base_prompt_version,
        source_language=job.source_language,
        target_language=job.target_language,
        scope_kind="project_pair",
        scope_key=project_pair_scope_key(job),
    )


def _resolution_supports_compatibility(
    resolution: HumanReviewResolutionRecord,
    compatibility: PromptCompatibilityTuple,
) -> bool:
    for candidate in resolution.reviewed_candidates:
        if (
            candidate.model_id == compatibility.model_id
            and candidate.prompt_variant_id == compatibility.prompt_variant_id
            and candidate.base_prompt_version == compatibility.base_prompt_version
        ):
            return True
    return False


def _resolution_failure_weight(
    resolution: HumanReviewResolutionRecord,
    compatibility: PromptCompatibilityTuple,
) -> float:
    selected = next(
        (
            candidate
            for candidate in resolution.reviewed_candidates
            if candidate.candidate_id == resolution.candidate_id
        ),
        None,
    )
    if resolution.resolution_kind == "approved_good":
        return (
            1.0 if selected and _candidate_matches_compatibility(selected, compatibility) else 0.0
        )
    if resolution.resolution_kind == "approved_best_available":
        return (
            0.35 if selected and _candidate_matches_compatibility(selected, compatibility) else 0.0
        )
    if any(
        _candidate_matches_compatibility(candidate, compatibility)
        for candidate in resolution.reviewed_candidates
    ):
        return -1.0
    return 0.0


def _candidate_matches_compatibility(
    candidate: HumanReviewedCandidateContext,
    compatibility: PromptCompatibilityTuple,
) -> bool:
    return (
        candidate.model_id == compatibility.model_id
        and candidate.prompt_variant_id == compatibility.prompt_variant_id
        and candidate.base_prompt_version == compatibility.base_prompt_version
    )


def _dominant_failure_tag(
    resolutions: list[HumanReviewResolutionRecord],
) -> tuple[FailureTag | None, float]:
    counts: dict[str, int] = {}
    tagged_runs = 0
    for resolution in resolutions:
        if not resolution.failure_tags:
            continue
        tagged_runs += 1
        for tag in set(resolution.failure_tags):
            counts[tag] = counts.get(tag, 0) + 1
    if not counts or tagged_runs == 0:
        return None, 0.0
    dominant_tag, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return dominant_tag, round(count / tagged_runs, 4)  # type: ignore[return-value]


def _build_feedback_prompt_proposal(
    *,
    context: ReviewSessionContext,
    compatibility: PromptCompatibilityTuple,
    dominant_tag: FailureTag,
    resolution_ref: str,
) -> PromptEvolutionProposal:
    proposal_id = (
        f"human-feedback-{compatibility.model_id}-{compatibility.prompt_variant_id}-"
        f"{compatibility.base_prompt_version}".replace("/", "-")
    )
    return PromptEvolutionProposal(
        proposal_id=proposal_id,
        job_id=context.job.job_id,
        source_consolidation_id=f"human-feedback-{context.run_record.run_id}",
        prompt_family="translation",
        target_model_id=compatibility.model_id,
        target_prompt_version=compatibility.base_prompt_version,
        target_prompt_variant_id=compatibility.prompt_variant_id,
        base_prompt_version=compatibility.base_prompt_version,
        compatibility=compatibility,
        status="proposed",
        rationale=(
            "Repeated human review resolutions found a dominant failure pattern for one exact "
            "translation compatibility tuple."
        ),
        suggested_changes=(
            PromptChange(section="system", instruction=_failure_tag_instruction(dominant_tag)),
        ),
        evidence_refs=(resolution_ref,),
        metadata={
            "proposal_origin": "human_review_feedback",
            "dominant_failure_tag": dominant_tag,
            "source_language": context.job.source_language,
            "target_language": context.job.target_language,
            "scope_kind": compatibility.scope_kind,
            "scope_key": compatibility.scope_key,
            "human_backed_support": True,
        },
    )


def _updated_feedback_proposal_status(
    proposal: PromptEvolutionProposal,
    resolutions: list[HumanReviewResolutionRecord],
    compatibility: PromptCompatibilityTuple,
) -> PromptEvolutionProposal:
    canary_scores = [
        _resolution_failure_weight(resolution, compatibility)
        for resolution in resolutions
        if any(
            candidate.prompt_resolution_mode == "canary"
            and _candidate_matches_compatibility(candidate, compatibility)
            for candidate in resolution.reviewed_candidates
        )
    ]
    control_scores = [
        _resolution_failure_weight(resolution, compatibility)
        for resolution in resolutions
        if any(
            candidate.prompt_resolution_mode in {"control", "active", None}
            and _candidate_matches_compatibility(candidate, compatibility)
            for candidate in resolution.reviewed_candidates
        )
    ]
    canary_count = len([score for score in canary_scores if score != 0.0])
    control_count = len([score for score in control_scores if score != 0.0])
    canary_score = _average_score(canary_scores)
    control_score = _average_score(control_scores)
    canary_rejected_rate = _rejected_rate(canary_scores)
    control_rejected_rate = _rejected_rate(control_scores)
    status = proposal.status
    rollback_reason = proposal.rollback_reason
    if status == "canary":
        if (
            canary_count >= 5
            and control_count >= 5
            and canary_score >= control_score + 0.15
            and canary_rejected_rate <= control_rejected_rate
        ):
            status = "active"
            rollback_reason = None
    elif status == "active":
        recent_active_scores = [score for score in canary_scores[-5:] if score != 0.0]
        if (
            len(recent_active_scores) >= 5
            and _average_score(recent_active_scores) <= control_score - 0.10
        ):
            status = "rolled_back"
            rollback_reason = "active proposal regressed on human-feedback score"
    return proposal.model_copy(
        update={
            "status": status,
            "rollback_reason": rollback_reason,
            "canary_run_count": canary_count,
            "control_run_count": control_count,
            "metadata": {
                **proposal.metadata,
                "canary_feedback_score": canary_score,
                "control_feedback_score": control_score,
                "canary_rejected_all_rate": canary_rejected_rate,
                "control_rejected_all_rate": control_rejected_rate,
            },
        }
    )


def _average_score(values: list[float]) -> float:
    non_zero = [value for value in values if value != 0.0]
    if not non_zero:
        return 0.0
    return round(sum(non_zero) / len(non_zero), 4)


def _rejected_rate(values: list[float]) -> float:
    non_zero = [value for value in values if value != 0.0]
    if not non_zero:
        return 0.0
    return round(sum(1 for value in non_zero if value < 0.0) / len(non_zero), 4)


def _save_feedback_prompt_proposal(
    context: ReviewSessionContext,
    proposal: PromptEvolutionProposal,
    save_proposal,
) -> str:
    proposal_ref = job_path(
        context.job,
        "memory",
        "prompt-evolution",
        f"{proposal.proposal_id}.json",
    )
    proposal = proposal.model_copy(
        update={"metadata": {**proposal.metadata, "proposal_ref": proposal_ref}}
    )
    write_model_artifact(context.runtime, proposal_ref, proposal)
    save_proposal(proposal)
    return proposal_ref


def _existing_routing_facts(context: ReviewSessionContext) -> tuple[RoutingFact, ...]:
    scorecard_ref = job_path(context.job, "published", "scorecard.json")
    if not context.blob_store.exists(scorecard_ref):
        return ()
    payload = json.loads(context.blob_store.read_bytes(scorecard_ref).decode("utf-8"))
    raw_facts = payload.get("routing_facts", [])
    if not isinstance(raw_facts, list):
        return ()
    return tuple(
        RoutingFact.model_validate(raw_fact) for raw_fact in raw_facts if isinstance(raw_fact, dict)
    )


def _optional_json(runtime: Any, ref: str | None) -> dict[str, Any] | None:
    if not ref or not runtime.blob_store.exists(ref):
        return None
    payload = json.loads(runtime.blob_store.read_bytes(ref).decode("utf-8"))
    return payload if isinstance(payload, dict) else None


def _optional_model(
    runtime: Any,
    ref: str | None,
    *,
    model_type: Any | None,
) -> Any | None:
    if model_type is None or not ref or not runtime.blob_store.exists(ref):
        return None
    return read_model_artifact(runtime, ref, model_type)


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _normalized_approved_by(approved_by: str | None) -> str:
    if approved_by is not None and approved_by.strip():
        return approved_by.strip()
    resolved = getpass.getuser().strip()
    return resolved or "human"
