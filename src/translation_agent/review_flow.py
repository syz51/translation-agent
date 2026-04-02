"""Reopenable translation review and approval helpers."""

from __future__ import annotations

import getpass
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from translation_agent.graph import GraphState, build_phase_two_runtime
from translation_agent.graph.state import RoutingFact
from translation_agent.models import (
    FinalTranslationDecision,
    HumanApprovalRecord,
    JobContext,
    MemoryWrite,
    MemoryWriteBatch,
    ReviewBundle,
    Segment,
    StructuredEvidence,
    TranscriptApprovalLearningEvent,
    TranscriptCandidate,
    TranscriptProviderQualityStats,
    TranslationCandidate,
)
from translation_agent.models.jobs import ReferenceMode
from translation_agent.nodes.common import (
    approval_record_key,
    memory_batch_key,
    memory_consolidation_key,
    operational_job_key,
    read_model_artifact,
    review_preview_key,
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

    return {
        "run_id": run_id,
        "job_id": context.job.job_id,
        "status": context.run_record.status,
        "review_required_stage": output_data.get("review_required_stage"),
        "review_available": output_data.get("review_required_stage") == "translation",
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
        ],
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
    """Approve one translation candidate and republish canonical artifacts in place."""

    context = _load_review_session(run_id, store=store, blob_store=blob_store)
    output_data = _dict_payload(context.run_record.output_data)
    if output_data.get("review_required_stage") != "translation":
        raise ValueError("run does not have a pending translation review")
    approval_ref = approval_record_key(context.job)
    if context.blob_store.exists(approval_ref):
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
    approved_candidate = translation_candidates.get(candidate_id)
    if approved_candidate is None:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    if not approved_candidate.source_transcript_candidate_id:
        raise ValueError("approved translation candidate is missing transcript provenance")
    transcript_candidates = {
        candidate.candidate_id: candidate
        for candidate in runtime.decision_store.list_transcript_candidates(
            context.job.job_id,
            storage_job_id=storage_job_id,
        )
    }
    approved_transcript = transcript_candidates.get(
        approved_candidate.source_transcript_candidate_id
    )
    if approved_transcript is None:
        raise ValueError("approved source transcript candidate was not found")

    from translation_agent.models import FinalTranslationDecision

    machine_translation_decision = read_model_artifact(
        runtime,
        output_data.get("translation_decision_ref") or translation_decision_key(context.job),
        FinalTranslationDecision,
    )
    approved_at = datetime.now(UTC)
    approved_by = _normalized_approved_by(approved_by)
    approval_record = HumanApprovalRecord(
        run_id=run_id,
        job_id=context.job.job_id,
        approved_candidate_id=approved_candidate.candidate_id,
        approved_source_transcript_candidate_id=approved_transcript.candidate_id,
        approved_by=approved_by,
        note=note or "",
        approved_at=approved_at,
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
        created_at=approved_at,
    )
    write_model_artifact(runtime, learning_ref, learning_event)

    _update_provider_quality_stats(
        store=store,
        transcript_candidate=approved_transcript,
        job=context.job,
        approved_at=approved_at,
    )
    memory_batch, consolidation_ref, routing_facts = _write_approval_learning_memory(
        context=context,
        approval_record=approval_record,
        learning_ref=learning_ref,
        approved_candidate=approved_candidate,
        approved_transcript=approved_transcript,
    )

    state = _approval_publish_state(
        context=context,
        output_data=output_data,
        approved_candidate=approved_candidate,
        approved_transcript=approved_transcript,
        approval_ref=approval_ref,
        learning_ref=learning_ref,
        memory_batch=memory_batch,
        routing_facts=routing_facts,
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
        "approval_ref": approval_ref,
        "approved_candidate_id": approved_candidate.candidate_id,
        "approved_source_transcript_candidate_id": approved_transcript.candidate_id,
        "review_required_stage": "translation",
        "human_review_required": False,
        "translation_failed": False,
        "final_stage": "approve_review",
        "published_artifact_refs": list(published_refs),
        "final_transcript_ref": artifacts.final_transcript_ref,
        "final_translation_ref": artifacts.final_translation_ref,
        "learning_ref": learning_ref,
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
        status="completed_after_human_review",
        output_data=updated_output_data,
        error=None,
    )
    return {
        "run_id": run_id,
        "job_id": context.job.job_id,
        "status": "completed_after_human_review",
        "approval_ref": approval_ref,
        "approved_candidate_id": approved_candidate.candidate_id,
        "approved_source_transcript_candidate_id": approved_transcript.candidate_id,
        "final_transcript_ref": artifacts.final_transcript_ref,
        "final_translation_ref": artifacts.final_translation_ref,
        "default_output_path": str(
            (
                context.blob_store.root / job_path(context.job, "exports", "translation.srt")
            ).resolve()
        ),
    }


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


def _update_provider_quality_stats(
    *,
    store: OperationalStore,
    transcript_candidate: TranscriptCandidate,
    job: JobContext,
    approved_at: datetime,
) -> None:
    current = store.get_transcript_provider_quality_stats(
        provider_id=transcript_candidate.provider_id,
        source_language=job.source_language,
        target_language=job.target_language,
    )
    approval_timestamps = list(current.approval_timestamps if current is not None else ())
    review_timestamps = list(current.review_escalation_timestamps if current is not None else ())
    approval_timestamps.append(approved_at)
    review_timestamps.append(approved_at)
    recent_cutoff = approved_at - timedelta(days=30)
    recent_approval_timestamps = [ts for ts in approval_timestamps if ts >= recent_cutoff]
    recent_review_timestamps = [ts for ts in review_timestamps if ts >= recent_cutoff]
    total_review_escalations = (current.total_review_escalations if current is not None else 0) + 1
    total_approved_outcomes = (current.total_approved_outcomes if current is not None else 0) + 1
    selected_count = (
        current.selected_via_approved_translation_count if current is not None else 0
    ) + 1
    indirect_approval_rate = round(
        selected_count / total_review_escalations if total_review_escalations else 0.0,
        4,
    )
    store.save_transcript_provider_quality_stats(
        TranscriptProviderQualityStats(
            provider_id=transcript_candidate.provider_id,
            source_language=job.source_language,
            target_language=job.target_language,
            total_review_escalations=total_review_escalations,
            total_approved_outcomes=total_approved_outcomes,
            selected_via_approved_translation_count=selected_count,
            recent_review_escalations_30d=len(recent_review_timestamps),
            recent_approved_outcomes_30d=len(recent_approval_timestamps),
            indirect_approval_rate=indirect_approval_rate,
            review_escalation_timestamps=tuple(review_timestamps),
            approval_timestamps=tuple(approval_timestamps),
            last_approved_at=approved_at,
            updated_at=approved_at,
        )
    )


def _write_approval_learning_memory(
    *,
    context: ReviewSessionContext,
    approval_record: HumanApprovalRecord,
    learning_ref: str,
    approved_candidate: TranslationCandidate,
    approved_transcript: TranscriptCandidate,
) -> tuple[MemoryWriteBatch, str, tuple[RoutingFact, ...]]:
    now = approval_record.approved_at
    scope_key = f"{context.job.source_language}::{context.job.target_language}"
    batch = MemoryWriteBatch(
        batch_id=f"batch-translation_human_approval-{context.run_record.run_id}",
        job_id=context.job.job_id,
        source_stage="translation_human_approval",
        decision_ref=approval_record.machine_translation_decision_ref,
        investigation_ref=approval_record.machine_investigation_ref,
        winner_candidate_id=approved_candidate.candidate_id,
        decision_mode="human_approval",
        decision_confidence=1.0,
        disagreement_bucket="unresolved",
        translation_model_winner=approved_candidate.model_id,
        prompt_variant_winner=approved_candidate.prompt_variant_id,
        prompt_version_winner=approved_candidate.prompt_version,
        semantic_writes=(
            MemoryWrite(
                kind="semantic",
                content=(
                    "Approved translations for "
                    f"{context.job.source_language}->{context.job.target_language} "
                    f"selected transcript provider {approved_transcript.provider_id}."
                ),
                scope_kind="pair",
                scope_key=scope_key,
                updated_at=now,
                score=0.92,
                source_ref=learning_ref,
                metadata={
                    "dedupe_key": (
                        "approval:transcript-source:"
                        f"{approved_transcript.provider_id}:{context.job.source_language}:"
                        f"{context.job.target_language}"
                    ),
                    "category": "transcript_source_learning",
                },
            ),
        ),
        episodic_writes=(
            MemoryWrite(
                kind="episodic",
                content=(
                    "Human approval chose "
                    f"{approved_candidate.candidate_id} and implied transcript "
                    f"{approved_transcript.candidate_id}."
                ),
                scope_kind="pair",
                scope_key=scope_key,
                updated_at=now,
                score=0.95,
                source_ref=approval_record.machine_translation_decision_ref,
                metadata={
                    "dedupe_key": f"approval:event:{context.run_record.run_id}",
                    "event_id": context.run_record.run_id,
                },
            ),
        ),
        procedural_writes=(
            MemoryWrite(
                kind="procedural",
                content=(
                    "Approved translation outcomes are strong indirect supervision for transcript "
                    "source ranking and translation candidate preference."
                ),
                scope_kind="pair",
                scope_key=scope_key,
                updated_at=now,
                score=0.9,
                source_ref=approval_record.machine_translation_decision_ref,
                metadata={
                    "dedupe_key": (
                        "approval:translation-learning:"
                        f"{approved_candidate.model_id}:{approved_candidate.prompt_variant_id}:"
                        f"{approved_candidate.prompt_version}"
                    ),
                    "prompt_family": "translation",
                },
            ),
        ),
        dedupe_keys=(
            f"approval:transcript-source:{approved_transcript.provider_id}:{scope_key}",
            f"approval:event:{context.run_record.run_id}",
            (
                "approval:translation-learning:"
                f"{approved_candidate.model_id}:{approved_candidate.prompt_variant_id}:"
                f"{approved_candidate.prompt_version}"
            ),
        ),
        metadata={
            "approved_candidate_id": approved_candidate.candidate_id,
            "approved_source_transcript_candidate_id": approved_transcript.candidate_id,
            "approved_by": approval_record.approved_by,
        },
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
            stage="approve_review",
            fact_type="memory_batch_staged",
            value=batch.batch_id,
            source_ref=batch_ref,
        ),
        RoutingFact(
            stage="approve_review",
            fact_type="memory_batch_consolidated",
            value=consolidation.consolidation_id,
            source_ref=consolidation_ref,
        ),
    )
    return updated_batch, consolidation_ref, routing_facts


def _approval_publish_state(
    *,
    context: ReviewSessionContext,
    output_data: dict[str, Any],
    approved_candidate: TranslationCandidate,
    approved_transcript: TranscriptCandidate,
    approval_ref: str,
    learning_ref: str,
    memory_batch: MemoryWriteBatch,
    routing_facts: tuple[RoutingFact, ...],
) -> GraphState:
    existing_facts = _existing_routing_facts(context)
    return GraphState(
        run_id=context.run_record.run_id,
        job=context.job,
        current_stage="approve_review",
        source_video_ref=context.job.source_video_ref,
        source_artifact_ref=context.source_artifact_ref,
        transcript_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in context.runtime.decision_store.list_transcript_candidates(
                context.job.job_id,
                storage_job_id=operational_job_key(context.job),
            )
        ),
        final_transcript_candidate_id=approved_transcript.candidate_id,
        final_transcript_decision_ref=output_data.get("transcript_decision_ref")
        or transcript_decision_key(context.job),
        translation_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in context.runtime.decision_store.list_translation_candidates(
                context.job.job_id,
                storage_job_id=operational_job_key(context.job),
            )
        ),
        final_translation_candidate_id=approved_candidate.candidate_id,
        final_translation_decision_ref=output_data.get("translation_decision_ref")
        or translation_decision_key(context.job),
        reference_transcript_ref=_optional_string(output_data.get("reference_transcript_ref")),
        evaluation_report_ref=_optional_string(output_data.get("evaluation_report_ref")),
        regenerated_translation_draft_ref=_optional_string(
            output_data.get("regenerated_translation_draft_ref")
        ),
        improvement_proposal_refs=tuple(
            str(ref) for ref in output_data.get("improvement_proposal_refs", [])
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
                stage="approve_review",
                fact_type="learning_artifact",
                value=learning_ref,
                source_ref=learning_ref,
            ),
            RoutingFact(
                stage="approve_review",
                fact_type="approval_artifact",
                value=approval_ref,
                source_ref=approval_ref,
            ),
        )
        + routing_facts,
        human_review_required=False,
        review_required_stage="translation",
        approval_ref=approval_ref,
        approved_candidate_id=approved_candidate.candidate_id,
        approved_source_transcript_candidate_id=approved_transcript.candidate_id,
        translation_failed=False,
    )


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
