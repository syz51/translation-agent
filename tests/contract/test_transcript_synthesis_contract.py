from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pytest

from translation_agent.graph.runtime import ReasoningProfile, WorkflowRuntime
from translation_agent.models import (
    JobContext,
    Segment,
    TranscriptCandidate,
    TranscriptSynthesisRecord,
    TranscriptSynthesisReview,
)
from translation_agent.transcript_synthesis import (
    build_canonical_transcript_spans,
    build_span_candidates,
    run_global_adjudicator,
    run_reviewer_agent,
    run_selector_agent,
)

pytestmark = pytest.mark.contract


@dataclass(frozen=True)
class _RuntimeStub:
    reasoning_profile: ReasoningProfile = ReasoningProfile(
        provider_id="openai",
        model_id="gpt-5.4",
        base_url_source="openai-sdk-default",
    )


def _runtime() -> WorkflowRuntime:
    return cast(WorkflowRuntime, _RuntimeStub())


def test_selector_output_contract_roundtrip() -> None:
    spans, span_candidates = _fixture_inputs()

    record = run_selector_agent(
        job=_job(),
        run_id="run-contract-selector",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )

    reparsed = TranscriptSynthesisRecord.model_validate(record.model_dump(mode="json"))
    assert reparsed.agent_role == "selector"
    assert reparsed.canonical_span_count == len(spans)
    assert reparsed.metadata["base_candidate_id"]
    assert reparsed.metadata["candidate_rankings"]


def test_reviewer_output_contract_roundtrip() -> None:
    spans, span_candidates = _fixture_inputs()
    selector = run_selector_agent(
        job=_job(),
        run_id="run-contract-reviewer",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )

    review = run_reviewer_agent(
        job=_job(),
        run_id="run-contract-reviewer",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
    )

    reparsed = TranscriptSynthesisReview.model_validate(review.model_dump(mode="json"))
    assert reparsed.review_id
    assert reparsed.reasoning_provider == "openai"


def test_global_adjudicator_output_contract_roundtrip() -> None:
    spans, span_candidates = _fixture_inputs()
    selector = run_selector_agent(
        job=_job(),
        run_id="run-contract-global",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )
    review = run_reviewer_agent(
        job=_job(),
        run_id="run-contract-global",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
    )

    adjudicated = run_global_adjudicator(
        job=_job(),
        run_id="run-contract-global",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
        review=review,
    )

    reparsed = TranscriptSynthesisRecord.model_validate(adjudicated.model_dump(mode="json"))
    assert reparsed.agent_role == "global_adjudicator"
    assert reparsed.reasoning_model_id == "gpt-5.4"
    assert reparsed.record_id != selector.record_id


def _fixture_inputs():
    candidates = (
        TranscriptCandidate(
            candidate_id="candidate-a",
            job_id="job-contract",
            provider_id="assemblyai",
            provider_request_id="req-a",
            language="en",
            segments=(
                Segment(
                    segment_id="seg-a1",
                    start_ms=0,
                    end_ms=1_000,
                    speaker="speaker-1",
                    source_text="Take the map.",
                ),
            ),
            full_text="Take the map.",
            speaker_map={"speaker-1": "Host"},
            timing_resolution="segment",
            raw_payload_ref="raw/a.json",
            normalization_version="test",
            metadata={"provider_rank": 0},
        ),
        TranscriptCandidate(
            candidate_id="candidate-b",
            job_id="job-contract",
            provider_id="speechmatics",
            provider_request_id="req-b",
            language="en",
            segments=(
                Segment(
                    segment_id="seg-b1",
                    start_ms=0,
                    end_ms=1_000,
                    speaker="speaker-1",
                    source_text="Take the map. Bring the radio.",
                ),
            ),
            full_text="Take the map. Bring the radio.",
            speaker_map={"speaker-1": "Host"},
            timing_resolution="segment",
            raw_payload_ref="raw/b.json",
            normalization_version="test",
            metadata={"provider_rank": 1},
        ),
    )
    spans = build_canonical_transcript_spans(candidates)
    return spans, build_span_candidates(spans, candidates)


def _job() -> JobContext:
    return JobContext(
        job_id="job-contract",
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester",
        created_at=datetime.now(UTC),
        profile_ref="profiles/default",
        asset_id="asset-1",
        media_fingerprint="fingerprint-1",
        media_key="media-1",
    )
