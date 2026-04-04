from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from translation_agent.graph.runtime import ReasoningProfile, WorkflowRuntime
from translation_agent.models import (
    CanonicalTranscriptSpan,
    JobContext,
    Segment,
    TranscriptCandidate,
    TranscriptSpanDecision,
    TranscriptSynthesisRecord,
    TranscriptSynthesisReview,
)
from translation_agent.parallelism import GlobalConcurrencyLimiter, RuntimeParallelismPolicy
from translation_agent.transcript_synthesis import (
    blocking_failures_for_artifact,
    build_canonical_transcript_spans,
    build_span_candidates,
    materialize_synthesized_transcript,
    run_global_adjudicator,
    run_reviewer_agent,
    run_selector_agent,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _RuntimeStub:
    reasoning_profile: ReasoningProfile = ReasoningProfile(
        provider_id="openai",
        model_id="gpt-5.4",
        base_url_source="openai-sdk-default",
    )


def _runtime() -> WorkflowRuntime:
    return cast(WorkflowRuntime, _RuntimeStub())


def test_build_canonical_transcript_spans_groups_overlapping_provider_segments() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Alpha"), ("seg-a2", 2_500, 3_100, "Omega")),
        ),
        _candidate(
            "candidate-b",
            "speechmatics",
            (("seg-b1", 900, 1_800, "Alpha continued"),),
        ),
    )

    spans = build_canonical_transcript_spans(candidates)

    assert len(spans) == 2
    assert spans[0].start_ms == 0
    assert spans[0].end_ms == 1_800
    assert spans[0].supporting_provider_ids == ("assemblyai", "speechmatics")
    assert spans[1].start_ms == 2_500
    assert spans[1].end_ms == 3_100


def test_selector_picks_single_provider_when_only_one_supports_span() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Only one provider spoke here"),),
        ),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)

    record = run_selector_agent(
        job=_job(),
        run_id="run-selector-single",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )

    assert record.unresolved_span_ids == ()
    assert record.decisions[0].decision_type == "select_provider_span"
    assert record.decisions[0].output_text == "Only one provider spoke here"


def test_selector_merges_complementary_provider_fragments() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_400, "We should go."),),
        ),
        _candidate(
            "candidate-b",
            "speechmatics",
            (("seg-b1", 0, 1_400, "We should go. Wait, take the map."),),
        ),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)

    record = run_selector_agent(
        job=_job(),
        run_id="run-selector-merge",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )

    assert record.decisions[0].decision_type in {
        "select_provider_span",
        "merge_provider_spans",
    }
    assert "take the map" in record.decisions[0].output_text.lower()


def test_selector_marks_unresolved_for_conflicting_unsupported_merge() -> None:
    span = CanonicalTranscriptSpan(
        canonical_span_id="canonical-span-0001",
        start_ms=0,
        end_ms=1_000,
        speaker=None,
        supporting_candidate_ids=("candidate-a", "candidate-b"),
        supporting_provider_ids=("assemblyai", "speechmatics"),
        metadata={},
    )
    candidates = (
        _candidate("candidate-a", "assemblyai", (("seg-a1", 0, 1_000, "red blue"),)),
        _candidate("candidate-b", "speechmatics", (("seg-b1", 0, 1_000, "orange green"),)),
    )
    span_candidates = build_span_candidates((span,), candidates)

    record = run_selector_agent(
        job=_job(),
        run_id="run-selector-unresolved",
        runtime=_runtime(),
        spans=(span,),
        span_candidates=span_candidates,
    )

    assert record.unresolved_span_ids == ("canonical-span-0001",)
    assert record.decisions[0].decision_type == "mark_unresolved"


def test_global_adjudicator_resolves_remaining_span_when_one_candidate_remains() -> None:
    candidates = (
        _candidate("candidate-a", "assemblyai", (("seg-a1", 0, 1_000, "resolved text"),)),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)
    selector_record = TranscriptSynthesisRecord(
        record_id="selector",
        job_id="job-1",
        run_id="run-1",
        agent_role="selector",
        reasoning_provider="openai",
        reasoning_model_id="gpt-5.4",
        canonical_span_count=1,
        decisions=(
            TranscriptSpanDecision(
                canonical_span_id=spans[0].canonical_span_id,
                decision_type="mark_unresolved",
                rationale="Needs escalation",
            ),
        ),
        unresolved_span_ids=(spans[0].canonical_span_id,),
        provider_support_summary={"assemblyai": 1},
        metadata={},
    )
    review = TranscriptSynthesisReview(
        review_id="review",
        job_id="job-1",
        run_id="run-1",
        reasoning_provider="openai",
        reasoning_model_id="gpt-5.4",
        accepted_span_ids=(),
        corrected_decisions=(),
        unresolved_span_ids=(spans[0].canonical_span_id,),
        dropped_supported_span_ids=(),
        issues=(),
        metadata={},
    )

    adjudicated = run_global_adjudicator(
        job=_job(),
        run_id="run-global",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector_record,
        review=review,
    )

    assert adjudicated.unresolved_span_ids == ()
    assert adjudicated.decisions[0].decision_type == "select_provider_span"
    assert adjudicated.decisions[0].output_text == "resolved text"


def test_materialized_transcript_blocks_when_supported_span_stays_unresolved() -> None:
    candidates = (
        _candidate("candidate-a", "assemblyai", (("seg-a1", 0, 1_000, "red blue"),)),
        _candidate("candidate-b", "speechmatics", (("seg-b1", 0, 1_000, "orange green"),)),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)
    unresolved_record = TranscriptSynthesisRecord(
        record_id="selector",
        job_id="job-1",
        run_id="run-1",
        agent_role="selector",
        reasoning_provider="openai",
        reasoning_model_id="gpt-5.4",
        canonical_span_count=1,
        decisions=(
            TranscriptSpanDecision(
                canonical_span_id=spans[0].canonical_span_id,
                decision_type="mark_unresolved",
                rationale="conflict",
            ),
        ),
        unresolved_span_ids=(spans[0].canonical_span_id,),
        provider_support_summary={"assemblyai": 1, "speechmatics": 1},
        metadata={},
    )
    review = TranscriptSynthesisReview(
        review_id="review",
        job_id="job-1",
        run_id="run-1",
        reasoning_provider="openai",
        reasoning_model_id="gpt-5.4",
        accepted_span_ids=(),
        corrected_decisions=(),
        unresolved_span_ids=(spans[0].canonical_span_id,),
        dropped_supported_span_ids=(),
        issues=(),
        metadata={},
    )

    artifact = materialize_synthesized_transcript(
        job=_job(),
        run_id="run-materialize",
        language="en",
        spans=spans,
        span_candidates=span_candidates,
        selector_record=unresolved_record,
        review=review,
        global_record=unresolved_record,
    )

    assert artifact.status == "blocked"
    assert artifact.quality_metrics.unresolved_span_count == 1
    assert blocking_failures_for_artifact(artifact.quality_metrics) == (
        "unresolved_supported_spans",
    )


def test_regression_window_2504_2604_preserves_missing_dialogue() -> None:
    start_ms = 25 * 60 * 1_000 + 4 * 1_000
    end_ms = 26 * 60 * 1_000 + 4 * 1_000
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", start_ms, end_ms, "We should leave now."),),
        ),
        _candidate(
            "candidate-b",
            "speechmatics",
            (("seg-b1", start_ms, end_ms, "We should leave now. Bring the radio."),),
        ),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)
    record = run_selector_agent(
        job=_job(),
        run_id="run-regression-window",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )

    assert "bring the radio" in record.decisions[0].output_text.lower()


def test_live_reasoning_requests_acquire_provider_io_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CollectingTraceSink:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def record(self, event: Any) -> None:
            self.events.append(event)

    class ResponseStub:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def model_dump(self, mode: str = "json") -> dict[str, object]:
            del mode
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(self._payload),
                        }
                    }
                ]
            }

    class CompletionsStub:
        def __init__(self, payloads: list[dict[str, object]]) -> None:
            self._payloads = payloads
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: Any) -> ResponseStub:
            self.calls.append(kwargs)
            if not self._payloads:
                raise AssertionError("unexpected live reasoning request")
            return ResponseStub(self._payloads.pop(0))

    completions = CompletionsStub(
        [
            {
                "decisions": [
                    {
                        "canonical_span_id": "canonical-span-0001",
                        "decision_type": "select_provider_span",
                        "selected_candidate_ids": ["candidate-a"],
                        "selected_span_candidate_ids": ["canonical-span-0001--candidate-a"],
                        "source_fragment_refs": ["candidate-a:seg-a1"],
                        "output_text": "Take the map.",
                        "speaker_label": "speaker-1",
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "rationale": "Grounded provider evidence.",
                    }
                ]
            },
            {
                "accepted_span_ids": ["canonical-span-0001"],
                "corrected_decisions": [],
                "unresolved_span_ids": [],
                "dropped_supported_span_ids": [],
                "issues": [],
            },
            {
                "decisions": [
                    {
                        "canonical_span_id": "canonical-span-0001",
                        "decision_type": "select_provider_span",
                        "selected_candidate_ids": ["candidate-a"],
                        "selected_span_candidate_ids": ["canonical-span-0001--candidate-a"],
                        "source_fragment_refs": ["candidate-a:seg-a1"],
                        "output_text": "Take the map.",
                        "speaker_label": "speaker-1",
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "rationale": "Grounded provider evidence.",
                    }
                ]
            },
        ]
    )

    class ReasoningClientStub:
        def __init__(self, completions_stub: CompletionsStub) -> None:
            self.chat = type(
                "ChatStub",
                (),
                {"completions": completions_stub},
            )()

    monkeypatch.setattr(
        "translation_agent.transcript_synthesis._reasoning_client",
        lambda runtime: ReasoningClientStub(completions),
    )

    trace_sink = CollectingTraceSink()
    runtime = cast(
        WorkflowRuntime,
        type(
            "LiveRuntimeStub",
            (),
            {
                "reasoning_profile": ReasoningProfile(
                    provider_id="openai",
                    model_id="gpt-5.4",
                    base_url_source="openai-sdk-default",
                    api_key="openai-test-key",  # pragma: allowlist secret
                    live_adapter_enabled=True,
                ),
                "parallelism": RuntimeParallelismPolicy(
                    global_max_parallel_tokens=2,
                    provider_io_token_cost=2,
                    local_compute_token_cost=1,
                    transcription_max_workers=None,
                    translation_candidate_max_workers=None,
                    translation_chunk_max_workers=None,
                    review_max_workers=None,
                    reference_evaluation_max_workers=None,
                    memory_drain_max_workers=None,
                ),
                "global_concurrency_limiter": GlobalConcurrencyLimiter(2),
                "trace_sink": trace_sink,
            },
        )(),
    )

    spans, span_candidates = _fixture_inputs()
    selector = run_selector_agent(
        job=_job(),
        run_id="run-live-reasoning",
        runtime=runtime,
        spans=spans,
        span_candidates=span_candidates,
    )
    review = run_reviewer_agent(
        job=_job(),
        run_id="run-live-reasoning",
        runtime=runtime,
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
    )
    adjudicated = run_global_adjudicator(
        job=_job(),
        run_id="run-live-reasoning",
        runtime=runtime,
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
        review=review,
    )

    assert selector.decisions[0].decision_type == "select_provider_span"
    assert review.accepted_span_ids == ("canonical-span-0001",)
    assert adjudicated.decisions[0].decision_type == "select_provider_span"
    assert len(completions.calls) == 3

    started_events = [event for event in trace_sink.events if event.name.endswith(".started")]
    completed_events = [event for event in trace_sink.events if event.name.endswith(".completed")]
    assert [event.name for event in started_events] == [
        "transcript_synthesis.selector.started",
        "transcript_synthesis.reviewer.started",
        "transcript_synthesis.global_adjudicator.started",
    ]
    assert [event.name for event in completed_events] == [
        "transcript_synthesis.selector.completed",
        "transcript_synthesis.reviewer.completed",
        "transcript_synthesis.global_adjudicator.completed",
    ]
    assert all(event.attributes["parallel_task_class"] == "provider_io" for event in started_events)
    assert all(event.attributes["global_parallel_tokens_total"] == 2 for event in started_events)
    assert all(event.attributes["global_parallel_tokens_acquired"] == 2 for event in started_events)
    assert all(event.attributes["effective_stage_workers"] == 1 for event in started_events)
    assert all(event.run_id == "run-live-reasoning" for event in trace_sink.events)


def _candidate(
    candidate_id: str,
    provider_id: str,
    segments: tuple[tuple[str, int, int, str], ...],
) -> TranscriptCandidate:
    return TranscriptCandidate(
        candidate_id=candidate_id,
        job_id="job-1",
        provider_id=provider_id,
        provider_request_id=f"req-{candidate_id}",
        language="en",
        segments=tuple(
            Segment(
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker="speaker-1",
                source_text=text,
            )
            for segment_id, start_ms, end_ms, text in segments
        ),
        full_text=" ".join(text for _segment_id, _start_ms, _end_ms, text in segments),
        speaker_map={"speaker-1": "Host"},
        timing_resolution="segment",
        raw_payload_ref=f"raw/{candidate_id}.json",
        normalization_version="test",
        metadata={"provider_rank": 0 if provider_id == "assemblyai" else 1},
    )


def _fixture_inputs() -> tuple[tuple[CanonicalTranscriptSpan, ...], tuple[Any, ...]]:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Take the map."),),
        ),
        _candidate(
            "candidate-b",
            "speechmatics",
            (("seg-b1", 0, 1_000, "Take the map. Bring the radio."),),
        ),
    )
    spans = build_canonical_transcript_spans(candidates)
    return spans, build_span_candidates(spans, candidates)


def _job() -> JobContext:
    return JobContext(
        job_id="job-1",
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
