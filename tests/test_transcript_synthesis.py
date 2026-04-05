from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from translation_agent.graph.runtime import ReasoningProfile, WorkflowRuntime
from translation_agent.models import (
    CanonicalTranscriptSpan,
    JobContext,
    Segment,
    TranscriptCandidate,
)
from translation_agent.observability import TraceEvent
from translation_agent.parallelism import GlobalConcurrencyLimiter, RuntimeParallelismPolicy
from translation_agent.transcript_synthesis import (
    _invoke_reasoning_json,
    blocking_failures_for_artifact,
    build_canonical_transcript_spans,
    build_span_candidates,
    materialize_synthesized_transcript,
    run_global_adjudicator,
    run_reviewer_agent,
    run_selector_agent,
)

pytestmark = pytest.mark.unit


@dataclass
class _RecordingTraceSink:
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def path(self) -> None:
        return None

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class _RuntimeStub:
    reasoning_profile: ReasoningProfile
    parallelism: RuntimeParallelismPolicy
    global_concurrency_limiter: GlobalConcurrencyLimiter
    trace_sink: _RecordingTraceSink


def _runtime(
    *,
    trace_sink: _RecordingTraceSink | None = None,
    live_adapter_enabled: bool = False,
    global_max_parallel_tokens: int = 4,
    provider_io_token_cost: int = 1,
) -> WorkflowRuntime:
    return cast(
        WorkflowRuntime,
        _RuntimeStub(
            reasoning_profile=ReasoningProfile(
                provider_id="openai",
                model_id="gpt-5.4",
                base_url_source="openai-sdk-default",
                api_key=(
                    "openai-test-key" if live_adapter_enabled else None
                ),  # pragma: allowlist secret
                live_adapter_enabled=live_adapter_enabled,
            ),
            parallelism=RuntimeParallelismPolicy(
                global_max_parallel_tokens=global_max_parallel_tokens,
                provider_io_token_cost=provider_io_token_cost,
                local_compute_token_cost=1,
                transcription_max_workers=None,
                translation_candidate_max_workers=None,
                translation_chunk_max_workers=None,
                review_max_workers=None,
                reference_evaluation_max_workers=None,
                memory_drain_max_workers=None,
            ),
            global_concurrency_limiter=GlobalConcurrencyLimiter(global_max_parallel_tokens),
            trace_sink=trace_sink or _RecordingTraceSink(),
        ),
    )


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


def test_selector_chooses_one_global_base_candidate() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Alpha"), ("seg-a2", 1_500, 2_300, "Beta")),
        ),
        _candidate(
            "candidate-b",
            "speechmatics",
            (("seg-b1", 0, 1_000, "Alpha"),),
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

    assert record.metadata["base_candidate_id"] == "candidate-a"
    assert record.metadata["base_provider_id"] == "assemblyai"
    assert record.decisions[0].metadata["decision_origin"] == "base"
    assert record.decisions[1].metadata["decision_origin"] == "base"
    assert len(record.metadata["candidate_rankings"]) == 2


def test_base_covered_span_survives_provider_disagreement() -> None:
    candidates = (
        _candidate("candidate-a", "assemblyai", (("seg-a1", 0, 1_000, "red blue"),)),
        _candidate(
            "candidate-b",
            "speechmatics",
            (("seg-b1", 0, 1_000, "orange green extra words"),),
        ),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)

    record = run_selector_agent(
        job=_job(),
        run_id="run-selector-unresolved",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )

    assert record.unresolved_span_ids == ()
    assert record.decisions[0].decision_type == "select_provider_span"
    assert record.decisions[0].output_text == "red blue"
    assert record.decisions[0].metadata["decision_origin"] == "base"


def test_global_adjudicator_fills_base_gap_from_non_base_candidate() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Alpha"), ("seg-a2", 3_000, 4_000, "Gamma")),
        ),
        _candidate("candidate-b", "speechmatics", (("seg-b1", 1_500, 2_100, "Beta"),)),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)
    selector_record = run_selector_agent(
        job=_job(),
        run_id="run-global",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )
    review = run_reviewer_agent(
        job=_job(),
        run_id="run-global",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector_record,
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
    assert len(adjudicated.decisions) == 1
    assert adjudicated.decisions[0].decision_type == "select_provider_span"
    assert adjudicated.decisions[0].output_text == "Beta"
    assert adjudicated.decisions[0].metadata["decision_origin"] == "fill"


def test_materialized_transcript_builds_base_plus_fill_without_unresolved_disagreement() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Alpha"), ("seg-a2", 3_000, 4_000, "Gamma")),
        ),
        _candidate("candidate-b", "speechmatics", (("seg-b1", 1_500, 2_100, "Beta"),)),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)
    selector_record = run_selector_agent(
        job=_job(),
        run_id="run-materialize",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )
    review = run_reviewer_agent(
        job=_job(),
        run_id="run-materialize",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector_record,
    )
    global_record = run_global_adjudicator(
        job=_job(),
        run_id="run-materialize",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector_record,
        review=review,
    )

    artifact = materialize_synthesized_transcript(
        job=_job(),
        run_id="run-materialize",
        language="en",
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector_record,
        review=review,
        global_record=global_record,
    )

    assert artifact.status == "ready"
    assert artifact.quality_metrics.unresolved_span_count == 0
    assert [segment.source_text for segment in artifact.final_segments] == [
        "Alpha",
        "Beta",
        "Gamma",
    ]
    assert artifact.full_text == "Alpha Beta Gamma"
    assert artifact.transcript_metadata["base_span_count"] == 2
    assert artifact.transcript_metadata["fill_span_count"] == 1
    assert blocking_failures_for_artifact(artifact.quality_metrics) == ()


def test_regression_window_2504_2604_preserves_base_text_under_disagreement() -> None:
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

    assert record.decisions[0].output_text == "We should leave now."
    assert record.decisions[0].metadata["decision_origin"] == "base"


def test_materialized_transcript_preserves_distinct_selector_and_global_refs() -> None:
    candidates = (
        _candidate(
            "candidate-a",
            "assemblyai",
            (("seg-a1", 0, 1_000, "Alpha"), ("seg-a2", 3_000, 4_000, "Gamma")),
        ),
        _candidate("candidate-b", "speechmatics", (("seg-b1", 1_500, 2_100, "Beta"),)),
    )
    spans = build_canonical_transcript_spans(candidates)
    span_candidates = build_span_candidates(spans, candidates)
    selector = run_selector_agent(
        job=_job(),
        run_id="run-distinct-refs",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
    )
    review = run_reviewer_agent(
        job=_job(),
        run_id="run-distinct-refs",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
    )
    global_record = run_global_adjudicator(
        job=_job(),
        run_id="run-distinct-refs",
        runtime=_runtime(),
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
        review=review,
    )
    artifact = materialize_synthesized_transcript(
        job=_job(),
        run_id="run-distinct-refs",
        language="en",
        spans=spans,
        span_candidates=span_candidates,
        selector_record=selector,
        review=review,
        global_record=global_record,
    )

    assert (
        artifact.transcript_metadata["selector_record_id"]
        != artifact.transcript_metadata["global_adjudicator_record_id"]
    )
    fill_provenance = next(
        item for item in artifact.provenance if item.candidate_ids == ("candidate-b",)
    )
    assert selector.record_id in fill_provenance.reasoning_refs
    assert global_record.record_id in fill_provenance.reasoning_refs


def test_live_reasoning_requests_acquire_provider_io_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
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

    class _Completions:
        def __init__(self, payloads: list[dict[str, object]]) -> None:
            self._payloads = payloads
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: Any) -> _Response:
            self.calls.append(kwargs)
            if not self._payloads:
                raise AssertionError("unexpected live reasoning request")
            return _Response(self._payloads.pop(0))

    completions = _Completions(
        [
            {"base_candidate_id": "candidate-a"},
            {
                "accepted_span_ids": ["canonical-span-0001"],
                "corrected_decisions": [],
                "unresolved_span_ids": [],
                "dropped_supported_span_ids": [],
                "issues": [],
            },
            {"decisions": []},
        ]
    )

    class _Client:
        def __init__(self, completions_stub: _Completions) -> None:
            self.chat = type("_Chat", (), {"completions": completions_stub})()

    monkeypatch.setattr(
        "translation_agent.transcript_synthesis._reasoning_client",
        lambda _runtime: _Client(completions),
    )

    trace_sink = _RecordingTraceSink()
    runtime = _runtime(
        trace_sink=trace_sink,
        live_adapter_enabled=True,
        global_max_parallel_tokens=2,
        provider_io_token_cost=2,
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
    assert adjudicated.decisions == ()
    assert len(completions.calls) == 3

    stage_started_events = [
        event for event in trace_sink.events if event.name.startswith("transcript_synthesis.")
    ]
    assert [event.name for event in stage_started_events if event.name.endswith(".started")] == [
        "transcript_synthesis.selector.started",
        "transcript_synthesis.reviewer.started",
        "transcript_synthesis.global_adjudicator.started",
    ]
    assert [event.name for event in stage_started_events if event.name.endswith(".completed")] == [
        "transcript_synthesis.selector.completed",
        "transcript_synthesis.reviewer.completed",
        "transcript_synthesis.global_adjudicator.completed",
    ]
    assert all(
        event.attributes["parallel_task_class"] == "provider_io"
        for event in stage_started_events
        if event.name.endswith(".started")
    )
    assert all(
        event.attributes["global_parallel_tokens_total"] == 2
        for event in stage_started_events
        if event.name.endswith(".started")
    )
    assert all(
        event.attributes["global_parallel_tokens_acquired"] == 2
        for event in stage_started_events
        if event.name.endswith(".started")
    )
    assert all(
        event.attributes["effective_stage_workers"] == 1
        for event in stage_started_events
        if event.name.endswith(".started")
    )
    assert all(event.run_id == "run-live-reasoning" for event in trace_sink.events)

    reasoning_events = [
        event.name for event in trace_sink.events if event.name.startswith("transcript.reasoning.")
    ]
    assert reasoning_events == [
        "transcript.reasoning.started",
        "transcript.reasoning.completed",
        "transcript.reasoning.started",
        "transcript.reasoning.completed",
        "transcript.reasoning.started",
        "transcript.reasoning.completed",
    ]


def test_invoke_reasoning_json_records_started_and_completed_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    class _Response:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            assert mode == "json"
            content = '{"decisions": [{"canonical_span_id": "canonical-span-0001"}]}'
            return {
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            }

    class _Completions:
        def create(self, **_kwargs: object) -> _Response:
            return _Response()

    class _Client:
        chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(
        "translation_agent.transcript_synthesis._reasoning_client",
        lambda _runtime: _Client(),
    )

    result = _invoke_reasoning_json(
        run_id="run-trace-success",
        runtime=runtime,
        schema_name="transcript_selector",
        schema={"type": "object"},
        system_prompt="system",
        user_prompt="user",
        trace_name_prefix="transcript_synthesis.selector",
        trace_attributes={"canonical_span_count": 1},
    )

    assert result == {"decisions": [{"canonical_span_id": "canonical-span-0001"}]}
    events = cast(_RecordingTraceSink, runtime.trace_sink).events
    assert [event.name for event in events if event.name.startswith("transcript.reasoning.")] == [
        "transcript.reasoning.started",
        "transcript.reasoning.completed",
    ]
    assert [event.name for event in events if event.name.startswith("transcript_synthesis.")] == [
        "transcript_synthesis.selector.started",
        "transcript_synthesis.selector.completed",
    ]
    assert events[0].attributes["schema_name"] == "transcript_selector"
    assert events[-1].attributes["response_choice_count"] == 1


def test_invoke_reasoning_json_records_failed_event_on_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    class _Completions:
        def create(self, **_kwargs: object) -> object:
            raise ValueError("boom")

    class _Client:
        chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(
        "translation_agent.transcript_synthesis._reasoning_client",
        lambda _runtime: _Client(),
    )

    result = _invoke_reasoning_json(
        run_id="run-trace-failed",
        runtime=runtime,
        schema_name="transcript_selector",
        schema={"type": "object"},
        system_prompt="system",
        user_prompt="user",
        trace_name_prefix="transcript_synthesis.selector",
        trace_attributes={"canonical_span_count": 1},
    )

    assert result == {}
    events = cast(_RecordingTraceSink, runtime.trace_sink).events
    assert [event.name for event in events if event.name.startswith("transcript.reasoning.")] == [
        "transcript.reasoning.started",
        "transcript.reasoning.failed",
    ]
    assert [event.name for event in events if event.name.startswith("transcript_synthesis.")] == [
        "transcript_synthesis.selector.started",
        "transcript_synthesis.selector.failed",
    ]
    assert events[-1].attributes["error_type"] == "ValueError"


def test_invoke_reasoning_json_records_parse_failed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    class _Response:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            assert mode == "json"
            return {"choices": [{"message": {"content": "not-json"}}]}

    class _Completions:
        def create(self, **_kwargs: object) -> _Response:
            return _Response()

    class _Client:
        chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(
        "translation_agent.transcript_synthesis._reasoning_client",
        lambda _runtime: _Client(),
    )

    result = _invoke_reasoning_json(
        run_id="run-trace-parse-failed",
        runtime=runtime,
        schema_name="transcript_selector",
        schema={"type": "object"},
        system_prompt="system",
        user_prompt="user",
        trace_name_prefix="transcript_synthesis.selector",
        trace_attributes={"canonical_span_count": 1},
    )

    assert result == {}
    events = cast(_RecordingTraceSink, runtime.trace_sink).events
    assert [event.name for event in events if event.name.startswith("transcript.reasoning.")] == [
        "transcript.reasoning.started",
        "transcript.reasoning.parse_failed",
    ]
    assert [event.name for event in events if event.name.startswith("transcript_synthesis.")] == [
        "transcript_synthesis.selector.started",
        "transcript_synthesis.selector.failed",
    ]
    assert events[-1].attributes["failure_stage"] == "content_parse_error"


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
