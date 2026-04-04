from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.run_status import (
    derive_run_status_snapshot,
    derive_run_timing_summary,
    normalize_trace_event,
)
from translation_agent.storage import NodeExecutionRecord, RunRecord

pytestmark = pytest.mark.unit


def _run_record(
    *,
    run_id: str = "run-123",
    status: str = "running",
    created_at: str = "2026-04-03T00:00:00+00:00",
    updated_at: str = "2026-04-03T00:00:05+00:00",
    output_data: dict[str, object] | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        tenant_id="tenant-local",
        project_id="project-local",
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        input_data={"job_id": "job-123", "source": "input.wav"},
        output_data=output_data,
        metadata=None,
        error=None,
    )


def _node_execution(
    *,
    execution_id: str,
    node_name: str,
    status: str,
    created_at: str,
    updated_at: str | None = None,
) -> NodeExecutionRecord:
    return NodeExecutionRecord(
        execution_id=execution_id,
        run_id="run-123",
        node_name=node_name,
        status=status,
        created_at=created_at,
        updated_at=updated_at or created_at,
        input_data=None,
        output_data=None,
        error=None,
    )


def test_normalize_trace_event_builds_short_domain_messages() -> None:
    completed = normalize_trace_event(
        {
            "name": "translation.variant.completed",
            "timestamp": "2026-04-03T00:00:02+00:00",
            "attributes": {
                "prompt_variant_id": "variant-a",
                "source_transcript_candidate_id": "tr-1",
                "variant_total": 4,
            },
        }
    )
    failed = normalize_trace_event(
        {
            "name": "review.bundle.failed",
            "timestamp": "2026-04-03T00:00:03+00:00",
            "attributes": {
                "review_stage": "review_translations",
                "reviewer_role": "accuracy",
                "bundle_total": 2,
                "error": "parser exploded",
            },
        }
    )

    assert completed is not None
    assert completed.message == "Completed translation variant variant-a:tr-1"
    assert completed.stage == "generate_translation_candidates"
    assert failed is not None
    assert failed.message == "Failed review bundle review_translations:accuracy: parser exploded"
    assert failed.stage == "review_translations"


def test_derive_run_status_snapshot_tracks_active_node_recent_events_and_counters() -> None:
    snapshot = derive_run_status_snapshot(
        _run_record(),
        (
            _node_execution(
                execution_id="exec-1",
                node_name="fanout_transcription",
                status="started",
                created_at="2026-04-03T00:00:01+00:00",
            ),
        ),
        (
            {
                "run_id": "run-123",
                "name": "run.started",
                "timestamp": "2026-04-03T00:00:00+00:00",
                "attributes": {"job_id": "job-123"},
            },
            {
                "run_id": "run-123",
                "name": "node.started",
                "timestamp": "2026-04-03T00:00:01+00:00",
                "attributes": {"node_name": "fanout_transcription", "execution_id": "exec-1"},
            },
            {
                "run_id": "run-123",
                "name": "transcription.provider.started",
                "timestamp": "2026-04-03T00:00:02+00:00",
                "attributes": {"provider_id": "deepgram", "provider_total": 2},
            },
            {
                "run_id": "run-123",
                "name": "transcription.provider.completed",
                "timestamp": "2026-04-03T00:00:03+00:00",
                "attributes": {
                    "provider_id": "deepgram",
                    "provider_total": 2,
                    "candidate_id": "tr-deepgram",
                },
            },
            {
                "run_id": "run-123",
                "name": "transcription.provider.failed",
                "timestamp": "2026-04-03T00:00:04+00:00",
                "attributes": {
                    "provider_id": "speechmatics",
                    "provider_total": 2,
                    "error": "timeout",
                },
            },
        ),
        trace_path=Path("/tmp/run-123.jsonl"),
        now=datetime(2026, 4, 3, 0, 0, 5, tzinfo=UTC),
    )

    assert snapshot.run_id == "run-123"
    assert snapshot.job_id == "job-123"
    assert snapshot.status == "running"
    assert snapshot.current_stage == "fanout_transcription"
    assert snapshot.active_node == "fanout_transcription"
    assert snapshot.elapsed_seconds == 5.0
    assert snapshot.transcription_providers is not None
    assert snapshot.transcription_providers.total == 2
    assert snapshot.transcription_providers.completed == 1
    assert snapshot.transcription_providers.failed == 1
    assert snapshot.transcription_providers.active == 0
    assert (
        snapshot.recent_events[-1].message == "Failed transcription provider speechmatics: timeout"
    )


def test_derive_run_timing_summary_tracks_completed_and_active_intervals() -> None:
    summary = derive_run_timing_summary(
        (
            {
                "run_id": "run-123",
                "name": "run.started",
                "timestamp": "2026-04-03T00:00:00+00:00",
                "attributes": {"job_id": "job-123"},
            },
            {
                "run_id": "run-123",
                "name": "node.started",
                "timestamp": "2026-04-03T00:00:01+00:00",
                "attributes": {"node_name": "fanout_transcription", "execution_id": "exec-1"},
            },
            {
                "run_id": "run-123",
                "name": "transcription.provider.started",
                "timestamp": "2026-04-03T00:00:02+00:00",
                "attributes": {"provider_id": "deepgram", "provider_total": 2},
            },
            {
                "run_id": "run-123",
                "name": "transcription.provider.completed",
                "timestamp": "2026-04-03T00:00:04+00:00",
                "attributes": {"provider_id": "deepgram", "provider_total": 2},
            },
            {
                "run_id": "run-123",
                "name": "translation.variant.started",
                "timestamp": "2026-04-03T00:00:05+00:00",
                "attributes": {
                    "prompt_variant_id": "variant-a",
                    "source_transcript_candidate_id": "tr-1",
                    "variant_total": 2,
                },
            },
        ),
        now=datetime(2026, 4, 3, 0, 0, 8, tzinfo=UTC),
    )

    assert summary.run_status == "active"
    assert summary.run_started_at == "2026-04-03T00:00:00+00:00"
    assert summary.run_completed_at is None
    assert summary.run_elapsed_seconds == 8.0
    assert len(summary.nodes) == 1
    assert summary.nodes[0].name == "fanout_transcription"
    assert summary.nodes[0].status == "active"
    assert summary.nodes[0].elapsed_seconds == 7.0
    assert len(summary.transcription_providers) == 1
    assert summary.transcription_providers[0].name == "deepgram"
    assert summary.transcription_providers[0].status == "completed"
    assert summary.transcription_providers[0].elapsed_seconds == 2.0
    assert len(summary.translation_variants) == 1
    assert summary.translation_variants[0].name == "variant-a:tr-1"
    assert summary.translation_variants[0].status == "active"
    assert summary.translation_variants[0].elapsed_seconds == 3.0


@pytest.mark.parametrize(
    ("executions", "record", "expected_stage", "expected_active_node"),
    [
        (
            (
                _node_execution(
                    execution_id="exec-1",
                    node_name="generate_translation_candidates",
                    status="started",
                    created_at="2026-04-03T00:00:01+00:00",
                ),
            ),
            _run_record(status="running"),
            "generate_translation_candidates",
            "generate_translation_candidates",
        ),
        (
            (
                _node_execution(
                    execution_id="exec-1",
                    node_name="normalize_translations",
                    status="completed",
                    created_at="2026-04-03T00:00:01+00:00",
                    updated_at="2026-04-03T00:00:02+00:00",
                ),
            ),
            _run_record(status="running"),
            "normalize_translations",
            None,
        ),
        (
            (
                _node_execution(
                    execution_id="exec-1",
                    node_name="review_translations",
                    status="failed",
                    created_at="2026-04-03T00:00:01+00:00",
                    updated_at="2026-04-03T00:00:02+00:00",
                ),
            ),
            _run_record(status="failed"),
            "review_translations",
            None,
        ),
        (
            (
                _node_execution(
                    execution_id="exec-1",
                    node_name="generate_translation_candidates",
                    status="completed",
                    created_at="2026-04-03T00:00:01+00:00",
                    updated_at="2026-04-03T00:00:02+00:00",
                ),
                _node_execution(
                    execution_id="exec-2",
                    node_name="background_memory_pipeline",
                    status="completed",
                    created_at="2026-04-03T00:00:03+00:00",
                    updated_at="2026-04-03T00:00:04+00:00",
                ),
                _node_execution(
                    execution_id="exec-3",
                    node_name="generate_translation_candidates",
                    status="completed",
                    created_at="2026-04-03T00:00:05+00:00",
                    updated_at="2026-04-03T00:00:06+00:00",
                ),
            ),
            _run_record(
                status="human_review_required",
                output_data={"final_stage": "adjudicate_translation"},
            ),
            "adjudicate_translation",
            None,
        ),
    ],
)
def test_derive_run_status_snapshot_resolves_current_stage_across_node_states(
    executions: tuple[NodeExecutionRecord, ...],
    record: RunRecord,
    expected_stage: str,
    expected_active_node: str | None,
) -> None:
    snapshot = derive_run_status_snapshot(
        record,
        executions,
        (),
        trace_path=Path("/tmp/run-123.jsonl"),
    )

    assert snapshot.current_stage == expected_stage
    assert snapshot.active_node == expected_active_node
