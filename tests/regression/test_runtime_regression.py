from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.config import sanitize_db_target
from translation_agent.graph import GraphState, build_phase_two_runtime, run_workflow
from translation_agent.models import JobContext
from translation_agent.observability import NoOpTraceSink
from translation_agent.replay import ReplayAdjudicationRequest, replay_adjudication
from translation_agent.review import content_risk_class_for_scenario
from translation_agent.storage import (
    LocalBlobStore,
    NodeExecutionRecord,
    RunRecord,
    job_path,
    job_scope_token,
)

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        (
            "postgresql://user:secret@db.example.com:5432/translation_agent?sslmode=require",
            "postgresql://db.example.com:5432/translation_agent",
        ),
        (
            "postgresql://user:secret@db.example.com/translation_agent?connect_timeout=1",
            "postgresql://db.example.com/translation_agent",
        ),
        ("not-a-dsn", "<invalid>"),
        (None, "<missing>"),
    ],
)
def test_sanitize_db_target_strips_secrets_and_noise_regression(
    dsn: str | None, expected: str
) -> None:
    assert sanitize_db_target(dsn) == expected


def test_blob_store_overwrite_does_not_leave_temporary_files_regression(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    store.put_bytes("jobs/run-1/request.json", b"first")
    store.put_bytes("jobs/run-1/request.json", b"second")

    assert store.read_bytes("jobs/run-1/request.json") == b"second"
    assert store.list_keys() == ["jobs/run-1/request.json"]
    assert not any(path.name.startswith(".tmp-blob-") for path in store.root.rglob("*"))


@dataclass
class InMemoryRunStore:
    runs: dict[str, RunRecord]
    node_executions: dict[str, NodeExecutionRecord]

    def __init__(self) -> None:
        self.runs = {}
        self.node_executions = {}

    def create_run(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        status: str = "queued",
        input_data=None,
        metadata=None,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> RunRecord:
        assert run_id is not None
        created_at = created_at or datetime.now(UTC).isoformat()
        record = RunRecord(
            run_id=run_id,
            tenant_id=tenant_id,
            project_id=project_id,
            status=status,
            created_at=created_at,
            updated_at=created_at,
            input_data=input_data,
            output_data=None,
            metadata=metadata,
            error=None,
        )
        self.runs[run_id] = record
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return list(self.runs.values())

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        output_data=None,
        metadata=None,
        error=None,
        updated_at: str | None = None,
    ) -> RunRecord:
        current = self.runs[run_id]
        record = RunRecord(
            run_id=current.run_id,
            tenant_id=current.tenant_id,
            project_id=current.project_id,
            status=current.status if status is None else status,
            created_at=current.created_at,
            updated_at=updated_at or datetime.now(UTC).isoformat(),
            input_data=current.input_data,
            output_data=current.output_data if output_data is None else output_data,
            metadata=current.metadata if metadata is None else metadata,
            error=current.error if error is None else error,
        )
        self.runs[run_id] = record
        return record

    def create_node_execution(
        self,
        *,
        run_id: str,
        node_name: str,
        status: str = "started",
        input_data=None,
        execution_id: str | None = None,
        created_at: str | None = None,
    ) -> NodeExecutionRecord:
        execution_id = execution_id or f"exec-{len(self.node_executions) + 1}"
        created_at = created_at or datetime.now(UTC).isoformat()
        record = NodeExecutionRecord(
            execution_id=execution_id,
            run_id=run_id,
            node_name=node_name,
            status=status,
            created_at=created_at,
            updated_at=created_at,
            input_data=input_data,
            output_data=None,
            error=None,
        )
        self.node_executions[execution_id] = record
        return record

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None:
        return self.node_executions.get(execution_id)

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]:
        return [record for record in self.node_executions.values() if record.run_id == run_id]

    def update_node_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        output_data=None,
        error=None,
        updated_at: str | None = None,
    ) -> NodeExecutionRecord:
        current = self.node_executions[execution_id]
        record = NodeExecutionRecord(
            execution_id=current.execution_id,
            run_id=current.run_id,
            node_name=current.node_name,
            status=current.status if status is None else status,
            created_at=current.created_at,
            updated_at=updated_at or datetime.now(UTC).isoformat(),
            input_data=current.input_data,
            output_data=current.output_data if output_data is None else output_data,
            error=current.error if error is None else error,
        )
        self.node_executions[execution_id] = record
        return record


def _job_context(
    *,
    job_id: str = "job-replay",
    tenant_id: str = "tenant-1",
    project_id: str = "project-1",
    source_language: str = "en",
    target_language: str = "fr",
) -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id=tenant_id,
        project_id=project_id,
        source_video_ref="input.mp4",
        target_language=target_language,
        source_language=source_language,
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
    )


def _run_workflow(
    tmp_path: Path,
    *,
    run_id: str,
    scenario: str,
    job: JobContext | None = None,
) -> tuple[GraphState, LocalBlobStore]:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id=run_id, status="running")
    blob_store = LocalBlobStore(tmp_path / run_id / "blobs")
    source_ref = f"jobs/{run_id}-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario=scenario,
    )
    initial_state = GraphState(
        run_id=run_id,
        job=job or _job_context(),
        current_stage="ingest",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
    )
    final_state = run_workflow(initial_state, runtime)
    return final_state, blob_store


def _load_json(blob_store: LocalBlobStore, path: str) -> dict[str, object]:
    return json.loads(blob_store.read_bytes(path).decode("utf-8"))


def _normalize_scorecard(payload: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload))
    normalized["run_id"] = "<normalized>"
    normalized["trace_refs"] = ["<normalized>"]
    for fact in normalized.get("routing_facts", []):
        source_ref = fact.get("source_ref")
        if isinstance(source_ref, str) and source_ref.startswith("jobs/run-"):
            fact["source_ref"] = "<normalized-request-artifact>"
    return normalized


def _normalize_failure_manifest(payload: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload))
    normalized["run_id"] = "<normalized>"
    return normalized


def test_replay_scorecards_memory_and_prompt_proposals_are_stable_regression(
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-replay-happy")
    _, first_blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-a",
        scenario="happy",
        job=job,
    )
    _, second_blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-b",
        scenario="happy",
        job=job,
    )

    scorecard_path = job_path(job, "published", "scorecard.json")
    scope_token = job_scope_token(job)
    consolidation_path = job_path(
        job,
        "memory",
        "consolidations",
        f"consolidation-batch-translation_adjudication-job-replay-happy-{scope_token}.json",
    )
    prompt_path = job_path(
        job,
        "memory",
        "prompt-evolution",
        (
            "prompt-evolution-"
            f"consolidation-batch-translation_adjudication-job-replay-happy-{scope_token}.json"
        ),
    )

    assert _normalize_scorecard(_load_json(first_blob_store, scorecard_path)) == (
        _normalize_scorecard(_load_json(second_blob_store, scorecard_path))
    )
    assert _load_json(first_blob_store, consolidation_path) == _load_json(
        second_blob_store,
        consolidation_path,
    )
    assert _load_json(first_blob_store, prompt_path) == _load_json(second_blob_store, prompt_path)


def test_replay_translation_failure_manifest_is_stable_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-failure")
    _, first_blob_store = _run_workflow(
        tmp_path,
        run_id="run-failure-a",
        scenario="translation_failed",
        job=job,
    )
    _, second_blob_store = _run_workflow(
        tmp_path,
        run_id="run-failure-b",
        scenario="translation_failed",
        job=job,
    )

    failure_path = job_path(job, "published", "translation-failed.json")
    assert _normalize_failure_manifest(_load_json(first_blob_store, failure_path)) == (
        _normalize_failure_manifest(_load_json(second_blob_store, failure_path))
    )


def test_replay_adjudication_uses_persisted_candidates_reviews_and_memory_refs(
    tmp_path: Path,
) -> None:
    job = _job_context(job_id="job-replay-adjudication")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-adjudication",
        scenario="translation_conflict",
        job=job,
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="translation_conflict",
    )
    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="translation",
            candidate_refs=tuple(
                job_path(job, "candidates", "translations", f"{candidate_id}.json")
                for candidate_id in final_state.translation_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "translation", f"{review_id}.json")
                for review_id in final_state.translation_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_translation"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("translation_conflict"),
        ),
    )

    stored_decision = _load_json(blob_store, job_path(job, "decisions", "translation.json"))
    replayed_decision = replayed.decision.model_dump(mode="json")

    assert replayed.decision.decision_mode == stored_decision["decision_mode"]
    assert replayed.decision.winner_candidate_id == stored_decision["winner_candidate_id"]
    assert replayed.decision.disagreement_bucket == stored_decision["disagreement_bucket"]
    assert replayed_decision["adjudication_scorecard"] == stored_decision["adjudication_scorecard"]


def test_replay_adjudication_supports_transcript_stage_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-transcript")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-transcript",
        scenario="happy",
        job=job,
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="happy",
    )

    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="transcript",
            candidate_refs=tuple(
                job_path(job, "candidates", "transcripts", f"{candidate_id}.json")
                for candidate_id in final_state.transcript_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "transcript", f"{review_id}.json")
                for review_id in final_state.transcript_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_transcript"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("happy"),
        ),
    )

    assert replayed.decision.decision_mode == "automatic_finalize"
    assert replayed.decision.winner_candidate_id == final_state.final_transcript_candidate_id


def test_replay_adjudication_preserves_timeout_escalation_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-timeout")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-timeout",
        scenario="translation_conflict_timeout",
        job=job,
    )
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="translation_conflict_timeout",
    )

    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="translation",
            candidate_refs=tuple(
                job_path(job, "candidates", "translations", f"{candidate_id}.json")
                for candidate_id in final_state.translation_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "translation", f"{review_id}.json")
                for review_id in final_state.translation_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_translation"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("translation_conflict_timeout"),
        ),
    )

    assert replayed.decision.decision_mode == "human_review"
    assert replayed.decision.winner_candidate_id is None
    assert replayed.decision.disagreement_bucket == "unresolved"


def test_replay_adjudication_ignores_missing_timeout_artifact_regression(tmp_path: Path) -> None:
    job = _job_context(job_id="job-replay-missing-timeout")
    final_state, blob_store = _run_workflow(
        tmp_path,
        run_id="run-replay-missing-timeout",
        scenario="translation_conflict_timeout",
        job=job,
    )
    blob_store.delete(job_path(job, "investigations", "translation.json"))
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=InMemoryRunStore(),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=f"jobs/{final_state.run_id}-request.json",
        scenario="translation_conflict_timeout",
    )

    replayed = replay_adjudication(
        runtime,
        ReplayAdjudicationRequest(
            run_id=final_state.run_id,
            job=job,
            stage="translation",
            candidate_refs=tuple(
                job_path(job, "candidates", "translations", f"{candidate_id}.json")
                for candidate_id in final_state.translation_candidate_ids
            ),
            review_refs=tuple(
                job_path(job, "reviews", "translation", f"{review_id}.json")
                for review_id in final_state.translation_review_ids
            ),
            memory_ref=next(
                fact.source_ref
                for fact in final_state.routing_facts
                if fact.fact_type == "adjudication_memory_bundle"
                and fact.stage == "adjudicate_translation"
                and fact.source_ref is not None
            ),
            content_risk_class=content_risk_class_for_scenario("translation_conflict_timeout"),
        ),
    )

    assert replayed.decision.decision_mode == "conflict_investigation"
    assert replayed.decision.human_review_required is False
