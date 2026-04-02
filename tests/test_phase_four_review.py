from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from translation_agent.graph import GraphState, build_phase_two_runtime, run_workflow
from translation_agent.models import (
    AdjudicationContext,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    JobContext,
    MemoryBundle,
    ReviewBundle,
    Segment,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.models.review import ReviewStage
from translation_agent.nodes.review import review_transcripts, review_translations
from translation_agent.observability import NoOpTraceSink
from translation_agent.review import (
    adjudicate_reviews,
    parse_reviewer_output,
    reviewer_roles_for_stage,
)
from translation_agent.storage import LocalBlobStore, NodeExecutionRecord, RunRecord, job_path
from translation_agent.storage.paths import operational_job_key

pytestmark = pytest.mark.unit


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


def _job_context(job_id: str = "job-phase-four") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
        media_key=f"source-ref:{job_id}",
    )


def _artifact_path(*parts: str) -> str:
    return job_path(_job_context(), *parts)


def _run_workflow(
    tmp_path: Path,
    *,
    scenario: str,
) -> tuple[GraphState, InMemoryRunStore, LocalBlobStore]:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-phase-four", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-phase-four-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario=scenario,
    )
    initial_state = GraphState(
        run_id="run-phase-four",
        job=_job_context(),
        current_stage="ingest",
        source_video_ref="input.mp4",
        source_artifact_ref=source_ref,
    )
    final_state = run_workflow(initial_state, runtime)
    return final_state, run_store, blob_store


def _translation_candidate(candidate_id: str, text: str, variant: str) -> TranslationCandidate:
    return TranslationCandidate(
        candidate_id=candidate_id,
        job_id="job-phase-four",
        source_transcript_candidate_id="tr-final",
        model_id="gpt-5.4-mini",
        prompt_variant_id=variant,
        prompt_version="phase-4-v1",
        language="fr",
        segments=(
            Segment(
                segment_id=f"{candidate_id}-seg-1",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                source_text="Hello world from the workflow skeleton.",
                target_text=text,
            ),
        ),
        full_text=text,
        raw_response_ref=f"raw/{candidate_id}.json",
        normalization_version="2026-03-30-phase-4",
        metadata={},
    )


def _transcript_candidate(candidate_id: str, text: str) -> TranscriptCandidate:
    return TranscriptCandidate(
        candidate_id=candidate_id,
        job_id="job-phase-four",
        provider_id="assemblyai",
        provider_request_id=f"req-{candidate_id}",
        language="en",
        segments=(
            Segment(
                segment_id=f"{candidate_id}-seg-1",
                start_ms=0,
                end_ms=1_000,
                speaker="speaker-1",
                source_text=text,
            ),
        ),
        full_text=text,
        speaker_map={"speaker-1": "Host"},
        timing_resolution="segment",
        raw_payload_ref=f"raw/{candidate_id}.json",
        normalization_version="2026-03-30-phase-4",
        metadata={},
    )


def _review_bundle(
    review_id: str,
    stage: ReviewStage,
    reviewer_role: str,
    raw_review_text: str,
) -> ReviewBundle:
    return ReviewBundle(
        review_id=review_id,
        job_id="job-phase-four",
        stage=stage,
        reviewer_role=reviewer_role,
        confidence=0.5,
        raw_review_text=raw_review_text,
        parser_version="phase-4-v1",
    )


def _adjudication_context(
    *,
    stage: ReviewStage,
    content_risk_class: str = "standard",
    candidate_ids: tuple[str, ...] = ("candidate-a", "candidate-b"),
) -> AdjudicationContext:
    return AdjudicationContext(
        run_id="run-phase-four",
        stage=stage,
        job=_job_context(),
        candidate_ids=candidate_ids,
        review_ids=("rev-1", "rev-2"),
        memory_bundle=MemoryBundle(),
        content_risk_class=content_risk_class,
    )


def test_parse_reviewer_output_extracts_required_fields() -> None:
    raw_review = """Winner: tl-variant-a
Confidence: 84%
Why:
- Candidate A preserves the workflow reference and reads clearly.
Key Errors By Candidate:
- tl-variant-b | terminology | major | Replaces workflow with pipeline.
Quoted Evidence:
- tl-variant-a | seg-a | Bonjour tout le monde depuis le workflow.
- tl-variant-b | seg-b | Salut tout le monde depuis le pipeline.
Suggested Fixes:
- terminology | tl-variant-b | Restore the workflow term in the disputed span.
Escalate?: yes
"""

    parsed = parse_reviewer_output(raw_review)

    assert parsed.winner_candidate_id == "tl-variant-a"
    assert parsed.confidence == pytest.approx(0.84)
    assert parsed.escalation_signal is True
    assert parsed.issues[0].category == "terminology"
    assert parsed.issues[0].severity == "major"
    assert parsed.quoted_evidence[1].segment_id == "seg-b"
    assert parsed.suggested_fixes[0].candidate_id == "tl-variant-b"


def test_parallel_review_generation_preserves_review_id_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store = InMemoryRunStore()
    run_store.create_run(run_id="run-review-order", status="running")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    source_ref = "jobs/run-review-order-request.json"
    blob_store.put_bytes(source_ref, b"{}\n")
    runtime = build_phase_two_runtime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=NoOpTraceSink(),
        source_artifact_ref=source_ref,
        scenario="happy",
    )
    job = _job_context(job_id="job-review-order")
    storage_job_id = operational_job_key(job)
    transcript = _transcript_candidate("tr-final", "Hello world from provider A.")
    translation_a = _translation_candidate("tl-a", "Bonjour du workflow.", "variant-a")
    translation_b = _translation_candidate("tl-b", "Salut du workflow.", "variant-b")
    runtime.decision_store.save_transcript_candidate(transcript, storage_job_id=storage_job_id)
    runtime.decision_store.save_translation_candidate(translation_a, storage_job_id=storage_job_id)
    runtime.decision_store.save_translation_candidate(translation_b, storage_job_id=storage_job_id)

    first_role = reviewer_roles_for_stage("transcript")[0].reviewer_role
    gate_started = threading.Event()
    gate_release = threading.Event()

    def delayed_render(review_context, candidates, prompt_text, final_transcript):  # noqa: ANN001
        del candidates, prompt_text, final_transcript
        if review_context.reviewer_role == first_role:
            gate_started.set()
            assert gate_release.wait(timeout=1)
        else:
            assert gate_started.wait(timeout=1)
            gate_release.set()
        return f"review:{review_context.stage}:{review_context.reviewer_role}"

    monkeypatch.setattr("translation_agent.nodes.review.render_reviewer_output", delayed_render)

    transcript_result = review_transcripts(
        GraphState(
            run_id="run-review-order",
            job=job,
            current_stage="review_transcripts",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
            transcript_candidate_ids=("tr-final",),
        ),
        runtime,
    )
    translation_result = review_translations(
        GraphState(
            run_id="run-review-order",
            job=job,
            current_stage="review_translations",
            source_video_ref="input.mp4",
            source_artifact_ref=source_ref,
            transcript_candidate_ids=("tr-final",),
            translation_candidate_ids=("tl-a", "tl-b"),
            final_transcript_candidate_id="tr-final",
        ),
        runtime,
    )

    transcript_roles = [
        ReviewBundle.model_validate_json(
            blob_store.read_bytes(job_path(job, "reviews", "transcript", f"{review_id}.json"))
        ).reviewer_role
        for review_id in cast(tuple[str, ...], transcript_result["transcript_review_ids"])
    ]
    translation_roles = [
        ReviewBundle.model_validate_json(
            blob_store.read_bytes(job_path(job, "reviews", "translation", f"{review_id}.json"))
        ).reviewer_role
        for review_id in cast(tuple[str, ...], translation_result["translation_review_ids"])
    ]
    assert transcript_roles == [
        spec.reviewer_role for spec in reviewer_roles_for_stage("transcript")
    ]
    assert translation_roles == [
        spec.reviewer_role for spec in reviewer_roles_for_stage("translation")
    ]


def test_parse_reviewer_output_rejects_missing_sections() -> None:
    with pytest.raises(ValueError):
        parse_reviewer_output(
            """Winner: tl-variant-a
Confidence: 0.8
Why:
- Missing the rest of the required sections.
"""
        )


def test_adjudication_uses_provider_quality_prior_as_bounded_tie_breaker() -> None:
    candidates = (
        _transcript_candidate("candidate-a", "Hello world from provider A."),
        _transcript_candidate("candidate-b", "Hello world from provider B."),
    )

    outcome = adjudicate_reviews(
        candidates=candidates,
        reviews=(),
        context=_adjudication_context(stage="transcript").model_copy(
            update={"ranking_priors": {"candidate-b": 0.2}}
        ),
    )

    assert outcome.decision_mode == "automatic_finalize"
    assert outcome.winner_candidate_id == "candidate-b"


def test_adjudicate_reviews_routes_medium_conflict_to_investigation() -> None:
    candidates = (
        _translation_candidate(
            "candidate-a",
            "Bonjour a tous depuis le flux de travail.",
            "variant-a",
        ),
        _translation_candidate(
            "candidate-b",
            "Salut tout le monde depuis le pipeline.",
            "variant-b",
        ),
    )
    reviews = (
        _review_bundle(
            "rev-1",
            "translation",
            "faithfulness_reviewer",
            """Winner: candidate-a
Confidence: 0.86
Why:
- Candidate A preserves the workflow concept without adding a pipeline metaphor.
Key Errors By Candidate:
- candidate-b | terminology | major | Introduces pipeline wording that is not in the source.
Quoted Evidence:
- candidate-a | candidate-a-seg-1 | Bonjour a tous depuis le flux de travail.
Suggested Fixes:
- terminology | candidate-b | Replace pipeline with the workflow concept.
Escalate?: yes
""",
        ),
        _review_bundle(
            "rev-2",
            "translation",
            "style_reviewer",
            """Winner: candidate-b
Confidence: 0.80
Why:
- Candidate B is more natural to read in conversational French.
Key Errors By Candidate:
- candidate-a | style | minor | Reads a bit stiff in the opening phrase.
Quoted Evidence:
- candidate-b | candidate-b-seg-1 | Salut tout le monde depuis le pipeline.
Suggested Fixes:
- style | candidate-a | Relax the opening while keeping the same meaning.
Escalate?: no
""",
        ),
    )

    outcome = adjudicate_reviews(
        candidates=candidates,
        reviews=reviews,
        context=_adjudication_context(stage="translation"),
    )

    assert outcome.decision_mode == "conflict_investigation"
    assert outcome.human_review_required is False
    assert outcome.winner_candidate_id == "candidate-a"
    assert outcome.investigation_payload is not None
    assert outcome.disagreement_bucket == "medium"


def test_adjudicate_reviews_routes_high_risk_to_stronger_adjudicator() -> None:
    candidates = (
        _translation_candidate(
            "candidate-a",
            "Bonjour a tous depuis le flux de travail.",
            "variant-a",
        ),
        _translation_candidate(
            "candidate-b",
            "Salut tout le monde depuis le pipeline.",
            "variant-b",
        ),
    )
    reviews = (
        _review_bundle(
            "rev-1",
            "translation",
            "faithfulness_reviewer",
            """Winner: candidate-a
Confidence: 0.88
Why:
- Candidate A stays closest to the approved transcript.
Key Errors By Candidate:
- candidate-b | terminology | major | Introduces pipeline wording that is not in the source.
Quoted Evidence:
- candidate-a | candidate-a-seg-1 | Bonjour a tous depuis le flux de travail.
Suggested Fixes:
- terminology | candidate-b | Restore the original workflow wording.
Escalate?: yes
""",
        ),
        _review_bundle(
            "rev-2",
            "translation",
            "style_reviewer",
            """Winner: candidate-b
Confidence: 0.82
Why:
- Candidate B reads more naturally, but the terminology tradeoff is risky.
Key Errors By Candidate:
- candidate-a | style | major | Feels overly stiff for a customer-facing line.
Quoted Evidence:
- candidate-b | candidate-b-seg-1 | Salut tout le monde depuis le pipeline.
Suggested Fixes:
- style | candidate-a | Loosen the opening while preserving the same meaning.
Escalate?: yes
""",
        ),
    )

    outcome = adjudicate_reviews(
        candidates=candidates,
        reviews=reviews,
        context=_adjudication_context(stage="translation", content_risk_class="high"),
    )

    assert outcome.decision_mode == "stronger_adjudicator"
    assert outcome.human_review_required is False
    assert outcome.winner_candidate_id == "candidate-a"
    assert outcome.disagreement_bucket == "high"


def test_adjudicate_reviews_marks_unresolved_for_human_review() -> None:
    candidates = (
        _transcript_candidate("candidate-a", "Hello world from the workflow skeleton."),
        _transcript_candidate("candidate-b", "Hello world from the escalation path."),
    )
    reviews = (
        _review_bundle(
            "rev-1",
            "transcript",
            "accuracy_reviewer",
            """Winner: candidate-a
Confidence: 0.78
Why:
- Candidate A looks closer to the expected terminology, but the disagreement is material.
Key Errors By Candidate:
- candidate-b | accuracy | critical | Rewrites the disputed span into a different meaning.
Quoted Evidence:
- candidate-a | candidate-a-seg-1 | Hello world from the workflow skeleton.
- candidate-b | candidate-b-seg-1 | Hello world from the escalation path.
Suggested Fixes:
- accuracy | candidate-b | Re-check the raw payload for the disputed span.
Escalate?: yes
""",
        ),
        _review_bundle(
            "rev-2",
            "transcript",
            "coherence_reviewer",
            """Winner: candidate-b
Confidence: 0.78
Why:
- Candidate B is internally coherent, but the disagreement is still unresolved.
Key Errors By Candidate:
- candidate-a | coherence | critical | Carries a conflicting meaning for the same segment.
Quoted Evidence:
- candidate-a | candidate-a-seg-1 | Hello world from the workflow skeleton.
- candidate-b | candidate-b-seg-1 | Hello world from the escalation path.
Suggested Fixes:
- coherence | candidate-a | Compare both candidates against the raw transcript payload.
Escalate?: yes
""",
        ),
    )

    outcome = adjudicate_reviews(
        candidates=candidates,
        reviews=reviews,
        context=_adjudication_context(stage="transcript", content_risk_class="critical"),
    )

    assert outcome.decision_mode == "human_review"
    assert outcome.human_review_required is True
    assert outcome.winner_candidate_id is None
    assert outcome.investigation_payload is not None


def test_adjudicate_reviews_keeps_single_transcript_candidate_on_reduced_confidence_path() -> None:
    candidate = _transcript_candidate("candidate-a", "Hello world from the workflow skeleton.")
    reviews = (
        _review_bundle(
            "rev-1",
            "transcript",
            "accuracy_reviewer",
            """Winner: candidate-a
Confidence: 0.92
Why:
- Candidate A matches the only available transcript evidence.
Key Errors By Candidate:
- candidate-a | accuracy | minor | No material errors found in the surviving transcript.
Quoted Evidence:
- candidate-a | candidate-a-seg-1 | Hello world from the workflow skeleton.
Suggested Fixes:
- accuracy | candidate-a | Preserve the surviving candidate as-is.
Escalate?: no
""",
        ),
    )

    outcome = adjudicate_reviews(
        candidates=(candidate,),
        reviews=reviews,
        context=_adjudication_context(
            stage="transcript",
            candidate_ids=("candidate-a",),
        ),
    )

    assert outcome.decision_mode == "automatic_finalize"
    assert outcome.winner_candidate_id == "candidate-a"
    assert outcome.human_review_required is False
    assert outcome.escalated is True
    assert outcome.investigation_payload is not None
    assert outcome.investigation_payload["strategy"] == "single-candidate escalation check"
    assert outcome.decision_confidence == pytest.approx(0.73)


def test_phase_four_workflow_routes_translation_escalation_to_stronger_adjudicator(
    tmp_path: Path,
) -> None:
    final_state, _, blob_store = _run_workflow(tmp_path, scenario="translation_escalation")

    decision = FinalTranslationDecision.model_validate_json(
        blob_store.read_bytes(_artifact_path("decisions", "translation.json"))
    )

    assert final_state.current_stage == "finalize_outputs"
    assert final_state.human_review_required is False
    assert final_state.translation_failed is False
    assert decision.decision_mode == "stronger_adjudicator"
    assert decision.investigation_ref == _artifact_path("investigations", "translation.json")
    assert blob_store.exists(_artifact_path("investigations", "translation.json"))


def test_phase_four_workflow_routes_transcript_escalation_to_human_review(
    tmp_path: Path,
) -> None:
    final_state, run_store, blob_store = _run_workflow(tmp_path, scenario="transcript_escalation")

    decision = FinalTranscriptDecision.model_validate_json(
        blob_store.read_bytes(_artifact_path("decisions", "transcript.json"))
    )

    assert final_state.human_review_required is False
    assert final_state.final_translation_decision_ref is not None
    assert decision.decision_mode == "human_review"
    assert decision.human_review_required is False
    assert decision.investigation_ref == _artifact_path("investigations", "transcript.json")
    assert blob_store.exists(_artifact_path("investigations", "transcript.json"))
    executed_nodes = [
        record.node_name for record in run_store.list_node_executions("run-phase-four")
    ]
    assert "generate_translation_candidates" in executed_nodes
