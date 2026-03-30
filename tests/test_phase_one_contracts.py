from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from translation_agent.adapters import (
    AudioExtractionAdapter,
    TranscriptionAdapter,
    TranslationAdapter,
)
from translation_agent.graph.state import GraphState, RoutingFact
from translation_agent.memory import MemoryRecallBackend, MemoryStagingBackend
from translation_agent.models import (
    AdjudicationContext,
    AudioArtifact,
    CandidatePreference,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    JobContext,
    MemoryBundle,
    MemoryEntry,
    MemoryQuery,
    MemoryWrite,
    MemoryWriteBatch,
    ProviderCaveat,
    PublishContext,
    PublishedArtifacts,
    QuotedEvidence,
    RequestContext,
    ReviewBundle,
    ReviewContext,
    RoutingContext,
    Segment,
    SuggestedFix,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.observability import JsonlTraceSink, NoOpTraceSink, TraceEvent, TraceSink
from translation_agent.storage import (
    BlobStore,
    DecisionStore,
    LocalBlobStore,
    MemoryBatchStore,
    RunStore,
)

pytestmark = pytest.mark.unit


def _job_context() -> JobContext:
    return JobContext(
        job_id="job-123",
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="videos/source.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester@example.com",
        created_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        profile_ref="profiles/default",
    )


def _segment() -> Segment:
    return Segment(
        segment_id="seg-1",
        start_ms=0,
        end_ms=1250,
        speaker="speaker-1",
        source_text="Hello world",
        annotations={"confidence": 0.95},
    )


def _memory_bundle() -> MemoryBundle:
    return MemoryBundle(
        semantic_memory=(
            MemoryEntry(
                memory_id="mem-sem-1",
                kind="semantic",
                content="Product names stay untranslated.",
                source_ref="memory/semantic/1",
                score=0.8,
            ),
        ),
        glossary=(
            MemoryEntry(
                memory_id="mem-glossary-1",
                kind="glossary",
                content="OpenAI -> OpenAI",
                source_ref="memory/glossary/1",
            ),
        ),
        provider_caveats=(
            ProviderCaveat(provider_id="assemblyai", note="Over-splits long pauses."),
        ),
    )


def test_phase_one_contract_models_round_trip_json() -> None:
    job = _job_context()
    request_context = RequestContext(
        run_id="run-123",
        attempt=2,
        job=job,
        source_artifact_ref="jobs/run-123-request.json",
        metadata={"priority": "high"},
    )
    routing_context = RoutingContext(
        run_id="run-123",
        stage="review_transcripts",
        job=job,
        available_candidate_ids=("tr-1", "tr-2"),
        review_ids=("rev-1",),
        escalation_signals=("low_confidence",),
        retryable_failures_present=True,
    )
    audio_artifact = AudioArtifact(
        artifact_id="audio-1",
        job_id=job.job_id,
        blob_ref="audio/job-123.wav",
        duration_ms=60_000,
        sample_rate_hz=48_000,
        channels=2,
        codec="pcm_s16le",
        extraction_metadata={"tool": "ffmpeg"},
    )
    transcript_candidate = TranscriptCandidate(
        candidate_id="tr-1",
        job_id=job.job_id,
        provider_id="assemblyai",
        provider_request_id="req-1",
        language="en",
        segments=(_segment(),),
        full_text="Hello world",
        speaker_map={"speaker-1": "Host"},
        timing_resolution="word",
        raw_payload_ref="payloads/tr-1.json",
        normalization_version="2026-03-30",
        metadata={"confidence": 0.95},
    )
    translation_candidate = TranslationCandidate(
        candidate_id="tl-1",
        job_id=job.job_id,
        source_transcript_candidate_id=transcript_candidate.candidate_id,
        model_id="gpt-5.4-mini",
        prompt_variant_id="variant-a",
        prompt_version="v1",
        language="fr",
        segments=(
            Segment(
                segment_id="seg-1",
                start_ms=0,
                end_ms=1250,
                speaker="speaker-1",
                source_text="Hello world",
                target_text="Bonjour le monde",
            ),
        ),
        full_text="Bonjour le monde",
        raw_response_ref="responses/tl-1.json",
        normalization_version="2026-03-30",
        metadata={"temperature": 0},
    )
    memory_bundle = _memory_bundle()
    memory_query = MemoryQuery(
        job=job,
        stage="review_transcripts",
        query_text="speaker normalization",
        candidate_ids=(transcript_candidate.candidate_id,),
    )
    review_context = ReviewContext(
        run_id="run-123",
        stage="transcript",
        reviewer_role="accuracy_reviewer",
        job=job,
        candidate_ids=(transcript_candidate.candidate_id,),
        memory_bundle=memory_bundle,
        policy_ref="policy/transcript-review-v1",
    )
    adjudication_context = AdjudicationContext(
        run_id="run-123",
        stage="translation",
        job=job,
        candidate_ids=(translation_candidate.candidate_id,),
        review_ids=("rev-1",),
        memory_bundle=memory_bundle,
        content_risk_class="standard",
    )
    review_bundle = ReviewBundle(
        review_id="rev-1",
        job_id=job.job_id,
        stage="translation",
        reviewer_role="style_reviewer",
        candidate_preferences=(
            CandidatePreference(candidate_id="tl-1", rank=1, rationale="Most natural phrasing."),
        ),
        confidence=0.91,
        raw_review_text="Winner: tl-1",
        quoted_evidence=(QuotedEvidence(quote="Bonjour le monde", candidate_id="tl-1"),),
        issue_categories=("style",),
        suggested_fixes=(
            SuggestedFix(
                issue_category="style",
                candidate_id="tl-1",
                description="Shorten greeting if UI space is tight.",
            ),
        ),
        escalation_signal=False,
        parser_version="2026-03-30",
    )
    transcript_decision = FinalTranscriptDecision(
        job_id=job.job_id,
        winner_candidate_id=transcript_candidate.candidate_id,
        decision_mode="automatic_finalize",
        decision_confidence=0.88,
        rationale_summary="Agreement across transcript reviewers.",
        review_refs=("rev-tr-1",),
    )
    translation_decision = FinalTranslationDecision(
        job_id=job.job_id,
        winner_candidate_id=translation_candidate.candidate_id,
        decision_mode="automatic_finalize",
        decision_confidence=0.89,
        rationale_summary="Translation reviewers aligned on tone and accuracy.",
        review_refs=(review_bundle.review_id,),
        prompt_variant_winner="variant-a",
        prompt_version_winner="v1",
    )
    write_batch = MemoryWriteBatch(
        batch_id="batch-1",
        job_id=job.job_id,
        source_stage="translation_adjudication",
        semantic_writes=(
            MemoryWrite(
                kind="semantic",
                content="UI greeting translated as Bonjour le monde.",
                source_ref="decisions/translation/job-123.json",
            ),
        ),
        dedupe_keys=("job-123:greeting",),
    )
    publish_context = PublishContext(
        run_id="run-123",
        job=job,
        transcript_decision_ref="decisions/transcript/job-123.json",
        translation_decision_ref="decisions/translation/job-123.json",
        trace_refs=("traces/run-123.jsonl",),
        export_targets=("srt", "json"),
        downstream_targets=("cms",),
    )
    final_transcript_ref = "published/job-123/transcript.json"
    final_translation_ref = "published/job-123/translation.json"
    artifacts = PublishedArtifacts(
        final_transcript_ref=final_transcript_ref,
        final_translation_ref=final_translation_ref,
        scorecard_refs=("published/job-123/scorecard.json",),
        trace_refs=("traces/run-123.jsonl",),
        export_refs=("exports/job-123.srt",),
        downstream_delivery_refs=("deliveries/job-123-cms.json",),
    )
    state = GraphState(
        run_id="run-123",
        job=job,
        current_stage="review_translations",
        source_video_ref=job.source_video_ref,
        audio_artifact_ref=audio_artifact.blob_ref,
        transcript_candidate_ids=(transcript_candidate.candidate_id,),
        transcript_review_ids=("rev-tr-1",),
        final_transcript_candidate_id=transcript_candidate.candidate_id,
        final_transcript_decision_ref="decisions/transcript/job-123.json",
        translation_candidate_ids=(translation_candidate.candidate_id,),
        translation_review_ids=(review_bundle.review_id,),
        final_translation_candidate_id=translation_candidate.candidate_id,
        final_translation_decision_ref="decisions/translation/job-123.json",
        memory_batch_ids=(write_batch.batch_id,),
        published_artifact_refs=(final_translation_ref,),
        routing_facts=(
            RoutingFact(
                stage="review_translations",
                fact_type="surviving_candidates",
                value="1",
                source_ref="reviews/rev-1.json",
            ),
        ),
    )

    payloads = [
        request_context,
        routing_context,
        audio_artifact,
        transcript_candidate,
        translation_candidate,
        memory_query,
        review_context,
        adjudication_context,
        review_bundle,
        transcript_decision,
        translation_decision,
        write_batch,
        publish_context,
        artifacts,
        state,
    ]

    for model in payloads:
        dumped = model.model_dump(mode="json")
        rebuilt = type(model).model_validate(dumped)
        assert rebuilt == model


def test_phase_one_contract_models_reject_invalid_shapes() -> None:
    with pytest.raises(ValidationError):
        Segment(segment_id="seg-1", start_ms=10, end_ms=5)

    with pytest.raises(ValidationError):
        TranslationCandidate(
            candidate_id="tl-1",
            job_id="job-123",
            model_id="gpt-5.4-mini",
            prompt_variant_id="variant-a",
            prompt_version="v1",
            language="fr",
            normalization_version="2026-03-30",
        )

    with pytest.raises(ValidationError):
        GraphState.model_validate(
            {
                "run_id": "run-123",
                "job": _job_context().model_dump(mode="json"),
                "current_stage": "ingest",
                "source_video_ref": "videos/source.mp4",
                "raw_payload": {"should": "not exist"},
            }
        )


def test_phase_one_protocols_accept_fake_implementations(tmp_path: Path) -> None:
    class FakeExtractor:
        adapter_id = "fake-extractor"

        def extract_audio(self, video_ref: str, job_context: RequestContext) -> AudioArtifact:
            return AudioArtifact(
                artifact_id="audio-1",
                job_id=job_context.job.job_id,
                blob_ref=f"audio/{video_ref}.wav",
                duration_ms=1,
                sample_rate_hz=16_000,
                channels=1,
                codec="pcm",
            )

    class FakeTranscriber:
        provider_id = "fake-stt"

        def transcribe(
            self,
            audio_artifact: AudioArtifact,
            request_context: RequestContext,
        ) -> TranscriptCandidate:
            return TranscriptCandidate(
                candidate_id="tr-1",
                job_id=request_context.job.job_id,
                provider_id=self.provider_id,
                language=request_context.job.source_language,
                full_text="hello",
                normalization_version="v1",
                raw_payload_ref=audio_artifact.blob_ref,
            )

    class FakeTranslator:
        model_id = "fake-translation-model"

        def generate_translation(
            self,
            final_transcript: TranscriptCandidate,
            prompt_variant_id: str,
            request_context: RequestContext,
        ) -> TranslationCandidate:
            return TranslationCandidate(
                candidate_id="tl-1",
                job_id=request_context.job.job_id,
                source_transcript_candidate_id=final_transcript.candidate_id,
                model_id=self.model_id,
                prompt_variant_id=prompt_variant_id,
                prompt_version="v1",
                language=request_context.job.target_language,
                full_text="bonjour",
                normalization_version="v1",
            )

    class FakeDecisionStore:
        def __init__(self) -> None:
            self.transcript_candidates: list[TranscriptCandidate] = []
            self.translation_candidates: list[TranslationCandidate] = []
            self.transcript_decision: FinalTranscriptDecision | None = None
            self.translation_decision: FinalTranslationDecision | None = None

        def save_transcript_candidate(self, candidate: TranscriptCandidate) -> None:
            self.transcript_candidates.append(candidate)

        def list_transcript_candidates(self, job_id: str) -> list[TranscriptCandidate]:
            return [
                candidate for candidate in self.transcript_candidates if candidate.job_id == job_id
            ]

        def save_transcript_decision(self, decision: FinalTranscriptDecision) -> None:
            self.transcript_decision = decision

        def get_transcript_decision(self, job_id: str) -> FinalTranscriptDecision | None:
            return (
                self.transcript_decision
                if self.transcript_decision and self.transcript_decision.job_id == job_id
                else None
            )

        def save_translation_candidate(self, candidate: TranslationCandidate) -> None:
            self.translation_candidates.append(candidate)

        def list_translation_candidates(self, job_id: str) -> list[TranslationCandidate]:
            return [
                candidate for candidate in self.translation_candidates if candidate.job_id == job_id
            ]

        def save_translation_decision(self, decision: FinalTranslationDecision) -> None:
            self.translation_decision = decision

        def get_translation_decision(self, job_id: str) -> FinalTranslationDecision | None:
            return (
                self.translation_decision
                if self.translation_decision and self.translation_decision.job_id == job_id
                else None
            )

    class FakeMemoryBatchStore:
        def __init__(self) -> None:
            self.batches: dict[str, MemoryWriteBatch] = {}

        def save_batch(self, batch: MemoryWriteBatch) -> None:
            self.batches[batch.batch_id] = batch

        def get_batch(self, batch_id: str) -> MemoryWriteBatch | None:
            return self.batches.get(batch_id)

        def list_batches(self, job_id: str) -> list[MemoryWriteBatch]:
            return [batch for batch in self.batches.values() if batch.job_id == job_id]

    class FakeRunStore:
        def create_run(self, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def get_run(self, run_id: str):
            return None

        def list_runs(self):
            return []

        def update_run(self, run_id: str, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def create_node_execution(self, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def get_node_execution(self, execution_id: str):
            return None

        def list_node_executions(self, run_id: str):
            return []

        def update_node_execution(self, execution_id: str, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    class FakeRecallBackend:
        def recall_memory(self, query: MemoryQuery) -> MemoryBundle:
            return MemoryBundle(
                semantic_memory=(
                    MemoryEntry(
                        memory_id=f"semantic:{query.job.job_id}",
                        kind="semantic",
                        content=query.query_text,
                    ),
                )
            )

    class FakeStagingBackend:
        def stage_memory_candidates(
            self,
            decision: FinalTranscriptDecision | FinalTranslationDecision,
            *,
            source_stage: str,
        ) -> MemoryWriteBatch:
            return MemoryWriteBatch(
                batch_id=f"batch:{decision.job_id}",
                job_id=decision.job_id,
                source_stage=source_stage,
            )

    job = _job_context()
    request_context = RequestContext(
        run_id="run-123",
        job=job,
        source_artifact_ref="jobs/run-123-request.json",
    )

    extractor = FakeExtractor()
    transcriber = FakeTranscriber()
    translator = FakeTranslator()
    decision_store = FakeDecisionStore()
    memory_batch_store = FakeMemoryBatchStore()
    run_store = FakeRunStore()
    recall_backend = FakeRecallBackend()
    staging_backend = FakeStagingBackend()
    blob_store = LocalBlobStore(tmp_path / "blobs")
    trace_sink = NoOpTraceSink()

    assert isinstance(extractor, AudioExtractionAdapter)
    assert isinstance(transcriber, TranscriptionAdapter)
    assert isinstance(translator, TranslationAdapter)
    assert isinstance(decision_store, DecisionStore)
    assert isinstance(memory_batch_store, MemoryBatchStore)
    assert isinstance(run_store, RunStore)
    assert isinstance(recall_backend, MemoryRecallBackend)
    assert isinstance(staging_backend, MemoryStagingBackend)
    assert isinstance(blob_store, BlobStore)
    assert isinstance(trace_sink, TraceSink)

    audio_artifact = extractor.extract_audio("source.mp4", request_context)
    transcript_candidate = transcriber.transcribe(audio_artifact, request_context)
    translation_candidate = translator.generate_translation(
        transcript_candidate,
        "variant-a",
        request_context,
    )
    decision_store.save_transcript_candidate(transcript_candidate)
    decision_store.save_translation_candidate(translation_candidate)
    transcript_decision = FinalTranscriptDecision(
        job_id=job.job_id,
        winner_candidate_id=transcript_candidate.candidate_id,
        decision_mode="automatic_finalize",
        decision_confidence=0.75,
        rationale_summary="single fake candidate",
    )
    translation_decision = FinalTranslationDecision(
        job_id=job.job_id,
        winner_candidate_id=translation_candidate.candidate_id,
        decision_mode="automatic_finalize",
        decision_confidence=0.75,
        rationale_summary="single fake candidate",
    )
    decision_store.save_transcript_decision(transcript_decision)
    decision_store.save_translation_decision(translation_decision)
    batch = staging_backend.stage_memory_candidates(
        translation_decision,
        source_stage="translation_adjudication",
    )
    memory_batch_store.save_batch(batch)
    recalled = recall_backend.recall_memory(
        MemoryQuery(job=job, stage="review", query_text="brand names")
    )
    blob_entry = blob_store.put_bytes("contracts/check.txt", b"ok")

    trace_path = tmp_path / "trace.jsonl"
    with JsonlTraceSink(trace_path) as jsonl_trace_sink:
        assert isinstance(jsonl_trace_sink, TraceSink)
        jsonl_trace_sink.record(TraceEvent(name="contracts.checked", run_id=request_context.run_id))

    assert decision_store.get_transcript_decision(job.job_id) == transcript_decision
    assert decision_store.get_translation_decision(job.job_id) == translation_decision
    assert memory_batch_store.get_batch(batch.batch_id) == batch
    assert recalled.semantic_memory[0].content == "brand names"
    assert blob_entry.size_bytes == 2
    assert trace_path.exists()
