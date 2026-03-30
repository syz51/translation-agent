"""Phase 2 workflow runtime dependencies and fake implementations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from translation_agent.adapters import (
    AudioExtractionAdapter,
    TranscriptionAdapter,
    TranslationAdapter,
)
from translation_agent.memory import MemoryRecallBackend, MemoryStagingBackend
from translation_agent.models import (
    AudioArtifact,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    MemoryBundle,
    MemoryEntry,
    MemoryQuery,
    MemoryWrite,
    MemoryWriteBatch,
    ProviderCaveat,
    RequestContext,
    Segment,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.observability import TraceSink
from translation_agent.storage import BlobStore, DecisionStore, MemoryBatchStore, RunStore

PHASE_TWO_NORMALIZATION_VERSION = "2026-03-30-phase-2"
DEFAULT_SCENARIO = "happy"


@dataclass(slots=True)
class WorkflowRuntime:
    """Dependencies required to execute the deterministic Phase 2 graph."""

    blob_store: BlobStore
    run_store: RunStore
    trace_sink: TraceSink
    decision_store: DecisionStore
    memory_batch_store: MemoryBatchStore
    audio_extractor: AudioExtractionAdapter
    transcription_adapters: tuple[TranscriptionAdapter, ...]
    translation_adapter: TranslationAdapter
    memory_recall_backend: MemoryRecallBackend
    memory_staging_backend: MemoryStagingBackend
    source_artifact_ref: str
    scenario: str = DEFAULT_SCENARIO


class InMemoryDecisionStore:
    """Reference implementation for candidate and decision persistence."""

    def __init__(self) -> None:
        self._transcript_candidates: dict[str, TranscriptCandidate] = {}
        self._translation_candidates: dict[str, TranslationCandidate] = {}
        self._transcript_decisions: dict[str, FinalTranscriptDecision] = {}
        self._translation_decisions: dict[str, FinalTranslationDecision] = {}

    def save_transcript_candidate(self, candidate: TranscriptCandidate) -> None:
        self._transcript_candidates[candidate.candidate_id] = candidate

    def list_transcript_candidates(self, job_id: str) -> list[TranscriptCandidate]:
        candidates = [
            candidate
            for candidate in self._transcript_candidates.values()
            if candidate.job_id == job_id
        ]
        return sorted(candidates, key=lambda candidate: candidate.candidate_id)

    def save_transcript_decision(self, decision: FinalTranscriptDecision) -> None:
        self._transcript_decisions[decision.job_id] = decision

    def get_transcript_decision(self, job_id: str) -> FinalTranscriptDecision | None:
        return self._transcript_decisions.get(job_id)

    def save_translation_candidate(self, candidate: TranslationCandidate) -> None:
        self._translation_candidates[candidate.candidate_id] = candidate

    def list_translation_candidates(self, job_id: str) -> list[TranslationCandidate]:
        candidates = [
            candidate
            for candidate in self._translation_candidates.values()
            if candidate.job_id == job_id
        ]
        return sorted(candidates, key=lambda candidate: candidate.candidate_id)

    def save_translation_decision(self, decision: FinalTranslationDecision) -> None:
        self._translation_decisions[decision.job_id] = decision

    def get_translation_decision(self, job_id: str) -> FinalTranslationDecision | None:
        return self._translation_decisions.get(job_id)


class InMemoryMemoryBatchStore:
    """Reference implementation for adjudication-boundary memory staging."""

    def __init__(self) -> None:
        self._batches: dict[str, MemoryWriteBatch] = {}

    def save_batch(self, batch: MemoryWriteBatch) -> None:
        self._batches[batch.batch_id] = batch

    def get_batch(self, batch_id: str) -> MemoryWriteBatch | None:
        return self._batches.get(batch_id)

    def list_batches(self, job_id: str) -> list[MemoryWriteBatch]:
        batches = [batch for batch in self._batches.values() if batch.job_id == job_id]
        return sorted(batches, key=lambda batch: batch.batch_id)


class FakeAudioExtractionAdapter:
    """Deterministic extraction adapter used by the dry-run workflow."""

    adapter_id = "fake-ffmpeg"

    def extract_audio(self, video_ref: str, job_context: RequestContext) -> AudioArtifact:
        return AudioArtifact(
            artifact_id=f"audio-{job_context.job.job_id}",
            job_id=job_context.job.job_id,
            blob_ref=f"audio/{job_context.job.job_id}.wav",
            duration_ms=61_000,
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
            extraction_metadata={
                "adapter_id": self.adapter_id,
                "source_video_ref": video_ref,
                "generated_at": datetime.now(UTC).isoformat(),
            },
        )


class FakeTranscriptionAdapter:
    """Deterministic STT adapter with scenario-driven failure injection."""

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate:
        scenario = str(request_context.metadata.get("scenario", DEFAULT_SCENARIO))
        failed_providers = _transcription_failures_for_scenario(scenario)
        if self.provider_id in failed_providers:
            raise RuntimeError(f"simulated transcription failure for {self.provider_id}")

        text = _transcript_text_for_provider(self.provider_id, scenario)
        return TranscriptCandidate(
            candidate_id=f"tr-{self.provider_id}-{request_context.job.job_id}",
            job_id=request_context.job.job_id,
            provider_id=self.provider_id,
            provider_request_id=f"req-{self.provider_id}-{request_context.run_id}",
            language=request_context.job.source_language,
            segments=(
                Segment(
                    segment_id=f"seg-{self.provider_id}-1",
                    start_ms=0,
                    end_ms=1_200,
                    speaker="speaker-1",
                    source_text=text,
                ),
            ),
            full_text=text,
            speaker_map={"speaker-1": "Host"},
            timing_resolution="segment",
            raw_payload_ref=f"raw/transcripts/{request_context.job.job_id}/{self.provider_id}.json",
            normalization_version=PHASE_TWO_NORMALIZATION_VERSION,
            metadata={"provider_rank": _provider_rank(self.provider_id)},
        )


class FakeTranslationAdapter:
    """Deterministic translation adapter with scenario-driven failure injection."""

    model_id = "gpt-5.4-mini"

    def generate_translation(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> TranslationCandidate:
        scenario = str(request_context.metadata.get("scenario", DEFAULT_SCENARIO))
        failed_variants = _translation_failures_for_scenario(scenario)
        if prompt_variant_id in failed_variants:
            raise RuntimeError(f"simulated translation failure for {prompt_variant_id}")

        text = _translation_text_for_variant(prompt_variant_id, scenario)
        return TranslationCandidate(
            candidate_id=f"tl-{prompt_variant_id}-{request_context.job.job_id}",
            job_id=request_context.job.job_id,
            source_transcript_candidate_id=final_transcript.candidate_id,
            model_id=self.model_id,
            prompt_variant_id=prompt_variant_id,
            prompt_version="phase-2-v1",
            language=request_context.job.target_language,
            segments=(
                Segment(
                    segment_id=f"seg-{prompt_variant_id}-1",
                    start_ms=0,
                    end_ms=1_200,
                    speaker="speaker-1",
                    source_text=final_transcript.full_text,
                    target_text=text,
                ),
            ),
            full_text=text,
            raw_response_ref=(
                f"raw/translations/{request_context.job.job_id}/{prompt_variant_id}.json"
            ),
            normalization_version=PHASE_TWO_NORMALIZATION_VERSION,
            metadata={"scenario": scenario},
        )


class FakeMemoryRecallBackend:
    """Dry-run recall backend returning a small deterministic memory slice."""

    def recall_memory(self, query: MemoryQuery) -> MemoryBundle:
        return MemoryBundle(
            semantic_memory=(
                MemoryEntry(
                    memory_id=f"semantic:{query.stage}:{query.job.project_id}",
                    kind="semantic",
                    content="Preserve named entities and product terms.",
                    source_ref="memory/semantic/dry-run",
                    score=0.74,
                ),
            ),
            glossary=(
                MemoryEntry(
                    memory_id=f"glossary:{query.job.target_language}",
                    kind="glossary",
                    content="OpenAI -> OpenAI",
                    source_ref="memory/glossary/dry-run",
                ),
            ),
            rules=(
                MemoryEntry(
                    memory_id=f"rule:{query.stage}",
                    kind="rule",
                    content=f"Prefer deterministic dry-run behavior for {query.stage}.",
                    source_ref="memory/rules/dry-run",
                ),
            ),
            provider_caveats=(
                ProviderCaveat(
                    provider_id="speechmatics",
                    note="The fake provider can be disabled for degraded-path tests.",
                ),
            ),
        )


class FakeMemoryStagingBackend:
    """Dry-run staging backend that emits deterministic memory write batches."""

    def stage_memory_candidates(
        self,
        decision: FinalTranscriptDecision | FinalTranslationDecision,
        *,
        source_stage: str,
    ) -> MemoryWriteBatch:
        summary = decision.rationale_summary
        semantic_writes: tuple[MemoryWrite, ...] = ()
        if decision.winner_candidate_id is not None:
            semantic_writes = (
                MemoryWrite(
                    kind="semantic",
                    content=summary,
                    source_ref=f"decisions/{source_stage}/{decision.job_id}.json",
                ),
            )
        return MemoryWriteBatch(
            batch_id=f"batch-{source_stage}-{decision.job_id}",
            job_id=decision.job_id,
            source_stage=source_stage,
            semantic_writes=semantic_writes,
            episodic_writes=(
                MemoryWrite(
                    kind="episodic",
                    content=summary,
                    source_ref=f"decisions/{source_stage}/{decision.job_id}.json",
                ),
            ),
            dedupe_keys=(f"{source_stage}:{decision.job_id}",),
        )


def build_phase_two_runtime(
    *,
    blob_store: BlobStore,
    run_store: RunStore,
    trace_sink: TraceSink,
    source_artifact_ref: str,
    scenario: str = DEFAULT_SCENARIO,
) -> WorkflowRuntime:
    """Construct the default dry-run runtime used by the public entrypoints."""

    return WorkflowRuntime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=trace_sink,
        decision_store=InMemoryDecisionStore(),
        memory_batch_store=InMemoryMemoryBatchStore(),
        audio_extractor=FakeAudioExtractionAdapter(),
        transcription_adapters=(
            FakeTranscriptionAdapter("assemblyai"),
            FakeTranscriptionAdapter("speechmatics"),
            FakeTranscriptionAdapter("deepgram"),
        ),
        translation_adapter=FakeTranslationAdapter(),
        memory_recall_backend=FakeMemoryRecallBackend(),
        memory_staging_backend=FakeMemoryStagingBackend(),
        source_artifact_ref=source_artifact_ref,
        scenario=scenario,
    )


def runtime_metadata(base_metadata: dict[str, Any], runtime: WorkflowRuntime) -> dict[str, Any]:
    """Attach workflow runtime metadata to a request context."""

    return {
        **base_metadata,
        "scenario": runtime.scenario,
    }


def _provider_rank(provider_id: str) -> int:
    order = {"assemblyai": 0, "speechmatics": 1, "deepgram": 2}
    return order.get(provider_id, 100)


def _transcription_failures_for_scenario(scenario: str) -> set[str]:
    mapping = {
        "degraded_stt": {"speechmatics"},
        "single_transcript_candidate": {"speechmatics", "deepgram"},
        "transcript_escalation": {"speechmatics"},
    }
    return mapping.get(scenario, set())


def _translation_failures_for_scenario(scenario: str) -> set[str]:
    mapping = {
        "translation_single_variant": {"variant-b"},
        "translation_failed": {"variant-a", "variant-b"},
    }
    return mapping.get(scenario, set())


def _transcript_text_for_provider(provider_id: str, scenario: str) -> str:
    if scenario == "transcript_escalation" and provider_id == "deepgram":
        return "Hello world from the escalation path."

    text_by_provider = {
        "assemblyai": "Hello world from the workflow skeleton.",
        "speechmatics": "Hello world from the workflow skeleton.",
        "deepgram": "Hello world from workflow skeleton.",
    }
    return text_by_provider.get(provider_id, "Hello world.")


def _translation_text_for_variant(prompt_variant_id: str, scenario: str) -> str:
    if scenario == "translation_escalation":
        text_by_variant = {
            "variant-a": "Bonjour a tous depuis le flux de travail.",
            "variant-b": "Salut tout le monde depuis le pipeline.",
        }
    else:
        text_by_variant = {
            "variant-a": "Bonjour tout le monde depuis le workflow.",
            "variant-b": "Salut tout le monde depuis le workflow.",
        }
    return text_by_variant[prompt_variant_id]
