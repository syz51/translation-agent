"""Workflow runtime dependencies plus fake and real adapter builders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from translation_agent.adapters import (
    AssemblyAITranscriptionAdapter,
    AudioExtractionAdapter,
    DeepgramTranscriptionAdapter,
    FFmpegAudioExtractionAdapter,
    OpenAITranslationAdapter,
    RetryPolicy,
    SpeechmaticsTranscriptionAdapter,
    TranscriptionAdapter,
    TranslationAdapter,
)
from translation_agent.config import (
    Settings,
    resolve_transcription_providers,
    validate_provider_configuration,
    validate_runtime_compatibility,
)
from translation_agent.graph._langgraph_compat import ensure_langgraph_runtime_supported
from translation_agent.memory import (
    BlobBackedLongTermMemoryStore,
    DeterministicMemoryConsolidationBackend,
    DeterministicMemoryStagingBackend,
    DeterministicPromptEvolutionBackend,
    LongTermMemoryRecallBackend,
    MemoryConsolidationBackend,
    MemoryRecallBackend,
    MemoryStagingBackend,
    PromptEvolutionBackend,
    PromptResolver,
    ProposalBackedPromptResolver,
)
from translation_agent.models import (
    AudioArtifact,
    FinalTranscriptDecision,
    FinalTranslationDecision,
    MemoryWriteBatch,
    RequestContext,
    Segment,
    TranscriptCandidate,
    TranslationCandidate,
)
from translation_agent.observability import TraceSink
from translation_agent.parallelism import RuntimeParallelismPolicy
from translation_agent.storage import (
    BlobStore,
    DecisionStore,
    MemoryBatchStore,
    RunStore,
    job_path,
    job_scope_token,
)

PHASE_TWO_NORMALIZATION_VERSION = "2026-03-30-phase-2"
PHASE_THREE_NORMALIZATION_VERSION = "2026-04-03-phase-4"
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
    memory_consolidation_backend: MemoryConsolidationBackend
    prompt_evolution_backend: PromptEvolutionBackend
    prompt_resolver: PromptResolver
    parallelism: RuntimeParallelismPolicy
    source_artifact_ref: str
    scenario: str = DEFAULT_SCENARIO
    adapter_mode: str = "fake"
    normalization_version: str = PHASE_TWO_NORMALIZATION_VERSION


@dataclass(slots=True)
class RealRuntimeOverrides:
    """Optional real-adapter overrides used in tests."""

    audio_extractor: AudioExtractionAdapter | None = None
    transcription_adapters: tuple[TranscriptionAdapter, ...] | None = None
    translation_adapter: TranslationAdapter | None = None


class InMemoryDecisionStore:
    """Reference implementation for candidate and decision persistence."""

    def __init__(self) -> None:
        self._transcript_candidates: dict[str, TranscriptCandidate] = {}
        self._transcript_candidate_job_ids: dict[str, str] = {}
        self._translation_candidates: dict[str, TranslationCandidate] = {}
        self._translation_candidate_job_ids: dict[str, str] = {}
        self._transcript_decisions: dict[str, FinalTranscriptDecision] = {}
        self._translation_decisions: dict[str, FinalTranslationDecision] = {}
        self._investigations: dict[tuple[str, str], dict[str, object]] = {}

    def save_transcript_candidate(
        self,
        candidate: TranscriptCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._transcript_candidates[candidate.candidate_id] = candidate
        self._transcript_candidate_job_ids[candidate.candidate_id] = (
            storage_job_id or candidate.job_id
        )

    def list_transcript_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranscriptCandidate]:
        resolved_job_id = storage_job_id or job_id
        candidates = [
            candidate
            for candidate in self._transcript_candidates.values()
            if self._transcript_candidate_job_ids.get(candidate.candidate_id) == resolved_job_id
        ]
        return sorted(candidates, key=lambda candidate: candidate.candidate_id)

    def save_transcript_decision(
        self,
        decision: FinalTranscriptDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._transcript_decisions[storage_job_id or decision.job_id] = decision

    def get_transcript_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranscriptDecision | None:
        return self._transcript_decisions.get(storage_job_id or job_id)

    def save_translation_candidate(
        self,
        candidate: TranslationCandidate,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._translation_candidates[candidate.candidate_id] = candidate
        self._translation_candidate_job_ids[candidate.candidate_id] = (
            storage_job_id or candidate.job_id
        )

    def list_translation_candidates(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[TranslationCandidate]:
        resolved_job_id = storage_job_id or job_id
        candidates = [
            candidate
            for candidate in self._translation_candidates.values()
            if self._translation_candidate_job_ids.get(candidate.candidate_id) == resolved_job_id
        ]
        return sorted(candidates, key=lambda candidate: candidate.candidate_id)

    def save_translation_decision(
        self,
        decision: FinalTranslationDecision,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._translation_decisions[storage_job_id or decision.job_id] = decision

    def get_translation_decision(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> FinalTranslationDecision | None:
        return self._translation_decisions.get(storage_job_id or job_id)

    def save_investigation(
        self,
        *,
        job_id: str,
        stage: str,
        payload: dict[str, object],
        storage_job_id: str | None = None,
    ) -> None:
        self._investigations[(storage_job_id or job_id, stage)] = payload

    def get_investigation(
        self,
        *,
        job_id: str,
        stage: str,
        storage_job_id: str | None = None,
    ) -> dict[str, object] | None:
        return self._investigations.get((storage_job_id or job_id, stage))


class InMemoryMemoryBatchStore:
    """Reference implementation for adjudication-boundary memory staging."""

    def __init__(self) -> None:
        self._batches: dict[str, MemoryWriteBatch] = {}
        self._batch_job_ids: dict[str, str] = {}

    def save_batch(
        self,
        batch: MemoryWriteBatch,
        *,
        storage_job_id: str | None = None,
    ) -> None:
        self._batches[batch.batch_id] = batch
        self._batch_job_ids[batch.batch_id] = storage_job_id or batch.job_id

    def get_batch(self, batch_id: str) -> MemoryWriteBatch | None:
        return self._batches.get(batch_id)

    def list_batches(
        self,
        job_id: str,
        *,
        storage_job_id: str | None = None,
    ) -> list[MemoryWriteBatch]:
        resolved_job_id = storage_job_id or job_id
        batches = [
            batch
            for batch in self._batches.values()
            if self._batch_job_ids.get(batch.batch_id) == resolved_job_id
        ]
        return sorted(batches, key=lambda batch: batch.batch_id)


class FakeAudioExtractionAdapter:
    """Deterministic extraction adapter used by the dry-run workflow."""

    adapter_id = "fake-ffmpeg"

    def extract_audio(self, video_ref: str, job_context: RequestContext) -> AudioArtifact:
        scope_token = job_scope_token(job_context.job)
        return AudioArtifact(
            artifact_id=f"audio-{job_context.job.job_id}-{scope_token}",
            job_id=job_context.job.job_id,
            blob_ref=job_path(job_context.job, "artifacts", "audio.wav"),
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

    def __init__(self, provider_id: str, *, blob_store: BlobStore) -> None:
        self.provider_id = provider_id
        self._blob_store = blob_store

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> TranscriptCandidate:
        candidate, raw_payload = self.transcribe_with_payload(audio_artifact, request_context)
        self._blob_store.put_bytes(
            candidate.raw_payload_ref or "",
            (json.dumps(raw_payload, sort_keys=True) + "\n").encode("utf-8"),
        )
        return candidate

    def transcribe_with_payload(
        self,
        audio_artifact: AudioArtifact,
        request_context: RequestContext,
    ) -> tuple[TranscriptCandidate, dict[str, object]]:
        scenario = str(request_context.metadata.get("scenario", DEFAULT_SCENARIO))
        failed_providers = _transcription_failures_for_scenario(scenario)
        if self.provider_id in failed_providers:
            raise RuntimeError(f"simulated transcription failure for {self.provider_id}")

        text = _transcript_text_for_provider(self.provider_id, scenario)
        raw_payload: dict[str, object] = {"provider": self.provider_id, "text": text}
        raw_payload_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"{self.provider_id}.json",
        )
        scope_token = job_scope_token(request_context.job)
        return (
            TranscriptCandidate(
                candidate_id=f"tr-{self.provider_id}-{request_context.job.job_id}-{scope_token}",
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
                raw_payload_ref=raw_payload_ref,
                normalization_version=PHASE_TWO_NORMALIZATION_VERSION,
                metadata={"provider_rank": _provider_rank(self.provider_id)},
            ),
            raw_payload,
        )


class FakeTranslationAdapter:
    """Deterministic translation adapter with scenario-driven failure injection."""

    model_id = "gpt-5-mini"

    def __init__(self, *, blob_store: BlobStore) -> None:
        self._blob_store = blob_store

    def generate_translation(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> TranslationCandidate:
        candidate, raw_payload = self.generate_translation_with_payload(
            final_transcript,
            prompt_variant_id,
            request_context,
        )
        self._blob_store.put_bytes(
            candidate.raw_response_ref or "",
            (json.dumps(raw_payload, sort_keys=True) + "\n").encode("utf-8"),
        )
        return candidate

    def generate_translation_with_payload(
        self,
        final_transcript: TranscriptCandidate,
        prompt_variant_id: str,
        request_context: RequestContext,
    ) -> tuple[TranslationCandidate, dict[str, object]]:
        scenario = str(request_context.metadata.get("scenario", DEFAULT_SCENARIO))
        failed_variants = _translation_failures_for_scenario(scenario)
        if prompt_variant_id in failed_variants:
            raise RuntimeError(f"simulated translation failure for {prompt_variant_id}")

        resolved_prompt = request_context.metadata.get("resolved_translation_prompt", {})
        text = _translation_text_for_variant(prompt_variant_id, scenario)
        raw_payload: dict[str, object] = {
            "translation": text,
            "variant": prompt_variant_id,
        }
        raw_response_ref = job_path(
            request_context.job,
            "raw",
            "provider-payloads",
            f"openai-{prompt_variant_id}.json",
        )
        scope_token = job_scope_token(request_context.job)
        return (
            TranslationCandidate(
                candidate_id=f"tl-{prompt_variant_id}-{request_context.job.job_id}-{scope_token}",
                job_id=request_context.job.job_id,
                source_transcript_candidate_id=final_transcript.candidate_id,
                model_id=self.model_id,
                prompt_variant_id=prompt_variant_id,
                prompt_version=str(resolved_prompt.get("effective_prompt_version", "phase-2-v1")),
                language=request_context.job.target_language,
                segments=(
                    Segment(
                        segment_id=(
                            f"seg-{prompt_variant_id}-1"
                            if prompt_variant_id in {"variant-a", "variant-b"}
                            else final_transcript.segments[0].segment_id
                            if final_transcript.segments
                            else f"seg-{prompt_variant_id}-1"
                        ),
                        start_ms=(
                            final_transcript.segments[0].start_ms
                            if final_transcript.segments
                            else 0
                        ),
                        end_ms=(
                            final_transcript.segments[0].end_ms
                            if final_transcript.segments
                            else 1_200
                        ),
                        speaker="speaker-1",
                        source_text=final_transcript.full_text,
                        target_text=text,
                    ),
                ),
                full_text=text,
                raw_response_ref=raw_response_ref,
                normalization_version=PHASE_TWO_NORMALIZATION_VERSION,
                metadata={"scenario": scenario, "prompt_resolver": resolved_prompt},
            ),
            raw_payload,
        )


def build_phase_two_runtime(
    *,
    blob_store: BlobStore,
    run_store: RunStore,
    decision_store: DecisionStore | None = None,
    memory_batch_store: MemoryBatchStore | None = None,
    trace_sink: TraceSink,
    source_artifact_ref: str,
    scenario: str = DEFAULT_SCENARIO,
) -> WorkflowRuntime:
    """Construct the default dry-run runtime used by the public entrypoints."""

    memory_store = BlobBackedLongTermMemoryStore(blob_store)
    resolved_decision_store = decision_store or _decision_store_for_run_store(run_store)
    resolved_memory_batch_store = memory_batch_store or _memory_batch_store_for_run_store(run_store)
    return WorkflowRuntime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=trace_sink,
        decision_store=resolved_decision_store,
        memory_batch_store=resolved_memory_batch_store,
        audio_extractor=FakeAudioExtractionAdapter(),
        transcription_adapters=(
            FakeTranscriptionAdapter("assemblyai", blob_store=blob_store),
            FakeTranscriptionAdapter("speechmatics", blob_store=blob_store),
            FakeTranscriptionAdapter("deepgram", blob_store=blob_store),
        ),
        translation_adapter=FakeTranslationAdapter(blob_store=blob_store),
        memory_recall_backend=LongTermMemoryRecallBackend(memory_store),
        memory_staging_backend=DeterministicMemoryStagingBackend(),
        memory_consolidation_backend=DeterministicMemoryConsolidationBackend(memory_store),
        prompt_evolution_backend=DeterministicPromptEvolutionBackend(),
        prompt_resolver=ProposalBackedPromptResolver(blob_store),
        parallelism=_default_parallelism_policy(provider_count=3),
        source_artifact_ref=source_artifact_ref,
        scenario=scenario,
        adapter_mode="fake",
        normalization_version=PHASE_TWO_NORMALIZATION_VERSION,
    )


def build_runtime(
    *,
    settings: Settings,
    blob_store: BlobStore,
    run_store: RunStore,
    decision_store: DecisionStore | None = None,
    memory_batch_store: MemoryBatchStore | None = None,
    trace_sink: TraceSink,
    source_artifact_ref: str,
    scenario: str = DEFAULT_SCENARIO,
    real_overrides: RealRuntimeOverrides | None = None,
) -> WorkflowRuntime:
    """Construct the configured runtime while preserving the Phase 2 fake path."""

    if settings.adapter_mode == "fake":
        return build_phase_two_runtime(
            blob_store=blob_store,
            run_store=run_store,
            decision_store=decision_store,
            memory_batch_store=memory_batch_store,
            trace_sink=trace_sink,
            source_artifact_ref=source_artifact_ref,
            scenario=scenario,
        )

    return build_phase_three_runtime(
        settings=settings,
        blob_store=blob_store,
        run_store=run_store,
        decision_store=decision_store,
        memory_batch_store=memory_batch_store,
        trace_sink=trace_sink,
        source_artifact_ref=source_artifact_ref,
        scenario=scenario,
        overrides=real_overrides,
    )


def build_phase_three_runtime(
    *,
    settings: Settings,
    blob_store: BlobStore,
    run_store: RunStore,
    decision_store: DecisionStore | None = None,
    memory_batch_store: MemoryBatchStore | None = None,
    trace_sink: TraceSink,
    source_artifact_ref: str,
    scenario: str = DEFAULT_SCENARIO,
    overrides: RealRuntimeOverrides | None = None,
) -> WorkflowRuntime:
    """Construct the real-adapter Phase 3 runtime."""

    config_error = validate_provider_configuration(settings)
    if config_error is not None:
        raise RuntimeError(config_error)
    compatibility_error = validate_runtime_compatibility(settings)
    if compatibility_error is not None:
        raise RuntimeError(compatibility_error)
    ensure_langgraph_runtime_supported()

    retry_policy = RetryPolicy(
        max_attempts=settings.adapter_retry_attempts,
        initial_backoff_seconds=settings.adapter_initial_backoff_seconds,
        max_backoff_seconds=settings.adapter_max_backoff_seconds,
        poll_interval_seconds=settings.adapter_poll_interval_seconds,
        max_polls=settings.adapter_poll_attempts,
    )
    overrides = overrides or RealRuntimeOverrides()
    audio_extractor = overrides.audio_extractor or FFmpegAudioExtractionAdapter(
        blob_store=blob_store,
        binary=settings.ffmpeg_binary,
        retry_policy=RetryPolicy(
            max_attempts=settings.adapter_retry_attempts,
            initial_backoff_seconds=settings.adapter_initial_backoff_seconds,
            max_backoff_seconds=settings.adapter_max_backoff_seconds,
        ),
    )
    transcription_adapters = overrides.transcription_adapters or _build_real_transcription_adapters(
        settings=settings,
        blob_store=blob_store,
        retry_policy=retry_policy,
    )
    translation_adapter = overrides.translation_adapter or OpenAITranslationAdapter(
        blob_store=blob_store,
        api_key=_required_setting(settings.openai_api_key, "TA_OPENAI_API_KEY"),
        model_id=settings.translation_model_id,
        prompt_version=settings.translation_prompt_version,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.translation_timeout_seconds,
        retry_policy=retry_policy,
        max_chunk_workers=settings.translation_chunk_max_workers,
        max_chunk_characters=settings.translation_max_chunk_characters,
        max_chunk_segments=settings.translation_max_chunk_segments,
        context_segment_window=settings.translation_context_segment_window,
    )

    memory_store = BlobBackedLongTermMemoryStore(blob_store)
    resolved_decision_store = decision_store or _decision_store_for_run_store(run_store)
    resolved_memory_batch_store = memory_batch_store or _memory_batch_store_for_run_store(run_store)
    return WorkflowRuntime(
        blob_store=blob_store,
        run_store=run_store,
        trace_sink=trace_sink,
        decision_store=resolved_decision_store,
        memory_batch_store=resolved_memory_batch_store,
        audio_extractor=audio_extractor,
        transcription_adapters=transcription_adapters,
        translation_adapter=translation_adapter,
        memory_recall_backend=LongTermMemoryRecallBackend(memory_store),
        memory_staging_backend=DeterministicMemoryStagingBackend(),
        memory_consolidation_backend=DeterministicMemoryConsolidationBackend(memory_store),
        prompt_evolution_backend=DeterministicPromptEvolutionBackend(),
        prompt_resolver=ProposalBackedPromptResolver(blob_store),
        parallelism=_settings_parallelism_policy(
            settings,
            provider_count=len(transcription_adapters),
        ),
        source_artifact_ref=source_artifact_ref,
        scenario=scenario,
        adapter_mode="real",
        normalization_version=PHASE_THREE_NORMALIZATION_VERSION,
    )


def runtime_metadata(base_metadata: dict[str, Any], runtime: WorkflowRuntime) -> dict[str, Any]:
    """Attach workflow runtime metadata to a request context."""

    return {
        **base_metadata,
        "scenario": runtime.scenario,
        "adapter_mode": runtime.adapter_mode,
        "blob_root": _blob_root(runtime.blob_store),
        "normalization_version": runtime.normalization_version,
    }


def _required_setting(value: str | None, env_var: str) -> str:
    if value:
        return value
    raise RuntimeError(f"{env_var} is required when TA_ADAPTER_MODE=real")


def _build_real_transcription_adapters(
    *,
    settings: Settings,
    blob_store: BlobStore,
    retry_policy: RetryPolicy,
) -> tuple[TranscriptionAdapter, ...]:
    adapters: list[TranscriptionAdapter] = []
    for provider_id in resolve_transcription_providers(settings):
        if provider_id == "assemblyai":
            adapters.append(
                AssemblyAITranscriptionAdapter(
                    blob_store=blob_store,
                    api_key=_required_setting(settings.assemblyai_api_key, "TA_ASSEMBLYAI_API_KEY"),
                    base_url=settings.assemblyai_base_url,
                    timeout_seconds=settings.assemblyai_timeout_seconds,
                    retry_policy=retry_policy,
                )
            )
            continue
        if provider_id == "speechmatics":
            adapters.append(
                SpeechmaticsTranscriptionAdapter(
                    blob_store=blob_store,
                    api_key=_required_setting(
                        settings.speechmatics_api_key,
                        "TA_SPEECHMATICS_API_KEY",
                    ),
                    base_url=settings.speechmatics_base_url,
                    timeout_seconds=settings.provider_timeout_seconds,
                    retry_policy=retry_policy,
                )
            )
            continue
        adapters.append(
            DeepgramTranscriptionAdapter(
                blob_store=blob_store,
                api_key=_required_setting(settings.deepgram_api_key, "TA_DEEPGRAM_API_KEY"),
                base_url=settings.deepgram_base_url,
                utterance_split_seconds=settings.deepgram_utterance_split_seconds,
                timeout_seconds=settings.provider_timeout_seconds,
                retry_policy=retry_policy,
            )
        )
    if not adapters:
        raise RuntimeError("TA_TRANSCRIPTION_PROVIDERS must select at least one provider when set")
    return tuple(adapters)


def _default_parallelism_policy(*, provider_count: int) -> RuntimeParallelismPolicy:
    return RuntimeParallelismPolicy(
        transcription_max_workers=min(max(provider_count, 1), 4),
        translation_candidate_max_workers=2,
        translation_chunk_max_workers=4,
        review_max_workers=2,
        reference_evaluation_max_workers=4,
        memory_drain_max_workers=2,
    )


def _settings_parallelism_policy(
    settings: Settings,
    *,
    provider_count: int,
) -> RuntimeParallelismPolicy:
    defaults = _default_parallelism_policy(provider_count=provider_count)
    return RuntimeParallelismPolicy(
        transcription_max_workers=settings.transcription_max_workers
        or defaults.transcription_max_workers,
        translation_candidate_max_workers=settings.translation_candidate_max_workers,
        translation_chunk_max_workers=settings.translation_chunk_max_workers,
        review_max_workers=settings.review_max_workers,
        reference_evaluation_max_workers=settings.reference_evaluation_max_workers,
        memory_drain_max_workers=settings.memory_drain_max_workers,
    )


def _blob_root(blob_store: BlobStore) -> str | None:
    root = getattr(blob_store, "root", None)
    if root is None:
        return None
    return str(root)


def _decision_store_for_run_store(run_store: RunStore) -> DecisionStore:
    if isinstance(run_store, DecisionStore):
        return run_store
    return InMemoryDecisionStore()


def _memory_batch_store_for_run_store(run_store: RunStore) -> MemoryBatchStore:
    if isinstance(run_store, MemoryBatchStore):
        return run_store
    return InMemoryMemoryBatchStore()


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
    if prompt_variant_id not in {"variant-a", "variant-b"}:
        prompt_variant_id = "variant-a"
    text_by_scenario = {
        "happy": {
            "variant-a": "Bonjour tout le monde depuis le workflow.",
            "variant-b": "Salut tout le monde depuis le workflow.",
        },
        "translation_conflict": {
            "variant-a": "Bonjour a tous depuis le flux de travail.",
            "variant-b": "Salut tout le monde depuis le pipeline.",
        },
        "translation_conflict_timeout": {
            "variant-a": "Bonjour a tous depuis le flux de travail.",
            "variant-b": "Salut tout le monde depuis le pipeline.",
        },
        "translation_high_risk": {
            "variant-a": "Bonjour a tous depuis le flux de travail.",
            "variant-b": "Salut tout le monde depuis le pipeline.",
        },
        "translation_escalation": {
            "variant-a": "Bonjour a tous. Le flux de travail annule le sens source.",
            "variant-b": "Salut, on improvise le pipeline au lieu du workflow.",
        },
    }
    text_by_variant = text_by_scenario.get(
        scenario,
        text_by_scenario["happy"],
    )
    return text_by_variant[prompt_variant_id]
