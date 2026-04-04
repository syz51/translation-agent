from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import translation_agent.graph.runtime as runtime_module
from translation_agent.config import Settings, load_settings
from translation_agent.graph.runtime import build_phase_three_runtime
from translation_agent.observability import NoOpTraceSink
from translation_agent.storage import LocalBlobStore

pytestmark = pytest.mark.unit


class InMemoryRunStore:
    pass


def test_phase_three_runtime_uses_translation_timeout_and_chunk_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    captured_kwargs: dict[str, Any] = {}

    class StubTranslationAdapter:
        provider_id = "gemini"
        model_id = "gemini-2.5-flash"

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(runtime_module, "ChatCompletionTranslationAdapter", StubTranslationAdapter)
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",  # pragma: allowlist secret
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        speechmatics_api_key="speech",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
        gemini_api_key="gemini",  # pragma: allowlist secret
        translation_timeout_seconds=95.0,
        translation_max_chunk_characters=4321,
        translation_max_chunk_segments=87,
        translation_context_segment_window=3,
    )

    build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=cast(Any, InMemoryRunStore()),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert captured_kwargs["timeout_seconds"] == 95.0
    assert captured_kwargs["provider_id"] == "gemini"
    assert captured_kwargs["max_chunk_workers"] is None
    assert captured_kwargs["max_chunk_characters"] == 4321
    assert captured_kwargs["max_chunk_segments"] == 87
    assert captured_kwargs["context_segment_window"] == 3


def test_phase_three_runtime_wires_parallelism_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",  # pragma: allowlist secret
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        speechmatics_api_key="speech",  # pragma: allowlist secret
        gemini_api_key="gemini",  # pragma: allowlist secret
        global_parallel_tokens=12,
        transcription_providers="assemblyai,speechmatics",
        transcription_max_workers=2,
        translation_candidate_max_workers=3,
        translation_chunk_max_workers=5,
        review_max_workers=4,
        reference_evaluation_max_workers=6,
        memory_drain_max_workers=2,
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=cast(Any, InMemoryRunStore()),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert runtime.parallelism.global_max_parallel_tokens == 12
    assert runtime.parallelism.transcription_max_workers == 2
    assert runtime.parallelism.translation_candidate_max_workers == 3
    assert runtime.parallelism.translation_chunk_max_workers == 5
    assert runtime.parallelism.review_max_workers == 4
    assert runtime.parallelism.reference_evaluation_max_workers == 6
    assert runtime.parallelism.memory_drain_max_workers == 2
    assert runtime.global_concurrency_limiter.total_tokens == 12


def test_phase_three_runtime_defaults_stage_caps_to_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",  # pragma: allowlist secret
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
        gemini_api_key="gemini",  # pragma: allowlist secret
        transcription_providers="assemblyai,deepgram",
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=cast(Any, InMemoryRunStore()),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert runtime.parallelism.global_max_parallel_tokens == 8
    assert runtime.parallelism.transcription_max_workers is None
    assert runtime.parallelism.translation_candidate_max_workers is None
    assert runtime.parallelism.translation_chunk_max_workers is None
    assert runtime.parallelism.review_max_workers is None
    assert runtime.parallelism.reference_evaluation_max_workers is None
    assert runtime.parallelism.memory_drain_max_workers is None


def test_load_settings_wires_global_parallel_tokens_and_auto_stage_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TA_GLOBAL_PARALLEL_TOKENS", "11")
    monkeypatch.delenv("TA_TRANSCRIPTION_MAX_WORKERS", raising=False)
    monkeypatch.delenv("TA_TRANSLATION_CANDIDATE_MAX_WORKERS", raising=False)
    monkeypatch.delenv("TA_TRANSLATION_CHUNK_MAX_WORKERS", raising=False)
    monkeypatch.delenv("TA_REVIEW_MAX_WORKERS", raising=False)
    monkeypatch.delenv("TA_REFERENCE_EVALUATION_MAX_WORKERS", raising=False)
    monkeypatch.delenv("TA_MEMORY_DRAIN_MAX_WORKERS", raising=False)

    settings = load_settings(env_file=None)

    assert settings.global_parallel_tokens == 11
    assert settings.transcription_max_workers is None
    assert settings.translation_candidate_max_workers is None
    assert settings.translation_chunk_max_workers is None
    assert settings.review_max_workers is None
    assert settings.reference_evaluation_max_workers is None
    assert settings.memory_drain_max_workers is None


def test_phase_three_runtime_exposes_default_reasoning_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",  # pragma: allowlist secret
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
        gemini_api_key="gemini",  # pragma: allowlist secret
        transcription_providers="assemblyai,deepgram",
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=cast(Any, InMemoryRunStore()),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert runtime.reasoning_profile.provider_id == "openai"
    assert runtime.reasoning_profile.model_id == "gpt-5.4"
    assert runtime.reasoning_profile.base_url_source == "openai-sdk-default"
    assert runtime.reasoning_profile.api_key is None
    assert runtime.reasoning_profile.base_url is None
    assert runtime.reasoning_profile.live_adapter_enabled is False
    assert runtime.global_concurrency_limiter.total_tokens == 8


def test_phase_three_runtime_exposes_live_reasoning_profile_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "translation_agent.graph.runtime.ensure_langgraph_runtime_supported",
        lambda: None,
    )
    settings = Settings(
        adapter_mode="real",
        allow_langgraph_py314_warning=True,
        state_db_dsn="postgresql://user:pass@db.example.com:5432/app",  # pragma: allowlist secret
        assemblyai_api_key="assembly",  # pragma: allowlist secret
        speechmatics_api_key="speech",  # pragma: allowlist secret
        deepgram_api_key="deepgram",  # pragma: allowlist secret
        gemini_api_key="gemini",  # pragma: allowlist secret
        openai_api_key="openai",  # pragma: allowlist secret
        openai_base_url="https://api.openai.example/v1/",
        reasoning_provider="openai",
    )

    runtime = build_phase_three_runtime(
        settings=settings,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        run_store=cast(Any, InMemoryRunStore()),
        trace_sink=NoOpTraceSink(),
        source_artifact_ref="jobs/request.json",
    )

    assert runtime.reasoning_profile.api_key == settings.openai_api_key
    assert runtime.reasoning_profile.base_url == "https://api.openai.example/v1/"
    assert runtime.reasoning_profile.live_adapter_enabled is True
