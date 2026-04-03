from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import translation_agent.graph.runtime as runtime_module
from translation_agent.config import Settings
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
    assert captured_kwargs["max_chunk_workers"] == 4
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

    assert runtime.parallelism.transcription_max_workers == 2
    assert runtime.parallelism.translation_candidate_max_workers == 3
    assert runtime.parallelism.translation_chunk_max_workers == 5
    assert runtime.parallelism.review_max_workers == 4
    assert runtime.parallelism.reference_evaluation_max_workers == 6
    assert runtime.parallelism.memory_drain_max_workers == 2


def test_phase_three_runtime_defaults_transcription_workers_to_selected_provider_count(
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

    assert runtime.parallelism.transcription_max_workers == 2


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
