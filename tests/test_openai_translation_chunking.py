from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.adapters import OpenAITranslationAdapter, RetryPolicy
from translation_agent.adapters import openai_translation as openai_translation_module
from translation_agent.adapters.common import HttpRequest, HttpResponse
from translation_agent.models import JobContext, RequestContext, Segment, TranscriptCandidate
from translation_agent.parallelism import GlobalConcurrencyLimiter
from translation_agent.storage import LocalBlobStore

pytestmark = pytest.mark.unit


class SequencedTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = responses
        self.requests: list[HttpRequest] = []

    def request(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected HTTP request")
        return self._responses.pop(0)


def _json_response(payload: object, *, status_code: int = 200) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def _job_context(job_id: str = "job-openai-chunks") -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-1",
        project_id="project-1",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="tester@example.com",
        created_at=datetime.now(UTC),
        media_key="media-fingerprint-test",
    )


def _request_context(job_id: str = "job-openai-chunks") -> RequestContext:
    return RequestContext(
        run_id="run-openai-chunks",
        job=_job_context(job_id),
        source_artifact_ref="artifacts/audio.wav",
        metadata={},
    )


def _transcript_candidate() -> TranscriptCandidate:
    segments = (
        Segment(
            segment_id="seg-1",
            start_ms=0,
            end_ms=1_000,
            speaker="speaker-1",
            source_text="Alpha source",
        ),
        Segment(
            segment_id="seg-2",
            start_ms=1_000,
            end_ms=2_000,
            speaker="speaker-1",
            source_text="Beta source",
        ),
        Segment(
            segment_id="seg-3",
            start_ms=2_000,
            end_ms=3_000,
            speaker="speaker-2",
            source_text="Gamma source",
        ),
    )
    return TranscriptCandidate(
        candidate_id="tr-openai-chunks",
        job_id="job-openai-chunks",
        provider_id="assemblyai",
        provider_request_id="provider-job",
        language="en",
        segments=segments,
        full_text=" ".join(segment.source_text or "" for segment in segments).strip(),
        speaker_map={"speaker-1": "speaker-1", "speaker-2": "speaker-2"},
        timing_resolution="segment",
        raw_payload_ref="raw/provider-payloads/assemblyai.json",
        normalization_version="raw",
        metadata={},
    )


def test_chunk_transcript_respects_limits_and_context() -> None:
    transcript = _transcript_candidate()
    exact_two_segment_budget = openai_translation_module._estimated_user_prompt_size(
        chunk_index=0,
        segments=transcript.segments[:2],
        transcript_segments=transcript.segments,
        start_index=0,
        end_index=2,
        context_segment_window=1,
    )

    chunks = openai_translation_module._chunk_transcript(
        transcript,
        max_chunk_characters=exact_two_segment_budget,
        max_chunk_segments=2,
        context_segment_window=1,
    )

    assert len(chunks) == 2
    assert [segment.segment_id for segment in chunks[0].segments] == ["seg-1", "seg-2"]
    assert [segment.segment_id for segment in chunks[1].segments] == ["seg-3"]
    assert chunks[0].context_before == ()
    assert chunks[0].context_after == ("Gamma source",)
    assert chunks[1].context_before == ("Beta source",)
    assert chunks[1].context_after == ()


def test_openai_translation_adapter_chunks_requests_and_aggregates_results(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate()
    exact_two_segment_budget = openai_translation_module._estimated_user_prompt_size(
        chunk_index=0,
        segments=transcript.segments[:2],
        transcript_segments=transcript.segments,
        start_index=0,
        end_index=2,
        context_segment_window=1,
    )
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "resp-1",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha Bonjour beta",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-2",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour gamma",
                            "segments": [
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                            ],
                        }
                    ),
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=exact_two_segment_budget,
        max_chunk_segments=2,
        context_segment_window=1,
    )

    candidate = adapter.generate_translation(transcript, "variant-a", _request_context())

    assert len(transport.requests) == 2
    assert [segment.segment_id for segment in candidate.segments] == ["seg-1", "seg-2", "seg-3"]
    assert [segment.target_text for segment in candidate.segments] == [
        "Bonjour alpha",
        "Bonjour beta",
        "Bonjour gamma",
    ]
    assert candidate.metadata["chunking"]["chunk_count"] == 2
    assert candidate.metadata["provider"]["response_ids"] == ["resp-1", "resp-2"]
    assert blob_store.exists(candidate.raw_response_ref or "")
    assert candidate.raw_response_ref is not None

    assert transport.requests[0].body is not None
    first_request = json.loads(transport.requests[0].body.decode("utf-8"))
    first_prompt = json.loads(first_request["input"][1]["content"][0]["text"])
    assert first_request["text"]["format"]["type"] == "json_schema"
    assert [segment["segment_id"] for segment in first_prompt["segments"]] == ["seg-1", "seg-2"]
    assert "full_text" not in first_prompt
    assert first_prompt["context_after"] == ["Gamma source"]

    stored_payload = json.loads(blob_store.read_bytes(candidate.raw_response_ref).decode("utf-8"))
    assert stored_payload["chunking"]["chunk_count"] == 2
    assert [chunk["segment_ids"] for chunk in stored_payload["chunks"]] == [
        ["seg-1", "seg-2"],
        ["seg-3"],
    ]


def test_gemini_translation_adapter_uses_chat_completions_endpoint(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "full_text": "Bonjour alpha Bonjour beta Bonjour gamma",
                                        "segments": [
                                            {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                                            {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                            {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                                        ],
                                    }
                                )
                            }
                        }
                    ],
                }
            )
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        provider_id="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=10_000,
        max_chunk_segments=10,
    )

    candidate = adapter.generate_translation(
        _transcript_candidate(),
        "variant-a",
        _request_context(),
    )

    assert candidate.full_text == "Bonjour alpha Bonjour beta Bonjour gamma"
    assert candidate.metadata["provider"]["provider_request_id"] == "chatcmpl-1"
    assert len(transport.requests) == 1
    assert transport.requests[0].url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )

    assert transport.requests[0].body is not None
    request_payload = json.loads(transport.requests[0].body.decode("utf-8"))
    assert [message["role"] for message in request_payload["messages"]] == ["system", "user"]
    assert request_payload["response_format"]["type"] == "json_schema"
    assert request_payload["response_format"]["json_schema"]["name"] == "translation_chunk"


def test_normalized_api_base_url_strips_endpoint_suffixes() -> None:
    assert (
        openai_translation_module._normalized_api_base_url(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert (
        openai_translation_module._normalized_api_base_url("https://api.openai.com/v1/responses")
        == "https://api.openai.com/v1/"
    )


def test_openai_translation_adapter_splits_retryable_failed_chunks(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate()
    transport = SequencedTransport(
        [
            _json_response({"error": {"message": "please retry"}}, status_code=500),
            _json_response(
                {
                    "id": "resp-left",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-right",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour beta Bonjour gamma",
                            "segments": [
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                            ],
                        }
                    ),
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=10_000,
        max_chunk_segments=10,
        context_segment_window=1,
    )

    candidate = adapter.generate_translation(transcript, "variant-a", _request_context())

    assert len(transport.requests) == 3
    assert [segment.target_text for segment in candidate.segments] == [
        "Bonjour alpha",
        "Bonjour beta",
        "Bonjour gamma",
    ]
    assert candidate.metadata["chunking"]["chunk_count"] == 2
    assert candidate.metadata["chunking"]["response_count"] == 2
    assert candidate.raw_response_ref is not None

    stored_payload = json.loads(blob_store.read_bytes(candidate.raw_response_ref).decode("utf-8"))
    assert stored_payload["chunking"]["planned_chunk_count"] == 1
    assert stored_payload["chunking"]["executed_request_count"] == 2
    assert stored_payload["chunks"][0]["status"] == "split_after_retryable_failure"
    assert stored_payload["chunks"][0]["fallback_children"] == ["chunk-0.a", "chunk-0.b"]


def test_openai_translation_adapter_splits_partial_segment_coverage_chunks(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate()
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "resp-parent",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-left",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-right",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour beta Bonjour gamma",
                            "segments": [
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                            ],
                        }
                    ),
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=10_000,
        max_chunk_segments=10,
        context_segment_window=1,
    )

    candidate = adapter.generate_translation(transcript, "variant-a", _request_context())

    assert len(transport.requests) == 3
    assert [segment.target_text for segment in candidate.segments] == [
        "Bonjour alpha",
        "Bonjour beta",
        "Bonjour gamma",
    ]
    assert candidate.metadata["chunking"]["chunk_count"] == 2
    assert candidate.metadata["chunking"]["response_count"] == 2
    assert candidate.raw_response_ref is not None

    stored_payload = json.loads(blob_store.read_bytes(candidate.raw_response_ref).decode("utf-8"))
    assert stored_payload["chunking"]["planned_chunk_count"] == 1
    assert stored_payload["chunking"]["executed_request_count"] == 2
    assert stored_payload["chunks"][0]["status"] == "split_after_validation_failure"
    assert stored_payload["chunks"][0]["fallback_children"] == ["chunk-0.a", "chunk-0.b"]
    assert stored_payload["chunks"][0]["error"]["message"].startswith(
        "translation payload was missing segment translations"
    )
    assert stored_payload["chunks"][0]["response"]["id"] == "resp-parent"


def test_openai_translation_adapter_splits_implausibly_long_segment_translations(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate()
    runaway_text = "Bonjour " * 20
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "resp-parent",
                    "output_text": json.dumps(
                        {
                            "full_text": f"{runaway_text} Bonjour beta Bonjour gamma",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": runaway_text},
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-left",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-right",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour beta Bonjour gamma",
                            "segments": [
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                            ],
                        }
                    ),
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=10_000,
        max_chunk_segments=10,
        context_segment_window=1,
    )

    candidate = adapter.generate_translation(transcript, "variant-a", _request_context())

    assert len(transport.requests) == 3
    assert [segment.target_text for segment in candidate.segments] == [
        "Bonjour alpha",
        "Bonjour beta",
        "Bonjour gamma",
    ]

    stored_payload = json.loads(
        blob_store.read_bytes(candidate.raw_response_ref or "").decode("utf-8")
    )
    assert stored_payload["chunks"][0]["status"] == "split_after_validation_failure"
    assert stored_payload["chunks"][0]["fallback_children"] == ["chunk-0.a", "chunk-0.b"]
    assert stored_payload["chunks"][0]["error"]["message"].startswith(
        "translation payload produced implausibly long text"
    )


def test_openai_translation_adapter_splits_segments_that_duplicate_later_text(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate().model_copy(
        update={
            "segments": (
                Segment(
                    segment_id="seg-1",
                    start_ms=0,
                    end_ms=1_000,
                    speaker="speaker-1",
                    source_text="Alpha source sentence with enough length",
                ),
                Segment(
                    segment_id="seg-2",
                    start_ms=1_000,
                    end_ms=2_000,
                    speaker="speaker-1",
                    source_text="Beta source sentence with enough length",
                ),
                Segment(
                    segment_id="seg-3",
                    start_ms=2_000,
                    end_ms=3_000,
                    speaker="speaker-2",
                    source_text="Gamma source sentence with enough length",
                ),
            ),
            "full_text": (
                "Alpha source sentence with enough length "
                "Beta source sentence with enough length "
                "Gamma source sentence with enough length"
            ),
        }
    )
    duplicated_later_text = (
        "Je suis perdu. Les deux personnes ont un charme different. Choisir est difficile."
    )
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "resp-parent",
                    "output_text": json.dumps(
                        {
                            "full_text": (
                                f"Bonjour alpha {duplicated_later_text} {duplicated_later_text}"
                            ),
                            "segments": [
                                {
                                    "segment_id": "seg-1",
                                    "target_text": f"Bonjour alpha {duplicated_later_text}",
                                },
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                {"segment_id": "seg-3", "target_text": duplicated_later_text},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-left",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-right",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour beta " + duplicated_later_text,
                            "segments": [
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                {"segment_id": "seg-3", "target_text": duplicated_later_text},
                            ],
                        }
                    ),
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=10_000,
        max_chunk_segments=10,
        context_segment_window=1,
    )

    candidate = adapter.generate_translation(transcript, "variant-a", _request_context())

    assert len(transport.requests) == 3
    assert [segment.target_text for segment in candidate.segments] == [
        "Bonjour alpha",
        "Bonjour beta",
        duplicated_later_text,
    ]

    stored_payload = json.loads(
        blob_store.read_bytes(candidate.raw_response_ref or "").decode("utf-8")
    )
    assert stored_payload["chunks"][0]["status"] == "split_after_validation_failure"
    assert stored_payload["chunks"][0]["fallback_children"] == ["chunk-0.a", "chunk-0.b"]
    assert stored_payload["chunks"][0]["error"]["message"].startswith(
        "translation payload duplicated later segment text"
    )


def test_chunk_transcript_uses_serialized_prompt_size_estimator() -> None:
    transcript = _transcript_candidate()
    exact_two_segment_budget = openai_translation_module._estimated_user_prompt_size(
        chunk_index=0,
        segments=transcript.segments[:2],
        transcript_segments=transcript.segments,
        start_index=0,
        end_index=2,
        context_segment_window=1,
    )

    chunks_at_budget = openai_translation_module._chunk_transcript(
        transcript,
        max_chunk_characters=exact_two_segment_budget,
        max_chunk_segments=10,
        context_segment_window=1,
    )
    chunks_below_budget = openai_translation_module._chunk_transcript(
        transcript,
        max_chunk_characters=exact_two_segment_budget - 1,
        max_chunk_segments=10,
        context_segment_window=1,
    )

    assert [segment.segment_id for segment in chunks_at_budget[0].segments] == ["seg-1", "seg-2"]
    assert [segment.segment_id for segment in chunks_below_budget[0].segments] == ["seg-1"]


def test_openai_translation_adapter_does_not_split_children_twice(
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate().model_copy(
        update={
            "segments": (
                *_transcript_candidate().segments,
                Segment(
                    segment_id="seg-4",
                    start_ms=3_000,
                    end_ms=4_000,
                    speaker="speaker-2",
                    source_text="Delta source",
                ),
            ),
            "full_text": "Alpha source Beta source Gamma source Delta source",
        }
    )
    transport = SequencedTransport(
        [
            _json_response(
                {
                    "id": "resp-parent",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-left",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour alpha Bonjour beta",
                            "segments": [
                                {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                                {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                            ],
                        }
                    ),
                }
            ),
            _json_response(
                {
                    "id": "resp-right",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour gamma",
                            "segments": [
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                            ],
                        }
                    ),
                }
            ),
        ]
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_characters=10_000,
        max_chunk_segments=10,
        context_segment_window=1,
    )

    with pytest.raises(Exception, match="missing segment translations"):
        adapter.generate_translation(transcript, "variant-a", _request_context())

    assert len(transport.requests) == 3


def test_openai_translation_adapter_preserves_chunk_order_under_parallel_completion(
    tmp_path: Path,
) -> None:
    class CoordinatedTransport:
        def __init__(self) -> None:
            self.requests: list[HttpRequest] = []
            self._chunk_zero_started = threading.Event()
            self._chunk_one_finished = threading.Event()

        def request(self, request: HttpRequest) -> HttpResponse:
            self.requests.append(request)
            assert request.body is not None
            payload = json.loads(request.body.decode("utf-8"))
            prompt = json.loads(payload["input"][1]["content"][0]["text"])
            chunk_index = int(prompt["chunk_index"])
            if chunk_index == 0:
                self._chunk_zero_started.set()
                assert self._chunk_one_finished.wait(timeout=1)
                return _json_response(
                    {
                        "id": "resp-0",
                        "output_text": json.dumps(
                            {
                                "full_text": "Bonjour alpha Bonjour beta",
                                "segments": [
                                    {"segment_id": "seg-1", "target_text": "Bonjour alpha"},
                                    {"segment_id": "seg-2", "target_text": "Bonjour beta"},
                                ],
                            }
                        ),
                    }
                )
            assert self._chunk_zero_started.wait(timeout=1)
            self._chunk_one_finished.set()
            return _json_response(
                {
                    "id": "resp-1",
                    "output_text": json.dumps(
                        {
                            "full_text": "Bonjour gamma Bonjour delta",
                            "segments": [
                                {"segment_id": "seg-3", "target_text": "Bonjour gamma"},
                                {"segment_id": "seg-4", "target_text": "Bonjour delta"},
                            ],
                        }
                    ),
                }
            )

    blob_store = LocalBlobStore(tmp_path / "blobs")
    transcript = _transcript_candidate().model_copy(
        update={
            "segments": (
                *_transcript_candidate().segments,
                Segment(
                    segment_id="seg-4",
                    start_ms=3_000,
                    end_ms=4_000,
                    speaker="speaker-2",
                    source_text="Delta source",
                ),
            ),
            "full_text": "Alpha source Beta source Gamma source Delta source",
        }
    )
    transport = CoordinatedTransport()
    exact_two_segment_budget = max(
        openai_translation_module._estimated_user_prompt_size(
            chunk_index=0,
            segments=transcript.segments[:2],
            transcript_segments=transcript.segments,
            start_index=0,
            end_index=2,
            context_segment_window=1,
        ),
        openai_translation_module._estimated_user_prompt_size(
            chunk_index=1,
            segments=transcript.segments[2:],
            transcript_segments=transcript.segments,
            start_index=2,
            end_index=4,
            context_segment_window=1,
        ),
    )
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_workers=2,
        max_chunk_characters=exact_two_segment_budget,
        max_chunk_segments=2,
        context_segment_window=1,
    )

    candidate = adapter.generate_translation(transcript, "variant-a", _request_context())

    assert [segment.segment_id for segment in candidate.segments] == [
        "seg-1",
        "seg-2",
        "seg-3",
        "seg-4",
    ]
    assert candidate.metadata["provider"]["response_ids"] == ["resp-0", "resp-1"]
    stored_payload = json.loads(
        blob_store.read_bytes(candidate.raw_response_ref or "").decode("utf-8")
    )
    assert [chunk["chunk_key"] for chunk in stored_payload["chunks"]] == ["chunk-0", "chunk-1"]
    assert [chunk["segment_ids"] for chunk in stored_payload["chunks"]] == [
        ["seg-1", "seg-2"],
        ["seg-3", "seg-4"],
    ]


def test_openai_translation_adapter_nested_calls_share_global_chunk_limit(
    tmp_path: Path,
) -> None:
    class CountingTransport:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def request(self, request: HttpRequest) -> HttpResponse:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                assert request.body is not None
                payload = json.loads(request.body.decode("utf-8"))
                prompt = json.loads(payload["input"][1]["content"][0]["text"])
                segments = prompt["segments"]
                time.sleep(0.05)
                return _json_response(
                    {
                        "id": f"resp-{prompt['chunk_index']}",
                        "output_text": json.dumps(
                            {
                                "full_text": " ".join(
                                    f"Bonjour {segment['segment_id']}" for segment in segments
                                ),
                                "segments": [
                                    {
                                        "segment_id": segment["segment_id"],
                                        "target_text": f"Bonjour {segment['segment_id']}",
                                    }
                                    for segment in segments
                                ],
                            }
                        ),
                    }
                )
            finally:
                with self._lock:
                    self.active -= 1

    transcript = _transcript_candidate().model_copy(
        update={
            "segments": (
                *_transcript_candidate().segments,
                Segment(
                    segment_id="seg-4",
                    start_ms=3_000,
                    end_ms=4_000,
                    speaker="speaker-2",
                    source_text="Delta source",
                ),
            ),
            "full_text": "Alpha source Beta source Gamma source Delta source",
        }
    )
    blob_store = LocalBlobStore(tmp_path / "blobs")
    transport = CountingTransport()
    adapter = OpenAITranslationAdapter(
        api_key="test-key",  # pragma: allowlist secret
        blob_store=blob_store,
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1),
        sleep=lambda _: None,
        max_chunk_workers=4,
        global_concurrency_limiter=GlobalConcurrencyLimiter(4),
        provider_io_token_cost=2,
        max_chunk_characters=25,
        max_chunk_segments=2,
        context_segment_window=1,
    )
    results: list[tuple[str, list[str]]] = []
    errors: list[BaseException] = []

    def run_candidate(prompt_variant_id: str) -> None:
        try:
            candidate = adapter.generate_translation(
                transcript,
                prompt_variant_id,
                _request_context(),
            )
        except BaseException as exc:  # pragma: no cover - test safety
            errors.append(exc)
            return
        results.append((prompt_variant_id, [segment.segment_id for segment in candidate.segments]))

    first = threading.Thread(target=run_candidate, args=("variant-a",))
    second = threading.Thread(target=run_candidate, args=("variant-b",))
    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert transport.max_active == 2
    assert sorted(results) == [
        ("variant-a", ["seg-1", "seg-2", "seg-3", "seg-4"]),
        ("variant-b", ["seg-1", "seg-2", "seg-3", "seg-4"]),
    ]
