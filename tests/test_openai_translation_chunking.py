from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.adapters import OpenAITranslationAdapter, RetryPolicy
from translation_agent.adapters import openai_translation as openai_translation_module
from translation_agent.adapters.common import HttpRequest, HttpResponse
from translation_agent.models import JobContext, RequestContext, Segment, TranscriptCandidate
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

    chunks = openai_translation_module._chunk_transcript(
        transcript,
        max_chunk_characters=25,
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
        max_chunk_characters=25,
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
    assert [segment["segment_id"] for segment in first_prompt["chunk"]["segments"]] == [
        "seg-1",
        "seg-2",
    ]
    assert "Gamma source" not in first_prompt["chunk"]["full_text"]
    assert first_prompt["context_after"] == ["Gamma source"]

    stored_payload = json.loads(blob_store.read_bytes(candidate.raw_response_ref).decode("utf-8"))
    assert stored_payload["chunking"]["chunk_count"] == 2
    assert [chunk["segment_ids"] for chunk in stored_payload["chunks"]] == [
        ["seg-1", "seg-2"],
        ["seg-3"],
    ]


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
        max_chunk_characters=200,
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
