from __future__ import annotations

from dataclasses import dataclass

import pytest

from translation_agent.adapters.common import (
    AdapterError,
    HttpRequest,
    HttpResponse,
    RetryPolicy,
    StdlibHttpTransport,
    blob_filename,
    build_multipart_form_data,
    classify_http_error,
    json_request,
    normalize_whitespace,
    perform_with_retries,
    poll_until_complete,
)

pytestmark = pytest.mark.unit


@dataclass
class RecordingTransport:
    response: HttpResponse
    requests: list[HttpRequest]

    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.requests = []

    def request(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        return self.response


def test_http_response_json_rejects_invalid_payload_shapes() -> None:
    with pytest.raises(AdapterError, match="valid JSON"):
        HttpResponse(status_code=200, headers={}, body=b"{invalid").json()

    with pytest.raises(AdapterError, match="must be an object"):
        HttpResponse(status_code=200, headers={}, body=b"[1, 2, 3]").json()


def test_json_request_sets_body_and_content_type() -> None:
    transport = RecordingTransport(HttpResponse(status_code=200, headers={}, body=b"{}"))

    response = json_request(
        transport,
        method="POST",
        url="https://api.example.com/jobs",
        headers={"authorization": "Bearer token"},
        timeout_seconds=12.5,
        body={"job": "123"},
    )

    assert response.status_code == 200
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url == "https://api.example.com/jobs"
    assert request.timeout_seconds == 12.5
    assert request.headers["content-type"] == "application/json"
    assert request.body == b'{"job": "123"}'


def test_classify_http_error_extracts_nested_message_and_retryability() -> None:
    retryable = classify_http_error(
        "assemblyai",
        HttpResponse(
            status_code=503,
            headers={},
            body=b'{"error": {"message": "try again later"}}',
        ),
    )
    terminal = classify_http_error(
        "assemblyai",
        HttpResponse(status_code=400, headers={}, body=b"plain failure"),
    )

    assert retryable is not None
    assert retryable.retryable is True
    assert str(retryable) == "try again later"
    assert terminal is not None
    assert terminal.retryable is False
    assert str(terminal) == "plain failure"


def test_perform_with_retries_retries_coerced_provider_errors() -> None:
    attempts = 0
    sleeps: list[float] = []

    def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AdapterError(
                provider_id="http",
                message="request timed out",
                category="timeout",
                retryable=True,
            )
        return "ok"

    result = perform_with_retries(
        flaky_operation,
        provider_id="deepgram",
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=1,
        ),
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == [0.1]


def test_perform_with_retries_stops_on_terminal_error() -> None:
    with pytest.raises(AdapterError, match="bad request") as exc_info:
        perform_with_retries(
            lambda: (_ for _ in ()).throw(
                AdapterError(
                    provider_id="speechmatics",
                    message="bad request",
                    category="http_error",
                    retryable=False,
                    status_code=400,
                )
            ),
            provider_id="speechmatics",
            retry_policy=RetryPolicy(max_attempts=3),
            sleep=lambda _: None,
        )

    assert exc_info.value.status_code == 400


def test_poll_until_complete_handles_success_error_unknown_and_timeout() -> None:
    pending_then_success = iter(({"status": "queued"}, {"status": "done", "id": "job-1"}))
    sleeps: list[float] = []

    payload = poll_until_complete(
        provider_id="assemblyai",
        fetch_status=lambda: next(pending_then_success),
        pending_statuses={"queued"},
        success_statuses={"done"},
        error_statuses={"error"},
        retry_policy=RetryPolicy(poll_interval_seconds=0.25, max_polls=3),
        sleep=sleeps.append,
    )

    assert payload == {"status": "done", "id": "job-1"}
    assert sleeps == [0.25]

    with pytest.raises(AdapterError, match="provider exploded"):
        poll_until_complete(
            provider_id="assemblyai",
            fetch_status=lambda: {"status": "error", "detail": {"message": "provider exploded"}},
            pending_statuses={"queued"},
            success_statuses={"done"},
            error_statuses={"error"},
            retry_policy=RetryPolicy(max_polls=1),
            sleep=lambda _: None,
        )

    with pytest.raises(AdapterError, match="unknown status"):
        poll_until_complete(
            provider_id="assemblyai",
            fetch_status=lambda: {"status": "mystery"},
            pending_statuses={"queued"},
            success_statuses={"done"},
            error_statuses={"error"},
            retry_policy=RetryPolicy(max_polls=1),
            sleep=lambda _: None,
        )

    with pytest.raises(AdapterError, match="polling exceeded 2 attempts"):
        poll_until_complete(
            provider_id="assemblyai",
            fetch_status=lambda: {"status": "queued"},
            pending_statuses={"queued"},
            success_statuses={"done"},
            error_statuses={"error"},
            retry_policy=RetryPolicy(poll_interval_seconds=0.0, max_polls=2),
            sleep=lambda _: None,
        )


def test_transport_helpers_cover_url_validation_multipart_and_filename_logic() -> None:
    transport = StdlibHttpTransport()
    with pytest.raises(AdapterError, match="unsupported URL scheme"):
        transport.request(HttpRequest(method="GET", url="http://example.com", headers={}))

    body, content_type = build_multipart_form_data(
        fields={"model": "general"},
        file_field="audio",
        filename="sample.wav",
        content=b"wave-data",
    )

    assert "multipart/form-data; boundary=translation-agent-" in content_type
    assert b'Content-Disposition: form-data; name="model"' in body
    assert b'filename="sample.wav"' in body
    assert b"Content-Type: audio/x-wav" in body
    assert blob_filename("https://cdn.example.com/media/audio.wav", "fallback.bin") == "audio.wav"
    assert blob_filename("https://cdn.example.com/media/", "fallback.bin") == "fallback.bin"
    assert normalize_whitespace(" hello\tworld \n again ") == "hello world again"
