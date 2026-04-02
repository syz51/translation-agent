"""Shared adapter utilities for Phase 3 provider integrations."""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry settings shared across adapter boundaries."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0
    poll_interval_seconds: float = 1.0
    max_polls: int = 120


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.body.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - exercised through callers
            raise AdapterError(
                provider_id="http",
                message="response body was not valid JSON",
                category="malformed_response",
                retryable=False,
            ) from exc
        if not isinstance(payload, dict):
            raise AdapterError(
                provider_id="http",
                message="response JSON must be an object",
                category="malformed_response",
                retryable=False,
            )
        return payload


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None = None
    timeout_seconds: float = 30.0


class HttpTransport(Protocol):
    def request(self, request: HttpRequest) -> HttpResponse: ...


class AdapterError(RuntimeError):
    """Structured adapter failure with retry classification."""

    def __init__(
        self,
        *,
        provider_id: str,
        message: str,
        category: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.category = category
        self.retryable = retryable
        self.status_code = status_code

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "category": self.category,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "message": str(self),
        }


class StdlibHttpTransport:
    """Minimal urllib-backed HTTP transport with byte payload support."""

    def request(self, request: HttpRequest) -> HttpResponse:
        _validate_transport_url(request.url)
        prepared = urllib.request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            with urllib.request.urlopen(  # nosec B310
                prepared,
                timeout=request.timeout_seconds,
            ) as response:
                headers = {key.lower(): value for key, value in response.headers.items()}
                return HttpResponse(
                    status_code=response.status,
                    headers=headers,
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            headers = {key.lower(): value for key, value in exc.headers.items()}
            return HttpResponse(status_code=exc.code, headers=headers, body=exc.read())
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise AdapterError(
                provider_id="http",
                message=f"network failure: {reason}",
                category="network_error",
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise AdapterError(
                provider_id="http",
                message="request timed out",
                category="timeout",
                retryable=True,
            ) from exc


def json_request(
    transport: HttpTransport,
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
    body: dict[str, Any] | None = None,
) -> HttpResponse:
    payload = None
    request_headers = dict(headers)
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        request_headers.setdefault("content-type", "application/json")
    return transport.request(
        HttpRequest(
            method=method,
            url=url,
            headers=request_headers,
            body=payload,
            timeout_seconds=timeout_seconds,
        )
    )


def classify_http_error(
    provider_id: str,
    response: HttpResponse,
    *,
    retryable_status_codes: set[int] | None = None,
) -> AdapterError | None:
    if 200 <= response.status_code < 300:
        return None

    error_payload = _parsed_error_payload(response.body)
    retryable_codes = retryable_status_codes or {408, 409, 425, 429, 500, 502, 503, 504}
    error_code = _error_code(error_payload)
    retryable = response.status_code in retryable_codes and error_code != "insufficient_quota"
    message = _extract_error_message(error_payload or response.body)
    if message is None:
        message = f"http {response.status_code}"
    return AdapterError(
        provider_id=provider_id,
        message=message,
        category="http_error",
        retryable=retryable,
        status_code=response.status_code,
    )


def perform_with_retries[T](
    operation: Callable[[], T],
    *,
    provider_id: str,
    retry_policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempt = 0
    backoff_seconds = retry_policy.initial_backoff_seconds
    last_error: AdapterError | None = None
    while attempt < retry_policy.max_attempts:
        attempt += 1
        try:
            return operation()
        except AdapterError as exc:
            actual_error = _coerce_provider_error(provider_id, exc)
            last_error = actual_error
            if not actual_error.retryable or attempt >= retry_policy.max_attempts:
                raise actual_error
            sleep(min(backoff_seconds, retry_policy.max_backoff_seconds))
            backoff_seconds = min(backoff_seconds * 2, retry_policy.max_backoff_seconds)
    if last_error is None:  # pragma: no cover - defensive
        raise AdapterError(
            provider_id=provider_id,
            message="retry loop exited without a captured error",
            category="retry_exhausted",
            retryable=False,
        )
    raise last_error


def poll_until_complete(
    *,
    provider_id: str,
    fetch_status: Callable[[], dict[str, Any]],
    pending_statuses: set[str],
    success_statuses: set[str],
    error_statuses: set[str],
    retry_policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    for _ in range(retry_policy.max_polls):
        payload = fetch_status()
        status = str(payload.get("status", "")).lower()
        if status in success_statuses:
            return payload
        if status in error_statuses:
            raise AdapterError(
                provider_id=provider_id,
                message=_extract_error_message(payload) or f"{provider_id} returned {status}",
                category="provider_error",
                retryable=False,
            )
        if status not in pending_statuses:
            raise AdapterError(
                provider_id=provider_id,
                message=f"{provider_id} returned unknown status {status!r}",
                category="malformed_response",
                retryable=False,
            )
        sleep(retry_policy.poll_interval_seconds)
    raise AdapterError(
        provider_id=provider_id,
        message=f"{provider_id} polling exceeded {retry_policy.max_polls} attempts",
        category="timeout",
        retryable=True,
    )


def _parsed_error_payload(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _error_code(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    candidates = [
        payload.get("code"),
        payload.get("type"),
    ]
    nested_error = payload.get("error")
    if isinstance(nested_error, dict):
        candidates.extend((nested_error.get("code"), nested_error.get("type")))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def build_multipart_form_data(
    *,
    fields: Mapping[str, str],
    file_field: str,
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> tuple[bytes, str]:
    boundary = f"translation-agent-{uuid4().hex}"
    content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    lines.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


def blob_filename(blob_ref: str, fallback_stem: str) -> str:
    parsed = urllib.parse.urlparse(blob_ref)
    path = parsed.path if parsed.scheme else blob_ref
    leaf = path.rsplit("/", 1)[-1]
    return leaf or fallback_stem


def _validate_transport_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise AdapterError(
            provider_id="http",
            message=f"unsupported URL scheme: {parsed.scheme or '<missing>'}",
            category="invalid_url",
            retryable=False,
        )


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _extract_error_message(payload: bytes | dict[str, Any]) -> str | None:
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                nested = _extract_error_message(value)
                if nested:
                    return nested
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        return payload.decode("utf-8", errors="ignore")[:200].strip() or None
    if isinstance(data, dict):
        return _extract_error_message(data)
    return None


def _coerce_provider_error(provider_id: str, error: AdapterError) -> AdapterError:
    if error.provider_id == provider_id:
        return error
    return AdapterError(
        provider_id=provider_id,
        message=str(error),
        category=error.category,
        retryable=error.retryable,
        status_code=error.status_code,
    )
