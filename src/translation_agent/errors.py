"""Shared workflow exceptions and error-payload helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TranscriptionProvidersFailedError(RuntimeError):
    """Structured failure raised when no transcription provider succeeds."""

    def __init__(self, provider_errors: Mapping[str, str]) -> None:
        normalized_provider_errors = {
            str(provider_id): str(message)
            for provider_id, message in provider_errors.items()
            if str(provider_id) and str(message)
        }
        self.provider_errors = normalized_provider_errors
        provider_summary = "; ".join(
            f"{provider_id}: {message}"
            for provider_id, message in normalized_provider_errors.items()
        )
        message = "all transcription providers failed"
        if provider_summary:
            message = f"{message}: {provider_summary}"
        super().__init__(message)

    def error_payload(self) -> dict[str, Any]:
        return {
            "message": "all transcription providers failed",
            "category": "transcription_failed",
            "reason": "all_transcription_providers_failed",
            "provider_errors": [
                {
                    "provider_id": provider_id,
                    "message": message,
                }
                for provider_id, message in self.provider_errors.items()
            ],
        }


def exception_error_payload(exc: Exception) -> dict[str, Any]:
    """Normalize an exception into a JSON-safe persisted error payload."""

    payload_getter = getattr(exc, "error_payload", None)
    if callable(payload_getter):
        payload = payload_getter()
        if isinstance(payload, dict):
            return payload
    return {"message": str(exc)}
