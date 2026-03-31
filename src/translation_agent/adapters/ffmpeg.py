"""ffmpeg extraction adapter."""

from __future__ import annotations

import subprocess  # nosec B404
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from translation_agent.adapters.common import (
    AdapterError,
    RetryPolicy,
    blob_filename,
    perform_with_retries,
)
from translation_agent.models import AudioArtifact, RequestContext
from translation_agent.storage import BlobStore, job_path, job_scope_token


@dataclass(frozen=True, slots=True)
class FfmpegCompletedProcess:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


CommandRunner = Callable[[Sequence[str], float], FfmpegCompletedProcess]


class FFmpegAudioExtractionAdapter:
    """Direct local extraction adapter that stores WAV audio in the blob store."""

    adapter_id = "ffmpeg"

    def __init__(
        self,
        *,
        blob_store: BlobStore,
        binary: str = "ffmpeg",
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        codec: str = "pcm_s16le",
        timeout_seconds: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        command_runner: CommandRunner | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._blob_store = blob_store
        self._binary = binary
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._codec = codec
        self._timeout_seconds = timeout_seconds
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=1)
        self._command_runner = command_runner or _default_command_runner
        self._sleep = sleep or (lambda seconds: __import__("time").sleep(seconds))

    def extract_audio(self, video_ref: str, job_context: RequestContext) -> AudioArtifact:
        source_path = Path(video_ref).expanduser().resolve()
        if not source_path.exists():
            raise AdapterError(
                provider_id=self.adapter_id,
                message=f"input media was not found: {source_path}",
                category="input_missing",
                retryable=False,
            )

        with tempfile.TemporaryDirectory(prefix="translation-agent-ffmpeg-") as temp_dir:
            output_path = Path(temp_dir) / f"audio-{job_scope_token(job_context.job)}.wav"
            command = (
                self._binary,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                str(self._channels),
                "-ar",
                str(self._sample_rate_hz),
                "-c:a",
                self._codec,
                str(output_path),
            )
            process = perform_with_retries(
                lambda: self._run_command(command),
                provider_id=self.adapter_id,
                retry_policy=self._retry_policy,
                sleep=self._sleep,
            )
            data = output_path.read_bytes()
            duration_ms, sample_rate_hz, channels = _audio_stats(output_path)

        scope_token = job_scope_token(job_context.job)
        blob_ref = job_path(job_context.job, "artifacts", "audio.wav")
        self._blob_store.put_bytes(blob_ref, data)
        return AudioArtifact(
            artifact_id=f"audio-{job_context.job.job_id}-{scope_token}",
            job_id=job_context.job.job_id,
            blob_ref=blob_ref,
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            codec=self._codec,
            extraction_metadata={
                "adapter_id": self.adapter_id,
                "binary": self._binary,
                "source_video_ref": str(source_path),
                "source_filename": blob_filename(video_ref, "input.media"),
                "generated_at": datetime.now(UTC).isoformat(),
                "stderr": process.stderr.decode("utf-8", errors="ignore").strip(),
            },
        )

    def _run_command(self, command: Sequence[str]) -> FfmpegCompletedProcess:
        try:
            return self._command_runner(command, self._timeout_seconds)
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise AdapterError(
                provider_id=self.adapter_id,
                message="ffmpeg timed out while extracting audio",
                category="timeout",
                retryable=True,
            ) from exc
        except FileNotFoundError as exc:
            raise AdapterError(
                provider_id=self.adapter_id,
                message=f"ffmpeg binary was not found: {self._binary}",
                category="binary_missing",
                retryable=False,
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise AdapterError(
                provider_id=self.adapter_id,
                message=exc.stderr.decode("utf-8", errors="ignore").strip() or "ffmpeg failed",
                category="process_error",
                retryable=False,
                status_code=exc.returncode,
            ) from exc


def _default_command_runner(
    command: Sequence[str],
    timeout_seconds: float,
) -> FfmpegCompletedProcess:
    completed = subprocess.run(  # nosec B603
        list(command),
        capture_output=True,
        check=True,
        timeout=timeout_seconds,
    )
    return FfmpegCompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _audio_stats(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        channels = handle.getnchannels()
        frame_count = handle.getnframes()
    duration_ms = int(round((frame_count / frame_rate) * 1000)) if frame_rate else 0
    return duration_ms, frame_rate, channels


FfmpegAudioExtractionAdapter = FFmpegAudioExtractionAdapter
