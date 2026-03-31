"""Blob key helpers for job-scoped artifact storage."""

from __future__ import annotations

from hashlib import sha256

from translation_agent.models.jobs import JobContext


def job_scope_prefix(job: JobContext) -> str:
    """Return a stable blob prefix for one tenant/project/language/job scope."""

    return "/".join(
        (
            "tenants",
            _segment(job.tenant_id),
            "projects",
            _segment(job.project_id),
            "languages",
            f"{_segment(job.source_language)}-to-{_segment(job.target_language)}",
            "jobs",
            _segment(job.job_id),
        )
    )


def job_path(job: JobContext, *parts: str) -> str:
    """Join a job-scoped prefix with relative artifact parts."""

    clean_parts = [part.strip("/") for part in parts if part]
    return "/".join((job_scope_prefix(job), *clean_parts))


def operational_job_key(job: JobContext) -> str:
    """Return the shared operational-store key for one scoped job identity."""

    return job_scope_prefix(job)


def job_scope_token(job: JobContext) -> str:
    """Return a compact deterministic token for scoped IDs."""

    return sha256(operational_job_key(job).encode("utf-8")).hexdigest()[:12]


def _segment(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = [
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in value.strip()
    ]
    cleaned = "".join(normalized).strip("-")
    return cleaned or "unknown"
