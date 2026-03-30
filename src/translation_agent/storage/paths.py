"""Blob key helpers for job-scoped artifact storage."""

from __future__ import annotations

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


def _segment(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = [
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in value.strip()
    ]
    cleaned = "".join(normalized).strip("-")
    return cleaned or "unknown"
