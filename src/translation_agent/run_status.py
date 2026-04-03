from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from translation_agent.storage import NodeExecutionRecord, RunRecord

_RECENT_EVENT_LIMIT = 5
_TRACE_TAIL_LIMIT = 64
_NON_TERMINAL_RUN_STATUSES = {"bootstrapped", "queued", "running"}


@dataclass(slots=True)
class PhaseCounters:
    total: int | None = None
    active: int = 0
    completed: int = 0
    failed: int = 0


@dataclass(slots=True)
class RecentRunEvent:
    timestamp: str
    name: str
    message: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunStatusSnapshot:
    run_id: str
    job_id: str
    status: str
    current_stage: str | None
    active_node: str | None
    elapsed_seconds: float
    trace_path: Path
    recent_events: tuple[RecentRunEvent, ...]
    transcription_providers: PhaseCounters | None = None
    translation_variants: PhaseCounters | None = None
    review_bundles: PhaseCounters | None = None


@dataclass(slots=True)
class _NormalizedTraceEvent:
    timestamp: str
    name: str
    message: str
    attributes: dict[str, Any]
    stage: str | None = None


@dataclass(slots=True)
class _PhaseState:
    total: int | None = None
    states: dict[str, str] = field(default_factory=dict)

    def update(self, item_key: str, state: str, *, total: int | None = None) -> None:
        if total is not None:
            self.total = total
        self.states[item_key] = state

    def snapshot(self) -> PhaseCounters | None:
        if not self.states and self.total is None:
            return None
        return PhaseCounters(
            total=self.total,
            active=sum(1 for state in self.states.values() if state == "active"),
            completed=sum(1 for state in self.states.values() if state == "completed"),
            failed=sum(1 for state in self.states.values() if state == "failed"),
        )


class RunStatusAccumulator:
    def __init__(
        self,
        *,
        trace_path: str | Path | None = None,
        recent_event_limit: int = _RECENT_EVENT_LIMIT,
    ) -> None:
        self.run_id = ""
        self.job_id: str | None = None
        self.status = "unknown"
        self.current_stage: str | None = None
        self.active_node: str | None = None
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.created_at: datetime | None = None
        self.updated_at: datetime | None = None
        self._recent_events: deque[RecentRunEvent] = deque(maxlen=recent_event_limit)
        self._trace_stage_hint: str | None = None
        self._final_stage: str | None = None
        self._transcription_phase = _PhaseState()
        self._translation_phase = _PhaseState()
        self._review_phase = _PhaseState()

    def apply_run_record(self, record: RunRecord, *, trace_path: str | Path | None = None) -> None:
        self.run_id = record.run_id
        input_data = _dict_payload(record.input_data)
        output_data = _dict_payload(record.output_data)
        self.job_id = _string_or_none(input_data.get("job_id")) or record.run_id
        self.status = record.status
        self.trace_path = Path(trace_path) if trace_path is not None else self.trace_path
        self.created_at = _parse_timestamp(record.created_at) or self.created_at
        self.updated_at = _parse_timestamp(record.updated_at) or self.updated_at
        self._final_stage = _string_or_none(output_data.get("final_stage"))

    def apply_node_executions(self, executions: Sequence[NodeExecutionRecord]) -> None:
        if not executions:
            if self.current_stage is None:
                self.current_stage = self._final_stage or self._default_stage()
            return

        ordered = sorted(executions, key=_node_execution_sort_key)
        active = [execution for execution in ordered if execution.status == "started"]
        if active:
            self.active_node = active[-1].node_name
            self.current_stage = self.active_node
            return

        latest = ordered[-1]
        self.active_node = None
        if self._final_stage is not None and self.status not in _NON_TERMINAL_RUN_STATUSES:
            self.current_stage = self._final_stage
            return
        self.current_stage = self._trace_stage_hint or latest.node_name

    def apply_trace_event(self, event: Mapping[str, Any]) -> None:
        normalized = normalize_trace_event(event)
        if normalized is None:
            return

        run_id = _string_or_none(event.get("run_id"))
        if run_id:
            self.run_id = run_id
        attributes = normalized.attributes
        if self.job_id is None:
            self.job_id = _string_or_none(attributes.get("job_id"))
        timestamp = _parse_timestamp(normalized.timestamp)
        if normalized.name == "run.bootstrapped":
            self.status = "bootstrapped"
            self.created_at = timestamp or self.created_at
        elif normalized.name == "run.started":
            self.status = "running"
            self.created_at = timestamp or self.created_at
        elif normalized.name == "run.completed":
            self.status = _string_or_none(attributes.get("status")) or "completed"
            self.updated_at = timestamp or self.updated_at
        elif normalized.name == "run.failed":
            self.status = "failed"
            self.updated_at = timestamp or self.updated_at

        if normalized.stage is not None:
            self._trace_stage_hint = normalized.stage
            if self.active_node is None:
                self.current_stage = normalized.stage

        if normalized.name == "node.started":
            node_name = _string_or_none(attributes.get("node_name"))
            if node_name is not None:
                self.active_node = node_name
                self.current_stage = node_name
        elif normalized.name in {"node.completed", "node.failed"}:
            node_name = _string_or_none(attributes.get("node_name"))
            if node_name is not None and self.active_node == node_name:
                self.active_node = None
            if node_name is not None:
                self.current_stage = node_name

        self._update_phase_state(normalized)
        self._recent_events.append(
            RecentRunEvent(
                timestamp=normalized.timestamp,
                name=normalized.name,
                message=normalized.message,
                attributes=normalized.attributes,
            )
        )

    def snapshot(self, *, now: datetime | None = None) -> RunStatusSnapshot:
        trace_path = self.trace_path or Path(f"{self.run_id}.jsonl")
        effective_job_id = self.job_id or self.run_id
        effective_now = now or datetime.now(UTC)
        elapsed_end = (
            self.updated_at
            if self.status not in _NON_TERMINAL_RUN_STATUSES and self.updated_at is not None
            else effective_now
        )
        elapsed_start = self.created_at or elapsed_end
        elapsed_seconds = max(0.0, (elapsed_end - elapsed_start).total_seconds())
        current_stage = self.current_stage or self._final_stage or self._default_stage()
        return RunStatusSnapshot(
            run_id=self.run_id,
            job_id=effective_job_id,
            status=self.status,
            current_stage=current_stage,
            active_node=self.active_node,
            elapsed_seconds=round(elapsed_seconds, 1),
            trace_path=trace_path,
            recent_events=tuple(self._recent_events),
            transcription_providers=self._transcription_phase.snapshot(),
            translation_variants=self._translation_phase.snapshot(),
            review_bundles=self._review_phase.snapshot(),
        )

    def _default_stage(self) -> str | None:
        if self._final_stage is not None:
            return self._final_stage
        return self._trace_stage_hint

    def _update_phase_state(self, event: _NormalizedTraceEvent) -> None:
        attributes = event.attributes
        if event.name.startswith("transcription.provider."):
            provider_id = _string_or_none(attributes.get("provider_id"))
            if provider_id is None:
                return
            state = _event_state(event.name)
            if state is None:
                return
            self._transcription_phase.update(
                provider_id,
                state,
                total=_int_or_none(attributes.get("provider_total")),
            )
            return
        if event.name.startswith("translation.variant."):
            item_key = _translation_item_key(attributes)
            if item_key is None:
                return
            state = _event_state(event.name)
            if state is None:
                return
            self._translation_phase.update(
                item_key,
                state,
                total=_int_or_none(attributes.get("variant_total")),
            )
            return
        if event.name.startswith("review.bundle."):
            item_key = _review_item_key(attributes)
            if item_key is None:
                return
            state = _event_state(event.name)
            if state is None:
                return
            self._review_phase.update(
                item_key,
                state,
                total=_int_or_none(attributes.get("bundle_total")),
            )


def derive_run_status_snapshot(
    record: RunRecord,
    node_executions: Sequence[NodeExecutionRecord],
    trace_events: Sequence[Mapping[str, Any]],
    *,
    trace_path: str | Path,
    now: datetime | None = None,
) -> RunStatusSnapshot:
    accumulator = RunStatusAccumulator(trace_path=trace_path)
    accumulator.apply_run_record(record, trace_path=trace_path)
    for event in trace_events:
        accumulator.apply_trace_event(event)
    accumulator.apply_node_executions(node_executions)
    return accumulator.snapshot(now=now)


def normalize_trace_event(event: Mapping[str, Any]) -> _NormalizedTraceEvent | None:
    name = _string_or_none(event.get("name"))
    timestamp = _string_or_none(event.get("timestamp"))
    if name is None or timestamp is None:
        return None
    attributes = dict(_dict_payload(event.get("attributes")))

    if name == "run.bootstrapped":
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=f"Run bootstrapped for job {attributes.get('job_id') or '-'}",
            attributes=attributes,
        )
    if name == "run.started":
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message="Run started",
            attributes=attributes,
        )
    if name == "run.completed":
        status = _string_or_none(attributes.get("status")) or "completed"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=f"Run completed with status {status}",
            attributes=attributes,
        )
    if name == "run.failed":
        error = _string_or_none(attributes.get("error"))
        message = "Run failed" if error is None else f"Run failed: {error}"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=message,
            attributes=attributes,
        )
    if name == "node.started":
        node_name = _string_or_none(attributes.get("node_name")) or "unknown"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=f"Started node {node_name}",
            attributes=attributes,
            stage=node_name,
        )
    if name == "node.completed":
        node_name = _string_or_none(attributes.get("node_name")) or "unknown"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=f"Completed node {node_name}",
            attributes=attributes,
            stage=node_name,
        )
    if name == "node.failed":
        node_name = _string_or_none(attributes.get("node_name")) or "unknown"
        error = _string_or_none(attributes.get("error"))
        message = f"Failed node {node_name}"
        if error is not None:
            message = f"{message}: {error}"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=message,
            attributes=attributes,
            stage=node_name,
        )
    if name.startswith("transcription.provider."):
        provider_id = _string_or_none(attributes.get("provider_id")) or "unknown"
        message = _domain_event_message(
            name=name,
            subject=provider_id,
            error=attributes.get("error"),
        )
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=message,
            attributes=attributes,
            stage="fanout_transcription",
        )
    if name.startswith("translation.variant."):
        item_key = _translation_item_key(attributes) or "unknown"
        message = _domain_event_message(name=name, subject=item_key, error=attributes.get("error"))
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=message,
            attributes=attributes,
            stage="generate_translation_candidates",
        )
    if name.startswith("review.bundle."):
        review_stage = _string_or_none(attributes.get("review_stage")) or "review"
        reviewer_role = _string_or_none(attributes.get("reviewer_role")) or "unknown"
        action = {
            "review.bundle.started": "Started",
            "review.bundle.completed": "Completed",
            "review.bundle.failed": "Failed",
        }[name]
        error = _string_or_none(attributes.get("error"))
        message = f"{action} review bundle {review_stage}:{reviewer_role}"
        if name.endswith(".failed") and error is not None:
            message = f"{message}: {error}"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=message,
            attributes=attributes,
            stage=review_stage,
        )
    if name == "memory_pipeline.failed":
        error = _string_or_none(attributes.get("error"))
        message = "Background memory pipeline failed"
        if error is not None:
            message = f"{message}: {error}"
        return _NormalizedTraceEvent(
            timestamp=timestamp,
            name=name,
            message=message,
            attributes=attributes,
            stage="background_memory_pipeline",
        )
    return _NormalizedTraceEvent(
        timestamp=timestamp,
        name=name,
        message=name,
        attributes=attributes,
    )


def tail_trace_events(
    path: str | Path,
    *,
    limit: int = _TRACE_TAIL_LIMIT,
) -> tuple[dict[str, Any], ...]:
    trace_path = Path(path)
    if not trace_path.exists():
        return ()
    tail: deque[str] = deque(maxlen=limit)
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tail.append(line)
    events: list[dict[str, Any]] = []
    for line in tail:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return tuple(events)


def is_terminal_run_status(status: str) -> bool:
    return status not in _NON_TERMINAL_RUN_STATUSES


def _domain_event_message(*, name: str, subject: str, error: Any) -> str:
    phase_name, item_name, state = name.split(".")
    verb = {
        "started": "Started",
        "completed": "Completed",
        "failed": "Failed",
    }[state]
    if state == "failed" and error is not None:
        return f"{verb} {phase_name} {item_name} {subject}: {error}"
    return f"{verb} {phase_name} {item_name} {subject}"


def _event_state(name: str) -> str | None:
    if name.endswith(".started"):
        return "active"
    if name.endswith(".completed"):
        return "completed"
    if name.endswith(".failed"):
        return "failed"
    return None


def _translation_item_key(attributes: Mapping[str, Any]) -> str | None:
    prompt_variant_id = _string_or_none(attributes.get("prompt_variant_id"))
    transcript_candidate_id = _string_or_none(attributes.get("source_transcript_candidate_id"))
    if prompt_variant_id is None or transcript_candidate_id is None:
        return None
    return f"{prompt_variant_id}:{transcript_candidate_id}"


def _review_item_key(attributes: Mapping[str, Any]) -> str | None:
    reviewer_role = _string_or_none(attributes.get("reviewer_role"))
    review_stage = _string_or_none(attributes.get("review_stage"))
    if reviewer_role is None or review_stage is None:
        return None
    return f"{review_stage}:{reviewer_role}"


def _node_execution_sort_key(execution: NodeExecutionRecord) -> tuple[datetime, datetime, str]:
    created_at = _parse_timestamp(execution.created_at) or datetime.min.replace(tzinfo=UTC)
    updated_at = _parse_timestamp(execution.updated_at) or created_at
    return (created_at, updated_at, execution.execution_id)


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
