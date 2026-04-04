"""Shared bounded parallelism helpers for deterministic internal fan-out.

Worker threads must return data only. Shared persistence, state mutation, and
artifact publication stay on the calling thread after gather.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from threading import BoundedSemaphore, Lock
from time import perf_counter
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeParallelismPolicy:
    """Per-stage worker limits for the synchronous workflow runtime."""

    global_max_parallel_tokens: int
    provider_io_token_cost: int
    local_compute_token_cost: int
    transcription_max_workers: int | None
    translation_candidate_max_workers: int | None
    translation_chunk_max_workers: int | None
    review_max_workers: int | None
    reference_evaluation_max_workers: int | None
    memory_drain_max_workers: int | None

    def resolve_stage_workers(
        self,
        configured_max_workers: int | None,
        *,
        task_count: int,
    ) -> int:
        return resolve_stage_workers(
            configured_max_workers,
            task_count=task_count,
            global_max_parallel_tokens=self.global_max_parallel_tokens,
        )

    def token_cost(self, task_class: ParallelTaskClass) -> int:
        if task_class is ParallelTaskClass.PROVIDER_IO:
            return self.provider_io_token_cost
        return self.local_compute_token_cost


class ParallelTaskClass(StrEnum):
    PROVIDER_IO = "provider_io"
    LOCAL_COMPUTE = "local_compute"


@dataclass(frozen=True, slots=True)
class LimiterAcquisition:
    task_class: ParallelTaskClass
    tokens_total: int
    tokens_acquired: int
    wait_ms: float
    limiter_blocked_count: int
    max_concurrent_provider_calls_observed: int


class GlobalConcurrencyLimiter:
    """Shared weighted concurrency limiter for synchronous leaf work."""

    def __init__(self, total_tokens: int) -> None:
        if total_tokens < 1:
            raise ValueError("total_tokens must be >= 1")
        self._total_tokens = total_tokens
        self._semaphore = BoundedSemaphore(total_tokens)
        self._lock = Lock()
        self._blocked_count = 0
        self._current_provider_calls = 0
        self._max_concurrent_provider_calls_observed = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def blocked_count(self) -> int:
        with self._lock:
            return self._blocked_count

    @property
    def max_concurrent_provider_calls_observed(self) -> int:
        with self._lock:
            return self._max_concurrent_provider_calls_observed

    @contextmanager
    def acquire(
        self,
        tokens: int,
        *,
        task_class: ParallelTaskClass,
    ) -> Iterator[LimiterAcquisition]:
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        if tokens > self._total_tokens:
            raise ValueError("tokens must be <= total limiter capacity")

        blocked = False
        acquired = 0
        started_at = perf_counter()
        try:
            for _ in range(tokens):
                if not self._semaphore.acquire(blocking=False):
                    blocked = True
                    self._semaphore.acquire()
                acquired += 1
        except Exception:
            for _ in range(acquired):
                self._semaphore.release()
            raise

        wait_ms = max((perf_counter() - started_at) * 1_000.0, 0.0)
        with self._lock:
            if blocked:
                self._blocked_count += 1
            if task_class is ParallelTaskClass.PROVIDER_IO:
                self._current_provider_calls += 1
                self._max_concurrent_provider_calls_observed = max(
                    self._max_concurrent_provider_calls_observed,
                    self._current_provider_calls,
                )
            acquisition = LimiterAcquisition(
                task_class=task_class,
                tokens_total=self._total_tokens,
                tokens_acquired=tokens,
                wait_ms=wait_ms,
                limiter_blocked_count=self._blocked_count,
                max_concurrent_provider_calls_observed=(
                    self._max_concurrent_provider_calls_observed
                ),
            )

        try:
            yield acquisition
        finally:
            with self._lock:
                if task_class is ParallelTaskClass.PROVIDER_IO:
                    self._current_provider_calls -= 1
            for _ in range(tokens):
                self._semaphore.release()


@dataclass(frozen=True, slots=True)
class ParallelTaskResult[ResultT]:
    """One gathered work item with stable ordering metadata."""

    input_index: int
    stable_sort_key: Any
    success_payload: ResultT | None = None
    captured_exception: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.captured_exception is None

    @property
    def value(self) -> ResultT | None:
        return self.success_payload

    @property
    def error(self) -> Exception | None:
        return self.captured_exception

    @property
    def ok(self) -> bool:
        return self.captured_exception is None


def resolve_stage_workers(
    configured_max_workers: int | None,
    *,
    task_count: int,
    global_max_parallel_tokens: int,
) -> int:
    """Resolve a stage cap, defaulting to the global token budget in auto mode."""

    if task_count < 1:
        return 0
    limit = configured_max_workers or global_max_parallel_tokens
    return max(1, min(limit, task_count))


def concurrency_trace_attributes(
    acquisition: LimiterAcquisition,
    *,
    effective_stage_workers: int,
) -> dict[str, object]:
    """Return common limiter/debug metadata for trace events."""

    return {
        "global_parallel_tokens_total": acquisition.tokens_total,
        "global_parallel_tokens_acquired": acquisition.tokens_acquired,
        "global_parallel_tokens_wait_ms": round(acquisition.wait_ms, 3),
        "parallel_task_class": acquisition.task_class.value,
        "effective_stage_workers": effective_stage_workers,
        "limiter_blocked_count": acquisition.limiter_blocked_count,
        "max_concurrent_provider_calls_observed": (
            acquisition.max_concurrent_provider_calls_observed
        ),
    }


def gather_in_input_order[TaskT, ResultT](
    task_specs: Sequence[TaskT],
    worker: Callable[[TaskT], ResultT],
    *,
    max_workers: int,
    sort_key: Callable[[TaskT], Any] | None = None,
) -> tuple[ParallelTaskResult[ResultT], ...]:
    """Run blocking sync work with bounded threads and ordered results."""

    if not task_specs:
        return ()

    key_fn = sort_key or (lambda _task: 0)
    effective_workers = max(1, min(max_workers, len(task_specs)))
    if effective_workers == 1:
        return tuple(
            _execute_task(
                task,
                input_index=input_index,
                stable_sort_key=key_fn(task),
                worker=worker,
            )
            for input_index, task in enumerate(task_specs)
        )

    results: list[ParallelTaskResult[ResultT]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(worker, task): (input_index, key_fn(task))
            for input_index, task in enumerate(task_specs)
        }
        for future in as_completed(futures):
            input_index, stable_sort_key = futures[future]
            try:
                results.append(
                    ParallelTaskResult(
                        input_index=input_index,
                        stable_sort_key=stable_sort_key,
                        success_payload=future.result(),
                    )
                )
            except Exception as exc:
                results.append(
                    ParallelTaskResult(
                        input_index=input_index,
                        stable_sort_key=stable_sort_key,
                        captured_exception=exc,
                    )
                )
    return tuple(sorted(results, key=lambda result: result.input_index))


def ordered_parallel_map[TaskT, ResultT](
    task_specs: Sequence[TaskT],
    *,
    max_workers: int,
    worker: Callable[[TaskT], ResultT],
    sort_key: Callable[[int, TaskT], Any] | None = None,
) -> tuple[ParallelTaskResult[ResultT], ...]:
    """Compatibility wrapper for callers that still use index-aware sort keys."""

    if not task_specs:
        return ()

    key_fn = sort_key or (lambda input_index, _task: input_index)
    effective_workers = max(1, min(max_workers, len(task_specs)))
    if effective_workers == 1:
        return tuple(
            _execute_task(
                task,
                input_index=input_index,
                stable_sort_key=key_fn(input_index, task),
                worker=worker,
            )
            for input_index, task in enumerate(task_specs)
        )

    results: list[ParallelTaskResult[ResultT]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(worker, task): (input_index, key_fn(input_index, task))
            for input_index, task in enumerate(task_specs)
        }
        for future in as_completed(futures):
            input_index, stable_sort_key = futures[future]
            try:
                results.append(
                    ParallelTaskResult(
                        input_index=input_index,
                        stable_sort_key=stable_sort_key,
                        success_payload=future.result(),
                    )
                )
            except Exception as exc:
                results.append(
                    ParallelTaskResult(
                        input_index=input_index,
                        stable_sort_key=stable_sort_key,
                        captured_exception=exc,
                    )
                )
    return tuple(sorted(results, key=lambda result: result.input_index))


def _execute_task[TaskT, ResultT](
    task: TaskT,
    *,
    input_index: int,
    stable_sort_key: Any,
    worker: Callable[[TaskT], ResultT],
) -> ParallelTaskResult[ResultT]:
    try:
        return ParallelTaskResult(
            input_index=input_index,
            stable_sort_key=stable_sort_key,
            success_payload=worker(task),
        )
    except Exception as exc:
        return ParallelTaskResult(
            input_index=input_index,
            stable_sort_key=stable_sort_key,
            captured_exception=exc,
        )
