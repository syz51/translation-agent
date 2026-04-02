"""Shared bounded parallelism helpers for deterministic internal fan-out.

Worker threads must return data only. Shared persistence, state mutation, and
artifact publication stay on the calling thread after gather.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeParallelismPolicy:
    """Per-stage worker limits for the synchronous workflow runtime."""

    transcription_max_workers: int
    translation_candidate_max_workers: int
    translation_chunk_max_workers: int
    review_max_workers: int
    reference_evaluation_max_workers: int
    memory_drain_max_workers: int


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
