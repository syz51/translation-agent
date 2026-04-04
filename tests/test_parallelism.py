from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from translation_agent.parallelism import (
    GlobalConcurrencyLimiter,
    ParallelTaskClass,
    resolve_stage_workers,
)

pytestmark = pytest.mark.unit


def test_resolve_stage_workers_uses_global_budget_in_auto_mode() -> None:
    assert resolve_stage_workers(None, task_count=10, global_max_parallel_tokens=8) == 8
    assert resolve_stage_workers(None, task_count=3, global_max_parallel_tokens=8) == 3
    assert resolve_stage_workers(2, task_count=10, global_max_parallel_tokens=8) == 2


def test_global_concurrency_limiter_blocks_and_tracks_provider_concurrency() -> None:
    limiter = GlobalConcurrencyLimiter(2)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    states: list[tuple[int, float, int]] = []

    def first_worker() -> None:
        with limiter.acquire(2, task_class=ParallelTaskClass.PROVIDER_IO) as acquisition:
            states.append(
                (
                    acquisition.tokens_acquired,
                    acquisition.wait_ms,
                    acquisition.max_concurrent_provider_calls_observed,
                )
            )
            first_entered.set()
            assert release_first.wait(timeout=1)

    def second_worker() -> None:
        assert first_entered.wait(timeout=1)
        with limiter.acquire(1, task_class=ParallelTaskClass.LOCAL_COMPUTE) as acquisition:
            states.append(
                (
                    acquisition.tokens_acquired,
                    acquisition.wait_ms,
                    acquisition.max_concurrent_provider_calls_observed,
                )
            )
            second_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first_worker)
        second_future = executor.submit(second_worker)
        assert first_entered.wait(timeout=1)
        assert not second_finished.wait(timeout=0.05)
        release_first.set()
        first_future.result(timeout=1)
        second_future.result(timeout=1)

    assert states[0][0] == 2
    assert states[0][1] == pytest.approx(0.0, abs=5.0)
    assert states[0][2] == 1
    assert states[1][0] == 1
    assert states[1][1] > 0.0
    assert limiter.blocked_count >= 1
    assert limiter.max_concurrent_provider_calls_observed == 1


def test_global_concurrency_limiter_releases_tokens_after_exception() -> None:
    limiter = GlobalConcurrencyLimiter(1)

    with pytest.raises(RuntimeError, match="boom"):
        with limiter.acquire(1, task_class=ParallelTaskClass.LOCAL_COMPUTE):
            raise RuntimeError("boom")

    with limiter.acquire(1, task_class=ParallelTaskClass.LOCAL_COMPUTE) as acquisition:
        assert acquisition.tokens_acquired == 1
