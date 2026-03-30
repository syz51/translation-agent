"""Storage backends and contracts for translation_agent."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from .blobs import BlobEntry, LocalBlobStore
from .decisions import DecisionStore
from .memory_batches import MemoryBatchStore
from .paths import job_path, job_scope_prefix
from .runs import NodeExecutionRecord, PostgresRunStore, RunRecord


@runtime_checkable
class BlobStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> BlobEntry: ...

    def read_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...

    def list_keys(self, prefix: str | None = None) -> list[str]: ...

    def iter_entries(self, prefix: str | None = None) -> Iterator[BlobEntry]: ...


@runtime_checkable
class RunStore(Protocol):
    def create_run(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        status: str = "queued",
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def list_runs(self) -> list[RunRecord]: ...

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        error: Any = None,
        updated_at: str | None = None,
    ) -> RunRecord: ...

    def create_node_execution(
        self,
        *,
        run_id: str,
        node_name: str,
        status: str = "started",
        input_data: Any = None,
        execution_id: str | None = None,
        created_at: str | None = None,
    ) -> NodeExecutionRecord: ...

    def get_node_execution(self, execution_id: str) -> NodeExecutionRecord | None: ...

    def list_node_executions(self, run_id: str) -> list[NodeExecutionRecord]: ...

    def update_node_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        output_data: Any = None,
        error: Any = None,
        updated_at: str | None = None,
    ) -> NodeExecutionRecord: ...


__all__ = [
    "BlobEntry",
    "BlobStore",
    "DecisionStore",
    "LocalBlobStore",
    "MemoryBatchStore",
    "NodeExecutionRecord",
    "PostgresRunStore",
    "RunRecord",
    "RunStore",
    "job_path",
    "job_scope_prefix",
]
