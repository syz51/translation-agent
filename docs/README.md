# Translation Agent Docs

This folder documents the repository as it exists today. The older phase-planning language has been removed: the codebase already contains a runnable workflow, real-provider adapters, persistence, replay, memory staging, and publishing logic.

## Read This First

Start here if you need to understand the repo quickly:

1. [`../README.md`](../README.md)
   Runtime modes, quick start, CLI, Python API, artifact layout, and verification commands.
2. [`architecture.md`](./architecture.md)
   System shape, deterministic versus agentic boundaries, persistence, replay, and memory lifecycle.
3. [`workflow.md`](./workflow.md)
   Actual graph nodes, routing behavior, branch conditions, statuses, and scenario-driven dry-run behavior.
4. [`interfaces-and-data-models.md`](./interfaces-and-data-models.md)
   Public contracts, core models, operational stores, artifact conventions, and output payloads.
5. [`implementation-plan.md`](./implementation-plan.md)
   Current package map plus practical guidance for changing the repo without drifting from the implementation.
6. [`roadmap.md`](./roadmap.md)
   Work intentionally deferred beyond the current implementation.

## Current Repo Reality

The current implementation includes:

- a deterministic LangGraph workflow
- fake adapters for fast local and test execution
- real adapters for `ffmpeg`, AssemblyAI, Speechmatics, Deepgram, and OpenAI Responses
- SQLite and Postgres operational state backends
- structured logging plus JSONL traces
- staged memory writes, background consolidation, and translation prompt-evolution proposals
- replay helpers that reconstruct adjudication from persisted refs
- contract, regression, unit, slice, integration, and migration tests

## How To Use These Docs

Use the docs according to the task:

- debugging a run: start with [`workflow.md`](./workflow.md) and [`architecture.md`](./architecture.md)
- adding or changing a model: start with [`interfaces-and-data-models.md`](./interfaces-and-data-models.md)
- adding a provider or changing runtime wiring: start with [`architecture.md`](./architecture.md) and [`implementation-plan.md`](./implementation-plan.md)
- deciding whether work belongs in scope now or later: start with [`roadmap.md`](./roadmap.md)

## Source Of Truth

For implementation claims, prefer the code and tests over prose:

- runtime entrypoints: `src/translation_agent/api.py`, `src/translation_agent/cli.py`
- graph wiring: `src/translation_agent/graph/`
- workflow nodes: `src/translation_agent/nodes/`
- adapter behavior: `src/translation_agent/adapters/`
- review and adjudication logic: `src/translation_agent/review/`
- memory behavior: `src/translation_agent/memory/`
- persistence: `src/translation_agent/storage/`
- publishing and replay: `src/translation_agent/publish/`, `src/translation_agent/replay.py`
- system guarantees: `tests/`
