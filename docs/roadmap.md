# Roadmap

This document tracks work intentionally deferred beyond the current implementation. The repo already has a runnable deterministic workflow, real-provider adapters, persistence, replay, and memory handling, so the items below are true future work rather than missing foundations.

## Product And Runtime Work Still Deferred

- operator-facing UI for reviewing escalations and rerunning jobs
- explicit human approval workflows in the runtime path
- richer manual override and guided recovery flows
- a second translation provider
- richer downstream delivery integrations
- external long-term memory backends instead of the current blob-backed store
- deeper observability beyond structured logs, node executions, and JSONL traces

## Engineering Improvements Worth Reopening Later

- remove the temporary LangGraph Python 3.14 real-mode gate once upstream compatibility is stable
- decide whether fake-review rendering should be replaced by real model-backed review calls in non-test environments
- expand contract coverage for exported artifacts and scorecards
- formalize provider-specific runbooks and failure taxonomies for real mode
- turn currently implicit scenario coverage into explicit developer-facing fixtures or docs

## Why These Are Still Deferred

The current implementation optimizes for:

- deterministic, inspectable execution
- replayable decisions
- stable local development with fake adapters
- minimal operator surface area
- enough persistence and observability to debug without building a full control plane

## Signals That Should Reopen These Decisions

- unresolved escalations become common enough that manual triage needs tooling
- debugging requires more than traces, routing facts, scorecards, and replay
- the single-translation-provider strategy limits quality or resilience
- blob-backed long-term memory becomes a bottleneck for recall quality or scale
- downstream consumers need stronger delivery guarantees than the current artifact manifests provide
