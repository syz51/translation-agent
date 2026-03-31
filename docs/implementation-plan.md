# Implementation Map

This document is no longer a future-state plan. It is a guide to the current package layout and the safest ways to extend it without drifting from the implementation.

## Package Map

```text
src/
  translation_agent/
    api.py
    cli.py
    config.py
    normalization.py
    replay.py
    adapters/
    graph/
    memory/
    models/
    nodes/
    observability/
    publish/
    review/
    storage/
tests/
  contract/
  regression/
  test_api_cli_config.py
  test_phase_two_workflow.py
  test_phase_three_adapters.py
  test_phase_four_review.py
  test_phase_five_memory_publish.py
  test_phase_six_hardening.py
alembic/
compose.yaml
```

## Module Responsibilities

### `api.py`

Owns the public Python API and run bootstrap:

- configuration validation
- blob-store initialization
- run-record creation
- trace setup
- runtime construction
- final status derivation

### `cli.py`

Owns the public CLI contract:

- `validate-config`
- `run-job`

Changes here should be mirrored in contract tests and, when necessary, in the golden files under `tests/contract/golden/`.

### `config.py`

Owns:

- settings loading
- runtime path resolution
- backend selection
- provider credential validation
- LangGraph compatibility gating
- DSN sanitization for user-facing output

### `graph/`

Owns:

- runtime construction
- fake versus real adapter selection
- graph wiring
- graph execution helpers
- post-run trace synchronization

If you change graph structure, also update the workflow docs and the phase workflow tests.

### `nodes/`

Owns the concrete workflow stages:

- ingest
- audio extraction
- transcription fanout
- normalization
- review
- adjudication
- translation generation
- memory staging
- final publishing

Node functions should stay deterministic and operate on typed models plus refs.

### `adapters/`

Owns external integration boundaries:

- `ffmpeg`
- AssemblyAI
- Speechmatics
- Deepgram
- OpenAI translation
- retry, timeout, polling, and HTTP helpers

If a change affects real-provider payload shape or retry behavior, extend `tests/test_phase_three_adapters.py`.

### `models/`

Owns canonical contracts:

- job and request models
- artifacts
- transcript and translation candidates
- review and adjudication models
- memory models

Changes here usually require corresponding updates in:

- workflow nodes
- persistence code
- tests
- docs

### `review/`

Owns:

- reviewer role definitions
- fixed review prompt contracts
- deterministic dry-run review rendering
- prose parsing
- disagreement scoring and escalation policy

If you change required review sections or adjudication thresholds, update both tests and docs because these are part of the repo’s conceptual model.

### `memory/`

Owns:

- long-term recall
- staging of candidate writes
- consolidation and dedupe
- prompt-evolution proposal generation

This package deliberately keeps raw full transcript and translation bodies out of long-term memory.

### `storage/`

Owns:

- blob storage
- operational store interfaces
- SQLite and Postgres implementations
- Alembic migration helper
- job-scoped path conventions

If you change storage shape, keep SQLite, Postgres, Alembic, and tests aligned.

### `publish/`

Owns final artifact generation:

- transcript and translation publication
- translation-failure manifest
- scorecard
- text and JSON exports
- downstream payload
- published artifact manifest

### `observability/`

Owns:

- structured logging helpers
- trace event model
- JSONL trace sink

### `replay.py`

Owns deterministic adjudication replay from persisted refs. Changes here should be validated against the regression suite because replay is one of the repo’s strongest inspectability guarantees.

## How To Change The Repo Safely

### If You Change CLI Or API Output

Update:

- unit tests in `tests/test_api_cli_config.py`
- contract tests in `tests/contract/test_cli_contract.py`
- any affected golden files in `tests/contract/golden/`
- README command examples if the public UX changes

### If You Change Workflow Routing Or Status Behavior

Update:

- `tests/test_phase_two_workflow.py`
- `tests/test_phase_four_review.py`
- `tests/regression/test_runtime_regression.py`
- `docs/workflow.md`
- `README.md` status documentation when public status values change

### If You Change Adapters Or Normalization

Update:

- `tests/test_phase_three_adapters.py`
- docs that describe runtime modes, providers, or normalization expectations

### If You Change Memory Behavior

Update:

- `tests/test_phase_five_memory_publish.py`
- `tests/test_phase_six_hardening.py`
- `docs/architecture.md`
- `docs/interfaces-and-data-models.md`

### If You Change Storage Or Migrations

Update:

- storage tests
- Alembic migrations
- any docs that describe operational state or local setup

### If You Change Replay Behavior

Update:

- regression tests in `tests/regression/test_runtime_regression.py`
- architecture and workflow docs when replay semantics or investigation behavior change

## Verification Guide

The repo conventions in `AGENTS.md` are the current baseline:

```bash
uv run ruff check .
uv run pre-commit run --all-files
uv run pyright
uv run pytest -m "unit or slice"
uv run pytest -m contract
uv run pytest -m "integration or migration" -n 2 --dist=loadfile
uv run pytest -m "regression and not staging_only"
```

Recommended narrowing:

- docs-only or metadata-only changes: `uv run ruff check .`
- CLI/API changes: `uv run pytest tests/test_api_cli_config.py tests/contract/test_cli_contract.py`
- workflow changes: relevant phase tests plus regression tests
- storage or Alembic changes: storage tests plus migration or integration coverage

## Design Rules Worth Preserving

Keep these constraints unless the architecture is intentionally changing:

- graph state should stay ref-only and lightweight
- deterministic code should own orchestration, routing, retries, normalization, and persistence
- review output should remain parseable into typed bundles
- prompt evolution should remain derived from consolidated translation outcomes, not raw reviewer prose
- tenant, project, and language scoping should continue to protect artifact and memory isolation
- replay should remain possible from persisted refs alone

## Near-Term Engineering Opportunities

These are logical next changes if the repo keeps expanding:

- add a dedicated docs page for real-provider setup and failure modes
- separate fake-runtime scenario docs from production runtime docs once real mode becomes more common
- add explicit golden coverage for published scorecards and exports
- wire currently unused settings such as console-log toggles or remove them from the config surface
