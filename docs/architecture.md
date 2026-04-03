# Architecture

## System Shape

`translation-agent` is a Python application with two public entrypoints:

- a CLI in `translation_agent.cli`
- a Python API in `translation_agent.api`

Both entrypoints execute the same LangGraph workflow. The graph is implemented in `src/translation_agent/graph/` and operates on a ref-only `GraphState`, which keeps runtime state small and pushes durable artifacts into blob storage and the operational store.

## Runtime Modes

### Fake Runtime

The default runtime is built by `build_phase_two_runtime()`.

It uses:

- deterministic fake audio extraction
- three deterministic fake transcription adapters
- one deterministic fake translation adapter with two prompt variants
- blob-backed long-term memory
- deterministic review rendering, parsing, adjudication, and prompt evolution

This mode is intentionally rich enough to exercise:

- routing and escalation
- persistence and trace writing
- output publishing
- memory staging and consolidation
- replay and regression behavior

### Real Runtime

The real-provider runtime is built by `build_runtime()` when `TA_ADAPTER_MODE=real`, which delegates to `build_phase_three_runtime()`.

It uses:

- `FFmpegAudioExtractionAdapter`
- `AssemblyAITranscriptionAdapter`
- `SpeechmaticsTranscriptionAdapter`
- `DeepgramTranscriptionAdapter`
- `ChatCompletionTranslationAdapter`

The translation role is provider-scoped:

- `translation` defaults to Gemini with `gemini-3-flash`
- `reasoning` defaults to OpenAI with `gpt-5.4`

Real mode is gated by:

- provider credential validation
- the current LangGraph Python 3.14 compatibility guard
- retry, backoff, timeout, and polling settings loaded from `Settings`

## Deterministic Versus Agentic Boundary

The current implementation is deterministic-first.

Deterministic code owns:

- workflow execution
- node routing
- provider invocation and retries
- normalization
- candidate persistence
- review parsing
- adjudication scoring
- memory staging
- artifact publishing
- replay

The review layer is still structured around reviewer roles, but in the current fake runtime those roles are rendered deterministically into fixed-section prose and parsed back into structured review bundles. The configured reasoning profile is already attached to `WorkflowRuntime`, but review and adjudication still use deterministic code paths in this pass. This matters because the design target is not “free-form multi-agent orchestration”; it is inspectable workflow execution with narrow judgment points.

## Workflow Layering

### 1. Public API Layer

`translation_agent.api.run_job()`:

- validates configuration
- creates a request manifest in blob storage
- inserts a run record into the operational store
- configures structured logging and JSONL tracing
- builds the runtime
- runs the workflow
- persists final run status and outputs

`translation_agent.cli.main()` is a thin wrapper over the API and config validation helpers.

### 2. Graph And Node Layer

The graph is wired in `graph/builder.py` and executes these nodes:

1. `ingest`
2. `extract_audio`
3. `fanout_transcription`
4. `normalize_transcripts`
5. `review_transcripts`
6. `adjudicate_transcript`
7. `background_memory_pipeline`
8. conditional route to either translation generation or finalization
9. `normalize_translations`
10. `review_translations`
11. `adjudicate_translation`
12. `background_memory_pipeline`
13. `finalize_outputs`

The graph itself stages memory synchronously, but best-effort memory consolidation happens after graph completion inside `run_workflow()`. This is an important implementation detail: finalization is not blocked on consolidation.

### 3. Adapter Layer

The adapter layer is intentionally narrow:

- audio extraction returns a canonical `AudioArtifact`
- transcription returns canonical `TranscriptCandidate` values plus raw payload refs
- translation returns canonical `TranslationCandidate` values plus raw response refs

Shared retry and transport behavior lives in `adapters/common.py`. Real adapters write raw payloads into blob storage so normalization and replay can operate on stable refs instead of transient provider objects.

### 4. Review And Adjudication Layer

Review contracts are stage-specific and role-specific:

- transcript reviewers: `accuracy_reviewer`, `coherence_reviewer`
- translation reviewers: `faithfulness_reviewer`, `style_reviewer`

Reviewer output is required to match a fixed prose shape with these sections:

- `Winner`
- `Confidence`
- `Why`
- `Key Errors By Candidate`
- `Quoted Evidence`
- `Suggested Fixes`
- `Escalate?`

Adjudication in `review/policy.py` scores disagreement using:

- winner mismatch
- confidence spread
- contradictory evidence count
- highest issue severity
- escalation signal count
- content-risk multiplier

Current thresholds:

- score `< 3.0`: `automatic_finalize`
- score `>= 3.0`: `conflict_investigation`
- score `>= 5.2`: `stronger_adjudicator`
- unresolved or high-risk cases: `human_review`

The adjudication output is deterministic and yields:

- final decision model
- disagreement bucket
- scorecard
- escalation flags
- optional investigation payload

## Persistence Model

The implementation uses two complementary persistence paths.

### Blob Storage

Blob storage is the durable source for:

- request manifests
- raw provider payloads
- staged candidates
- normalized candidates
- reviews
- decisions
- investigations
- memory artifacts
- traces
- published artifacts and exports

By default the blob store is local filesystem storage rooted at `.translation-agent/blobs/`.

### Operational Store

The operational store tracks:

- runs
- node executions
- transcript candidates
- translation candidates
- transcript decisions
- translation decisions
- investigations
- memory batches

It can run on:

- SQLite by default
- Postgres when `TA_STATE_DB_DSN` is set

Postgres schema management is handled through Alembic migrations under `alembic/`.

## Artifact Scoping

Artifact keys are scoped by:

- tenant
- project
- source language
- target language
- job ID

This prevents collisions when:

- the same job ID appears in different tenants
- the same job ID is translated into different target languages

The helper functions in `storage/paths.py` define this convention and should be treated as the source of truth for new artifact keys.

## Memory Architecture

The current memory implementation is local and blob-backed, but the workflow already enforces useful boundaries.

### Recall

Recall is tenant, project, and language scoped. The recall backend returns:

- semantic memory
- episodic memory
- glossary entries
- rules
- provider caveats

Review and adjudication then further narrow those slices so prompts stay bounded and deterministic.

### Staging

Memory staging happens at adjudication boundaries.

The workflow creates `MemoryWriteBatch` artifacts that can include:

- semantic writes
- episodic writes
- procedural writes

Staging intentionally avoids writing raw transcript or translation bodies into long-term memory.

### Consolidation

Consolidation is a best-effort background step that:

- deduplicates writes
- persists semantic and episodic entries
- records skipped dedupe keys
- counts procedural writes

### Prompt Evolution

Prompt evolution proposals are generated only from consolidated translation adjudication outcomes. The current implementation:

- never derives prompt changes directly from raw reviewer prose
- only targets translation prompts
- can mark low-disagreement translation outcomes as `auto_activate_eligible`
- keeps reviewer and adjudicator prompt changes approval-gated

## Observability

The repo emits:

- structured logs through `observability/events.py`
- JSONL traces through `observability/tracing.py`
- node execution records in the operational store

This gives three complementary views of a run:

- coarse run lifecycle
- per-node execution status
- append-only trace events with timestamps and attributes

## Replay And Inspectability

`replay.py` reconstructs adjudication from persisted refs:

- candidate refs
- review refs
- memory ref

This is important for debugging because it lets you prove whether a decision changed because of:

- normalized inputs
- review parsing
- adjudication policy
- stored investigation artifacts

Replay also preserves the translation conflict-timeout regression path when the stored investigation payload reports `status="timed_out"`.

## Non-Goals Of The Current Implementation

The repo does not yet provide:

- a human-in-the-loop operator UI
- runtime manual override workflows
- free-form multi-agent orchestration
- live reasoning adapters for review, adjudication, or escalation
- external vector or semantic databases for long-term memory

Those remain roadmap items rather than hidden assumptions in the current code.
