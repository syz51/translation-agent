# Implementation Plan

## Goal

Turn the current empty scaffold into a runnable v1 system without reopening core architecture choices during implementation.

## Recommended Package Layout

Use `src/` layout when the real code lands.

```text
src/
  translation_agent/
    __init__.py
    api.py
    cli.py
    config.py
    graph/
      __init__.py
      builder.py
      state.py
      routing.py
    nodes/
      __init__.py
      ingest.py
      extract_audio.py
      transcription.py
      normalize.py
      review.py
      adjudicate.py
      translate.py
      finalize.py
      memory_pipeline.py
    adapters/
      __init__.py
      ffmpeg.py
      openai_translation.py
      assemblyai.py
      speechmatics.py
      deepgram.py
    models/
      __init__.py
      jobs.py
      artifacts.py
      transcript.py
      translation.py
      review.py
      memory.py
    review/
      __init__.py
      prompts.py
      parser.py
      policy.py
    memory/
      __init__.py
      recall.py
      staging.py
      consolidation.py
      prompt_evolution.py
    storage/
      __init__.py
      blobs.py
      runs.py
      decisions.py
      memory_batches.py
    publish/
      __init__.py
      outputs.py
    observability/
      __init__.py
      tracing.py
      events.py
tests/
  adapters/
  graph/
  nodes/
  memory/
  integration/
```

## Delivery Order

### Phase 1: Foundation

Implement first:

- package layout
- config loading
- canonical models
- provider interface abstractions
- lightweight persistence interfaces
- tracing and event interface

Exit criteria:

- codebase imports cleanly
- model and adapter contracts are stable

### Phase 2: Deterministic Workflow Skeleton

Implement next:

- graph state model
- node signatures
- routing logic
- stub node implementations
- CLI entrypoint for running a job
- keep node inputs/outputs on the Phase 1 typed contracts and keep graph state ref-only

Exit criteria:

- a dry-run graph can execute end to end with fake adapters

### Phase 3: Adapter Integration

Implement:

- `ffmpeg` extraction adapter
- AssemblyAI adapter
- Speechmatics adapter
- Deepgram adapter
- OpenAI translation adapter
- make concrete adapters satisfy the Phase 1 protocol surface in `translation_agent.adapters`

Exit criteria:

- raw adapter contract tests pass
- normalization entrypoints have realistic inputs

### Phase 4: Review And Adjudication

Implement:

- reviewer prompt templates
- reviewer output parser
- disagreement scoring
- conflict investigation routing
- stronger escalation hook

Exit criteria:

- adjudication path is deterministic for fixed inputs
- parser extracts required fields from reviewer prose

### Phase 5: Memory And Publishing

Implement:

- memory recall
- memory write staging
- async consolidation pipeline
- translation prompt evolution logic
- artifact publishing
- keep long-term memory inputs scoped to Phase 1 memory queries and write batches instead of loose dict payloads

Exit criteria:

- adjudication emits memory batches
- finalized runs persist outputs and trace refs

### Phase 6: Hardening

Implement:

- failure injection tests
- replay tests
- isolation tests for tenant and project memory
- observability validation

Exit criteria:

- test matrix from this document is covered

## Test Plan To Preserve

### Adapter Contract Tests

Each transcription adapter needs tests for:

- success
- timeout
- malformed output
- partial metadata
- retryable failure

### Translation Adapter Tests

Need tests for:

- both prompt variants
- prompt-version tracking
- output normalization

### Graph And Routing Tests

Need tests for:

- replay of normalized inputs and memory refs
- one STT provider down
- two reviewers disagree
- conflict investigator timeout
- one translation variant failure
- missing blob fetch

### Memory Verification

Need tests for:

- project and language isolation
- duplicate consolidation
- no raw full transcripts in long-term memory
- no raw full translations in long-term memory

### Prompt Evolution Verification

Need tests for:

- translation prompt variants update only from consolidated outcomes
- reviewer and adjudicator prompts never auto-activate

### Observability Verification

Need tests for:

- every run emits trace events
- every run emits structured summary records

## Recommended Near-Term Repo Changes

These are the next practical coding steps after this docs pass:

1. Convert the project to `src/` layout.
2. Add real dependencies to `pyproject.toml` using `uv`.
3. Replace `main.py` with package-based CLI entrypoints.
4. Implement the canonical models before touching provider integrations.
5. Add fake adapters so the graph can run before external services are wired.

## Non-Goals During Initial Buildout

- perfect prompt tuning
- UI work
- operator dashboard
- LangSmith-first observability
- adding more providers before the base path works
