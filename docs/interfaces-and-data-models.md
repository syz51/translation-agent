# Interfaces And Data Models

## Public Entry Points

### CLI

`translation_agent.cli` exposes:

- `translation-agent validate-config [--json]`
- `translation-agent list-runs [--json]`
- `translation-agent run-job <source> [--job-id <id>] [--source-language <code>] [--target-language <code>] [--json]`

### Python API

`translation_agent.api` exposes:

- `RunJobRequest`
- `RunJobResult`
- `list_runs()`
- `run_job()`

These are the stable entrypoints other tooling should build against.

## Runtime Configuration Contract

`translation_agent.config.Settings` is loaded from environment variables with the `TA_` prefix.

The most important current settings are:

- local path configuration
- SQLite versus Postgres operational state
- fake versus real adapter mode
- provider credentials
- retry, timeout, and polling controls
- default source and target languages
- translation model ID and prompt version

Configuration validation returns a `ValidationResult` with:

- path checks
- backend choice
- connectivity result
- provider-configuration result
- runtime-compatibility result
- sanitized database target

## Core Runtime Models

### `JobContext`

Immutable job identity and request facts:

- `job_id`
- `tenant_id`
- `project_id`
- `source_video_ref`
- `target_language`
- `source_language`
- `requested_by`
- `created_at`
- `profile_ref`

### `RequestContext`

Per-request runtime context for extraction, transcription, and translation:

- `run_id`
- `attempt`
- `job`
- `source_artifact_ref`
- `metadata`

### `GraphState`

Lean ref-only workflow state:

- run identity and current stage
- request and source refs
- audio, raw payload, candidate, review, decision, and published artifact refs
- memory batch IDs
- routing facts
- escalation and failure flags

The graph state is intentionally not a warehouse for large payloads. Durable data lives in blob storage and the operational store.

### `RoutingFact`

Small immutable fact record used to explain workflow decisions:

- `stage`
- `fact_type`
- `value`
- `source_ref`

The scorecard publishes routing facts so runs remain inspectable after completion.

## Canonical Artifact Models

### `AudioArtifact`

Canonical output of audio extraction:

- `artifact_id`
- `job_id`
- `blob_ref`
- `duration_ms`
- `sample_rate_hz`
- `channels`
- `codec`
- `extraction_metadata`

### `Segment`

Shared segment model for transcript and translation candidates:

- `segment_id`
- `start_ms`
- `end_ms`
- `speaker`
- `source_text`
- `target_text`
- `annotations`

### `TranscriptCandidate`

Normalized STT output:

- `candidate_id`
- `job_id`
- `provider_id`
- `provider_request_id`
- `language`
- `segments`
- `full_text`
- `speaker_map`
- `timing_resolution`
- `raw_payload_ref`
- `normalization_version`
- `metadata`

### `TranslationCandidate`

Normalized translation output:

- `candidate_id`
- `job_id`
- `source_transcript_candidate_id`
- `final_transcript_ref`
- `model_id`
- `prompt_variant_id`
- `prompt_version`
- `language`
- `segments`
- `full_text`
- `raw_response_ref`
- `normalization_version`
- `metadata`

## Review And Adjudication Models

### `ReviewContext`

Inputs passed into the review layer:

- run ID
- stage
- reviewer role
- job context
- candidate IDs
- stage-scoped memory bundle
- optional policy ref

### `ReviewBundle`

Parsed reviewer output:

- `review_id`
- `job_id`
- `stage`
- `reviewer_role`
- `candidate_preferences`
- `confidence`
- `raw_review_text`
- `quoted_evidence`
- `issue_categories`
- `suggested_fixes`
- `escalation_signal`
- `parser_version`

### `AdjudicationContext`

Context for deterministic adjudication:

- run ID
- stage
- job
- candidate IDs
- review IDs
- narrowed memory bundle
- optional investigation ref
- content-risk class

### `AdjudicationScorecard`

Structured scoring inputs used to explain a decision:

- `candidate_count`
- `preferred_candidate_id`
- `average_confidence`
- `confidence_spread`
- `contradictory_evidence_count`
- `highest_issue_severity`
- `winner_mismatch`
- `escalation_signal_count`
- `total_score`
- `content_risk_class`

### `FinalTranscriptDecision`

Transcript decision payload:

- `winner_candidate_id`
- `decision_mode`
- `decision_confidence`
- `rationale_summary`
- `review_refs`
- `investigation_ref`
- `disagreement_bucket`
- `adjudication_scorecard`
- `escalated`
- `human_review_required`

### `FinalTranslationDecision`

Translation decision payload extends the transcript pattern with:

- `winner_model_id`
- `prompt_variant_winner`
- `prompt_version_winner`

These fields are important because they feed prompt-evolution proposals and make translation decisions replayable.

## Memory Models

### `MemoryQuery`

Recall request with built-in scope:

- `job`
- `stage`
- `query_text`
- `candidate_ids`
- `max_items`

### `MemoryBundle`

Read-only recall output:

- `semantic_memory`
- `episodic_memory`
- `glossary`
- `rules`
- `provider_caveats`

### `MemoryWriteBatch`

Staged memory writes emitted at adjudication boundaries:

- `batch_id`
- `job_id`
- `source_stage`
- decision and winner metadata
- semantic, episodic, and procedural write lists
- dedupe keys
- consolidation status
- scope metadata

### `MemoryConsolidation`

Result of background consolidation:

- `consolidation_id`
- `batch_id`
- `job_id`
- source decision metadata
- inserted semantic and episodic memory IDs
- skipped dedupe keys
- procedural write count

### `PromptEvolutionProposal`

Translation-only prompt-improvement proposal derived from consolidated outcomes:

- `proposal_id`
- `job_id`
- `source_consolidation_id`
- `prompt_family`
- `target_model_id`
- `target_prompt_version`
- `target_prompt_variant_id`
- `status`
- `activation_mode`
- `auto_activate`
- `rationale`
- `suggested_changes`
- `evidence_refs`
- `metadata`

## Publishing Models

### `PublishContext`

Context for finalization and downstream publishing:

- run ID
- job
- transcript decision ref
- translation decision ref
- trace refs
- export targets
- downstream targets

### `PublishedArtifacts`

Stable references emitted by publishing:

- `final_transcript_ref`
- `final_translation_ref`
- `recoverable_translation_failure_ref`
- `scorecard_refs`
- `trace_refs`
- `export_refs`
- `downstream_delivery_refs`
- `memory_batch_refs`
- `memory_consolidation_refs`
- `prompt_evolution_refs`

## Storage Contracts

### Blob Store

The blob store contract supports:

- `put_bytes`
- `read_bytes`
- `exists`
- `delete`
- `list_keys`
- `iter_entries`

The default implementation is `LocalBlobStore`.

### Run Store

The run store contract supports:

- creating and updating runs
- creating and updating node executions
- listing historical runs and node executions

Current implementations:

- `SQLiteOperationalStore`
- `PostgresOperationalStore`

### Decision Store

The decision store persists:

- transcript candidates
- translation candidates
- transcript decisions
- translation decisions
- investigations
- transcript provider quality stats

### Memory Batch Store

The memory batch store persists:

- staged `MemoryWriteBatch` models

## Artifact Key Conventions

`storage/paths.py` defines the current job scope:

```text
tenants/<tenant>/projects/<project>/languages/<source>-to-<target>/jobs/<job_id>/
```

Everything published for a job hangs beneath that prefix. Any new feature that persists job-scoped artifacts should use the same convention instead of inventing a new storage layout.

## Output Payloads

### `validate-config --json`

The CLI contract includes:

- `ok`
- `checked_paths`
- `state_backend`
- `state_db_ok`
- `state_db_target`
- `adapter_mode`
- `runtime_compatibility_ok`
- `runtime_compatibility_error`
- `provider_config_ok`
- `provider_config_error`
- `state_db_error`

### `list-runs --json`

The CLI payload is a reverse-chronological array of persisted `RunRecord` objects. Each item includes:

- `run_id`
- `tenant_id`
- `project_id`
- `status`
- `created_at`
- `updated_at`
- `input_data`
- `output_data`
- `metadata`
- `error`

### `run-job --json`

The API and CLI result contract includes:

- `run_id`
- `job_id`
- `status`
- `source`
- `source_language`
- `target_language`
- `blob_root`
- `trace_path`
- `state_backend`
- `state_db_target`
- `review_required_stage`
- `approval_ref`
- `approved_candidate_id`
- `approved_source_transcript_candidate_id`
- `resume_commands`

### `review-job --json`

The review payload includes:

- `run_id`
- `job_id`
- `status`
- `review_required_stage`
- `summary`
- ranked translation candidates
- span-aligned `review_spans` records keyed by `source_span_id`
- optional `draft_resolution` for save/resume
- additive legacy `review_diffs` cards ordered for compatibility only
- source transcript provenance per candidate
- transcript disagreement and investigation summary
- preview paths for rendered translation and transcript artifacts

Each `review_spans[*]` record includes:

- `source_span_id`
- `start_ms`
- `end_ms`
- `time_range`
- `severity_summary`
- `blocking`
- `reviewer_roles`
- `evidence_summary`
- `source_excerpt`
- `transcript_provenance_options`
- `variants`
- `current_draft_decision`

Each `review_diffs[*]` card includes:

- `diff_id`
- `source_span_id`
- `time_range`
- `dimension`
- `severity`
- `blocking_hard_contradiction`
- `evidence_text`
- `normalized_value`
- `reviewer_roles`
- `source_excerpt`
- `left_candidate`
- `right_candidate`

`review_diffs` is now legacy and should not be treated as the primary TUI contract.
Interactive `review-job` launches a Textual app that reviews one span at a time, saves resumable
draft decisions, and finalizes a synthetic human-reviewed `TranslationCandidate`. The current TUI
uses Textual's themed command-palette surface, a tabbed span workspace, and toast notifications
instead of the earlier flat terminal layout.

### `approve-review --json`

The approval payload includes:

- `run_id`
- `job_id`
- `status`
- `approval_ref`
- `approved_candidate_id`
- `final_translation_candidate_id`
- `approved_source_transcript_candidate_id`
- final transcript and translation refs
- `default_output_path`

### Published Scorecard

The scorecard is the main audit payload and includes:

- job and run identity
- final status flags
- review and approval metadata
- transcript and translation refs
- trace refs
- export refs
- downstream refs
- memory refs
- routing facts
- transcript decision payload
- translation decision payload

### Human Approval And Learning Artifacts

New approval-side artifacts include:

- `HumanApprovalRecord` at `approvals/translation.json`
- `TranscriptApprovalLearningEvent` at `learning/transcript-approval.json`

The operational store also keeps `TranscriptProviderQualityStats` keyed by:

- transcript provider ID
- source language
- target language

These provider stats are used only as bounded transcript-ranking priors; they do not override hard
validation failures or invent transcript winners when no candidates survive.

## Replay Contract

`ReplayAdjudicationRequest` reconstructs a deterministic decision from:

- `run_id`
- `job`
- `stage`
- `candidate_refs`
- `review_refs`
- `memory_ref`
- `content_risk_class`

This contract is what makes the adjudication path inspectable and regression-testable after the original run completes.
