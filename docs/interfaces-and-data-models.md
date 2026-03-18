# Interfaces And Data Models

## Core Interface Contracts

These are the implementation-facing interfaces from the plan. Keep them stable unless the architecture changes.

```python
extract_audio(video_ref, job_context) -> AudioArtifact
transcribe(audio_artifact, provider_id, request_context) -> TranscriptCandidate
normalize_transcript_candidates(candidates) -> list[TranscriptCandidate]
review_transcripts(candidates, review_context) -> list[ReviewBundle]
adjudicate_transcripts(candidates, reviews, adjudication_context) -> FinalTranscriptDecision
generate_translation(final_transcript, prompt_variant_id, request_context) -> TranslationCandidate
normalize_translation_candidates(candidates) -> list[TranslationCandidate]
review_translations(candidates, review_context) -> list[ReviewBundle]
adjudicate_translations(candidates, reviews, adjudication_context) -> FinalTranslationDecision
recall_memory(memory_query) -> MemoryBundle
stage_memory_candidates(adjudication_result) -> MemoryWriteBatch
finalize_run(run_result) -> PublishedArtifacts
```

## Canonical Entity Set

These names should become concrete models early.

### `JobContext`

Fields should include:

- `job_id`
- `tenant_id`
- `project_id`
- `source_video_ref`
- `target_language`
- `source_language`
- `requested_by`
- `created_at`
- `profile_ref`

### `AudioArtifact`

Fields should include:

- `artifact_id`
- `job_id`
- `blob_ref`
- `duration_ms`
- `sample_rate_hz`
- `channels`
- `codec`
- `extraction_metadata`

### `TranscriptCandidate`

Fields should include:

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

Fields should include:

- `candidate_id`
- `job_id`
- `source_transcript_candidate_id` or final transcript ref
- `model_id`
- `prompt_variant_id`
- `prompt_version`
- `language`
- `segments`
- `full_text`
- `raw_response_ref`
- `normalization_version`
- `metadata`

### `Segment`

Fields should include:

- `segment_id`
- `start_ms`
- `end_ms`
- `speaker`
- `source_text`
- `target_text`
- `annotations`

Notes:

- transcript candidates will populate `source_text`
- translation candidates may reuse segment timing and fill `target_text`

### `ReviewBundle`

Fields should include:

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

### `FinalTranscriptDecision`

Fields should include:

- `job_id`
- `winner_candidate_id`
- `decision_mode`
- `decision_confidence`
- `rationale_summary`
- `review_refs`
- `investigation_ref`
- `escalated`
- `human_review_required`

### `FinalTranslationDecision`

Fields should include the same decision pattern as transcript adjudication, plus:

- `prompt_variant_winner`
- `prompt_version_winner`

### `MemoryBundle`

Fields should include:

- semantic memory slice
- episodic memory slice
- glossary slice
- rules slice
- provider caveats

### `MemoryWriteBatch`

Fields should include:

- `batch_id`
- `job_id`
- `source_stage`
- proposed semantic writes
- proposed episodic writes
- proposed procedural writes
- dedupe keys
- consolidation status

### `PublishedArtifacts`

Fields should include:

- final transcript ref
- final translation ref
- scorecard refs
- trace refs
- export refs
- downstream delivery refs

## Provider Adapter Contracts

Keep adapters narrow.

### Transcription Adapter

Must support:

- request transcription
- return provider IDs and raw payload ref
- normalize provider-specific metadata into canonical candidate shape
- classify retryable vs terminal errors

Providers in v1:

- `AssemblyAI` as primary
- `Speechmatics` as comparison or backup
- `Deepgram` as comparison or backup

### Translation Adapter

Must support:

- model invocation for prompt variant A
- model invocation for prompt variant B
- prompt version tracking
- raw response capture
- retryable error classification

V1 model:

- `gpt-5.4-mini`

## Adjudication Inputs

The deterministic adjudication node should compare at least:

- winner mismatch
- confidence spread
- contradictory evidence count
- severity of issues
- content risk class

This should produce one of:

- finalize automatically
- investigate conflict
- escalate to stronger adjudicator
- require human review

## Storage Shape

### Blob Store

Should hold:

- videos
- extracted audio
- raw provider payloads
- exported outputs

### Operational DB

Should hold tables or equivalent records for:

- runs
- node executions
- transcript candidates
- transcript decisions
- translation candidates
- translation decisions
- escalation events
- memory write batches

### Observability

V1 requires:

- structured run records
- structured evaluation records
- trace events through an abstract tracing interface

V1 does not require LangSmith, but should not block adding it later.
