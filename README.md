# translation-agent

`translation-agent` is a workflow-first translation pipeline built in Python on top of LangGraph. The repository is no longer a bootstrap scaffold: it already contains a runnable dry-run runtime, a real-adapter runtime, operational persistence, replay helpers, memory staging and consolidation, artifact publishing, and a test suite that exercises the end-to-end workflow.

## What It Does

The current pipeline executes these stages:

```text
ingest
-> extract_audio
-> fanout_transcription
-> normalize_transcripts
-> review_transcripts
-> adjudicate_transcript
-> background_memory_pipeline
-> generate_translation_candidates
-> normalize_translations
-> review_translations
-> adjudicate_translation
-> background_memory_pipeline
-> finalize_outputs
```

The workflow is deterministic-first:

- orchestration, retries, routing, persistence, normalization, and memory staging are implemented as deterministic nodes
- transcript and translation review are generated into a fixed prose contract and parsed back into structured review bundles
- adjudication is deterministic scoring with explicit escalation modes: `automatic_finalize`, `conflict_investigation`, `stronger_adjudicator`, and `human_review`
- translation failures are recoverable: the run can preserve the approved transcript, publish a failure manifest, and stop short of a final translation

## Runtime Modes

The repo supports two execution modes.

| Mode | How it is selected | What it uses | Best for |
| --- | --- | --- | --- |
| `fake` | default | deterministic fake adapters and scenario-driven outputs | local development, docs examples, fast tests |
| `real` | `TA_ADAPTER_MODE=real` | `ffmpeg`, a selectable subset of AssemblyAI, Speechmatics, and Deepgram, plus the OpenAI Responses API | real provider integration |

Notes:

- fake mode requires no provider credentials
- real mode defaults to all three transcription providers when `TA_TRANSCRIPTION_PROVIDERS` is unset
- real mode optionally supports a comma-separated transcription-provider subset via `TA_TRANSCRIPTION_PROVIDERS`
- real mode credential requirements depend on the selected transcription providers and still always require `TA_OPENAI_API_KEY`
- real mode is currently gated behind the LangGraph Python 3.14 compatibility check unless `TA_ALLOW_LANGGRAPH_PY314_WARNING=1` is set
- fake mode still writes the full run record, trace, blob artifacts, scorecards, memory batches, consolidations, and prompt-evolution proposals

## Quick Start

### Local Dry Run

```bash
uv run translation-agent validate-config --json
uv run translation-agent run-job input.wav --job-id demo --json
uv run translation-agent run-job input.wav --job-id demo-zh --target-language zh --json
uv run translation-agent convert-json-to-srt .translation-agent/blobs/.../published/translation.json
```

Defaults:

- data root: `.translation-agent/`
- blob root: `.translation-agent/blobs/`
- local state DB: `.translation-agent/state.sqlite3`
- traces: `.translation-agent/traces/`

### Postgres-Backed Runtime

```bash
docker compose up --build
uv run translation-agent validate-config --json
```

What `docker compose up --build` now does:

- loads repo-root `.env`
- starts Postgres 18 on port `55432`
- runs `translation-agent migrate-db` automatically after Postgres is healthy
- keeps an `app` container alive with the repo mounted, a Linux virtualenv cached in a named volume, and `validate-config` already checked

If you want to run the CLI on the host after the stack is up, no manual `export` is needed. `translation-agent` auto-loads the repo-root `.env`, so `uv run translation-agent ...` will use the Postgres DSN from that file by default.

If you want to run the CLI inside the container instead, use:

```bash
docker compose exec app uv run translation-agent run-job input.wav --job-id demo --json
```

The tracked example is [`.env.example`](./.env.example). This workspace also includes a local `.env` with the same safe development defaults.

### Real Adapter Mode

```bash
export TA_ADAPTER_MODE=real
export TA_ASSEMBLYAI_API_KEY=...
export TA_SPEECHMATICS_API_KEY=...
export TA_DEEPGRAM_API_KEY=...
export TA_OPENAI_API_KEY=...
export TA_ALLOW_LANGGRAPH_PY314_WARNING=1
uv run translation-agent validate-config --json
```

If the LangGraph compatibility warning disappears in a future dependency update, the explicit opt-in should no longer be necessary.

`TA_TRANSCRIPTION_PROVIDERS` is optional in real mode. If it is unset, the runtime behaves exactly as before and uses `assemblyai,speechmatics,deepgram`. If it is set, real mode uses exactly the selected non-empty subset in the configured order.

AssemblyAI-only:

```bash
export TA_ADAPTER_MODE=real
export TA_TRANSCRIPTION_PROVIDERS=assemblyai
export TA_ASSEMBLYAI_API_KEY=...
export TA_OPENAI_API_KEY=...
export TA_ALLOW_LANGGRAPH_PY314_WARNING=1
uv run translation-agent validate-config --json
uv run translation-agent run-job /absolute/path/to/input.mp4 --job-id demo-real
```

AssemblyAI + Deepgram:

```bash
export TA_ADAPTER_MODE=real
export TA_TRANSCRIPTION_PROVIDERS=assemblyai,deepgram
export TA_ASSEMBLYAI_API_KEY=...
export TA_DEEPGRAM_API_KEY=...
export TA_OPENAI_API_KEY=...
export TA_ALLOW_LANGGRAPH_PY314_WARNING=1
uv run translation-agent validate-config --json
```

## CLI

The CLI entrypoint is `translation-agent`.

### `validate-config`

```bash
uv run translation-agent validate-config
uv run translation-agent validate-config --json
```

It verifies:

- runtime directories exist or can be created
- SQLite or Postgres connectivity works
- secrets are not leaked in DSN output
- provider credentials are present for the selected real-mode providers when `TA_ADAPTER_MODE=real`
- real mode is allowed by the current LangGraph compatibility gate

### `run-job`

```bash
uv run translation-agent run-job input.wav --job-id demo
uv run translation-agent run-job input.wav --job-id demo --review auto
uv run translation-agent run-job input.wav --job-id demo --json
uv run translation-agent run-job input.wav --job-id demo-ja --target-language ja --json
uv run translation-agent review-job <run-id> --json
uv run translation-agent approve-review <run-id> --candidate-id <candidate-id> --json
```

`review-job` now has two operator surfaces:

- `--json` returns the existing candidate payload plus additive `review_diffs` cards for pairwise review
- interactive mode opens a one-diff-at-a-time terminal viewer with left/right finalization shortcuts

In interactive mode, choosing left or right finalizes the whole review with that candidate. It does
not mark only the current diff as accepted and continue to the next conflict.

The public result payload contains:

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
- `failure_ref`
- `failure_summary`
- `failure_reasons`
- `review_required_stage`
- `approval_ref`
- `approved_candidate_id`
- `approved_source_transcript_candidate_id`
- `resume_commands`

Final statuses currently emitted by the API are:

- `completed`
- `completed_with_degraded_transcription`
- `completed_after_human_review`
- `human_review_required`
- `translation_failed`

Human review is translation-only in the current CLI. Transcript disagreement no longer blocks the
runtime path; instead, translation generation fans out across all surviving transcript candidates,
and a later approved translation implicitly selects the canonical transcript candidate.

### `convert-json-to-srt`

```bash
uv run translation-agent convert-json-to-srt .translation-agent/blobs/.../published/translation.json
uv run translation-agent convert-json-to-srt .translation-agent/blobs/.../published/translation.json --output ./translation-backfill.srt --json
```

This command converts a persisted `TranslationCandidate` artifact such as `published/translation.json`
or `candidates/translations/*.json` into `.srt`. It does not accept the summary payload at
`exports/translation.json` because that file does not include timed segments.

## Python API

```python
from translation_agent.api import RunJobRequest, run_job

result = run_job(
    RunJobRequest(
        source="input.wav",
        job_id="demo",
        tenant_id="tenant-local",
        project_id="project-local",
        source_language="en",
        target_language="zh",
    )
)

print(result.status)
print(result.trace_path)
```

The API validates configuration first, persists a request manifest, creates a run record, executes the workflow, writes a JSONL trace, and returns the run summary.

## Artifact Layout

A sample successful dry-run writes artifacts like this under the blob root:

```text
jobs/<run_id>-request.json
memory/long-term/store.json
tenants/<tenant>/projects/<project>/languages/<src>-to-<dst>/jobs/<job_id>/
  artifacts/audio.json
  artifacts/audio.wav
  staging/transcripts/*.json
  staging/translations/*.json
  raw/provider-payloads/*.json
  candidates/transcripts/*.json
  candidates/translations/*.json
  reviews/transcript/*.json
  reviews/translation/*.json
  approvals/translation.json
  learning/transcript-approval.json
  decisions/transcript.json
  decisions/translation.json
  investigations/*.json
  memory/recall/*.json
  memory/batches/*.json
  memory/consolidations/*.json
  memory/prompt-evolution/*.json
  published/artifacts.json
  published/scorecard.json
  published/transcript.json
  published/translation.json
  exports/translation.srt
  exports/translation.json
  deliveries/translation.json
  traces/<run_id>.jsonl
```

Important behavior:

- artifact keys are scoped by tenant, project, source language, target language, and job ID
- the same job ID can exist safely across tenants or language pairs
- translation failure publishes `translation-failed.json` instead of `published/translation.json`
- the scorecard includes routing facts, decision payloads, trace refs, export refs, downstream refs, memory refs, and prompt-evolution refs

To backfill subtitles from a persisted translation artifact:

```bash
uv run translation-agent convert-json-to-srt path/to/published/translation.json
```

## Configuration

The settings model accepts more fields than most users need. These are the ones that materially change current behavior.

| Variable | Purpose |
| --- | --- |
| `TA_DATA_DIR` | root for local blobs, traces, and SQLite state |
| `TA_BLOB_DIR` | override blob directory |
| `TA_TRACE_DIR` | override trace directory |
| `TA_STATE_DB_PATH` | override SQLite database path |
| `TA_STATE_DB_DSN` | switch operational state to Postgres |
| `TA_ADAPTER_MODE` | choose `fake` or `real` runtime |
| `TA_TRANSCRIPTION_PROVIDERS` | comma-separated subset of real-mode transcription providers: `assemblyai`, `speechmatics`, `deepgram` |
| `TA_ALLOW_LANGGRAPH_PY314_WARNING` | opt into real mode despite the current warning gate |
| `TA_FFMPEG_BINARY` | override the `ffmpeg` executable path |
| `TA_PROVIDER_TIMEOUT_SECONDS` | provider HTTP timeout |
| `TA_ADAPTER_RETRY_ATTEMPTS` | retry attempts for provider calls |
| `TA_ADAPTER_INITIAL_BACKOFF_SECONDS` | initial retry backoff |
| `TA_ADAPTER_MAX_BACKOFF_SECONDS` | max retry backoff |
| `TA_ADAPTER_POLL_INTERVAL_SECONDS` | poll interval for async providers |
| `TA_ADAPTER_POLL_ATTEMPTS` | max provider polling attempts |
| `TA_ASSEMBLYAI_API_KEY` | AssemblyAI credential for real mode |
| `TA_SPEECHMATICS_API_KEY` | Speechmatics credential for real mode |
| `TA_DEEPGRAM_API_KEY` | Deepgram credential for real mode |
| `TA_OPENAI_API_KEY` | OpenAI credential for real mode |
| `TA_DEFAULT_SOURCE_LANGUAGE` | default source language for API and CLI runs when omitted |
| `TA_DEFAULT_TARGET_LANGUAGE` | default target language for API and CLI runs when omitted |
| `TA_TRANSLATION_MODEL_ID` | translation model ID for real mode |
| `TA_TRANSLATION_PROMPT_VERSION` | translation prompt version recorded in outputs |

The settings model also exposes `TA_WORKSPACE_DIR`, `TA_LOG_LEVEL`, `TA_EMIT_CONSOLE_LOGS`, and provider base URL overrides. At the moment those are configuration surface area, but the repo’s behavior is primarily driven by the variables listed above.

Environment loading order:

- process environment variables
- repo-root `.env`
- model defaults

## Testing And Verification

Project conventions in [`AGENTS.md`](./AGENTS.md) require `uv` and `ruff`.

Useful commands:

```bash
uv run ruff check .
uv run detect-secrets scan > .secrets.baseline
uv run pre-commit run --all-files
uv run pyright
uv run pytest -m "unit or slice"
uv run pytest -m contract
uv run pytest -m "integration or migration" -n 2 --dist=loadfile
uv run pytest -m "regression and not staging_only"
```

The pre-commit suite now includes `detect-secrets` with [`.secrets.baseline`](./.secrets.baseline) so new credential-like additions are blocked while the current fake/test fixtures stay allowlisted. Regenerate the baseline only when you intentionally add or change known non-secret fixtures.

The highest-signal workflow coverage lives in:

- `tests/test_phase_two_workflow.py`
- `tests/test_phase_three_adapters.py`
- `tests/test_phase_four_review.py`
- `tests/test_phase_five_memory_publish.py`
- `tests/test_phase_six_hardening.py`
- `tests/regression/test_runtime_regression.py`

## Docs

Implementation docs live under [`docs/`](./docs):

1. [`docs/README.md`](./docs/README.md)
2. [`docs/architecture.md`](./docs/architecture.md)
3. [`docs/workflow.md`](./docs/workflow.md)
4. [`docs/interfaces-and-data-models.md`](./docs/interfaces-and-data-models.md)
5. [`docs/implementation-plan.md`](./docs/implementation-plan.md)
6. [`docs/roadmap.md`](./docs/roadmap.md)
