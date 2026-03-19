# translation-agent

Phase 0 bootstrap is in place. The implementation spec still lives in [`docs/README.md`](docs/README.md).

Quick start:

1. `uv sync`
2. `docker compose up -d postgres`
3. `export TA_STATE_DB_DSN=postgresql://translation_agent:translation_agent@127.0.0.1:55432/translation_agent`
4. `uv run translation-agent validate-config --json`
5. `uv run translation-agent run-job path/to/input.media --json`
6. `uv run pytest`

Recommended reading order:

1. [`docs/README.md`](docs/README.md)
2. [`docs/architecture.md`](docs/architecture.md)
3. [`docs/workflow.md`](docs/workflow.md)
4. [`docs/interfaces-and-data-models.md`](docs/interfaces-and-data-models.md)
5. [`docs/implementation-plan.md`](docs/implementation-plan.md)
6. [`docs/roadmap.md`](docs/roadmap.md)
