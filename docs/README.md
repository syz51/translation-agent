# Translation Agent Docs

This folder is the implementation-facing source of truth for v1.

The original planning document was broad and idea-dense. These docs restate it in a form that a coding agent can follow without re-synthesizing the same decisions each time.

## What This System Is

V1 is a workflow-first translation pipeline built in Python with LangGraph.

It:

- ingests a video job
- extracts audio with `ffmpeg`
- fans out transcription to multiple providers
- reviews and adjudicates transcript candidates
- generates 2 translation candidates with `gpt-5.4-mini`
- reviews and adjudicates translation candidates
- publishes final artifacts
- writes memory candidates asynchronously after adjudication boundaries

It is not, in v1:

- an interactive operator console
- a human-in-the-loop runtime by default
- a free-form multi-agent system with agent-owned control flow

## Defaults To Preserve

- Architecture is workflow-first, not agent-first.
- Deterministic nodes own orchestration, retries, normalization, routing, persistence, and memory staging.
- Agentic layers are limited to critique, comparison, conflict investigation, and escalation.
- Transcript generation uses external transcription providers, not LLM transcription.
- Translation generation uses 1 model with 2 prompt variants.
- Translation-generation prompts may auto-evolve from consolidated review outcomes.
- Reviewer and adjudicator prompts must remain approval-gated.
- The runtime is fully automated unless escalation remains unresolved.
- The system is multi-tenant.
- Translation stays inside the same graph as transcription.

## Read Order

1. [`architecture.md`](./architecture.md)
   Explains system boundaries, execution model, and agent vs deterministic split.
2. [`workflow.md`](./workflow.md)
   Defines graph nodes, inputs, outputs, and routing behavior.
3. [`interfaces-and-data-models.md`](./interfaces-and-data-models.md)
   Gives the implementation contracts and canonical entities.
4. [`implementation-plan.md`](./implementation-plan.md)
   Gives the recommended package layout and build order.
5. [`roadmap.md`](./roadmap.md)
   Lists intentionally deferred work.

## Fast Start For A Coding Agent

If implementing from scratch, follow this order:

1. Create the package layout from [`implementation-plan.md`](./implementation-plan.md).
2. Implement canonical models and adapter interfaces from [`interfaces-and-data-models.md`](./interfaces-and-data-models.md).
3. Implement deterministic workflow nodes from [`workflow.md`](./workflow.md).
4. Add provider adapters and review/adjudication logic.
5. Add persistence, artifact publishing, and background memory writes.
6. Add tests in the order listed in [`implementation-plan.md`](./implementation-plan.md).

## Current Repo Reality

The current repository is still mostly empty. Treat these docs as the current spec, not as secondary documentation for an already-built system.
