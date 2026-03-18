# Architecture

## System Shape

V1 is a Python application with two public entrypoints:

- CLI for local and batch execution
- Python API for programmatic use

The core runtime is a LangGraph workflow. Most execution is deterministic. Agents are used only where comparison, critique, or escalation adds value.

## Primary Design Choice

Use a deterministic workflow to control:

- artifact movement
- provider calls
- retries
- normalization
- routing
- persistence
- memory staging

Use agents to perform:

- transcript critique
- translation critique
- evidence-based comparison
- targeted dispute investigation

This keeps the system inspectable and replayable while still allowing agent judgment where it matters.

## Runtime Layers

### 1. Workflow Layer

Owns end-to-end job execution and state transitions.

Responsibilities:

- receive job input
- manage node execution
- keep graph state lean
- pass IDs or references instead of large blobs

### 2. Tool And Provider Layer

Owns direct integrations.

V1 policy:

- `ffmpeg` runs as a direct local tool
- transcription providers use direct Python adapters
- OpenAI translation/review calls use a direct Python adapter
- MCP is optional and limited to auxiliary tools

### 3. Agent Review Layer

Owns candidate critique and conflict analysis.

Actors:

- Transcript Reviewer A
- Transcript Reviewer B
- Translation Reviewer A
- Translation Reviewer B
- Conflict Investigator
- Stronger adjudicator model for high-risk escalation

### 4. Persistence And Artifact Layer

Stores:

- operational records
- provider payload references
- candidate artifacts
- final outputs
- escalation records
- memory write batches

### 5. Memory Layer

Owns read/write policy across short-term and long-term memory stores.

## Memory Architecture

### Layer 0: Run-Local State

LangGraph thread or checkpoint state for the active run.

Use for:

- current node outputs
- IDs and lightweight refs
- routing decisions

### Layer 1: Human-Maintained Working Files

Deep Agents `/memories/` files for:

- project instructions
- runbooks
- correction ledgers
- manual notes

### Layer 2: Semantic Memory

LangMem semantic storage for:

- glossary entries
- transcription rules
- translation rules
- provider caveats
- tenant or project preferences

### Layer 3: Episodic Memory

LangMem episodic storage for:

- hard wins
- hard failures
- edge cases worth recalling later

### Layer 4: Procedural Memory

LangMem storage for prompt improvement proposals, especially around translation prompt evolution.

## Memory Write Boundaries

- Review agents can read scoped memory slices.
- Review and adjudication agents do not write long-term memory directly.
- Adjudication emits the trusted memory-candidate bundle.
- Background consolidation is the only writer to semantic and episodic long-term memory.
- Translation prompt auto-evolution consumes consolidated outcomes, not raw reviewer prose alone.

## Deterministic vs Agentic Boundary

### Deterministic Nodes Must Own

- media extraction
- provider fanout
- prompt variant selection
- normalization
- retries
- persistence
- escalation routing
- memory batching

### Agentic Nodes Must Own

- candidate critique
- quoted evidence gathering
- issue comparison
- limited conflict investigation

## Adjudication Model

Adjudication is deterministic-first.

Flow:

1. Parse reviewer outputs into structured signals.
2. Compare winners, confidence, evidence quality, and severity.
3. Finalize automatically when disagreement is low.
4. Spawn a conflict investigation step when disagreement is medium.
5. Escalate to a stronger adjudicator when disagreement is high or content risk is high.
6. Mark for human review if still unresolved.

## Core Non-Goals For V1

- operator supervisor in the runtime path
- recursive multi-agent debate loops
- human approval gates for normal runs
- a second translation provider
- mandatory LangSmith dependency

## Suggested Package Boundaries

The code should eventually separate into modules roughly like:

- `translation_agent.cli`
- `translation_agent.api`
- `translation_agent.graph`
- `translation_agent.nodes`
- `translation_agent.adapters`
- `translation_agent.models`
- `translation_agent.review`
- `translation_agent.memory`
- `translation_agent.storage`
- `translation_agent.publish`
- `translation_agent.observability`
