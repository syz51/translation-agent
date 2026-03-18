# Roadmap

This document records work intentionally deferred out of v1.

## Deferred Runtime Features

- operator supervisor agent
- interactive reruns
- manual override UX
- richer MCP integrations
- LangSmith integration

## Deferred Product Decisions

- reevaluate transcription provider mix after v1 data
- consider a second translation provider later
- revisit escalation model selection after baseline metrics exist

## Why These Are Deferred

V1 is optimizing for:

- a stable automated core workflow
- inspectable routing and decision behavior
- a small number of hard dependencies
- enough observability to debug without building an operator platform first

## Trigger To Reopen These Decisions

Revisit deferred items when one of these becomes true:

- unresolved escalations are too frequent
- debugging needs exceed trace tables and stored artifacts
- translation quality plateaus under the single-model, two-prompt strategy
- provider failure rates or cost profiles justify a different mix
- operators need guided reruns more than engineering needs more automation
