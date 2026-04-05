"""Routing helpers for the deterministic dry-run workflow."""

from __future__ import annotations

from translation_agent.graph.state import GraphState


def route_after_memory_pipeline(state: GraphState) -> str:
    """Choose the next step after transcript or translation memory staging."""

    if state.final_translation_decision_ref is not None:
        return "finalize_outputs"
    if state.transcript_failed:
        return "finalize_outputs"
    if state.human_review_required:
        return "finalize_outputs"
    return "generate_translation_candidates"
