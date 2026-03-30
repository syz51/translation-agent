"""Deterministic LangGraph builder for the Phase 2 dry-run workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from translation_agent.graph.routing import route_after_memory_pipeline
from translation_agent.graph.runtime import WorkflowRuntime
from translation_agent.graph.state import GraphState
from translation_agent.nodes.adjudicate import adjudicate_transcript, adjudicate_translation
from translation_agent.nodes.extract_audio import extract_audio
from translation_agent.nodes.finalize import finalize_outputs
from translation_agent.nodes.ingest import ingest_job
from translation_agent.nodes.memory_pipeline import background_memory_pipeline
from translation_agent.nodes.normalize import normalize_transcripts, normalize_translations
from translation_agent.nodes.review import review_transcripts, review_translations
from translation_agent.nodes.transcription import fanout_transcription
from translation_agent.nodes.translate import generate_translation_candidates
from translation_agent.observability import TraceEvent

NodeFn = Callable[[GraphState, WorkflowRuntime], dict[str, object]]


def build_workflow_graph(runtime: WorkflowRuntime):
    """Compile the deterministic Phase 2 workflow graph."""

    builder = StateGraph(GraphState)
    builder.add_node("ingest", _instrumented_node("ingest", runtime, ingest_job))
    builder.add_node("extract_audio", _instrumented_node("extract_audio", runtime, extract_audio))
    builder.add_node(
        "fanout_transcription",
        _instrumented_node("fanout_transcription", runtime, fanout_transcription),
    )
    builder.add_node(
        "normalize_transcripts",
        _instrumented_node("normalize_transcripts", runtime, normalize_transcripts),
    )
    builder.add_node(
        "review_transcripts",
        _instrumented_node("review_transcripts", runtime, review_transcripts),
    )
    builder.add_node(
        "adjudicate_transcript",
        _instrumented_node("adjudicate_transcript", runtime, adjudicate_transcript),
    )
    builder.add_node(
        "generate_translation_candidates",
        _instrumented_node(
            "generate_translation_candidates",
            runtime,
            generate_translation_candidates,
        ),
    )
    builder.add_node(
        "normalize_translations",
        _instrumented_node("normalize_translations", runtime, normalize_translations),
    )
    builder.add_node(
        "review_translations",
        _instrumented_node("review_translations", runtime, review_translations),
    )
    builder.add_node(
        "adjudicate_translation",
        _instrumented_node("adjudicate_translation", runtime, adjudicate_translation),
    )
    builder.add_node(
        "background_memory_pipeline",
        _instrumented_node(
            "background_memory_pipeline",
            runtime,
            background_memory_pipeline,
        ),
    )
    builder.add_node(
        "finalize_outputs",
        _instrumented_node("finalize_outputs", runtime, finalize_outputs),
    )

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract_audio")
    builder.add_edge("extract_audio", "fanout_transcription")
    builder.add_edge("fanout_transcription", "normalize_transcripts")
    builder.add_edge("normalize_transcripts", "review_transcripts")
    builder.add_edge("review_transcripts", "adjudicate_transcript")
    builder.add_edge("adjudicate_transcript", "background_memory_pipeline")
    builder.add_conditional_edges(
        "background_memory_pipeline",
        route_after_memory_pipeline,
        {
            "generate_translation_candidates": "generate_translation_candidates",
            "finalize_outputs": "finalize_outputs",
        },
    )
    builder.add_edge("generate_translation_candidates", "normalize_translations")
    builder.add_edge("normalize_translations", "review_translations")
    builder.add_edge("review_translations", "adjudicate_translation")
    builder.add_edge("adjudicate_translation", "background_memory_pipeline")
    builder.add_edge("finalize_outputs", END)
    return builder.compile(name="translation_agent_phase_two")


def run_workflow(initial_state: GraphState, runtime: WorkflowRuntime) -> GraphState:
    """Execute the compiled workflow and validate the final state."""

    compiled = build_workflow_graph(runtime)
    result = compiled.invoke(initial_state)
    return GraphState.model_validate(result)


def _instrumented_node(name: str, runtime: WorkflowRuntime, node_fn: NodeFn):
    def wrapped(state: GraphState) -> dict[str, object]:
        execution = runtime.run_store.create_node_execution(
            run_id=state.run_id,
            node_name=name,
            status="started",
            input_data=state.model_dump(mode="json"),
        )
        runtime.trace_sink.record(
            TraceEvent(
                run_id=state.run_id,
                name="node.started",
                attributes={"node_name": name, "execution_id": execution.execution_id},
            )
        )
        try:
            output = node_fn(state, runtime)
        except Exception as exc:
            runtime.run_store.update_node_execution(
                execution.execution_id,
                status="failed",
                error={"message": str(exc)},
            )
            runtime.trace_sink.record(
                TraceEvent(
                    run_id=state.run_id,
                    name="node.failed",
                    attributes={
                        "node_name": name,
                        "execution_id": execution.execution_id,
                        "error": str(exc),
                    },
                )
            )
            raise

        runtime.run_store.update_node_execution(
            execution.execution_id,
            status="completed",
            output_data=_json_safe(output),
        )
        runtime.trace_sink.record(
            TraceEvent(
                run_id=state.run_id,
                name="node.completed",
                attributes={"node_name": name, "execution_id": execution.execution_id},
            )
        )
        return output

    return wrapped


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
