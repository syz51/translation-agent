from __future__ import annotations

import importlib


MODULES = [
    "translation_agent",
    "translation_agent.adapters.assemblyai",
    "translation_agent.adapters.deepgram",
    "translation_agent.adapters.ffmpeg",
    "translation_agent.adapters.openai_translation",
    "translation_agent.adapters.speechmatics",
    "translation_agent.api",
    "translation_agent.cli",
    "translation_agent.config",
    "translation_agent.graph.builder",
    "translation_agent.graph.routing",
    "translation_agent.graph.state",
    "translation_agent.memory.consolidation",
    "translation_agent.memory.prompt_evolution",
    "translation_agent.memory.recall",
    "translation_agent.memory.staging",
    "translation_agent.models.artifacts",
    "translation_agent.models.jobs",
    "translation_agent.models.memory",
    "translation_agent.models.review",
    "translation_agent.models.transcript",
    "translation_agent.models.translation",
    "translation_agent.nodes.adjudicate",
    "translation_agent.nodes.extract_audio",
    "translation_agent.nodes.finalize",
    "translation_agent.nodes.ingest",
    "translation_agent.nodes.memory_pipeline",
    "translation_agent.nodes.normalize",
    "translation_agent.nodes.review",
    "translation_agent.nodes.transcription",
    "translation_agent.nodes.translate",
    "translation_agent.observability.events",
    "translation_agent.observability.tracing",
    "translation_agent.publish.outputs",
    "translation_agent.review.parser",
    "translation_agent.review.policy",
    "translation_agent.review.prompts",
    "translation_agent.storage.blobs",
    "translation_agent.storage.decisions",
    "translation_agent.storage.memory_batches",
    "translation_agent.storage.runs",
]


def test_phase_zero_modules_import_cleanly() -> None:
    for module in MODULES:
        importlib.import_module(module)
