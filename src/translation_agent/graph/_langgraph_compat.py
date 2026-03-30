"""LangGraph imports with targeted warning suppression for Python 3.14."""

from __future__ import annotations

import warnings
from importlib.metadata import metadata, version
from typing import TypedDict

_PY314_PYDANTIC_V1_WARNING = (
    r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\."
)

# langchain-core 1.2.23 still imports its legacy pydantic.v1 helpers during
# StateGraph import on Python 3.14, even when callers only use Pydantic v2 APIs.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=_PY314_PYDANTIC_V1_WARNING,
        category=UserWarning,
        module=r"langchain_core\._api\.deprecation",
    )
    from langgraph.graph import END, START, StateGraph


def ensure_langgraph_runtime_supported() -> None:
    """Fail fast if the installed LangGraph stack is incompatible with the runtime."""

    class _CompatibilityProbeState(TypedDict):
        ok: bool

    langgraph_requires_python = metadata("langgraph").get("Requires-Python")
    langchain_requires_python = metadata("langchain-core").get("Requires-Python")
    if langgraph_requires_python is None or langchain_requires_python is None:
        raise RuntimeError("LangGraph compatibility metadata is unavailable")

    # Build and compile a minimal graph so the runtime surface is exercised,
    # not only the package metadata.
    graph = StateGraph(_CompatibilityProbeState)
    graph.add_node("noop", lambda state: state)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    graph.compile(name="compatibility_probe")


def langgraph_runtime_versions() -> dict[str, str]:
    """Return the installed dependency versions used by the compatibility gate."""

    return {
        "langgraph": version("langgraph"),
        "langchain-core": version("langchain-core"),
    }


__all__ = [
    "END",
    "START",
    "StateGraph",
    "ensure_langgraph_runtime_supported",
    "langgraph_runtime_versions",
]
