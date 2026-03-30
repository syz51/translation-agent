"""LangGraph imports with targeted warning suppression for Python 3.14."""

from __future__ import annotations

import warnings

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

__all__ = ["END", "START", "StateGraph"]
