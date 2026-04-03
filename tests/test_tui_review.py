from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Static, TextArea

from translation_agent.tui.review import ReviewTerminalApp

pytestmark = pytest.mark.unit


def _payload() -> dict[str, Any]:
    return {
        "run_id": "run-123",
        "job_id": "job-123",
        "status": "review_required",
        "candidates": [
            {
                "rank": 1,
                "candidate_id": "candidate-a",
                "prompt_variant_id": "prompt-a",
                "source_transcript_candidate_id": "transcript-a",
                "source_transcript": {"provider_id": "assemblyai"},
                "contradiction_count": 1,
                "blocking_hard_contradiction_count": 1,
            },
            {
                "rank": 2,
                "candidate_id": "candidate-b",
                "prompt_variant_id": "prompt-b",
                "source_transcript_candidate_id": "transcript-b",
                "source_transcript": {"provider_id": "deepgram"},
                "contradiction_count": 1,
                "blocking_hard_contradiction_count": 1,
            },
        ],
        "review_spans": [
            {
                "source_span_id": "span:0:1000",
                "start_ms": 0,
                "end_ms": 1000,
                "time_range": "00:00.000-00:01.000",
                "severity_summary": "critical",
                "blocking": True,
                "reviewer_roles": ["faithfulness_reviewer"],
                "evidence_summary": [
                    {
                        "candidate_id": "candidate-a",
                        "dimension": "meaning",
                        "severity": "critical",
                        "normalized_value": "conflict",
                        "evidence_text": "Meaning conflict detected.",
                        "reviewer_role": "faithfulness_reviewer",
                    }
                ],
                "source_excerpt": "Hello world.",
                "transcript_provenance_options": [
                    {
                        "source_transcript_candidate_id": "transcript-a",
                        "transcript_provider_id": "assemblyai",
                        "transcript_excerpt": "Hello world.",
                    },
                    {
                        "source_transcript_candidate_id": "transcript-b",
                        "transcript_provider_id": "deepgram",
                        "transcript_excerpt": "Hello world there.",
                    },
                ],
                "variants": [
                    {
                        "candidate_id": "candidate-a",
                        "rank": 1,
                        "model_id": "gpt-5.4",
                        "prompt_variant_id": "prompt-a",
                        "prompt_version": "v1",
                        "source_transcript_candidate_id": "transcript-a",
                        "transcript_provider_id": "assemblyai",
                        "target_excerpt": "Hello world translated.",
                        "machine_preferred": True,
                    },
                    {
                        "candidate_id": "candidate-b",
                        "rank": 2,
                        "model_id": "gpt-5.4",
                        "prompt_variant_id": "prompt-b",
                        "prompt_version": "v1",
                        "source_transcript_candidate_id": "transcript-b",
                        "transcript_provider_id": "deepgram",
                        "target_excerpt": "Alternate translation.",
                        "machine_preferred": False,
                    },
                ],
                "current_draft_decision": {
                    "selected_base_variant_id": "candidate-a",
                    "edited_text": None,
                    "resolution_status": "unresolved",
                    "dirty": False,
                    "reviewer_note": "",
                },
            },
            {
                "source_span_id": "span:1000:2000",
                "start_ms": 1000,
                "end_ms": 2000,
                "time_range": "00:01.000-00:02.000",
                "severity_summary": "minor",
                "blocking": False,
                "reviewer_roles": [],
                "evidence_summary": [],
                "source_excerpt": "Second line.",
                "transcript_provenance_options": [
                    {
                        "source_transcript_candidate_id": "transcript-a",
                        "transcript_provider_id": "assemblyai",
                        "transcript_excerpt": "Second line.",
                    }
                ],
                "variants": [
                    {
                        "candidate_id": "candidate-a",
                        "rank": 1,
                        "model_id": "gpt-5.4",
                        "prompt_variant_id": "prompt-a",
                        "prompt_version": "v1",
                        "source_transcript_candidate_id": "transcript-a",
                        "transcript_provider_id": "assemblyai",
                        "target_excerpt": "Second translated line.",
                        "machine_preferred": True,
                    }
                ],
                "current_draft_decision": {
                    "selected_base_variant_id": "candidate-a",
                    "edited_text": None,
                    "resolution_status": "unresolved",
                    "dirty": False,
                    "reviewer_note": "",
                },
            },
        ],
        "draft_resolution": None,
    }


def _build_app(
    *,
    payload: dict[str, Any] | None = None,
    saved_drafts: list[dict[str, object]] | None = None,
    resolved_payloads: list[dict[str, object]] | None = None,
) -> ReviewTerminalApp:
    payload = deepcopy(payload or _payload())
    saved_drafts = saved_drafts if saved_drafts is not None else []
    resolved_payloads = resolved_payloads if resolved_payloads is not None else []

    def _save_review_draft(
        run_id: str,
        *,
        draft_resolution: dict[str, object],
        settings: object,
    ) -> dict[str, object]:
        saved_drafts.append(draft_resolution)
        return {"run_id": run_id, "draft_ref": "jobs/job-123/review/draft-resolution.json"}

    def _resolve_review(
        run_id: str,
        *,
        resolution: str,
        candidate_id: str | None,
        reviewed_span_decisions: tuple[dict[str, object], ...],
        failure_tags: tuple[str, ...],
        approved_by: str | None,
        note: str | None,
        settings: object,
    ) -> dict[str, object]:
        resolved_payloads.append(
            {
                "run_id": run_id,
                "resolution": resolution,
                "candidate_id": candidate_id,
                "reviewed_span_decisions": reviewed_span_decisions,
                "failure_tags": failure_tags,
                "approved_by": approved_by,
                "note": note,
            }
        )
        return {"run_id": run_id, "status": "completed_after_human_review"}

    return ReviewTerminalApp(
        run_id="run-123",
        payload=payload,
        settings=SimpleNamespace(),
        save_review_draft=_save_review_draft,
        resolve_review=_resolve_review,
    )


def test_app_boots_from_payload_and_highlights_first_unresolved_span() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            current_span = app._current_span()
            assert current_span is not None
            assert current_span["source_span_id"] == "span:0:1000"

    asyncio.run(_scenario())


def test_selecting_variant_updates_draft_state() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("2")
            assert app._drafts["span:0:1000"]["selected_base_variant_id"] == "candidate-b"

    asyncio.run(_scenario())


def test_editing_span_text_updates_preview_and_marks_span_dirty() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            editor = app.query_one("#editor", TextArea)
            editor.load_text("Edited translation.")
            await pilot.pause()
            assert app._drafts["span:0:1000"]["edited_text"] == "Edited translation."
            assert app._drafts["span:0:1000"]["dirty"] is True

    asyncio.run(_scenario())


def test_save_resume_round_trips_draft_state() -> None:
    async def _scenario() -> None:
        saved_drafts: list[dict[str, object]] = []
        app = _build_app(saved_drafts=saved_drafts)
        async with app.run_test() as pilot:
            await pilot.pause()
            editor = app.query_one("#editor", TextArea)
            editor.load_text("Saved draft text.")
            await pilot.pause()
            await pilot.press("ctrl+s")
            assert saved_drafts

        resumed_payload = _payload()
        resumed_payload["draft_resolution"] = saved_drafts[-1]
        resumed_app = _build_app(payload=resumed_payload)
        async with resumed_app.run_test() as pilot:
            await pilot.pause()
            assert resumed_app._drafts["span:0:1000"]["edited_text"] == "Saved draft text."

    asyncio.run(_scenario())


def test_finalize_is_blocked_until_every_blocking_span_is_resolved() -> None:
    async def _scenario() -> None:
        resolved_payloads: list[dict[str, object]] = []
        app = _build_app(resolved_payloads=resolved_payloads)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+enter")
            await pilot.pause()
            assert not resolved_payloads
            status = app.query_one("#status-line", Static).content
            assert "resolve every blocking span" in str(status)

    asyncio.run(_scenario())


def test_finalize_writes_expected_structured_resolution_payload() -> None:
    async def _scenario() -> None:
        resolved_payloads: list[dict[str, object]] = []
        app = _build_app(resolved_payloads=resolved_payloads)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_mark_resolved()
            await pilot.press("j")
            app.action_mark_resolved()
            app.action_finalize_review()
            await pilot.pause()
            assert resolved_payloads
            payload = resolved_payloads[-1]
            assert payload["resolution"] == "approved_good"
            decisions = payload["reviewed_span_decisions"]
            assert isinstance(decisions, tuple)
            assert len(decisions) == 2
            assert decisions[0]["source_span_id"] == "span:0:1000"
            assert decisions[0]["selected_candidate_id"] == "candidate-a"

    asyncio.run(_scenario())


def test_navigation_keys_and_jump_shortcuts_work() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("j")
            current_span = app._current_span()
            assert current_span is not None
            assert current_span["source_span_id"] == "span:1000:2000"
            await pilot.press("[")
            current_span = app._current_span()
            assert current_span is not None
            assert current_span["source_span_id"] == "span:0:1000"
            await pilot.press("enter")
            await pilot.press("]")
            current_span = app._current_span()
            assert current_span is not None
            assert current_span["source_span_id"] == "span:0:1000"

    asyncio.run(_scenario())
