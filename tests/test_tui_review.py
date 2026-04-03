from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
from textual.widgets import Static, TextArea

from translation_agent.tui.review import ReviewTerminalApp

pytestmark = pytest.mark.unit


def _base_span(
    *,
    source_span_id: str,
    start_ms: int,
    end_ms: int,
    source_excerpt: str,
    severity: str,
    blocking: bool,
    variants: list[dict[str, Any]],
    evidence_summary: list[dict[str, Any]],
    transcript_provenance_options: list[dict[str, Any]],
) -> dict[str, Any]:
    recommended_variant_id = next(
        (
            str(variant["candidate_id"])
            for variant in variants
            if bool(variant.get("machine_preferred"))
        ),
        str(variants[0]["candidate_id"]) if variants else None,
    )
    return {
        "source_span_id": source_span_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "time_range": f"00:0{start_ms // 1000}.000-00:0{end_ms // 1000}.000",
        "severity_summary": severity,
        "severity": severity,
        "blocking": blocking,
        "source_excerpt": source_excerpt,
        "evidence_summary": evidence_summary,
        "recommended_variant_id": recommended_variant_id,
        "transcript_provenance_options": transcript_provenance_options,
        "variants": variants,
        "current_draft_decision": {
            "selected_base_variant_id": recommended_variant_id,
            "selected_variant_id": recommended_variant_id,
            "recommended_variant_id": recommended_variant_id,
            "edited_text": None,
            "acknowledged": False,
            "resolution_status": "unresolved",
            "dirty": False,
            "reviewer_note": "",
        },
    }


def _payload() -> dict[str, Any]:
    flagged_blocking = _base_span(
        source_span_id="span:0:1000",
        start_ms=0,
        end_ms=1000,
        source_excerpt="Hello world.",
        severity="critical",
        blocking=True,
        transcript_provenance_options=[
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
        evidence_summary=[
            {
                "candidate_id": "candidate-b",
                "dimension": "meaning",
                "severity": "critical",
                "normalized_value": "conflict",
                "evidence_text": "Meaning conflict detected.",
                "reviewer_role": "faithfulness_reviewer",
            }
        ],
        variants=[
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
    )
    flagged_warning = _base_span(
        source_span_id="span:1000:2000",
        start_ms=1000,
        end_ms=2000,
        source_excerpt="Second line.",
        severity="major",
        blocking=False,
        transcript_provenance_options=[
            {
                "source_transcript_candidate_id": "transcript-a",
                "transcript_provider_id": "assemblyai",
                "transcript_excerpt": "Second line.",
            }
        ],
        evidence_summary=[
            {
                "candidate_id": "candidate-a",
                "dimension": "terminology",
                "severity": "major",
                "normalized_value": "workflow",
                "evidence_text": "Terminology disagreement remains.",
                "reviewer_role": "terminology_reviewer",
            }
        ],
        variants=[
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
            },
            {
                "candidate_id": "candidate-b",
                "rank": 2,
                "model_id": "gpt-5.4",
                "prompt_variant_id": "prompt-b",
                "prompt_version": "v1",
                "source_transcript_candidate_id": "transcript-a",
                "transcript_provider_id": "assemblyai",
                "target_excerpt": "Alternate second line.",
                "machine_preferred": False,
            },
        ],
    )
    auto_accepted = _base_span(
        source_span_id="span:2000:3000",
        start_ms=2000,
        end_ms=3000,
        source_excerpt="Third line.",
        severity="none",
        blocking=False,
        transcript_provenance_options=[
            {
                "source_transcript_candidate_id": "transcript-a",
                "transcript_provider_id": "assemblyai",
                "transcript_excerpt": "Third line.",
            }
        ],
        evidence_summary=[],
        variants=[
            {
                "candidate_id": "candidate-a",
                "rank": 1,
                "model_id": "gpt-5.4",
                "prompt_variant_id": "prompt-a",
                "prompt_version": "v1",
                "source_transcript_candidate_id": "transcript-a",
                "transcript_provider_id": "assemblyai",
                "target_excerpt": "Third translated line.",
                "machine_preferred": True,
            }
        ],
    )
    return {
        "run_id": "run-123",
        "job_id": "job-123",
        "status": "review_required",
        "review_mode": "exception_only",
        "auto_accepted_span_count": 1,
        "blocking_span_count": 1,
        "warning_span_count": 1,
        "recommended_candidate_id": "candidate-a",
        "review_spans": [flagged_blocking, flagged_warning, auto_accepted],
        "flagged_spans": [
            {
                "source_span_id": flagged_blocking["source_span_id"],
                "time_range": flagged_blocking["time_range"],
                "source_excerpt": flagged_blocking["source_excerpt"],
                "blocking": flagged_blocking["blocking"],
                "severity": flagged_blocking["severity"],
                "issue_summary": (
                    "critical meaning disagreement; blocking review item raised by 1 reviewer(s)."
                ),
                "recommended_variant_id": flagged_blocking["recommended_variant_id"],
                "variants": flagged_blocking["variants"],
                "selected_variant_id": flagged_blocking["current_draft_decision"][
                    "selected_variant_id"
                ],
                "edited_text": None,
                "acknowledged": False,
                "evidence_summary": flagged_blocking["evidence_summary"],
                "transcript_provenance_options": flagged_blocking["transcript_provenance_options"],
            },
            {
                "source_span_id": flagged_warning["source_span_id"],
                "time_range": flagged_warning["time_range"],
                "source_excerpt": flagged_warning["source_excerpt"],
                "blocking": flagged_warning["blocking"],
                "severity": flagged_warning["severity"],
                "issue_summary": (
                    "major terminology disagreement; warning review item raised by 1 reviewer(s)."
                ),
                "recommended_variant_id": flagged_warning["recommended_variant_id"],
                "variants": flagged_warning["variants"],
                "selected_variant_id": flagged_warning["current_draft_decision"][
                    "selected_variant_id"
                ],
                "edited_text": None,
                "acknowledged": False,
                "evidence_summary": flagged_warning["evidence_summary"],
                "transcript_provenance_options": flagged_warning["transcript_provenance_options"],
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


def test_app_boots_on_first_flagged_span() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            current_span = app._current_flagged_span()
            assert current_span is not None
            assert current_span["source_span_id"] == "span:0:1000"

    asyncio.run(_scenario())


def test_candidate_selection_updates_draft_and_autosaves() -> None:
    async def _scenario() -> None:
        saved_drafts: list[dict[str, object]] = []
        app = _build_app(saved_drafts=saved_drafts)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
            assert app._drafts["span:0:1000"]["selected_variant_id"] == "candidate-b"
            assert saved_drafts
            latest_draft = cast(dict[str, object], saved_drafts[-1])
            span_decisions = cast(
                list[dict[str, object]],
                latest_draft["span_decisions"],
            )
            assert span_decisions[0]["selected_variant_id"] == "candidate-b"

    asyncio.run(_scenario())


def test_edit_drawer_stays_hidden_until_invoked_and_edit_autosaves() -> None:
    async def _scenario() -> None:
        saved_drafts: list[dict[str, object]] = []
        app = _build_app(saved_drafts=saved_drafts)
        async with app.run_test() as pilot:
            await pilot.pause()
            drawer = app.query_one("#edit-drawer")
            assert "hidden" in drawer.classes
            app.action_toggle_editor()
            await pilot.pause()
            assert "hidden" not in drawer.classes
            editor = app.query_one("#editor", TextArea)
            editor.load_text("Edited translation.")
            app.on_text_area_changed(TextArea.Changed(editor))
            await pilot.pause()
            assert app._drafts["span:0:1000"]["edited_text"] == "Edited translation."
            assert app._drafts["span:0:1000"]["dirty"] is True
            assert saved_drafts

    asyncio.run(_scenario())


def test_publish_is_blocked_until_blocking_span_is_acknowledged() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_publish_review()
            await pilot.pause()
            assert len(app.screen_stack) == 1
            status = app.query_one("#status-line", Static).content
            assert "Confirm every blocking span before publishing." in str(status)

    asyncio.run(_scenario())


def test_publish_allows_warning_span_to_remain_unacknowledged() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_confirm_span()
            await pilot.pause()
            assert app._drafts["span:0:1000"]["acknowledged"] is True
            app.action_publish_review()
            await pilot.pause()
            assert len(app.screen_stack) == 2

    asyncio.run(_scenario())


def test_reject_opens_structured_reason_flow() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_reject_review()
            await pilot.pause()
            assert len(app.screen_stack) == 2

    asyncio.run(_scenario())


def test_removed_primary_controls_are_absent() -> None:
    async def _scenario() -> None:
        app = _build_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert len(app.query("#mode-select")) == 0
            assert len(app.query("#main-tabs")) == 0
            assert len(app.query("#failure-tags")) == 0

    asyncio.run(_scenario())


def test_publish_builds_decisions_for_flagged_and_auto_accepted_spans() -> None:
    async def _scenario() -> None:
        resolved_payloads: list[dict[str, object]] = []
        app = _build_app(resolved_payloads=resolved_payloads)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_confirm_span()
            await pilot.pause()
            app._handle_resolution_result(("publish", ()))
            await pilot.pause()
            assert resolved_payloads
            payload = resolved_payloads[-1]
            assert payload["resolution"] == "approved_good"
            decisions = payload["reviewed_span_decisions"]
            assert isinstance(decisions, tuple)
            assert len(decisions) == 3
            assert decisions[-1]["source_span_id"] == "span:2000:3000"

    asyncio.run(_scenario())
