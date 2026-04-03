"""Textual review UI for span-level translation synthesis."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)


class ResolveReviewCallable(Protocol):
    """Callable contract for finalizing a human review."""

    def __call__(
        self,
        run_id: str,
        *,
        resolution: str,
        candidate_id: str | None,
        reviewed_span_decisions: tuple[dict[str, object], ...],
        failure_tags: tuple[str, ...],
        approved_by: str | None,
        note: str | None,
        settings: Any,
    ) -> dict[str, object]: ...


class SaveReviewDraftCallable(Protocol):
    """Callable contract for persisting in-progress review state."""

    def __call__(
        self,
        run_id: str,
        *,
        draft_resolution: dict[str, object],
        settings: Any,
    ) -> dict[str, object]: ...


class ScoreboardScreen(ModalScreen[None]):
    """Modal surface for global provenance and candidate scoreboards."""

    BINDINGS = [Binding("escape", "dismiss_screen", show=False)]

    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        yield Static(self._content, id="scoreboard-content")

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)


class ReviewTerminalApp(App[None]):
    """Span-driven synthesis review UI backed by Textual."""

    REVIEW_MODES = ("approved_good", "approved_best_available", "rejected_all")

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #summary {
        height: 3;
        margin: 0 1;
        padding: 0 1 0 1;
        content-align: left middle;
        background: $panel;
        border: round $accent 30%;
    }

    #toolbar {
        height: 3;
        margin: 0 1 1 1;
    }

    #filter {
        width: 1fr;
    }

    #left-pane {
        width: 34;
        min-width: 30;
        border: solid $panel;
        background: $boost;
    }

    #workspace {
        height: 1fr;
        margin: 0 1 1 1;
    }

    #main-pane {
        width: 1fr;
        margin-left: 1;
    }

    #navigator {
        height: 1fr;
        margin: 0 1 1 1;
    }

    #main-tabs {
        height: 1fr;
    }

    #decision-grid {
        height: 1fr;
    }

    #variant-pane {
        width: 1fr;
        border: solid $panel;
        background: $surface;
    }

    #editor-pane {
        width: 44;
        min-width: 36;
        margin-left: 1;
        border: solid $panel;
        background: $boost;
    }

    .pane-title {
        padding: 0 1;
        height: 3;
        content-align: left middle;
        background: $panel-darken-1;
        text-style: bold;
    }

    #mode-select,
    #failure-tags,
    #note {
        margin: 0 1 1 1;
    }

    #mode-select {
        width: 30;
        margin-left: 1;
    }

    #span-meta,
    #variant-view,
    #evidence-view,
    #provenance,
    #scoreboard-content {
        padding: 1;
        overflow-y: auto;
    }

    #span-meta {
        height: auto;
        min-height: 10;
        margin: 1;
        background: $boost;
        border: round $panel-lighten-1;
    }

    #status-line,
    #editor-label {
        padding: 0 1;
        height: 3;
        content-align: left middle;
    }

    #variant-view {
        height: 1fr;
        margin: 0 1 1 1;
        background: $surface-darken-1;
        border: round $panel-lighten-1;
    }

    #editor {
        height: 14;
        margin: 0 1;
    }

    #evidence-view,
    #provenance {
        height: 1fr;
        border: solid $panel;
        background: $surface;
    }

    #status-line {
        height: 3;
        margin: 0 1 1 1;
        color: $text-muted;
        background: $surface-darken-1;
        border: round $panel-lighten-1;
    }

    TabbedContent Tabs {
        background: transparent;
    }

    TabbedContent ContentSwitcher {
        border: solid $panel;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("j", "next_span", "Next"),
        Binding("k", "previous_span", "Prev"),
        Binding("down", "next_span", show=False),
        Binding("up", "previous_span", show=False),
        Binding("1", "pick_variant('1')", "Pick 1"),
        Binding("2", "pick_variant('2')", "Pick 2"),
        Binding("3", "pick_variant('3')", "Pick 3"),
        Binding("4", "pick_variant('4')", "Pick 4"),
        Binding("5", "pick_variant('5')", "Pick 5"),
        Binding("6", "pick_variant('6')", "Pick 6"),
        Binding("7", "pick_variant('7')", "Pick 7"),
        Binding("8", "pick_variant('8')", "Pick 8"),
        Binding("9", "pick_variant('9')", "Pick 9"),
        Binding("e", "toggle_editor", "Edit"),
        Binding("enter", "mark_resolved", "Resolve"),
        Binding("u", "clear_current_span", "Clear"),
        Binding("[", "jump_unresolved", "Jump Unresolved"),
        Binding("]", "jump_blocking", "Jump Blocking"),
        Binding("p", "open_scoreboard", "Scoreboard"),
        Binding("m", "cycle_mode", "Mode"),
        Binding("ctrl+s", "save_draft", "Save"),
        Binding("ctrl+enter", "finalize_review", "Finalize"),
        Binding("escape", "escape_state", show=False),
    ]

    def __init__(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        settings: Any,
        save_review_draft: SaveReviewDraftCallable,
        resolve_review: ResolveReviewCallable,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.payload = payload
        self.settings = settings
        self.save_review_draft = save_review_draft
        self.resolve_review = resolve_review
        self._spans = [span for span in payload.get("review_spans", []) if isinstance(span, dict)]
        self._span_by_id = {
            str(span["source_span_id"]): span
            for span in self._spans
            if isinstance(span.get("source_span_id"), str)
        }
        self._filtered_span_ids: list[str] = []
        self._current_index = 0
        self._loading_editor = False
        draft_resolution = payload.get("draft_resolution") or {}
        draft_decisions = {
            str(decision["source_span_id"]): dict(decision)
            for decision in draft_resolution.get("span_decisions", [])
            if isinstance(decision, dict) and isinstance(decision.get("source_span_id"), str)
        }
        self._drafts: dict[str, dict[str, Any]] = {}
        for span in self._spans:
            span_id = str(span["source_span_id"])
            current = span.get("current_draft_decision", {})
            if not isinstance(current, dict):
                current = {}
            draft = {
                "selected_base_variant_id": current.get("selected_base_variant_id"),
                "edited_text": current.get("edited_text"),
                "resolution_status": current.get("resolution_status") or "unresolved",
                "dirty": bool(current.get("dirty")),
                "reviewer_note": current.get("reviewer_note") or "",
            }
            draft.update(draft_decisions.get(span_id, {}))
            self._drafts[span_id] = draft
        self._mode = str(draft_resolution.get("resolution_kind") or "approved_good")
        self.title = "Translation Review"
        self.sub_title = run_id
        self._last_status = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="summary")
        with Horizontal(id="toolbar"):
            yield Input(
                placeholder="Filter spans, severity, source text, candidate id",
                id="filter",
            )
            yield Select.from_values(
                self.REVIEW_MODES,
                allow_blank=False,
                value=self._mode,
                prompt="Resolution mode",
                id="mode-select",
            )
        with Horizontal(id="workspace"):
            with Vertical(id="left-pane"):
                yield Static("Span Navigator", classes="pane-title")
                yield OptionList(id="navigator")
            with Vertical(id="main-pane"):
                with TabbedContent(initial="decision-tab", id="main-tabs"):
                    with TabPane("Decision", id="decision-tab"):
                        with Horizontal(id="decision-grid"):
                            with Vertical(id="variant-pane"):
                                yield Static(id="span-meta")
                                yield Static(id="variant-view")
                            with Vertical(id="editor-pane"):
                                yield Static(
                                    "Synthesis Editor",
                                    id="editor-label",
                                    classes="pane-title",
                                )
                                yield TextArea("", id="editor")
                                yield Input(
                                    placeholder="Failure tags: tag-a, tag-b",
                                    id="failure-tags",
                                )
                                yield Input(placeholder="Final note", id="note")
                                yield Static(id="status-line")
                    with TabPane("Evidence", id="evidence-tab"):
                        yield Static(id="evidence-view")
                    with TabPane("Provenance", id="provenance-tab"):
                        yield Static(id="provenance")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "nord"
        self._refresh_span_filter()
        initial_span_id = self._first_unresolved_span_id()
        if initial_span_id is None and self._filtered_span_ids:
            initial_span_id = self._filtered_span_ids[0]
        if initial_span_id is not None and initial_span_id in self._filtered_span_ids:
            self._current_index = self._filtered_span_ids.index(initial_span_id)
        self._refresh_current_span()
        self.query_one("#navigator", OptionList).focus()
        self._set_status("ctrl+p opens the command palette for theme switches and other commands")

    def action_next_span(self) -> None:
        if not self._filtered_span_ids:
            return
        self._current_index = min(self._current_index + 1, len(self._filtered_span_ids) - 1)
        self._refresh_current_span()

    def action_previous_span(self) -> None:
        if not self._filtered_span_ids:
            return
        self._current_index = max(self._current_index - 1, 0)
        self._refresh_current_span()

    def action_pick_variant(self, index_text: str) -> None:
        span = self._current_span()
        if span is None:
            return
        try:
            variant_index = int(index_text) - 1
        except ValueError:
            return
        variants = self._variants_for_span(span)
        if variant_index < 0 or variant_index >= len(variants):
            self._set_status(f"variant {index_text} is unavailable", severity="warning")
            return
        selected = variants[variant_index]
        draft = self._drafts[str(span["source_span_id"])]
        draft["selected_base_variant_id"] = selected["candidate_id"]
        draft["edited_text"] = None
        draft["dirty"] = False
        self._load_editor_text(self._base_text_for_span(span))
        self._refresh_current_span()
        self._set_status(f"selected {selected['candidate_id']}")

    def action_toggle_editor(self) -> None:
        editor = self.query_one("#editor", TextArea)
        navigator = self.query_one("#navigator", OptionList)
        if editor.has_focus:
            navigator.focus()
            return
        editor.focus()

    def action_mark_resolved(self) -> None:
        span = self._current_span()
        if span is None:
            return
        draft = self._drafts[str(span["source_span_id"])]
        if not draft.get("selected_base_variant_id"):
            self._set_status("pick a variant before resolving the span", severity="warning")
            return
        draft["resolution_status"] = "resolved"
        self._refresh_span_filter()
        self._refresh_current_span()
        self._set_status(f"resolved {span['source_span_id']}")

    def action_clear_current_span(self) -> None:
        span = self._current_span()
        if span is None:
            return
        draft = self._drafts[str(span["source_span_id"])]
        draft.update(
            {
                "selected_base_variant_id": None,
                "edited_text": None,
                "resolution_status": "unresolved",
                "dirty": False,
                "reviewer_note": "",
            }
        )
        self._load_editor_text("")
        self._refresh_span_filter()
        self._refresh_current_span()
        self._set_status(f"cleared {span['source_span_id']}", severity="warning")

    def action_jump_unresolved(self) -> None:
        self._jump_to(lambda span_id: self._drafts[span_id]["resolution_status"] != "resolved")

    def action_jump_blocking(self) -> None:
        self._jump_to(
            lambda span_id: (
                bool(self._span_by_id[span_id].get("blocking"))
                and self._drafts[span_id]["resolution_status"] != "resolved"
            )
        )

    def action_open_scoreboard(self) -> None:
        candidates = [
            candidate
            for candidate in self.payload.get("candidates", [])
            if isinstance(candidate, dict)
        ]
        lines = ["Candidate Scoreboard"]
        for candidate in candidates:
            source = candidate.get("source_transcript", {})
            provider_id = source.get("provider_id") if isinstance(source, dict) else None
            lines.append(
                f"{candidate.get('rank')}. {candidate.get('candidate_id')} "
                f"provider={provider_id or 'unknown'} "
                f"prompt={candidate.get('prompt_variant_id')} "
                f"contradictions={candidate.get('contradiction_count', 0)} "
                f"blocking={candidate.get('blocking_hard_contradiction_count', 0)}"
            )
        self.push_screen(ScoreboardScreen("\n".join(lines)))

    def action_cycle_mode(self) -> None:
        current_index = (
            self.REVIEW_MODES.index(self._mode) if self._mode in self.REVIEW_MODES else 0
        )
        self._mode = self.REVIEW_MODES[(current_index + 1) % len(self.REVIEW_MODES)]
        self.query_one("#mode-select", Select[str]).value = self._mode
        self._refresh_summary()
        self._set_status(f"mode set to {self._mode}")

    def action_save_draft(self) -> None:
        payload = self._draft_payload()
        result = self.save_review_draft(
            self.run_id,
            draft_resolution=payload,
            settings=self.settings,
        )
        draft_ref = result.get("draft_ref")
        self._set_status(f"draft saved: {draft_ref}")

    def action_finalize_review(self) -> None:
        failure_tags = self._failure_tags()
        note = self.query_one("#note", Input).value.strip() or None
        if self._mode == "rejected_all":
            if not failure_tags:
                self._set_status(
                    "rejected_all requires at least one failure tag",
                    severity="warning",
                )
                return
            self.resolve_review(
                self.run_id,
                resolution=self._mode,
                candidate_id=None,
                reviewed_span_decisions=(),
                failure_tags=failure_tags,
                approved_by=None,
                note=note,
                settings=self.settings,
            )
            self._set_status("review finalized as rejected_all")
            self.exit()
            return
        unresolved_blocking = [
            span_id
            for span_id, span in self._span_by_id.items()
            if bool(span.get("blocking"))
            and self._drafts[span_id]["resolution_status"] != "resolved"
        ]
        if unresolved_blocking:
            self._set_status(
                "resolve every blocking span before finalizing",
                severity="warning",
            )
            return
        decisions: list[dict[str, object]] = []
        for span in self._spans:
            span_id = str(span["source_span_id"])
            draft = self._drafts[span_id]
            selected_id = draft.get("selected_base_variant_id") or self._default_variant_id(span)
            if not isinstance(selected_id, str) or not selected_id:
                self._set_status(
                    f"span {span_id} has no selected base variant",
                    severity="error",
                )
                return
            variant = self._variant_by_candidate_id(span, selected_id)
            if variant is None:
                self._set_status(
                    f"span {span_id} selected an unavailable variant",
                    severity="error",
                )
                return
            base_text = str(variant["target_excerpt"])
            final_text = (
                str(draft["edited_text"]).strip()
                if isinstance(draft.get("edited_text"), str) and draft["edited_text"].strip()
                else base_text
            )
            decisions.append(
                {
                    "source_span_id": span_id,
                    "start_ms": int(span["start_ms"]),
                    "end_ms": int(span["end_ms"]),
                    "selected_candidate_id": selected_id,
                    "selected_source_transcript_candidate_id": variant.get(
                        "source_transcript_candidate_id"
                    ),
                    "selected_transcript_provider_id": variant.get("transcript_provider_id"),
                    "base_target_text": base_text,
                    "final_target_text": final_text,
                    "edited": final_text != base_text,
                    "reviewer_note": str(draft.get("reviewer_note") or ""),
                }
            )
        self.resolve_review(
            self.run_id,
            resolution=self._mode,
            candidate_id=None,
            reviewed_span_decisions=tuple(decisions),
            failure_tags=failure_tags,
            approved_by=None,
            note=note,
            settings=self.settings,
        )
        self._set_status("review finalized")
        self.exit()

    def action_escape_state(self) -> None:
        if self.screen_stack and len(self.screen_stack) > 1:
            self.pop_screen()
            return
        navigator = self.query_one("#navigator", OptionList)
        navigator.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self._refresh_span_filter()
            self._refresh_current_span()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "mode-select" or not isinstance(event.value, str):
            return
        self._mode = event.value
        self._refresh_summary()
        self._set_status(f"mode set to {self._mode}")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is None:
            return
        if event.option_index < 0 or event.option_index >= len(self._filtered_span_ids):
            return
        self._current_index = event.option_index
        self._refresh_current_span()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._loading_editor or event.text_area.id != "editor":
            return
        span = self._current_span()
        if span is None:
            return
        span_id = str(span["source_span_id"])
        draft = self._drafts[span_id]
        base_text = self._base_text_for_span(span)
        current_text = event.text_area.text.rstrip()
        if current_text == base_text:
            draft["edited_text"] = None
            draft["dirty"] = False
        else:
            draft["edited_text"] = current_text
            draft["dirty"] = True
        self._refresh_span_filter()
        self._refresh_current_span()

    def _jump_to(self, predicate: Callable[[str], bool]) -> None:
        for index, span_id in enumerate(self._filtered_span_ids):
            if predicate(span_id):
                self._current_index = index
                self._refresh_current_span()
                return
        self._set_status("no matching span in the current filter", severity="warning")

    def _refresh_span_filter(self) -> None:
        filter_value = self.query_one("#filter", Input).value.strip().casefold()
        self._filtered_span_ids = [
            span_id
            for span_id in self._span_by_id
            if not filter_value or filter_value in self._span_filter_blob(self._span_by_id[span_id])
        ]
        navigator = self.query_one("#navigator", OptionList)
        navigator.clear_options()
        for span_id in self._filtered_span_ids:
            span = self._span_by_id[span_id]
            navigator.add_option(self._navigator_label(span))
        if self._filtered_span_ids:
            self._current_index = max(0, min(self._current_index, len(self._filtered_span_ids) - 1))
            navigator.highlighted = self._current_index
        self._refresh_summary()

    def _refresh_current_span(self) -> None:
        navigator = self.query_one("#navigator", OptionList)
        if not self._filtered_span_ids:
            self.query_one("#span-meta", Static).update("No spans match the current filter.")
            self.query_one("#variant-view", Static).update("")
            self.query_one("#evidence-view", Static).update("")
            self.query_one("#provenance", Static).update("")
            self._load_editor_text("")
            return
        self._current_index = max(0, min(self._current_index, len(self._filtered_span_ids) - 1))
        navigator.highlighted = self._current_index
        span = self._current_span()
        if span is None:
            return
        self.sub_title = f"{self.run_id}  {span.get('time_range') or span.get('source_span_id')}"
        self.query_one("#span-meta", Static).update(self._span_meta_text(span))
        self.query_one("#variant-view", Static).update(self._variant_view_text(span))
        self.query_one("#evidence-view", Static).update(self._evidence_text(span))
        self.query_one("#provenance", Static).update(self._provenance_text(span))
        current_text = self._current_text_for_span(span)
        self._load_editor_text(current_text)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        blocking_count = sum(1 for span in self._spans if bool(span.get("blocking")))
        unresolved_count = sum(
            1
            for span_id, draft in self._drafts.items()
            if draft.get("resolution_status") != "resolved"
        )
        summary = (
            f"run={self.run_id}  status={self.payload.get('status')}  "
            f"unresolved={unresolved_count}  blocking={blocking_count}  mode={self._mode}  "
            "theme=nord"
        )
        self.query_one("#summary", Static).update(summary)

    def _set_status(
        self,
        message: str,
        *,
        severity: Literal["information", "warning", "error"] = "information",
    ) -> None:
        self._last_status = message
        self.query_one("#status-line", Static).update(message)
        self.notify(message, severity=severity, markup=False)

    def _current_span(self) -> dict[str, Any] | None:
        if not self._filtered_span_ids:
            return None
        return self._span_by_id[self._filtered_span_ids[self._current_index]]

    def _variants_for_span(self, span: dict[str, Any]) -> list[dict[str, Any]]:
        return [variant for variant in span.get("variants", []) if isinstance(variant, dict)]

    def _variant_by_candidate_id(
        self, span: dict[str, Any], candidate_id: str
    ) -> dict[str, Any] | None:
        for variant in self._variants_for_span(span):
            if variant.get("candidate_id") == candidate_id:
                return variant
        return None

    def _default_variant_id(self, span: dict[str, Any]) -> str | None:
        variants = self._variants_for_span(span)
        preferred = next(
            (variant for variant in variants if variant.get("machine_preferred")),
            None,
        )
        if preferred is not None:
            return str(preferred["candidate_id"])
        if not variants:
            return None
        return str(variants[0]["candidate_id"])

    def _base_text_for_span(self, span: dict[str, Any]) -> str:
        span_id = str(span["source_span_id"])
        selected_id = self._drafts[span_id].get("selected_base_variant_id")
        if not isinstance(selected_id, str) or not selected_id:
            selected_id = self._default_variant_id(span)
        if not isinstance(selected_id, str):
            return ""
        variant = self._variant_by_candidate_id(span, selected_id)
        if variant is None:
            return ""
        return str(variant.get("target_excerpt") or "")

    def _current_text_for_span(self, span: dict[str, Any]) -> str:
        span_id = str(span["source_span_id"])
        draft = self._drafts[span_id]
        if isinstance(draft.get("edited_text"), str) and draft["edited_text"] is not None:
            return str(draft["edited_text"])
        return self._base_text_for_span(span)

    def _load_editor_text(self, text: str) -> None:
        editor = self.query_one("#editor", TextArea)
        if editor.text == text:
            return
        self._loading_editor = True
        editor.load_text(text)
        self._loading_editor = False

    def _navigator_label(self, span: dict[str, Any]) -> str:
        span_id = str(span["source_span_id"])
        draft = self._drafts[span_id]
        resolved = "R" if draft.get("resolution_status") == "resolved" else "U"
        dirty = "*" if draft.get("dirty") else "-"
        blocking = "!" if span.get("blocking") else " "
        return (
            f"{blocking} {resolved}{dirty} {span.get('time_range')} {span.get('severity_summary')}"
        )

    def _span_meta_text(self, span: dict[str, Any]) -> str:
        evidence = [item for item in span.get("evidence_summary", []) if isinstance(item, dict)]
        draft = self._drafts[str(span["source_span_id"])]
        selected_id = draft.get("selected_base_variant_id") or self._default_variant_id(span)
        reviewers = ", ".join(str(role) for role in span.get("reviewer_roles", []))
        return "\n".join(
            [
                f"Span  {span.get('source_span_id')}",
                f"Window  {span.get('time_range')}",
                (
                    f"Severity  {span.get('severity_summary')}    "
                    f"Blocking  {bool(span.get('blocking'))}"
                ),
                f"Selected base  {selected_id or 'none'}",
                f"State  {draft.get('resolution_status')}    Dirty  {bool(draft.get('dirty'))}",
                f"Reviewers  {reviewers or 'none'}",
                "",
                "Source excerpt",
                str(span.get("source_excerpt") or "[no source excerpt]"),
                "",
                f"Evidence items  {len(evidence)}",
            ]
        )

    def _variant_view_text(self, span: dict[str, Any]) -> str:
        span_id = str(span["source_span_id"])
        selected_id = self._drafts[span_id].get("selected_base_variant_id")
        blocks = []
        for index, variant in enumerate(self._variants_for_span(span), start=1):
            marker = "*" if variant.get("candidate_id") == selected_id else " "
            preferred = " preferred" if variant.get("machine_preferred") else ""
            blocks.append(
                "\n".join(
                    [
                        (
                            f"{marker}[{index}] {variant.get('candidate_id')} "
                            f"rank={variant.get('rank')}{preferred}"
                        ),
                        (
                            f"    model={variant.get('model_id')} "
                            f"prompt={variant.get('prompt_variant_id')}"
                        ),
                        (
                            "    transcript="
                            f"{variant.get('source_transcript_candidate_id')} "
                            f"provider={variant.get('transcript_provider_id')}"
                        ),
                        "",
                        f"    {variant.get('target_excerpt')}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _evidence_text(self, span: dict[str, Any]) -> str:
        evidence = [item for item in span.get("evidence_summary", []) if isinstance(item, dict)]
        if not evidence:
            return "No evidence items for this span."
        lines = ["Evidence summary"]
        for item in evidence:
            lines.extend(
                [
                    "",
                    (
                        f"- candidate={item.get('candidate_id')} "
                        f"dimension={item.get('dimension')} "
                        f"severity={item.get('severity')}"
                    ),
                    f"  normalized_value={item.get('normalized_value')}",
                    f"  reviewer_role={item.get('reviewer_role')}",
                    f"  {item.get('evidence_text')}",
                ]
            )
        return "\n".join(lines)

    def _provenance_text(self, span: dict[str, Any]) -> str:
        options = [
            option
            for option in span.get("transcript_provenance_options", [])
            if isinstance(option, dict)
        ]
        lines = ["Provenance"]
        for option in options:
            lines.append(
                f"- transcript={option.get('source_transcript_candidate_id')} "
                f"provider={option.get('transcript_provider_id')}"
            )
            lines.append("")
            lines.append(str(option.get("transcript_excerpt") or "[no transcript excerpt]"))
            lines.append("")
        return "\n".join(lines)

    def _span_filter_blob(self, span: dict[str, Any]) -> str:
        parts = [
            str(span.get("source_span_id", "")),
            str(span.get("time_range", "")),
            str(span.get("severity_summary", "")),
            str(span.get("source_excerpt", "")),
        ]
        for item in span.get("evidence_summary", []):
            if isinstance(item, dict):
                parts.append(str(item.get("dimension", "")))
                parts.append(str(item.get("evidence_text", "")))
        for variant in span.get("variants", []):
            if isinstance(variant, dict):
                parts.append(str(variant.get("candidate_id", "")))
                parts.append(str(variant.get("target_excerpt", "")))
        return " ".join(parts).casefold()

    def _first_unresolved_span_id(self) -> str | None:
        for span in self._spans:
            span_id = str(span["source_span_id"])
            if self._drafts[span_id]["resolution_status"] != "resolved":
                return span_id
        return None

    def _draft_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "job_id": self.payload["job_id"],
            "resolution_kind": self._mode,
            "failure_tags": list(self._failure_tags()),
            "note": self.query_one("#note", Input).value,
            "approved_by": None,
            "span_decisions": [
                {
                    "source_span_id": span_id,
                    "selected_base_variant_id": draft.get("selected_base_variant_id"),
                    "edited_text": draft.get("edited_text"),
                    "resolution_status": draft.get("resolution_status"),
                    "dirty": bool(draft.get("dirty")),
                    "reviewer_note": draft.get("reviewer_note") or "",
                }
                for span_id, draft in self._drafts.items()
            ],
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def _failure_tags(self) -> tuple[str, ...]:
        raw = self.query_one("#failure-tags", Input).value
        return tuple(part.strip() for part in raw.split(",") if part.strip())
