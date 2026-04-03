"""Exception-only Textual review UI for flagged translation spans."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, OptionList, Static, TextArea

_FAILURE_REASON_OPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "subtitle_gibberish",
        "Unreadable output",
        "Rendered text is broken or unusable.",
    ),
    (
        "ungrounded_addition",
        "Adds unsupported meaning",
        "The translation says more than the source supports.",
    ),
    (
        "literal_but_wrong_semantics",
        "Literal but wrong",
        "Words are close, but the meaning is still wrong.",
    ),
    (
        "romanization_leak",
        "Romanization leak",
        "Raw transliteration leaked into the target language.",
    ),
    (
        "honorific_leak",
        "Honorific leak",
        "Source honorifics leaked into the target subtitle.",
    ),
    (
        "late_run_degeneration",
        "Late-run degeneration",
        "Output quality collapsed near the end of the run.",
    ),
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


class ResolutionReasonScreen(ModalScreen[tuple[str, tuple[str, ...]] | None]):
    """Compact structured reason picker for publish and reject actions."""

    CSS = """
    ResolutionReasonScreen {
        align: center middle;
        background: $background 65%;
    }

    #resolution-modal {
        width: 72;
        max-width: 90%;
        height: auto;
        padding: 1;
        border: solid $border;
        background: $surface;
    }

    #resolution-title {
        height: auto;
        padding: 0 0 1 0;
        text-style: bold;
    }

    .reason-checkbox {
        margin: 0 0 1 0;
    }

    #resolution-status {
        height: auto;
        color: $text-muted;
        padding: 1 0 0 0;
    }

    #resolution-actions {
        height: auto;
        padding: 1 0 0 0;
    }

    Button {
        margin-right: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", show=False)]

    def __init__(self, *, resolution: Literal["publish", "reject"]) -> None:
        super().__init__()
        self._resolution = resolution

    def compose(self) -> ComposeResult:
        title = "Publish review" if self._resolution == "publish" else "Reject review"
        body = (
            "Optional residual issues become soft feedback on publish."
            if self._resolution == "publish"
            else "Pick at least one structured reason before rejecting."
        )
        with Vertical(id="resolution-modal"):
            yield Static(title, id="resolution-title")
            yield Static(body)
            for tag, label, description in _FAILURE_REASON_OPTIONS:
                yield Checkbox(
                    f"{label}: {description}",
                    id=f"reason-{tag}",
                    classes="reason-checkbox",
                )
            with Horizontal(id="resolution-actions"):
                yield Button(
                    "Publish" if self._resolution == "publish" else "Reject",
                    id="confirm-resolution",
                    variant="primary",
                )
                yield Button("Cancel", id="cancel-resolution")
            yield Static("", id="resolution-status")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-resolution":
            self.dismiss(None)
            return
        selected = tuple(
            tag
            for tag, _, _ in _FAILURE_REASON_OPTIONS
            if self.query_one(f"#reason-{tag}", Checkbox).value
        )
        if self._resolution == "reject" and not selected:
            self.query_one("#resolution-status", Static).update(
                "Reject requires at least one reason."
            )
            return
        self.dismiss((self._resolution, selected))


class ReviewTerminalApp(App[None]):
    """Flagged-span compare workspace backed by Textual."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        color: $text;
    }

    #summary {
        height: auto;
        margin: 0 1 1 1;
        padding: 0 1;
        border: solid $border;
        background: $surface;
    }

    #workspace {
        height: 1fr;
        margin: 0 1;
    }

    #left-rail {
        width: 34;
        min-width: 30;
        border: solid $border;
        background: $surface;
    }

    #rail-title,
    #source-title,
    #edit-title {
        height: auto;
        padding: 0 1;
        text-style: bold;
        background: $panel;
        border-bottom: solid $border;
    }

    #navigator {
        height: 1fr;
        padding: 0 0 1 0;
    }

    #compare-pane {
        width: 1fr;
        margin-left: 1;
        border: solid $border;
        background: $surface;
    }

    #source-view {
        height: auto;
        min-height: 6;
        padding: 1;
        border-bottom: solid $border;
        background: $panel;
    }

    #issue-line {
        height: auto;
        padding: 0 1 1 1;
        color: $text-muted;
        border-bottom: solid $border;
    }

    #variant-grid {
        height: 1fr;
        overflow-x: auto;
        overflow-y: hidden;
        padding: 1;
    }

    .variant-card {
        width: 32;
        min-width: 32;
        height: 1fr;
        margin-right: 1;
        padding: 1;
        border: solid $border;
        background: $background;
    }

    .variant-selected {
        border: solid $accent;
    }

    .variant-recommended {
        background: $boost;
    }

    .hidden {
        display: none;
    }

    #edit-drawer {
        height: 16;
        margin: 1 1 0 1;
        border: solid $border;
        background: $surface;
    }

    #editor {
        height: 1fr;
        margin: 1;
    }

    #action-bar {
        height: auto;
        margin: 1;
        padding: 0 1;
        border: solid $border;
        background: $surface;
    }

    #action-bar Button {
        margin-right: 1;
    }

    #status-line {
        height: auto;
        margin: 0 1 1 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("up", "previous_span", show=False),
        Binding("down", "next_span", show=False),
        Binding("left", "previous_span", "Previous"),
        Binding("right", "next_span", "Next"),
        Binding("1", "pick_variant('1')", show=False),
        Binding("2", "pick_variant('2')", show=False),
        Binding("3", "pick_variant('3')", show=False),
        Binding("4", "pick_variant('4')", show=False),
        Binding("5", "pick_variant('5')", show=False),
        Binding("6", "pick_variant('6')", show=False),
        Binding("7", "pick_variant('7')", show=False),
        Binding("8", "pick_variant('8')", show=False),
        Binding("9", "pick_variant('9')", show=False),
        Binding("e", "toggle_editor", "Edit"),
        Binding("enter", "confirm_span", "Confirm"),
        Binding("p", "publish_review", "Publish"),
        Binding("r", "reject_review", "Reject"),
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
        self._all_spans = [
            span for span in payload.get("review_spans", []) if isinstance(span, dict)
        ]
        fallback_flagged = [
            span
            for span in self._all_spans
            if bool(span.get("blocking")) or span.get("evidence_summary")
        ]
        self._flagged_spans = [
            span
            for span in payload.get("flagged_spans", fallback_flagged)
            if isinstance(span, dict)
        ]
        self._span_by_id = {
            str(span["source_span_id"]): span
            for span in self._all_spans
            if isinstance(span.get("source_span_id"), str)
        }
        self._flagged_by_id = {
            str(span["source_span_id"]): span
            for span in self._flagged_spans
            if isinstance(span.get("source_span_id"), str)
        }
        self._flagged_span_ids = [
            str(span["source_span_id"])
            for span in self._flagged_spans
            if isinstance(span.get("source_span_id"), str)
        ]
        draft_resolution = payload.get("draft_resolution") or {}
        draft_decisions = {
            str(decision["source_span_id"]): dict(decision)
            for decision in draft_resolution.get("span_decisions", [])
            if isinstance(decision, dict) and isinstance(decision.get("source_span_id"), str)
        }
        self._drafts: dict[str, dict[str, Any]] = {}
        for span in self._all_spans:
            span_id = str(span["source_span_id"])
            current = span.get("current_draft_decision", {})
            if not isinstance(current, dict):
                current = {}
            selected_variant_id = (
                current.get("selected_variant_id")
                or current.get("selected_base_variant_id")
                or span.get("recommended_variant_id")
                or self._default_variant_id(span)
            )
            acknowledged = bool(current.get("acknowledged"))
            if not acknowledged and current.get("resolution_status") == "resolved":
                acknowledged = True
            draft = {
                "selected_variant_id": selected_variant_id,
                "edited_text": current.get("edited_text"),
                "acknowledged": acknowledged,
                "dirty": bool(current.get("dirty")),
                "reviewer_note": current.get("reviewer_note") or "",
            }
            persisted = draft_decisions.get(span_id, {})
            if persisted:
                draft["selected_variant_id"] = (
                    persisted.get("selected_variant_id")
                    or persisted.get("selected_base_variant_id")
                    or draft["selected_variant_id"]
                )
                draft["edited_text"] = persisted.get("edited_text")
                draft["acknowledged"] = bool(
                    persisted.get("acknowledged")
                    or persisted.get("resolution_status") == "resolved"
                )
                draft["dirty"] = bool(persisted.get("dirty"))
                draft["reviewer_note"] = persisted.get("reviewer_note") or ""
            self._drafts[span_id] = draft
        self._current_index = 0
        self._loading_editor = False
        self._editor_open = False
        self.title = "Review Compare"
        self.sub_title = run_id

    def compose(self) -> ComposeResult:
        yield Static(id="summary")
        with Horizontal(id="workspace"):
            with Vertical(id="left-rail"):
                yield Static("Flagged spans", id="rail-title")
                yield OptionList(id="navigator")
            with Vertical(id="compare-pane"):
                yield Static("Source", id="source-title")
                yield Static(id="source-view")
                yield Static(id="issue-line")
                with Horizontal(id="variant-grid"):
                    for index in range(1, 10):
                        yield Static("", id=f"variant-card-{index}", classes="variant-card hidden")
        with Vertical(id="edit-drawer", classes="hidden"):
            yield Static("Edit span", id="edit-title")
            yield TextArea("", id="editor")
        with Horizontal(id="action-bar"):
            yield Button("Publish", id="action-publish", variant="primary")
            yield Button("Reject", id="action-reject")
            yield Button("Edit span", id="action-edit")
            yield Button("Next", id="action-next")
            yield Button("Previous", id="action-previous")
        yield Static("", id="status-line")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_navigator()
        initial_span_id = self._first_pending_flagged_span_id()
        if initial_span_id is not None and initial_span_id in self._flagged_span_ids:
            self._current_index = self._flagged_span_ids.index(initial_span_id)
        self._refresh_current_span()
        if self._flagged_span_ids:
            self.query_one("#navigator", OptionList).focus()

    def action_next_span(self) -> None:
        if not self._flagged_span_ids:
            return
        self._current_index = min(self._current_index + 1, len(self._flagged_span_ids) - 1)
        self._refresh_current_span()

    def action_previous_span(self) -> None:
        if not self._flagged_span_ids:
            return
        self._current_index = max(self._current_index - 1, 0)
        self._refresh_current_span()

    def action_pick_variant(self, index_text: str) -> None:
        span = self._current_flagged_span()
        if span is None:
            return
        try:
            variant_index = int(index_text) - 1
        except ValueError:
            return
        variants = self._sorted_variants(span)
        if variant_index < 0 or variant_index >= len(variants):
            self._set_status(f"Candidate {index_text} is unavailable.")
            return
        draft = self._drafts[str(span["source_span_id"])]
        draft["selected_variant_id"] = variants[variant_index]["candidate_id"]
        draft["edited_text"] = None
        draft["dirty"] = False
        draft["acknowledged"] = False
        self._load_editor_text(self._current_text_for_span(span))
        self._persist_draft()
        self._refresh_navigator()
        self._refresh_current_span()
        self._set_status(f"Selected {variants[variant_index]['candidate_id']}.")

    def action_toggle_editor(self) -> None:
        drawer = self.query_one("#edit-drawer", Vertical)
        editor = self.query_one("#editor", TextArea)
        navigator = self.query_one("#navigator", OptionList)
        self._editor_open = not self._editor_open
        drawer.set_class(not self._editor_open, "hidden")
        if self._editor_open:
            span = self._current_flagged_span()
            self._load_editor_text(self._current_text_for_span(span) if span is not None else "")
            editor.focus()
        else:
            navigator.focus()

    def action_confirm_span(self) -> None:
        span = self._current_flagged_span()
        if span is None:
            return
        span_id = str(span["source_span_id"])
        if not self._drafts[span_id].get("selected_variant_id"):
            self._set_status("Pick a candidate before confirming the span.")
            return
        self._drafts[span_id]["acknowledged"] = True
        self._persist_draft()
        self._refresh_navigator()
        self._refresh_current_span()
        self._jump_to_next_pending_flagged_span()
        self._set_status(f"Confirmed {span_id}.")

    def action_publish_review(self) -> None:
        unresolved_blocking = [
            span_id
            for span_id in self._flagged_span_ids
            if bool(self._flagged_by_id[span_id].get("blocking"))
            and not bool(self._drafts[span_id].get("acknowledged"))
        ]
        if unresolved_blocking:
            self._set_status("Confirm every blocking span before publishing.")
            return
        self.push_screen(
            ResolutionReasonScreen(resolution="publish"),
            self._handle_resolution_result,
        )

    def action_reject_review(self) -> None:
        self.push_screen(
            ResolutionReasonScreen(resolution="reject"),
            self._handle_resolution_result,
        )

    def action_escape_state(self) -> None:
        if self.screen_stack and len(self.screen_stack) > 1:
            self.pop_screen()
            return
        if self._editor_open:
            self.action_toggle_editor()
            return
        self.query_one("#navigator", OptionList).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "action-publish":
            self.action_publish_review()
        elif button_id == "action-reject":
            self.action_reject_review()
        elif button_id == "action-edit":
            self.action_toggle_editor()
        elif button_id == "action-next":
            self.action_next_span()
        elif button_id == "action-previous":
            self.action_previous_span()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is None:
            return
        if event.option_index < 0 or event.option_index >= len(self._flagged_span_ids):
            return
        self._current_index = event.option_index
        self._refresh_current_span()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._loading_editor or event.text_area.id != "editor":
            return
        span = self._current_flagged_span()
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
        draft["acknowledged"] = False
        self._persist_draft()
        self._refresh_navigator()
        self._refresh_current_span()

    def _handle_resolution_result(
        self,
        result: tuple[str, tuple[str, ...]] | None,
    ) -> None:
        if result is None:
            return
        resolution, failure_tags = result
        if resolution == "publish":
            self._finalize_publish(failure_tags)
            return
        self._finalize_reject(failure_tags)

    def _finalize_publish(self, failure_tags: tuple[str, ...]) -> None:
        decisions = self._final_decisions()
        if decisions is None:
            return
        resolution_kind = "approved_best_available" if failure_tags else "approved_good"
        self.resolve_review(
            self.run_id,
            resolution=resolution_kind,
            candidate_id=None,
            reviewed_span_decisions=decisions,
            failure_tags=failure_tags,
            approved_by=None,
            note=None,
            settings=self.settings,
        )
        self._set_status("Review published.")
        self.exit()

    def _finalize_reject(self, failure_tags: tuple[str, ...]) -> None:
        self.resolve_review(
            self.run_id,
            resolution="rejected_all",
            candidate_id=None,
            reviewed_span_decisions=(),
            failure_tags=failure_tags,
            approved_by=None,
            note=None,
            settings=self.settings,
        )
        self._set_status("Review rejected.")
        self.exit()

    def _refresh_navigator(self) -> None:
        try:
            navigator = self.query_one("#navigator", OptionList)
        except NoMatches:
            return
        navigator.clear_options()
        for span_id in self._flagged_span_ids:
            navigator.add_option(self._navigator_label(self._flagged_by_id[span_id]))
        if self._flagged_span_ids:
            navigator.highlighted = max(
                0,
                min(self._current_index, len(self._flagged_span_ids) - 1),
            )
        self._refresh_summary()

    def _refresh_current_span(self) -> None:
        try:
            navigator = self.query_one("#navigator", OptionList)
        except NoMatches:
            return
        if not self._flagged_span_ids:
            self.query_one("#source-view", Static).update(
                "No flagged spans. Machine defaults are ready to publish."
            )
            self.query_one("#issue-line", Static).update("")
            for index in range(1, 10):
                self.query_one(f"#variant-card-{index}", Static).update("")
                self.query_one(f"#variant-card-{index}", Static).set_class(True, "hidden")
            self._load_editor_text("")
            self._refresh_summary()
            return
        self._current_index = max(0, min(self._current_index, len(self._flagged_span_ids) - 1))
        navigator.highlighted = self._current_index
        span = self._current_flagged_span()
        if span is None:
            return
        self.sub_title = f"{self.run_id}  {span.get('time_range')}"
        self.query_one("#source-view", Static).update(self._source_text(span))
        self.query_one("#issue-line", Static).update(self._issue_text(span))
        self._refresh_variant_cards(span)
        if self._editor_open:
            self._load_editor_text(self._current_text_for_span(span))
        self._refresh_summary()

    def _refresh_variant_cards(self, span: dict[str, Any]) -> None:
        variants = self._sorted_variants(span)
        selected_variant_id = self._selected_variant_id(span)
        recommended_variant_id = self._recommended_variant_id(span)
        for index in range(1, 10):
            card = self.query_one(f"#variant-card-{index}", Static)
            if index <= len(variants):
                variant = variants[index - 1]
                card.update(self._variant_card_text(index=index, span=span, variant=variant))
                card.set_class(False, "hidden")
                card.set_class(
                    variant.get("candidate_id") == selected_variant_id,
                    "variant-selected",
                )
                card.set_class(
                    variant.get("candidate_id") == recommended_variant_id,
                    "variant-recommended",
                )
            else:
                card.update("")
                card.set_class(True, "hidden")
                card.set_class(False, "variant-selected")
                card.set_class(False, "variant-recommended")

    def _refresh_summary(self) -> None:
        blocking_pending = sum(
            1
            for span_id in self._flagged_span_ids
            if bool(self._flagged_by_id[span_id].get("blocking"))
            and not bool(self._drafts[span_id].get("acknowledged"))
        )
        acknowledged = sum(
            1
            for span_id in self._flagged_span_ids
            if bool(self._drafts[span_id].get("acknowledged"))
        )
        summary = (
            f"run={self.run_id}  mode=exception_only  "
            f"flagged={len(self._flagged_span_ids)}  "
            f"auto_accepted={self.payload.get('auto_accepted_span_count', 0)}  "
            f"blocking_pending={blocking_pending}  "
            f"acknowledged={acknowledged}"
        )
        self.query_one("#summary", Static).update(summary)

    def _set_status(self, message: str) -> None:
        self.query_one("#status-line", Static).update(message)
        self.notify(message, markup=False)

    def _persist_draft(self) -> None:
        self.save_review_draft(
            self.run_id,
            draft_resolution=self._draft_payload(),
            settings=self.settings,
        )

    def _draft_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "job_id": self.payload["job_id"],
            "resolution_kind": "approved_good",
            "failure_tags": [],
            "note": "",
            "approved_by": None,
            "span_decisions": [
                {
                    "source_span_id": span_id,
                    "selected_variant_id": draft.get("selected_variant_id"),
                    "selected_base_variant_id": draft.get("selected_variant_id"),
                    "edited_text": draft.get("edited_text"),
                    "acknowledged": bool(draft.get("acknowledged")),
                    "resolution_status": (
                        "resolved" if draft.get("acknowledged") else "unresolved"
                    ),
                    "dirty": bool(draft.get("dirty")),
                    "reviewer_note": draft.get("reviewer_note") or "",
                }
                for span_id, draft in self._drafts.items()
            ],
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def _final_decisions(self) -> tuple[dict[str, object], ...] | None:
        decisions: list[dict[str, object]] = []
        for span in self._all_spans:
            span_id = str(span["source_span_id"])
            selected_id = self._drafts[span_id].get("selected_variant_id") or (
                self._default_variant_id(span)
            )
            if not isinstance(selected_id, str) or not selected_id:
                self._set_status(f"Span {span_id} has no selected candidate.")
                return None
            variant = self._variant_by_candidate_id(span, selected_id)
            if variant is None:
                self._set_status(f"Span {span_id} selected an unavailable candidate.")
                return None
            base_text = str(variant.get("target_excerpt") or "")
            edited_text = self._drafts[span_id].get("edited_text")
            final_text = (
                str(edited_text).strip()
                if isinstance(edited_text, str) and edited_text.strip()
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
                    "reviewer_note": "",
                }
            )
        return tuple(decisions)

    def _current_flagged_span(self) -> dict[str, Any] | None:
        if not self._flagged_span_ids:
            return None
        return self._flagged_by_id[self._flagged_span_ids[self._current_index]]

    def _first_pending_flagged_span_id(self) -> str | None:
        for span_id in self._flagged_span_ids:
            if not bool(self._drafts[span_id].get("acknowledged")):
                return span_id
        return self._flagged_span_ids[0] if self._flagged_span_ids else None

    def _jump_to_next_pending_flagged_span(self) -> None:
        for index, span_id in enumerate(self._flagged_span_ids):
            if not bool(self._drafts[span_id].get("acknowledged")):
                self._current_index = index
                self._refresh_current_span()
                return

    def _sorted_variants(self, span: dict[str, Any]) -> list[dict[str, Any]]:
        variants = [variant for variant in span.get("variants", []) if isinstance(variant, dict)]
        recommended_variant_id = cast(str | None, span.get("recommended_variant_id"))
        return sorted(
            variants,
            key=lambda variant: (
                0 if variant.get("candidate_id") == recommended_variant_id else 1,
                cast(int, variant.get("rank", 10**9)),
                str(variant.get("candidate_id")),
            ),
        )

    def _variant_by_candidate_id(
        self,
        span: dict[str, Any],
        candidate_id: str,
    ) -> dict[str, Any] | None:
        for variant in self._sorted_variants(span):
            if variant.get("candidate_id") == candidate_id:
                return variant
        return None

    def _recommended_variant_id(self, span: dict[str, Any]) -> str | None:
        recommended = span.get("recommended_variant_id")
        if isinstance(recommended, str) and recommended:
            return recommended
        return self._default_variant_id(span)

    def _default_variant_id(self, span: dict[str, Any]) -> str | None:
        variants = [variant for variant in span.get("variants", []) if isinstance(variant, dict)]
        for variant in sorted(
            variants,
            key=lambda item: (
                cast(int, item.get("rank", 10**9)),
                str(item.get("candidate_id")),
            ),
        ):
            candidate_id = variant.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                return candidate_id
        return None

    def _selected_variant_id(self, span: dict[str, Any]) -> str | None:
        draft = self._drafts[str(span["source_span_id"])]
        selected = draft.get("selected_variant_id")
        if isinstance(selected, str) and selected:
            return selected
        return self._recommended_variant_id(span)

    def _base_text_for_span(self, span: dict[str, Any]) -> str:
        selected_id = self._selected_variant_id(span)
        if not isinstance(selected_id, str):
            return ""
        variant = self._variant_by_candidate_id(span, selected_id)
        return str(variant.get("target_excerpt") or "") if variant is not None else ""

    def _current_text_for_span(self, span: dict[str, Any] | None) -> str:
        if span is None:
            return ""
        draft = self._drafts[str(span["source_span_id"])]
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
        selected_variant_id = draft.get("selected_variant_id") or self._recommended_variant_id(span)
        edited = "edit" if draft.get("dirty") else "base"
        acknowledged = "done" if draft.get("acknowledged") else "open"
        marker = "!" if span.get("blocking") else "~"
        return (
            f"{marker} {span.get('time_range')}  {span.get('severity')}  "
            f"{acknowledged}  {edited}  {selected_variant_id}"
        )

    def _source_text(self, span: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"{span.get('time_range')}  {'blocking' if span.get('blocking') else 'warning'}",
                "",
                str(span.get("source_excerpt") or "[no source excerpt]"),
            ]
        )

    def _issue_text(self, span: dict[str, Any]) -> str:
        draft = self._drafts[str(span["source_span_id"])]
        state = "confirmed" if draft.get("acknowledged") else "pending"
        return f"{span.get('issue_summary')}  Current state: {state}."

    def _variant_card_text(
        self,
        *,
        index: int,
        span: dict[str, Any],
        variant: dict[str, Any],
    ) -> str:
        candidate_id = str(variant.get("candidate_id") or "")
        recommended = candidate_id == self._recommended_variant_id(span)
        selected = candidate_id == self._selected_variant_id(span)
        provenance_lines = self._provenance_lines(span, variant)
        header = [f"[{index}] {candidate_id}"]
        if recommended:
            header.append("Recommended")
        if selected:
            header.append("Selected")
        lines = [
            " | ".join(header),
            f"model={variant.get('model_id')}  prompt={variant.get('prompt_variant_id')}",
            str(variant.get("target_excerpt") or "[no candidate text]"),
        ]
        if provenance_lines:
            lines.extend(["", *provenance_lines])
        return "\n".join(lines)

    def _provenance_lines(
        self,
        span: dict[str, Any],
        variant: dict[str, Any],
    ) -> list[str]:
        transcript_options = [
            option
            for option in span.get("transcript_provenance_options", [])
            if isinstance(option, dict)
        ]
        if len(transcript_options) <= 1:
            return []
        matching = next(
            (
                option
                for option in transcript_options
                if option.get("source_transcript_candidate_id")
                == variant.get("source_transcript_candidate_id")
            ),
            None,
        )
        if matching is None:
            return []
        return [
            "Transcript provenance",
            (
                f"{matching.get('source_transcript_candidate_id')} "
                f"via {matching.get('transcript_provider_id')}"
            ),
            str(matching.get("transcript_excerpt") or "[no transcript excerpt]"),
        ]
