"""Prompt resolution over exact-match project-pair and promoted pair overlays."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Protocol, cast, runtime_checkable

from translation_agent.memory.recall import build_scope_key
from translation_agent.models import (
    PromptCompatibilityTuple,
    PromptEvolutionProposal,
    ResolvedTranslationPrompt,
)
from translation_agent.storage import asset_path


class _ProposalQueryStore(Protocol):
    def list_prompt_evolution_proposals(
        self,
        *,
        status: str | None = None,
        prompt_family: str | None = None,
        target_model_id: str | None = None,
        target_language: str | None = None,
        source_language: str | None = None,
        prompt_variant_id: str | None = None,
        base_prompt_version: str | None = None,
        scope_kind: str | None = None,
        scope_key: str | None = None,
        media_key: str | None = None,
    ) -> list[PromptEvolutionProposal]: ...


class _BlobProposalStore(Protocol):
    def list_keys(self, prefix: str | None = None) -> list[str]: ...

    def read_bytes(self, key: str) -> bytes: ...


@runtime_checkable
class PromptResolver(Protocol):
    """Resolve effective translation prompt instructions for one run."""

    def resolve_translation_prompt(
        self,
        *,
        base_prompt_version: str,
        prompt_variant_id: str,
        model_id: str,
        source_language: str,
        target_language: str,
        tenant_id: str = "",
        project_id: str = "",
        media_key: str | None = None,
        run_id: str | None = None,
    ) -> ResolvedTranslationPrompt: ...


class ProposalBackedPromptResolver:
    """Apply project-pair overlays first, then promoted pair overlays."""

    def __init__(
        self, proposal_store: object | None = None, blob_store: object | None = None
    ) -> None:
        self._proposal_store = proposal_store
        self._blob_store = blob_store

    def resolve_translation_prompt(
        self,
        *,
        base_prompt_version: str,
        prompt_variant_id: str,
        model_id: str,
        source_language: str,
        target_language: str,
        tenant_id: str = "",
        project_id: str = "",
        media_key: str | None = None,
        run_id: str | None = None,
    ) -> ResolvedTranslationPrompt:
        project_pair = PromptCompatibilityTuple(
            prompt_family="translation",
            model_id=model_id,
            prompt_variant_id=prompt_variant_id,
            base_prompt_version=base_prompt_version,
            source_language=source_language,
            target_language=target_language,
            scope_kind="project_pair",
            scope_key=build_scope_key(
                scope_kind="project_pair",
                tenant_id=tenant_id,
                project_id=project_id,
                source_language=source_language,
                target_language=target_language,
            ),
        )
        pair = PromptCompatibilityTuple(
            prompt_family="translation",
            model_id=model_id,
            prompt_variant_id=prompt_variant_id,
            base_prompt_version=base_prompt_version,
            source_language=source_language,
            target_language=target_language,
            scope_kind="pair",
            scope_key=build_scope_key(
                scope_kind="pair",
                tenant_id=tenant_id,
                project_id=project_id,
                source_language=source_language,
                target_language=target_language,
            ),
        )

        self._promote_proposed_if_needed(compatibility=project_pair, media_key=media_key)
        project_pair_selection = self._select_overlay(
            compatibility=project_pair,
            media_key=media_key,
            run_id=run_id,
            pair_scope=False,
        )
        pair_selection = (
            None
            if project_pair_selection is not None
            else self._select_overlay(
                compatibility=pair,
                media_key=media_key,
                run_id=run_id,
                pair_scope=True,
            )
        )
        selected = project_pair_selection or pair_selection
        selected_scope = project_pair if project_pair_selection is not None else pair
        if selected is None:
            selected_scope = project_pair
        instructions = tuple(
            change.instruction
            for proposal in selected or ()
            for change in proposal.suggested_changes
        )
        proposal_refs = tuple(
            ref for proposal in selected or () for ref in _proposal_refs(proposal)
        )
        effective_prompt_version = base_prompt_version
        resolution_mode = "control"
        selected_proposal_id = None
        if selected:
            resolution_mode = "canary" if selected[0].status == "canary" else "active"
            selected_proposal_id = selected[0].proposal_id
            version_suffix = sha256(
                "|".join(proposal.proposal_id for proposal in selected).encode("utf-8")
            ).hexdigest()[:8]
            effective_prompt_version = f"{base_prompt_version}+{resolution_mode}-{version_suffix}"
        return ResolvedTranslationPrompt(
            prompt_variant_id=prompt_variant_id,
            base_prompt_version=base_prompt_version,
            effective_prompt_version=effective_prompt_version,
            resolution_mode=resolution_mode,  # type: ignore[arg-type]
            selected_proposal_id=selected_proposal_id,
            scope_kind=selected_scope.scope_kind,
            scope_key=selected_scope.scope_key,
            instructions=instructions,
            applied_proposal_refs=proposal_refs,
        )

    def _select_overlay(
        self,
        *,
        compatibility: PromptCompatibilityTuple,
        media_key: str | None,
        run_id: str | None,
        pair_scope: bool,
    ) -> tuple[PromptEvolutionProposal, ...] | None:
        active = self._matching_proposals(
            compatibility=compatibility,
            status="active",
            media_key=media_key,
            pair_scope=pair_scope,
        )
        canary = self._matching_proposals(
            compatibility=compatibility,
            status="canary",
            media_key=media_key,
            pair_scope=pair_scope,
        )
        if canary and run_id is not None and _is_canary_run(run_id):
            return canary[:1]
        if active:
            return active
        return None

    def _matching_proposals(
        self,
        *,
        compatibility: PromptCompatibilityTuple,
        status: str,
        media_key: str | None,
        pair_scope: bool,
    ) -> tuple[PromptEvolutionProposal, ...]:
        matched: list[PromptEvolutionProposal] = []
        if self._proposal_store is None:
            return self._matching_blob_proposals(
                compatibility=compatibility,
                status=status,
                media_key=media_key,
                pair_scope=pair_scope,
            )

        if hasattr(self._proposal_store, "list_prompt_evolution_proposals"):
            query_store = cast(_ProposalQueryStore, self._proposal_store)
            matched.extend(
                proposal
                for proposal in query_store.list_prompt_evolution_proposals(
                    status=status,
                    prompt_family=compatibility.prompt_family,
                    target_model_id=compatibility.model_id,
                    target_language=compatibility.target_language,
                    source_language=compatibility.source_language,
                    prompt_variant_id=compatibility.prompt_variant_id,
                    base_prompt_version=compatibility.base_prompt_version,
                    scope_kind=compatibility.scope_kind,
                    scope_key=compatibility.scope_key,
                    media_key=None,
                )
                if _proposal_matches(
                    proposal,
                    compatibility,
                    media_key=media_key,
                    pair_scope=pair_scope,
                )
            )

        for proposal in self._matching_blob_proposals(
            compatibility=compatibility,
            status=status,
            media_key=media_key,
            pair_scope=pair_scope,
        ):
            if all(existing.proposal_id != proposal.proposal_id for existing in matched):
                matched.append(proposal)
        return tuple(sorted(matched, key=lambda proposal: proposal.proposal_id))

    def _matching_blob_proposals(
        self,
        *,
        compatibility: PromptCompatibilityTuple,
        status: str,
        media_key: str | None,
        pair_scope: bool,
    ) -> tuple[PromptEvolutionProposal, ...]:
        store = self._blob_store if self._blob_store is not None else self._proposal_store
        if (
            media_key is None
            or store is None
            or not hasattr(store, "list_keys")
            or not hasattr(store, "read_bytes")
        ):
            return ()
        blob_store = cast(_BlobProposalStore, store)
        matched: list[PromptEvolutionProposal] = []
        keys = sorted(blob_store.list_keys(asset_path(media_key, "improvement-proposals")))
        for key in keys:
            proposal = PromptEvolutionProposal.model_validate_json(blob_store.read_bytes(key))
            if proposal.status != status or not _proposal_matches(
                proposal,
                compatibility,
                media_key=media_key,
                pair_scope=pair_scope,
            ):
                continue
            matched.append(
                proposal.model_copy(
                    update={"metadata": {**proposal.metadata, "asset_proposal_ref": key}}
                )
            )
        return tuple(matched)

    def _promote_proposed_if_needed(
        self,
        *,
        compatibility: PromptCompatibilityTuple,
        media_key: str | None,
    ) -> None:
        if self._proposal_store is None or not hasattr(
            self._proposal_store, "save_prompt_evolution_proposal"
        ):
            return
        if self._matching_proposals(
            compatibility=compatibility,
            status="active",
            media_key=media_key,
            pair_scope=False,
        ):
            return
        if self._matching_proposals(
            compatibility=compatibility,
            status="canary",
            media_key=media_key,
            pair_scope=False,
        ):
            return
        proposed = self._matching_proposals(
            compatibility=compatibility,
            status="proposed",
            media_key=media_key,
            pair_scope=False,
        )
        proposed = tuple(
            proposal
            for proposal in proposed
            if proposal.metadata.get("proposal_origin")
            in {"human_review_feedback", "mainline_adjudication"}
        )
        if not proposed:
            return
        updated = proposed[0].model_copy(update={"status": "canary"})
        save = getattr(self._proposal_store, "save_prompt_evolution_proposal", None)
        if callable(save):
            save(updated)
        self._persist_proposal_artifact(updated)

    def _persist_proposal_artifact(self, proposal: PromptEvolutionProposal) -> None:
        store = self._blob_store if self._blob_store is not None else self._proposal_store
        if store is None or not hasattr(store, "put_bytes"):
            return
        proposal_ref = proposal.metadata.get("proposal_ref")
        if not isinstance(proposal_ref, str) or not proposal_ref.strip():
            media_key = proposal.metadata.get("media_key")
            if not isinstance(media_key, str) or not media_key.strip():
                return
            proposal_ref = asset_path(
                media_key, "improvement-proposals", f"{proposal.proposal_id}.json"
            )
        blob_store = cast(object, store)
        getattr(blob_store, "put_bytes")(
            proposal_ref,
            (json.dumps(proposal.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )


def _proposal_matches(
    proposal: PromptEvolutionProposal,
    compatibility: PromptCompatibilityTuple,
    *,
    media_key: str | None,
    pair_scope: bool,
) -> bool:
    if proposal.compatibility != compatibility:
        return False
    if proposal.compatibility is None:
        return False
    if proposal.compatibility.scope_kind == "asset":
        return False
    if pair_scope and proposal.compatibility.scope_kind == "pair":
        promotion_status = proposal.promotion_status
        if promotion_status != "promoted":
            legacy_status = proposal.metadata.get("promotion_status")
            if legacy_status not in {None, "promoted"}:
                return False
    proposal_media_key = proposal.metadata.get("media_key")
    if isinstance(proposal_media_key, str) and proposal_media_key.strip():
        return proposal_media_key == media_key
    return True


def _proposal_refs(proposal: PromptEvolutionProposal) -> tuple[str, ...]:
    asset_ref = proposal.metadata.get("asset_proposal_ref")
    if isinstance(asset_ref, str) and asset_ref.strip():
        return (asset_ref,)
    proposal_ref = proposal.metadata.get("proposal_ref")
    if isinstance(proposal_ref, str) and proposal_ref.strip():
        return (proposal_ref,)
    media_key = proposal.metadata.get("media_key")
    if isinstance(media_key, str) and media_key.strip():
        return (
            asset_path(
                media_key,
                "improvement-proposals",
                f"{proposal.proposal_id}.json",
            ),
        )
    return ()


def _is_canary_run(run_id: str) -> bool:
    digest = sha256(run_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10 == 0
