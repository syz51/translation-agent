"""Prompt resolution over exact-match active and canary proposal overlays."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, cast, runtime_checkable

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
        media_key: str | None = None,
        run_id: str | None = None,
        scope_kind: str = "pair",
        scope_key: str | None = None,
    ) -> ResolvedTranslationPrompt: ...


class ProposalBackedPromptResolver:
    """Apply exact-match active proposals and deterministic canary overlays."""

    def __init__(self, proposal_store: object | None = None) -> None:
        self._proposal_store = proposal_store

    def resolve_translation_prompt(
        self,
        *,
        base_prompt_version: str,
        prompt_variant_id: str,
        model_id: str,
        source_language: str,
        target_language: str,
        media_key: str | None = None,
        run_id: str | None = None,
        scope_kind: str = "pair",
        scope_key: str | None = None,
    ) -> ResolvedTranslationPrompt:
        compatibility = PromptCompatibilityTuple(
            prompt_family="translation",
            model_id=model_id,
            prompt_variant_id=prompt_variant_id,
            base_prompt_version=base_prompt_version,
            source_language=source_language,
            target_language=target_language,
            scope_kind=scope_kind,  # type: ignore[arg-type]
            scope_key=scope_key or f"{source_language}::{target_language}",
        )
        active_proposals = self._matching_proposals(
            compatibility=compatibility, status="active", media_key=media_key
        )
        canary_proposals = self._matching_proposals(
            compatibility=compatibility, status="canary", media_key=media_key
        )
        selected: tuple[PromptEvolutionProposal, ...]
        resolution_mode = "control"
        if canary_proposals and run_id is not None and _is_canary_run(run_id):
            selected = canary_proposals[:1]
            resolution_mode = "canary"
        else:
            selected = active_proposals
            if active_proposals:
                resolution_mode = "active"
        instructions = tuple(
            change.instruction for proposal in selected for change in proposal.suggested_changes
        )
        proposal_refs = tuple(ref for proposal in selected for ref in _proposal_refs(proposal))
        effective_prompt_version = base_prompt_version
        selected_proposal_id = selected[0].proposal_id if selected else None
        if selected:
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
            instructions=instructions,
            applied_proposal_refs=proposal_refs,
        )

    def _matching_proposals(
        self,
        *,
        compatibility: PromptCompatibilityTuple,
        status: str,
        media_key: str | None,
    ) -> tuple[PromptEvolutionProposal, ...]:
        matched: list[PromptEvolutionProposal] = []
        if self._proposal_store is None:
            return ()

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
                    media_key=media_key,
                )
                if _proposal_matches(proposal, compatibility)
            )

        if (
            media_key is not None
            and hasattr(self._proposal_store, "list_keys")
            and hasattr(self._proposal_store, "read_bytes")
        ):
            blob_store = cast(_BlobProposalStore, self._proposal_store)
            keys = sorted(blob_store.list_keys(asset_path(media_key, "improvement-proposals")))
            for key in keys:
                proposal = PromptEvolutionProposal.model_validate_json(blob_store.read_bytes(key))
                if proposal.status != status or not _proposal_matches(proposal, compatibility):
                    continue
                if all(existing.proposal_id != proposal.proposal_id for existing in matched):
                    matched.append(
                        proposal.model_copy(
                            update={
                                "metadata": {
                                    **proposal.metadata,
                                    "asset_proposal_ref": key,
                                }
                            }
                        )
                    )
        return tuple(sorted(matched, key=lambda proposal: proposal.proposal_id))


def _proposal_matches(
    proposal: PromptEvolutionProposal,
    compatibility: PromptCompatibilityTuple,
) -> bool:
    return proposal.compatibility == compatibility


def _proposal_refs(proposal: PromptEvolutionProposal) -> tuple[str, ...]:
    asset_ref = proposal.metadata.get("asset_proposal_ref")
    if isinstance(asset_ref, str) and asset_ref.strip():
        return (asset_ref,)
    media_key = proposal.metadata.get("media_key")
    if isinstance(media_key, str) and media_key.strip():
        return (
            asset_path(
                media_key,
                "improvement-proposals",
                f"{proposal.proposal_id}.json",
            ),
        )
    proposal_ref = proposal.metadata.get("proposal_ref")
    if isinstance(proposal_ref, str) and proposal_ref.strip():
        return (proposal_ref,)
    return ()


def _is_canary_run(run_id: str) -> bool:
    digest = sha256(run_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10 == 0
