"""Prompt resolution over approved proposal overlays."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, cast, runtime_checkable

from translation_agent.models import PromptEvolutionProposal, ResolvedTranslationPrompt
from translation_agent.storage import asset_path


class _ProposalQueryStore(Protocol):
    def list_prompt_evolution_proposals(
        self,
        *,
        status: str | None = None,
        target_model_id: str | None = None,
        target_language: str | None = None,
        source_language: str | None = None,
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
    ) -> ResolvedTranslationPrompt: ...


class ProposalBackedPromptResolver:
    """Apply approved translation proposals as prompt overlays."""

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
    ) -> ResolvedTranslationPrompt:
        proposals = tuple(
            self._matching_proposals(
                model_id=model_id,
                source_language=source_language,
                target_language=target_language,
                prompt_variant_id=prompt_variant_id,
                media_key=media_key,
            )
        )
        instructions = tuple(
            change.instruction for proposal in proposals for change in proposal.suggested_changes
        )
        proposal_refs = tuple(ref for proposal in proposals for ref in _proposal_refs(proposal))
        effective_prompt_version = base_prompt_version
        if proposals:
            version_suffix = sha256(
                "|".join(proposal.proposal_id for proposal in proposals).encode("utf-8")
            ).hexdigest()[:8]
            effective_prompt_version = f"{base_prompt_version}+approved-{version_suffix}"
        return ResolvedTranslationPrompt(
            prompt_variant_id=prompt_variant_id,
            base_prompt_version=base_prompt_version,
            effective_prompt_version=effective_prompt_version,
            instructions=instructions,
            applied_proposal_refs=proposal_refs,
        )

    def _matching_proposals(
        self,
        *,
        model_id: str,
        source_language: str,
        target_language: str,
        prompt_variant_id: str,
        media_key: str | None,
    ) -> list[PromptEvolutionProposal]:
        matched: list[PromptEvolutionProposal] = []
        if self._proposal_store is None:
            return matched

        if hasattr(self._proposal_store, "list_prompt_evolution_proposals"):
            query_store = cast(_ProposalQueryStore, self._proposal_store)
            proposals = query_store.list_prompt_evolution_proposals(
                status="approved",
                target_model_id=model_id,
                target_language=target_language,
                source_language=source_language,
                media_key=media_key,
            )
            matched.extend(
                proposal
                for proposal in proposals
                if proposal.prompt_family == "translation"
                and proposal.target_prompt_variant_id in {None, prompt_variant_id}
                and proposal.metadata.get("media_key", media_key) == media_key
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
                if proposal.status != "approved":
                    continue
                if proposal.prompt_family != "translation":
                    continue
                if proposal.target_model_id != model_id:
                    continue
                if proposal.target_prompt_variant_id not in {None, prompt_variant_id}:
                    continue
                if proposal.metadata.get("source_language", source_language) != source_language:
                    continue
                if proposal.metadata.get("target_language", target_language) != target_language:
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
        return sorted(matched, key=lambda proposal: proposal.proposal_id)


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
