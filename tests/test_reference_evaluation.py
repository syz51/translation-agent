from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from translation_agent.api import RunJobRequest, run_job
from translation_agent.config import Settings
from translation_agent.memory import ProposalBackedPromptResolver
from translation_agent.models import JobContext, PromptChange, PromptEvolutionProposal
from translation_agent.storage import LocalBlobStore, SQLiteOperationalStore, asset_path, job_path

pytestmark = pytest.mark.unit


def _job(job_id: str) -> JobContext:
    return JobContext(
        job_id=job_id,
        tenant_id="tenant-local",
        project_id="project-local",
        source_video_ref="input.mp4",
        target_language="fr",
        source_language="en",
        requested_by="system@local",
        created_at=datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        profile_ref="profiles/default",
        asset_id="asset-1",
        media_key="asset-id:asset-1",
    )


def test_run_job_rejects_invalid_reference_request(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"media")

    with pytest.raises(ValueError, match="reference transcript source is required"):
        run_job(
            RunJobRequest(
                source=str(source),
                reference_mode="evaluate_and_regenerate",
            ),
            settings=settings,
        )

    with pytest.raises(ValueError, match="unsupported reference transcript format"):
        run_job(
            RunJobRequest(
                source=str(source),
                reference_transcript_source=str(tmp_path / "reference.txt"),
                reference_transcript_format="txt",  # type: ignore[arg-type]
            ),
            settings=settings,
        )


def test_reference_evaluation_path_publishes_asset_artifacts_and_keeps_canonical_output(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "runtime")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"media-for-reference-evaluation")
    reference_path = tmp_path / "reference.srt"
    reference_path.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:01,000",
                "OpenAI workflow",
                "",
                "2",
                "00:00:01,000 --> 00:00:02,000",
                "OpenAI workflow",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_job(
        RunJobRequest(source=str(source), job_id="job-prev-1", asset_id="asset-1"),
        settings=settings,
    )
    run_job(
        RunJobRequest(source=str(source), job_id="job-prev-2", asset_id="asset-1"),
        settings=settings,
    )
    result = run_job(
        RunJobRequest(
            source=str(source),
            job_id="job-eval",
            asset_id="asset-1",
            reference_transcript_source=str(reference_path),
            reference_mode="evaluate_and_regenerate",
        ),
        settings=settings,
    )

    blob_store = LocalBlobStore(result.blob_root)
    manifest = json.loads(
        blob_store.read_bytes(job_path(_job("job-eval"), "published", "artifacts.json")).decode(
            "utf-8"
        )
    )
    evaluation_report = json.loads(
        blob_store.read_bytes(manifest["evaluation_report_refs"][0]).decode("utf-8")
    )
    regenerated_draft = json.loads(
        blob_store.read_bytes(manifest["regenerated_draft_refs"][0]).decode("utf-8")
    )
    proposal = json.loads(
        blob_store.read_bytes(manifest["improvement_proposal_refs"][0]).decode("utf-8")
    )

    assert blob_store.exists(
        asset_path("asset-id:asset-1", "references", "transcript", "latest.json")
    )
    assert blob_store.exists(manifest["final_translation_ref"])
    assert manifest["regenerated_draft_refs"][0] != manifest["final_translation_ref"]
    assert len(evaluation_report["evaluated_runs"]) == 2
    assert regenerated_draft["generated_from_reference_transcript"] is True
    assert regenerated_draft["replaces_canonical"] is False
    assert proposal["status"] == "proposed"
    assert proposal["activation_mode"] == "approval_required"


def test_approved_prompt_proposals_affect_prompt_resolution(tmp_path: Path) -> None:
    store = SQLiteOperationalStore(tmp_path / "state.sqlite3")
    try:
        proposal = PromptEvolutionProposal(
            proposal_id="proposal-approved",
            job_id="job-1",
            source_consolidation_id="consolidation-1",
            prompt_family="translation",
            target_model_id="gpt-5.4-mini",
            target_prompt_version="phase-5-v1",
            target_prompt_variant_id="variant-a",
            status="approved",
            activation_mode="approval_required",
            auto_activate=False,
            rationale="Approved correction.",
            suggested_changes=(
                PromptChange(
                    section="system",
                    instruction="Preserve OpenAI exactly across the translation.",
                ),
            ),
            metadata={
                "source_language": "en",
                "target_language": "fr",
                "media_key": "asset-id:asset-1",
                "proposal_ref": (
                    "assets/asset-id-asset-1/improvement-proposals/proposal-approved.json"
                ),
            },
        )
        store.save_prompt_evolution_proposal(proposal)
        resolver = ProposalBackedPromptResolver(store)

        resolved = resolver.resolve_translation_prompt(
            base_prompt_version="phase-5-v1",
            prompt_variant_id="variant-a",
            model_id="gpt-5.4-mini",
            source_language="en",
            target_language="fr",
            media_key="asset-id:asset-1",
        )
    finally:
        store.close()

    assert resolved.effective_prompt_version != "phase-5-v1"
    assert "Preserve OpenAI exactly across the translation." in resolved.instructions
    assert resolved.applied_proposal_refs == (
        "assets/asset-id-asset-1/improvement-proposals/proposal-approved.json",
    )
