"""Deterministic parser for reviewer prose with a fixed section contract."""

from __future__ import annotations

import re
from typing import Literal, cast

from pydantic import Field

from translation_agent.models import QuotedEvidence, SuggestedFix
from translation_agent.models.base import ContractModel
from translation_agent.review.prompts import PARSER_VERSION, REQUIRED_REVIEW_SECTIONS

IssueSeverity = Literal["minor", "major", "critical"]

_SECTION_PATTERN = re.compile(
    (
        r"^(Winner|Confidence|Why|Key Errors By Candidate|Quoted Evidence|Suggested Fixes|"
        r"Escalate\?)\s*:\s*(.*)$"
    ),
    re.IGNORECASE | re.MULTILINE,
)


class ParsedReviewIssue(ContractModel):
    """Structured issue extracted from reviewer prose."""

    candidate_id: str | None = None
    category: str
    severity: IssueSeverity
    description: str


class ParsedReview(ContractModel):
    """Normalized review signals extracted from the prose contract."""

    winner_candidate_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    why: str
    issues: tuple[ParsedReviewIssue, ...] = ()
    quoted_evidence: tuple[QuotedEvidence, ...] = ()
    suggested_fixes: tuple[SuggestedFix, ...] = ()
    escalation_signal: bool = False
    parser_version: str = PARSER_VERSION


def parse_reviewer_output(raw_review_text: str) -> ParsedReview:
    """Extract deterministic adjudication signals from fixed-section prose."""

    sections = _split_sections(raw_review_text)
    winner_text = _first_value(sections["Winner"])
    why_text = sections["Why"].strip() or "No rationale provided."
    return ParsedReview(
        winner_candidate_id=None if winner_text.lower() in {"none", "n/a"} else winner_text,
        confidence=_parse_confidence(sections["Confidence"]),
        why=why_text,
        issues=_parse_issues(sections["Key Errors By Candidate"]),
        quoted_evidence=_parse_quoted_evidence(sections["Quoted Evidence"]),
        suggested_fixes=_parse_suggested_fixes(sections["Suggested Fixes"]),
        escalation_signal=_parse_escalation(sections["Escalate?"]),
    )


def _split_sections(raw_review_text: str) -> dict[str, str]:
    matches = list(_SECTION_PATTERN.finditer(raw_review_text))
    if len(matches) != len(REQUIRED_REVIEW_SECTIONS):
        found = [match.group(1) for match in matches]
        raise ValueError(f"review output missing required sections: found {found!r}")

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        header = _canonical_header(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_review_text)
        same_line = match.group(2).strip()
        body = raw_review_text[start:end].strip()
        value = "\n".join(part for part in (same_line, body) if part).strip()
        sections[header] = value

    missing = [header for header in REQUIRED_REVIEW_SECTIONS if header not in sections]
    if missing:
        raise ValueError(f"review output missing required sections: {missing!r}")
    return sections


def _canonical_header(header: str) -> str:
    normalized = header.lower().rstrip("?")
    mapping = {
        "winner": "Winner",
        "confidence": "Confidence",
        "why": "Why",
        "key errors by candidate": "Key Errors By Candidate",
        "quoted evidence": "Quoted Evidence",
        "suggested fixes": "Suggested Fixes",
        "escalate": "Escalate?",
    }
    return mapping[normalized]


def _first_value(section_body: str) -> str:
    for line in section_body.splitlines():
        stripped = _strip_bullet(line)
        if stripped:
            return stripped
    return ""


def _parse_confidence(section_body: str) -> float:
    token = _first_value(section_body)
    if not token:
        raise ValueError("review output is missing a confidence value")
    token = token.rstrip("%")
    value = float(token)
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(value, 1.0))


def _parse_issues(section_body: str) -> tuple[ParsedReviewIssue, ...]:
    issues: list[ParsedReviewIssue] = []
    for line in _iter_content_lines(section_body):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            raise ValueError(f"invalid issue line: {line!r}")
        candidate_id, category, severity, description = parts
        issues.append(
            ParsedReviewIssue(
                candidate_id=None if candidate_id.lower() in {"none", "general"} else candidate_id,
                category=category,
                severity=_normalize_severity(severity),
                description=description,
            )
        )
    return tuple(issues)


def _parse_quoted_evidence(section_body: str) -> tuple[QuotedEvidence, ...]:
    evidence_items: list[QuotedEvidence] = []
    for line in _iter_content_lines(section_body):
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3:
            raise ValueError(f"invalid evidence line: {line!r}")
        candidate_id, segment_id, quote = parts
        evidence_items.append(
            QuotedEvidence(
                quote=quote,
                candidate_id=None if candidate_id.lower() in {"none", "general"} else candidate_id,
                segment_id=None if segment_id.lower() in {"none", "general"} else segment_id,
            )
        )
    return tuple(evidence_items)


def _parse_suggested_fixes(section_body: str) -> tuple[SuggestedFix, ...]:
    fixes: list[SuggestedFix] = []
    for line in _iter_content_lines(section_body):
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) != 3:
            raise ValueError(f"invalid suggested-fix line: {line!r}")
        category, candidate_id, description = parts
        fixes.append(
            SuggestedFix(
                issue_category=category,
                candidate_id=None if candidate_id.lower() in {"none", "general"} else candidate_id,
                description=description,
            )
        )
    return tuple(fixes)


def _parse_escalation(section_body: str) -> bool:
    token = _first_value(section_body).lower()
    return token in {"yes", "true", "1", "escalate", "required"}


def _normalize_severity(value: str) -> IssueSeverity:
    normalized = value.strip().lower()
    if normalized not in {"minor", "major", "critical"}:
        raise ValueError(f"unsupported issue severity: {value!r}")
    return cast(IssueSeverity, normalized)


def _iter_content_lines(section_body: str):
    for line in section_body.splitlines():
        stripped = _strip_bullet(line)
        if stripped:
            yield stripped


def _strip_bullet(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith(("-", "*")):
        return stripped[1:].strip()
    return stripped
