from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable


REFERENCE_DATE = date(2026, 6, 21)
CAREER_DURATION_TOLERANCE_MONTHS = 3
ZERO_DURATION_EXPERT_THRESHOLD = 10


@dataclass(frozen=True)
class DefenseResult:
    candidate_id: str
    should_drop: bool
    reasons: tuple[str, ...]
    career_months: int


def total_career_months(candidate: dict[str, Any]) -> int:
    return sum(
        int(role.get("duration_months") or 0)
        for role in candidate.get("career_history", [])
    )


def has_skill_duration_honeypot(candidate: dict[str, Any]) -> bool:
    career_months = total_career_months(candidate)

    for skill in candidate.get("skills", []):
        skill_months = int(skill.get("duration_months") or 0)
        if skill_months > career_months:
            return True

    return False


def month_span(
    start_date: str, end_date: str | None, reference_date: date = REFERENCE_DATE
) -> int:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else reference_date
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def has_career_duration_mismatch(candidate: dict[str, Any]) -> bool:
    for role in candidate.get("career_history", []):
        declared = int(role.get("duration_months") or 0)
        observed = month_span(role["start_date"], role.get("end_date"))

        if abs(declared - observed) > CAREER_DURATION_TOLERANCE_MONTHS:
            return True

    return False


def has_zero_duration_expert_anomaly(candidate: dict[str, Any]) -> bool:
    expert_zero_count = sum(
        1
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert"
        and int(skill.get("duration_months") or 0) == 0
    )
    return expert_zero_count >= ZERO_DURATION_EXPERT_THRESHOLD


def evaluate_candidate(candidate: dict[str, Any]) -> DefenseResult:
    reasons: list[str] = []

    if has_skill_duration_honeypot(candidate):
        reasons.append("skill_duration_exceeds_total_career")

    if has_career_duration_mismatch(candidate):
        reasons.append("career_duration_date_mismatch")

    if has_zero_duration_expert_anomaly(candidate):
        reasons.append("zero_duration_expert_anomaly")

    return DefenseResult(
        candidate_id=str(candidate.get("candidate_id", "")),
        should_drop=bool(reasons),
        reasons=tuple(reasons),
        career_months=total_career_months(candidate),
    )


def filter_candidates(
    candidates: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[DefenseResult]]:
    kept: list[dict[str, Any]] = []
    dropped: list[DefenseResult] = []

    for candidate in candidates:
        result = evaluate_candidate(candidate)
        if result.should_drop:
            dropped.append(result)
        else:
            kept.append(candidate)

    return kept, dropped

