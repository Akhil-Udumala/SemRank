from __future__ import annotations

import re
from typing import Any


MAX_REASONING_CHARS = 500
TECHNICAL_SKILL_HINTS = (
    "ai",
    "airflow",
    "aws",
    "bentoml",
    "dbt",
    "docker",
    "embedding",
    "elasticsearch",
    "faiss",
    "fine-tuning",
    "flink",
    "gcp",
    "hugging face",
    "kafka",
    "kubeflow",
    "llm",
    "lora",
    "ml",
    "mlops",
    "milvus",
    "model",
    "nlp",
    "opensearch",
    "pinecone",
    "python",
    "qdrant",
    "ranking",
    "recommendation",
    "retrieval",
    "search",
    "spark",
    "sql",
    "transformer",
    "vector",
    "weaviate",
)


def generate_candidate_reasoning(
    candidate_id: str,
    candidate_dict: dict[str, Any],
    gemini_features_dict: dict[str, Any],
) -> str:
    """Return a factual, cleaned reasoning string for a shortlist row."""
    feature_row = gemini_features_dict.get(str(candidate_id), {})
    if isinstance(feature_row, dict):
        for key in ("reasoning_snippet", "reasoning"):
            value = feature_row.get(key)
            if isinstance(value, str) and value.strip():
                return _clean_reasoning_string(value)

    return _build_local_reasoning(candidate_dict)


def _build_local_reasoning(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) if isinstance(candidate, dict) else {}
    title = _safe_value(profile.get("current_title"), default="Candidate")
    years = _safe_value(profile.get("years_of_experience"), default="unknown")
    location = _safe_value(
        profile.get("location") or profile.get("country"), default="location not listed"
    )
    top_skills = _top_technical_skills(candidate.get("skills", []))
    skill_text = ", ".join(top_skills) if top_skills else "no highlighted technical skills"

    return _clean_reasoning_string(
        f"{title} with {years} years of experience in {location}; profile-listed "
        f"technical strengths include {skill_text}."
    )


def _top_technical_skills(skills: Any) -> list[str]:
    if not isinstance(skills, list):
        return []

    scored: list[tuple[float, int, str]] = []
    for index, skill in enumerate(skills):
        if not isinstance(skill, dict):
            continue

        name = _safe_value(skill.get("name"), default="")
        if not name:
            continue

        lowered = name.lower()
        has_technical_hint = any(hint in lowered for hint in TECHNICAL_SKILL_HINTS)
        if not has_technical_hint:
            continue

        endorsements = _safe_float(skill.get("endorsements"))
        duration = _safe_float(skill.get("duration_months"))
        proficiency = str(skill.get("proficiency") or "").lower()
        proficiency_score = {
            "expert": 4.0,
            "advanced": 3.0,
            "intermediate": 2.0,
            "beginner": 1.0,
        }.get(proficiency, 0.0)

        score = proficiency_score + min(duration / 24.0, 3.0) + min(endorsements / 25.0, 2.0)
        scored.append((score, -index, name))

    scored.sort(reverse=True)
    return [name for _, _, name in scored[:3]]


def _clean_reasoning_string(text: str) -> str:
    cleaned = re.sub(r"[\n\t\r]+", " ", str(text or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned[:MAX_REASONING_CHARS]


def _safe_value(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

