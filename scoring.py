from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REFERENCE_DATE = date(2026, 6, 21)
MIN_BEHAVIORAL_MODIFIER = 0.75
MAX_BEHAVIORAL_MODIFIER = 1.15

BASE_WEIGHTS = {
    "career_domain": 0.30,
    "title_fit": 0.18,
    "production_shipping": 0.12,
    "eval_framework": 0.10,
    "trusted_skills": 0.10,
    "experience_fit": 0.08,
    "product_company": 0.07,
    "location_fit": 0.05,
}

STRONG_TITLES = {
    "recommendation systems engineer",
    "search engineer",
    "senior ai engineer",
    "lead ai engineer",
    "senior machine learning engineer",
    "staff machine learning engineer",
    "senior nlp engineer",
    "senior applied scientist",
    "applied ml engineer",
    "machine learning engineer",
}

GOOD_TITLES = {
    "ml engineer",
    "ai engineer",
    "nlp engineer",
    "data scientist",
    "senior data scientist",
    "senior software engineer (ml)",
    "ai specialist",
}

ADJACENT_TITLES = {
    "backend engineer",
    "data engineer",
    "senior data engineer",
    "analytics engineer",
    "software engineer",
    "senior software engineer",
    "cloud engineer",
    "devops engineer",
}

LOW_FIT_TITLES = {
    "hr manager",
    "marketing manager",
    "sales executive",
    "accountant",
    "content writer",
    "graphic designer",
    "civil engineer",
    "mechanical engineer",
    "customer support",
    "operations manager",
    "project manager",
    "business analyst",
}

CORE_DOMAIN_TERMS = (
    "embedding",
    "embeddings",
    "retrieval",
    "information retrieval",
    "ranking",
    "ranker",
    "search",
    "semantic search",
    "hybrid search",
    "recommendation",
    "recommendations",
    "recommender",
    "matching",
    "candidate matching",
)

VECTOR_TERMS = (
    "vector",
    "vector database",
    "faiss",
    "milvus",
    "pinecone",
    "qdrant",
    "weaviate",
    "opensearch",
    "elasticsearch",
    "ann",
    "nearest neighbor",
)

AI_SYSTEM_TERMS = (
    "nlp",
    "llm",
    "llms",
    "rag",
    "fine-tuning",
    "finetuning",
    "lora",
    "qlora",
    "peft",
    "transformers",
    "sentence-transformers",
    "hugging face",
)

PRODUCTION_TERMS = (
    "production",
    "deployed",
    "ship",
    "shipped",
    "shipping",
    "real users",
    "scale",
    "latency",
    "monitoring",
    "observability",
    "index refresh",
    "regression",
    "pipeline",
    "serving",
    "online",
    "a/b",
    "experiment",
    "recruiter",
    "marketplace",
)

EVAL_TERMS = (
    "ndcg",
    "mrr",
    "map",
    "precision",
    "recall",
    "relevance",
    "ranking metrics",
    "evaluation",
    "eval",
    "offline benchmark",
    "online metric",
    "a/b test",
    "ab test",
    "feedback loop",
)

PRODUCT_COMPANIES = {
    "swiggy",
    "razorpay",
    "cred",
    "zomato",
    "flipkart",
    "meesho",
    "inmobi",
    "nykaa",
    "phonepe",
    "dream11",
    "freshworks",
    "paytm",
    "ola",
    "zoho",
    "policybazaar",
    "unacademy",
    "byju's",
    "upgrad",
    "pharmeasy",
    "google",
    "netflix",
    "amazon",
    "salesforce",
    "uber",
    "meta",
    "adobe",
    "microsoft",
    "apple",
    "linkedin",
    "haptik",
    "glance",
    "observe.ai",
    "yellow.ai",
    "sarvam ai",
    "krutrim",
    "wysa",
    "mad street den",
    "saarthi.ai",
    "niramai",
    "aganitha",
    "rephrase.ai",
    "verloop.io",
    "locobuzz",
    "genpact ai",
    "vedantu",
}

PRODUCT_INDUSTRIES = {
    "software",
    "saas",
    "fintech",
    "e-commerce",
    "food delivery",
    "edtech",
    "ai/ml",
    "adtech",
    "transportation",
    "insurance tech",
}

PREFERRED_LOCATIONS = ("pune", "noida")
TIER1_LOCATIONS = (
    "delhi",
    "gurgaon",
    "mumbai",
    "hyderabad",
    "bangalore",
    "bengaluru",
)

SKILL_WEIGHTS = {
    "information retrieval": 1.25,
    "recommendation systems": 1.25,
    "search ranking": 1.20,
    "embeddings": 1.15,
    "vector databases": 1.10,
    "faiss": 1.05,
    "milvus": 1.05,
    "pinecone": 1.05,
    "qdrant": 1.05,
    "weaviate": 1.05,
    "opensearch": 1.00,
    "elasticsearch": 1.00,
    "nlp": 0.95,
    "llms": 0.90,
    "fine-tuning llms": 0.85,
    "hugging face transformers": 0.85,
    "lora": 0.75,
    "mlops": 0.80,
    "model serving": 0.80,
    "kubeflow": 0.75,
    "bentoml": 0.70,
    "python": 0.65,
    "spark": 0.45,
    "kafka": 0.45,
    "airflow": 0.40,
    "data pipelines": 0.40,
    "sql": 0.35,
}

PROFICIENCY_SCORES = {
    "beginner": 0.20,
    "intermediate": 0.50,
    "advanced": 0.80,
    "expert": 1.00,
}

COMPONENT_COLUMNS = tuple(BASE_WEIGHTS) + (
    "base_relevance",
    "behavioral_modifier",
    "score",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL candidate records for local scripts and tests."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def prepare_candidate_frame(candidates: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw candidate dicts into one row per candidate."""
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        salary = signals.get("expected_salary_range_inr_lpa", {}) or {}

        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "headline": profile.get("headline", ""),
                "summary": profile.get("summary", ""),
                "location": profile.get("location", ""),
                "country": profile.get("country", ""),
                "years_of_experience": profile.get("years_of_experience", 0.0),
                "current_title": profile.get("current_title", ""),
                "current_company": profile.get("current_company", ""),
                "current_industry": profile.get("current_industry", ""),
                "career_history": candidate.get("career_history", []) or [],
                "skills": candidate.get("skills", []) or [],
                "profile_completeness_score": signals.get(
                    "profile_completeness_score", 0.0
                ),
                "signup_date": signals.get("signup_date"),
                "last_active_date": signals.get("last_active_date"),
                "open_to_work_flag": signals.get("open_to_work_flag", False),
                "profile_views_received_30d": signals.get(
                    "profile_views_received_30d", 0
                ),
                "applications_submitted_30d": signals.get(
                    "applications_submitted_30d", 0
                ),
                "recruiter_response_rate": signals.get(
                    "recruiter_response_rate", 0.0
                ),
                "avg_response_time_hours": signals.get(
                    "avg_response_time_hours", 0.0
                ),
                "skill_assessment_scores": signals.get(
                    "skill_assessment_scores", {}
                )
                or {},
                "connection_count": signals.get("connection_count", 0),
                "endorsements_received": signals.get("endorsements_received", 0),
                "notice_period_days": signals.get("notice_period_days", 180),
                "expected_salary_min_lpa": salary.get("min", 0.0),
                "expected_salary_max_lpa": salary.get("max", 0.0),
                "preferred_work_mode": signals.get("preferred_work_mode", ""),
                "willing_to_relocate": signals.get("willing_to_relocate", False),
                "github_activity_score": signals.get("github_activity_score", -1),
                "search_appearance_30d": signals.get("search_appearance_30d", 0),
                "saved_by_recruiters_30d": signals.get(
                    "saved_by_recruiters_30d", 0
                ),
                "interview_completion_rate": signals.get(
                    "interview_completion_rate", 0.0
                ),
                "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1.0),
                "verified_email": signals.get("verified_email", False),
                "verified_phone": signals.get("verified_phone", False),
                "linkedin_connected": signals.get("linkedin_connected", False),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    history_features = _career_history_features(df)
    skill_features = _trusted_skill_features(df)

    return df.join(history_features).join(skill_features)


def score_candidates(
    candidates: Iterable[dict[str, Any]],
    gemini_features: dict[str, Any],
    include_components: bool = True,
) -> pd.DataFrame:
    """Score raw candidate dicts using pre-computed offline features."""
    frame = prepare_candidate_frame(candidates)
    return score_frame(
        frame, gemini_features=gemini_features, include_components=include_components
    )


def score_frame(
    df: pd.DataFrame, gemini_features: dict[str, Any], include_components: bool = True
) -> pd.DataFrame:
    """Compute approved Redrob ranking scores with Gemini inputs."""
    if df.empty:
        columns = ["candidate_id", "score"]
        if include_components:
            columns.extend(col for col in COMPONENT_COLUMNS if col != "score")
        return pd.DataFrame(columns=columns)

    work = _ensure_feature_columns(df.copy())

    local_prod = _production_shipping_score(work)
    local_eval = _eval_framework_score(work)
    work["production_shipping"] = _offline_feature_score(
        work["candidate_id"],
        gemini_features,
        feature_name="production_ml_depth",
        fallback=local_prod,
    )
    work["eval_framework"] = _offline_feature_score(
        work["candidate_id"],
        gemini_features,
        feature_name="eval_framework",
        fallback=local_eval,
    )

    work["career_domain"] = _career_domain_score(work)
    work["title_fit"] = _title_fit_score(work)
    work["trusted_skills"] = _series_01(work["trusted_skills"])
    work["experience_fit"] = _experience_fit_score(work)
    work["product_company"] = _product_company_score(work)
    work["location_fit"] = _location_fit_score(work)

    title_lower = _lower(work["current_title"]).str.strip()
    stuffer_mask = title_lower.isin(LOW_FIT_TITLES)
    work.loc[stuffer_mask, "career_domain"] *= 0.10
    work.loc[stuffer_mask, "trusted_skills"] *= 0.10

    base = np.zeros(len(work), dtype=float)
    for component, weight in BASE_WEIGHTS.items():
        base += weight * _series_01(work[component]).to_numpy(dtype=float)

    work["base_relevance"] = np.clip(base, 0.0, 1.0)
    work["behavioral_modifier"] = _behavioral_modifier(work)
    work["score"] = np.clip(
        work["base_relevance"].to_numpy(dtype=float)
        * work["behavioral_modifier"].to_numpy(dtype=float),
        0.0,
        MAX_BEHAVIORAL_MODIFIER,
    )

    output_cols = ["candidate_id", "score"]
    if include_components:
        output_cols.extend(col for col in COMPONENT_COLUMNS if col != "score")

    return work[output_cols].copy()


def _offline_feature_score(
    candidate_ids: pd.Series,
    gemini_features: dict[str, Any],
    feature_name: str,
    fallback: pd.Series,
) -> pd.Series:
    if not gemini_features:
        return _series_01(fallback)

    feature_map = {
        candidate_id: feature_values.get(feature_name)
        for candidate_id, feature_values in gemini_features.items()
        if isinstance(feature_values, dict) and feature_name in feature_values
    }
    mapped = candidate_ids.map(feature_map)
    values = pd.to_numeric(mapped, errors="coerce").combine_first(fallback)
    return _series_01(values)


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "career_text": "",
        "history_product_company_count": 0,
        "trusted_skills": 0.0,
        "headline": "",
        "summary": "",
        "location": "",
        "country": "",
        "years_of_experience": 0.0,
        "current_title": "",
        "current_company": "",
        "current_industry": "",
        "recruiter_response_rate": 0.0,
        "last_active_date": None,
        "open_to_work_flag": False,
        "profile_views_received_30d": 0,
        "applications_submitted_30d": 0,
        "avg_response_time_hours": 168.0,
        "notice_period_days": 180,
        "github_activity_score": -1,
        "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0,
        "interview_completion_rate": 0.0,
        "offer_acceptance_rate": -1.0,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
        "profile_completeness_score": 0.0,
        "willing_to_relocate": False,
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    return df


def _career_history_features(df: pd.DataFrame) -> pd.DataFrame:
    history = _explode_dict_column(df, "career_history")
    output = pd.DataFrame(index=df.index)

    if history.empty:
        output["career_text"] = ""
        output["history_product_company_count"] = 0
        return output

    for column in ("company", "title", "industry", "description"):
        if column not in history.columns:
            history[column] = ""

    text_parts = history[["company", "title", "industry", "description"]].fillna("")
    career_text = text_parts.agg(" ".join, axis=1).groupby(level=0).agg(" ".join)
    company_lower = _lower(history["company"])
    product_counts = company_lower.isin(PRODUCT_COMPANIES).groupby(level=0).sum()

    output["career_text"] = career_text.reindex(df.index, fill_value="")
    output["history_product_company_count"] = (
        product_counts.reindex(df.index, fill_value=0).astype(float)
    )
    return output


def _trusted_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    skills = _explode_dict_column(df, "skills")
    output = pd.DataFrame(index=df.index)

    if skills.empty:
        output["trusted_skills"] = 0.0
        return output

    for column in ("name", "proficiency", "endorsements", "duration_months"):
        if column not in skills.columns:
            skills[column] = 0 if column in {"endorsements", "duration_months"} else ""

    skill_names = _lower(skills["name"])
    skill_weight = skill_names.map(SKILL_WEIGHTS).fillna(0.0).astype(float)

    proficiency = (
        _lower(skills["proficiency"]).map(PROFICIENCY_SCORES).fillna(0.0).astype(float)
    )
    duration = np.clip(
        pd.to_numeric(skills["duration_months"], errors="coerce").fillna(0.0)
        / 48.0,
        0.0,
        1.0,
    )
    endorsements = np.clip(
        np.log1p(
            pd.to_numeric(skills["endorsements"], errors="coerce").fillna(0.0)
        )
        / math.log1p(60.0),
        0.0,
        1.0,
    )

    per_skill = skill_weight * (
        0.45 * proficiency + 0.35 * duration + 0.20 * endorsements
    )
    score = np.clip(per_skill.groupby(level=0).sum() / 5.0, 0.0, 1.0)

    output["trusted_skills"] = score.reindex(df.index, fill_value=0.0).astype(float)
    return output


def _explode_dict_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns or df.empty:
        return pd.DataFrame(index=pd.Index([], dtype=df.index.dtype))

    exploded = df[[column]].explode(column)
    valid = exploded[column].notna()

    if not bool(valid.any()):
        return pd.DataFrame(index=pd.Index([], dtype=df.index.dtype))

    records = exploded.loc[valid, column].tolist()
    normalized = pd.json_normalize(records)
    normalized.index = exploded.loc[valid].index
    return normalized


def _career_domain_score(df: pd.DataFrame) -> pd.Series:
    text = _combined_text(df, ("headline", "summary", "career_text"))
    core = _term_ratio(text, CORE_DOMAIN_TERMS, cap=4)
    vector = _term_ratio(text, VECTOR_TERMS, cap=3)
    ai_systems = _term_ratio(text, AI_SYSTEM_TERMS, cap=4)
    return _clip_series(0.50 * core + 0.30 * vector + 0.20 * ai_systems)


def _title_fit_score(df: pd.DataFrame) -> pd.Series:
    title = _lower(df["current_title"]).str.strip()
    score = np.full(len(df), 0.25, dtype=float)

    score[title.isin(LOW_FIT_TITLES).to_numpy()] = 0.05
    score[title.str.contains(r"\bresearch\b", na=False).to_numpy()] = 0.45
    score[title.isin(ADJACENT_TITLES).to_numpy()] = 0.55
    score[title.isin(GOOD_TITLES).to_numpy()] = 0.85
    score[title.isin(STRONG_TITLES).to_numpy()] = 1.00

    return pd.Series(score, index=df.index)


def _production_shipping_score(df: pd.DataFrame) -> pd.Series:
    text = _combined_text(df, ("headline", "summary", "career_text"))
    return _term_ratio(text, PRODUCTION_TERMS, cap=5)


def _eval_framework_score(df: pd.DataFrame) -> pd.Series:
    text = _combined_text(df, ("headline", "summary", "career_text"))
    return _term_ratio(text, EVAL_TERMS, cap=3)


def _experience_fit_score(df: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(df["years_of_experience"], errors="coerce").fillna(0.0)
    # Peak at 7 years, still tolerant of strong candidates just outside 5-9.
    score = np.exp(-((years.to_numpy(dtype=float) - 7.0) ** 2) / (2 * 2.2**2))
    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)


def _product_company_score(df: pd.DataFrame) -> pd.Series:
    current_company_match = _lower(df["current_company"]).isin(PRODUCT_COMPANIES)
    industry_match = _lower(df["current_industry"]).isin(PRODUCT_INDUSTRIES)
    history_count = pd.to_numeric(
        df["history_product_company_count"], errors="coerce"
    ).fillna(0.0)

    score = (
        np.minimum(history_count.to_numpy(dtype=float), 2.0) / 2.0
        + 0.25 * current_company_match.to_numpy(dtype=float)
        + 0.15 * industry_match.to_numpy(dtype=float)
    )
    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)


def _location_fit_score(df: pd.DataFrame) -> pd.Series:
    location = _lower(df["location"])
    country = _lower(df["country"])
    relocate = _bool_series(df["willing_to_relocate"])

    preferred = _contains_any(location, PREFERRED_LOCATIONS)
    tier1 = _contains_any(location, TIER1_LOCATIONS)
    india = country.eq("india")

    score = np.select(
        [
            preferred,
            tier1,
            india & relocate,
            india,
            relocate,
        ],
        [1.00, 0.90, 0.72, 0.58, 0.42],
        default=0.25,
    )
    return pd.Series(score, index=df.index)


def _behavioral_modifier(df: pd.DataFrame) -> pd.Series:
    response = _series_01(df["recruiter_response_rate"])
    recency = _activity_recency_score(df["last_active_date"])
    notice = 1.0 - np.clip(
        pd.to_numeric(df["notice_period_days"], errors="coerce").fillna(180.0)
        / 180.0,
        0.0,
        1.0,
    )
    open_to_work = _bool_series(df["open_to_work_flag"]).astype(float)
    interview = _series_01(df["interview_completion_rate"])
    offer = pd.to_numeric(df["offer_acceptance_rate"], errors="coerce").fillna(-1.0)
    offer = pd.Series(np.where(offer < 0, 0.50, offer), index=df.index)

    recruiter_interest = _recruiter_interest_score(df)
    verification = (
        0.45 * _bool_series(df["verified_email"]).astype(float)
        + 0.35 * _bool_series(df["verified_phone"]).astype(float)
        + 0.20 * _bool_series(df["linkedin_connected"]).astype(float)
    )
    response_speed = 1.0 - np.clip(
        pd.to_numeric(df["avg_response_time_hours"], errors="coerce").fillna(168.0)
        / 168.0,
        0.0,
        1.0,
    )
    github = pd.to_numeric(df["github_activity_score"], errors="coerce").fillna(-1.0)
    github = pd.Series(np.where(github < 0, 0.20, np.clip(github / 100.0, 0, 1)), index=df.index)
    completeness = np.clip(
        pd.to_numeric(df["profile_completeness_score"], errors="coerce").fillna(0.0)
        / 100.0,
        0.0,
        1.0,
    )

    availability = (
        0.22 * response
        + 0.16 * recency
        + 0.13 * notice
        + 0.10 * open_to_work
        + 0.09 * interview
        + 0.07 * offer
        + 0.07 * recruiter_interest
        + 0.06 * verification
        + 0.04 * response_speed
        + 0.03 * github
        + 0.03 * completeness
    )
    modifier = MIN_BEHAVIORAL_MODIFIER + (
        MAX_BEHAVIORAL_MODIFIER - MIN_BEHAVIORAL_MODIFIER
    ) * _series_01(availability)
    return pd.Series(
        np.clip(modifier, MIN_BEHAVIORAL_MODIFIER, MAX_BEHAVIORAL_MODIFIER),
        index=df.index,
    )


def _activity_recency_score(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.date
    days = parsed.map(
        lambda value: (REFERENCE_DATE - value).days if pd.notna(value) else 365
    )
    days = pd.to_numeric(days, errors="coerce").fillna(365)

    score = np.select(
        [days <= 30, days <= 90, days <= 180],
        [1.00, 0.82, 0.60],
        default=0.35,
    )
    return pd.Series(score, index=values.index)


def _recruiter_interest_score(df: pd.DataFrame) -> pd.Series:
    saved = _log_cap(df["saved_by_recruiters_30d"], cap=12)
    search = _log_cap(df["search_appearance_30d"], cap=400)
    views = _log_cap(df["profile_views_received_30d"], cap=80)
    applications = _log_cap(df["applications_submitted_30d"], cap=20)
    return _clip_series(0.35 * saved + 0.25 * search + 0.25 * views + 0.15 * applications)


def _log_cap(values: pd.Series, cap: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    score = np.log1p(np.maximum(numeric.to_numpy(dtype=float), 0.0)) / math.log1p(cap)
    return pd.Series(np.clip(score, 0.0, 1.0), index=values.index)


def _combined_text(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    parts = []
    for column in columns:
        if column in df.columns:
            parts.append(df[column].fillna("").astype(str))
    if not parts:
        return pd.Series("", index=df.index)

    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " " + part
    return _lower(combined)


def _term_ratio(series: pd.Series, terms: tuple[str, ...], cap: int) -> pd.Series:
    text = _lower(series)
    counts = text.str.count(_terms_pattern(terms)).to_numpy(dtype=float)
    return pd.Series(np.clip(counts / float(cap), 0.0, 1.0), index=series.index)


def _term_pattern(term: str) -> str:
    escaped = re.escape(term.lower())
    if re.fullmatch(r"[a-z0-9 ]+", term.lower()):
        return rf"\b{escaped}\b"
    return escaped


@lru_cache(maxsize=None)
def _terms_pattern(terms: tuple[str, ...]) -> str:
    ordered = sorted({term.lower() for term in terms}, key=len, reverse=True)
    return "|".join(f"(?:{_term_pattern(term)})" for term in ordered)


def _contains_any(series: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    text = _lower(series)
    mask = text.str.contains(_terms_pattern(terms), regex=True, na=False).to_numpy()
    return pd.Series(mask, index=series.index)


def _lower(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _series_01(series: pd.Series | np.ndarray) -> pd.Series:
    values = pd.Series(series) if not isinstance(series, pd.Series) else series
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return pd.Series(np.clip(numeric.to_numpy(dtype=float), 0.0, 1.0), index=values.index)


def _clip_series(series: pd.Series | np.ndarray) -> pd.Series:
    values = pd.Series(series) if not isinstance(series, pd.Series) else series
    return pd.Series(np.clip(values.to_numpy(dtype=float), 0.0, 1.0), index=values.index)
