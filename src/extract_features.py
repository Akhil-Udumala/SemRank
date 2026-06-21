from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree

import pandas as pd


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_JD_PATH = Path("data/job_description.docx")
DEFAULT_CANDIDATES_PATH = Path("data/candidates.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/candidate_features.parquet")
DEFAULT_ENV_PATH = Path(".env")

FEATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "production_ml_depth": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Hands-on production ML deployment depth vs theoretical knowledge."
            ),
        },
        "eval_framework": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Experience with ranking/recommendation evaluation metrics such as "
                "NDCG, MRR, MAP, offline benchmarks, or A/B tests."
            ),
        },
        "reasoning_snippet": {
            "type": "string",
            "description": (
                "One or two factual sentences justifying the scores from the profile."
            ),
        },
    },
    "required": ["production_ml_depth", "eval_framework", "reasoning_snippet"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Gemini feature extraction for Redrob candidate ranking."
    )
    parser.add_argument("--job-description", type=Path, default=DEFAULT_JD_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--cooldown", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--backoff-max", type=float, default=90.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing output file instead of resuming it.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def resolve_api_key(args: argparse.Namespace) -> str | None:
    return (
        args.api_key
        or os.environ.get("GEMINI_API_KEY")
        or load_dotenv(args.env_file).get("GEMINI_API_KEY")
        or load_dotenv(args.env_file).get("GOOGLE_API_KEY")
    )


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value

    return values


def read_job_description(path: Path) -> str:
    path = resolve_job_description_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {path}")

    if path.suffix.lower() == ".docx":
        return read_docx_text(path)

    return path.read_text(encoding="utf-8")


def resolve_job_description_path(path: Path) -> Path:
    if path.exists() or path.suffix.lower() != ".docx":
        return path

    for suffix in (".txt", ".md"):
        fallback = path.with_suffix(suffix)
        if fallback.exists():
            logging.info("Using plain-text job description fallback: %s", fallback)
            return fallback

    return path


def read_docx_text(path: Path) -> str:
    paragraphs: list[str] = []
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml)
    for paragraph in root.findall(".//w:p", namespace):
        text_chunks = [
            node.text or "" for node in paragraph.findall(".//w:t", namespace)
        ]
        text = "".join(text_chunks).strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def iter_candidates(path: Path, offset: int = 0, limit: int | None = None) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    yielded = 0

    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number < offset or not line.strip():
                continue
            yield json.loads(line)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def load_existing_features(path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return {}

    if path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
        return {
            str(row["candidate_id"]): {
                "production_ml_depth": row.get("production_ml_depth", 0.0),
                "eval_framework": row.get("eval_framework", 0.0),
                "reasoning_snippet": row.get("reasoning_snippet", ""),
            }
            for row in rows
            if row.get("candidate_id")
        }

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return {
            str(row["candidate_id"]): row
            for row in data
            if isinstance(row, dict) and row.get("candidate_id")
        }

    if not isinstance(data, dict):
        raise ValueError(f"Existing feature file must be JSON object/list: {path}")

    return data


def save_features(path: Path, features: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate_id, values in sorted(features.items()):
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "production_ml_depth": clamp01(values.get("production_ml_depth", 0.0)),
                "eval_framework": clamp01(values.get("eval_framework", 0.0)),
                "reasoning_snippet": clean_reasoning(
                    values.get("reasoning_snippet", "")
                ),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "production_ml_depth",
            "eval_framework",
            "reasoning_snippet",
        ],
    )

    if path.suffix.lower() == ".parquet":
        tmp_path = path.with_suffix(".tmp.parquet")
        try:
            frame.to_parquet(tmp_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Writing Parquet requires `pyarrow` or `fastparquet`. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc
    else:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                {row["candidate_id"]: row for row in rows},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    tmp_path.replace(path)


def make_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install the official SDK with `pip install google-genai`."
        ) from exc

    return genai.Client(api_key=api_key)


def build_prompt(job_description: str, candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])

    career_lines = []
    for role in career_history:
        career_lines.append(
            "\n".join(
                [
                    f"Company: {role.get('company', '')}",
                    f"Title: {role.get('title', '')}",
                    f"Duration months: {role.get('duration_months', '')}",
                    f"Description: {role.get('description', '')}",
                ]
            )
        )

    return f"""
You are scoring a candidate for the target Redrob Senior AI Engineer role.

Return only JSON matching the configured schema. Use only the candidate facts
below. Do not infer employers, skills, metrics, deployments, or tools that are
not present. The reasoning_snippet must be 1-2 sentences and factual.

Target role:
{job_description[:6000]}

Candidate:
candidate_id: {candidate.get("candidate_id")}
current_title: {profile.get("current_title", "")}
headline: {profile.get("headline", "")}
summary: {profile.get("summary", "")}
years_of_experience: {profile.get("years_of_experience", "")}
current_company: {profile.get("current_company", "")}
current_industry: {profile.get("current_industry", "")}

Career history:
{chr(10).join(career_lines)[:8000]}
""".strip()


def call_gemini_with_backoff(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
) -> dict[str, Any]:
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FEATURE_SCHEMA,
        temperature=0.0,
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            payload = parse_model_json(response)
            return normalize_feature_payload(payload)
        except Exception as exc:  # noqa: BLE001 - SDK exceptions vary by version.
            if not is_rate_limit_error(exc) or attempt >= max_retries:
                raise

            delay = min(backoff_max, backoff_base ** attempt)
            delay += random.uniform(0.0, min(1.0, delay * 0.10))
            logging.warning(
                "Gemini rate limit hit; retrying in %.1fs (attempt %s/%s).",
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable Gemini retry state.")


def parse_model_json(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini response did not include JSON text.")

    return json.loads(text)


def normalize_feature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "production_ml_depth": clamp01(payload.get("production_ml_depth", 0.0)),
        "eval_framework": clamp01(payload.get("eval_framework", 0.0)),
        "reasoning_snippet": clean_reasoning(payload.get("reasoning_snippet", "")),
    }


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def clean_reasoning(value: Any) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    return text[:500]


def is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


async def extract_feature_for_candidate(
    client: Any,
    model: str,
    job_description: str,
    candidate: dict[str, Any],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any] | None]:
    candidate_id = str(candidate.get("candidate_id", ""))
    prompt = build_prompt(job_description, candidate)

    async with semaphore:
        try:
            payload = await asyncio.to_thread(
                call_gemini_with_backoff,
                client,
                model,
                prompt,
                args.max_retries,
                args.backoff_base,
                args.backoff_max,
            )
            if args.cooldown > 0:
                await asyncio.sleep(args.cooldown)
            return candidate_id, payload
        except Exception:
            logging.exception("Feature extraction failed for %s.", candidate_id)
            return candidate_id, None


async def process_batch(
    batch: list[dict[str, Any]],
    client: Any,
    model: str,
    job_description: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, dict[str, Any]]:
    tasks = [
        extract_feature_for_candidate(
            client=client,
            model=model,
            job_description=job_description,
            candidate=candidate,
            args=args,
            semaphore=semaphore,
        )
        for candidate in batch
    ]
    results: dict[str, dict[str, Any]] = {}

    for task in asyncio.as_completed(tasks):
        candidate_id, payload = await task
        if payload is not None:
            results[candidate_id] = payload

    return results


async def run_async(args: argparse.Namespace) -> None:
    api_key = resolve_api_key(args)
    if not api_key:
        raise ValueError(
            "Provide a Gemini key via --api-key, GEMINI_API_KEY, or .env."
        )

    job_description = read_job_description(args.job_description)
    features = load_existing_features(args.out, overwrite=args.overwrite)
    client = make_client(api_key)

    logging.info("Loaded %s existing feature rows.", len(features))
    processed = 0
    skipped = 0
    pending_batch: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    next_flush_at = max(1, args.flush_every)

    for candidate in iter_candidates(args.candidates, args.offset, args.limit):
        candidate_id = str(candidate.get("candidate_id", ""))
        if candidate_id in features:
            skipped += 1
            continue

        pending_batch.append(candidate)
        if len(pending_batch) < max(1, args.batch_size):
            continue

        new_features = await process_batch(
            pending_batch,
            client=client,
            model=args.model,
            job_description=job_description,
            args=args,
            semaphore=semaphore,
        )
        features.update(new_features)
        processed += len(new_features)
        pending_batch.clear()

        if processed >= next_flush_at:
            save_features(args.out, features)
            logging.info(
                "Saved %s total features (%s new, %s skipped).",
                len(features),
                processed,
                skipped,
            )
            next_flush_at = processed + max(1, args.flush_every)

    if pending_batch:
        new_features = await process_batch(
            pending_batch,
            client=client,
            model=args.model,
            job_description=job_description,
            args=args,
            semaphore=semaphore,
        )
        features.update(new_features)
        processed += len(new_features)

    save_features(args.out, features)
    logging.info(
        "Done. Saved %s total features to %s (%s new, %s skipped).",
        len(features),
        args.out,
        processed,
        skipped,
    )


def run(args: argparse.Namespace) -> None:
    asyncio.run(run_async(args))


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
