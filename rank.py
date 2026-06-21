from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from defenses import filter_candidates
from reasoning import generate_candidate_reasoning
from scoring import prepare_candidate_frame, score_frame


REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
TOP_K = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Redrob candidates using local features only."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz.",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to pre-computed offline gemini_features.json.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def read_candidates(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    candidates: list[dict[str, Any]] = []

    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return candidates


def load_gemini_features(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return _feature_list_to_map(pd.read_parquet(path).to_dict("records"))

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        if "candidates" in data and isinstance(data["candidates"], list):
            return _feature_list_to_map(data["candidates"])
        return data

    if isinstance(data, list):
        return _feature_list_to_map(data)

    raise ValueError("Feature file must be a dict or a list of candidate feature rows.")


def _feature_list_to_map(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if candidate_id:
            mapped[str(candidate_id)] = row
    return mapped


def build_submission(
    candidates: list[dict[str, Any]], gemini_features: dict[str, Any]
) -> pd.DataFrame:
    kept_candidates, dropped = filter_candidates(candidates)
    if len(kept_candidates) < TOP_K:
        raise ValueError(
            f"Only {len(kept_candidates)} candidates remain after filters; need {TOP_K}."
        )

    candidate_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in kept_candidates
    }

    candidate_frame = prepare_candidate_frame(kept_candidates)
    scored = score_frame(
        candidate_frame, gemini_features=gemini_features, include_components=False
    )
    scored["score"] = pd.to_numeric(scored["score"], errors="coerce").fillna(0.0)
    scored["score"] = scored["score"].round(6)

    df_final = scored.sort_values(
        by=["score", "candidate_id"], ascending=[False, True]
    ).head(TOP_K)
    df_final = df_final.reset_index(drop=True)
    df_final["rank"] = df_final.index + 1
    df_final["reasoning"] = df_final["candidate_id"].map(
        lambda cid: generate_candidate_reasoning(
            str(cid), candidate_by_id[str(cid)], gemini_features
        )
    )

    return df_final[REQUIRED_COLUMNS]


def write_submission(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    candidates = read_candidates(args.candidates)
    gemini_features = load_gemini_features(args.features)
    submission = build_submission(candidates, gemini_features)
    write_submission(submission, args.out)


if __name__ == "__main__":
    main()
