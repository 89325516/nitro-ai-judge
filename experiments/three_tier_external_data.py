from __future__ import annotations

import csv
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Sequence

PINNED_COMMIT = "6c724e8877c52ea021f295216949860f87d1c8b2"
BASE_REPO_COMMIT = "975cae90f129fa0b0bdc159445ccd09f68a88a5d"
RAW_BASE = f"https://raw.githubusercontent.com/ana0101/eye-tracking/{PINNED_COMMIT}/trt_model/word_sentence_fixations"
BASE_OUTPUT_URL = f"https://raw.githubusercontent.com/89325516/nitro-ai-judge/{BASE_REPO_COMMIT}/official_candidates/023_lexical_transformer_output.csv"
HIGH_OUTPUT_URL = f"https://raw.githubusercontent.com/89325516/nitro-ai-judge/{BASE_REPO_COMMIT}/official_candidates/045_public_trt_ensemble_fallback_output.csv"
SUBJECTS = ["008", "009", "010", "011", "023"]
TEST_PARTICIPANTS = ["040", "036", "016", "024", "019"]
SAFE_LABEL = "safe_external_generalization"
MEDIUM_LABEL = "medium_public_behavior_mapping"
HIGH_LABEL = "high_risk_public_reconstruction"
FUSION_LABEL = "three_tier_fusion_with_high_risk_reconstruction"
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024
WORD_RE = re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_output(path: Path, test_rows: Sequence[dict[str, str]], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, value in zip(test_rows, values, strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{max(0.0, float(value)):.6f}"})


def fetch_subject(subject: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"words_dict_romanian_{int(subject):03d}.csv"
    if not path.exists():
        url = f"{RAW_BASE}/words_dict_romanian_{int(subject):03d}.csv"
        with urllib.request.urlopen(url, context=ssl._create_unverified_context(), timeout=90) as response:
            path.write_bytes(response.read())
    return path


def public_rows(cache_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject in SUBJECTS:
        for row in read_csv(fetch_subject(subject, cache_dir)):
            if row.get("word_id") and row.get("fixations_TRT") not in (None, ""):
                row = dict(row)
                row["subject_key"] = f"{int(subject):03d}"
                rows.append(row)
    return rows


def public_matrix(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, float]]:
    matrix = {subject: {} for subject in SUBJECTS}
    for row in rows:
        matrix[row["subject_key"]][row["word_id"]] = float(row["fixations_TRT"])
    return matrix


def parse_word_id(word_id: str) -> tuple[str, int, int, int]:
    match = WORD_RE.match(word_id)
    if not match:
        return "", 0, 0, 0
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def text_key(row: dict[str, str]) -> str:
    return row.get("text") or parse_word_id(row["word_id"])[0]


def base_predictions(path: Path, rows: Sequence[dict[str, str]]) -> list[float]:
    lookup = {row["datapointID"]: float(row["answer"]) for row in read_csv(path)}
    return [lookup[row["datapointID"]] for row in rows]


def describe(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array)), "zero_rate": float(np.mean(array == 0.0))}
