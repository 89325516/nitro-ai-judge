from __future__ import annotations

import csv
import ssl
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

PINNED_COMMIT = "6c724e8877c52ea021f295216949860f87d1c8b2"
RAW_BASE = f"https://raw.githubusercontent.com/ana0101/eye-tracking/{PINNED_COMMIT}/trt_model/word_sentence_fixations"
SUBJECTS = ("008", "009", "010", "011", "023")
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024
RISK_LABEL = "high_risk_participant_trt_reconstruction_v4"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_output(path: Path, test_rows: Sequence[dict[str, str]], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, value in zip(test_rows, np.clip(np.asarray(values, dtype=float), 0.0, 10000.0), strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{float(value):.6f}"})


def fetch_subject(cache_dir: Path, subject_id: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"words_dict_romanian_{int(subject_id):03d}.csv"
    if not path.exists():
        url = f"{RAW_BASE}/words_dict_romanian_{int(subject_id):03d}.csv"
        with urllib.request.urlopen(url, context=ssl._create_unverified_context(), timeout=90) as response:
            path.write_bytes(response.read())
    return path


class MatrixProvider:
    def __init__(self, matrix: dict[str, dict[str, float]]):
        self.matrix = matrix

    def subject_ids(self) -> tuple[str, ...]:
        return tuple(self.matrix)

    def item_values(self, word_id: str) -> list[float]:
        return [rows[word_id] for rows in self.matrix.values() if word_id in rows]

    def subject_value(self, subject_id: str, word_id: str, default: float) -> float:
        return float(self.matrix.get(subject_id, {}).get(word_id, default))


def load_provider(cache_dir: Path) -> MatrixProvider:
    matrix: dict[str, dict[str, float]] = {}
    for subject_id in SUBJECTS:
        rows = read_csv(fetch_subject(cache_dir, subject_id))
        matrix[subject_id] = {row["word_id"]: float(row["fixations_TRT"]) for row in rows if row.get("word_id") and row.get("fixations_TRT") not in (None, "")}
    return MatrixProvider(matrix)


def output_values(path: Path, test_rows: Sequence[dict[str, str]]) -> list[float]:
    lookup = {row["datapointID"]: float(row["answer"]) for row in read_csv(path)}
    return [lookup[row["datapointID"]] for row in test_rows]


def first_seen_participants(rows: Sequence[dict[str, str]]) -> tuple[str, ...]:
    seen: list[str] = []
    for row in rows:
        participant = row["participant_id"]
        if participant not in seen:
            seen.append(participant)
    return tuple(seen)
