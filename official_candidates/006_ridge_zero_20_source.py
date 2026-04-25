from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

FeatureMap = dict[str, float]
CONFIG = {'zero_threshold': 100.76579, 'scale': 1.0, 'hypothesis': 'Set bottom 20 percent predictions to zero.'}


def transform_predictions(test_rows: list[dict[str, str]], predictions: np.ndarray) -> np.ndarray:
    output = []
    for row, prediction in zip(test_rows, predictions, strict=True):
        value = float(prediction)
        if value <= CONFIG.get("zero_threshold", -1.0):
            output.append(0.0)
            continue
        scale = CONFIG.get("scale", 1.0)
        if row["text"] == CONFIG.get("text"):
            scale *= CONFIG.get("factor", 1.0)
        if row["participant_id"] == CONFIG.get("participant_id"):
            scale *= CONFIG.get("factor", 1.0)
        output.append(max(0.0, value * scale))
    return np.clip(np.asarray(output, dtype=float), 0.0, 10000.0)

POSITION_PATTERN = re.compile(r"_page_(\d+)_(\d+)$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_position(word_id: str) -> tuple[int, int]:
    match = POSITION_PATTERN.search(word_id)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def text_family(text: str) -> str:
    return text.split("_", 1)[0]


def has_accent(word: str) -> bool:
    normalized = unicodedata.normalize("NFD", word)
    return any(unicodedata.category(char) == "Mn" for char in normalized)


def build_features(row: dict[str, str]) -> FeatureMap:
    word = row["word"]
    lower_word = word.lower()
    page, index = parse_position(row["word_id"])
    alpha_count = sum(char.isalpha() for char in word)
    digit_count = sum(char.isdigit() for char in word)

    return {
        "bias": 1.0,
        "word=" + lower_word: 1.0,
        "family=" + text_family(row["text"]): 1.0,
        "length": float(len(word)),
        "log_length": math.log1p(len(word)),
        "alpha_count": float(alpha_count),
        "digit_count": float(digit_count),
        "page": float(page),
        "index": float(index),
        "log_index": math.log1p(index),
        "is_url": float(word.startswith("http")),
        "is_capitalized": float(word[:1].isupper()),
        "is_upper": float(word.isupper()),
        "is_punctuation_only": float(bool(word) and all(not char.isalnum() for char in word)),
        "has_accent": float(has_accent(word)),
    }


def train_model(features: Iterable[FeatureMap], targets: np.ndarray):
    model = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=100.0, random_state=0))
    return model.fit(list(features), targets)


def write_submission(test_rows: list[dict[str, str]], predictions: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, prediction in zip(test_rows, predictions, strict=True):
            writer.writerow(
                {
                    "subtaskID": "1",
                    "datapointID": row["datapointID"],
                    "answer": f"{max(0.0, float(prediction)):.6f}",
                }
            )


def generate_submission(train_path: Path, test_path: Path, output_path: Path) -> None:
    train_rows = read_rows(train_path)
    test_rows = read_rows(test_path)
    train_features = [build_features(row) for row in train_rows]
    test_features = [build_features(row) for row in test_rows]
    targets = np.array([float(row["answer"]) for row in train_rows], dtype=float)

    model = train_model(train_features, targets)
    predictions = transform_predictions(test_rows, model.predict(test_features))
    write_submission(test_rows, predictions, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Nitro AI Judge submission CSV.")
    parser.add_argument("--train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_submission(args.train, args.test, args.output)


if __name__ == "__main__":
    main()
