from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

STRIP_CHARS = ".,;:!?()[]{}\"'-–—«»„“”"
WORD_ID_PATTERN = re.compile(r"^(.*)_(\d+)_page_(\d+)_(\d+)$")
FAMILIES = ["arg", "enc", "ins", "lit", "popsci"]
ROMANIAN_MARKS = set("ăâîșțĂÂÎȘȚ")
VOWELS = set("aeiouăâîAEIOUĂÂÎ")

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

def clean_word(word: str) -> str:
    return word.strip(STRIP_CHARS)

def family(text: str) -> str:
    return text.split("_", 1)[0]

def parse_word_id(word_id: str) -> tuple[int, int, int]:
    match = WORD_ID_PATTERN.match(word_id)
    return (int(match.group(2)), int(match.group(3)), int(match.group(4))) if match else (0, 0, 0)

def build_context(rows: Sequence[dict[str, str]]) -> list[tuple[str, str, str, str, int, int]]:
    parsed = [parse_word_id(row["word_id"]) for row in rows]
    by_page: dict[tuple[str, int], list[int]] = defaultdict(list)
    text_counts = Counter(row["text"] for row in rows)
    for index, row in enumerate(rows):
        by_page[(row["text"], parsed[index][1])].append(index)
    context = [("", "", "", "", 1, text_counts[row["text"]]) for row in rows]
    for page_indexes in by_page.values():
        page_indexes.sort(key=lambda item: parsed[item][2])
        page_size = len(page_indexes)
        for offset, row_index in enumerate(page_indexes):
            words = []
            for step in (-2, -1, 1, 2):
                neighbor = offset + step
                words.append(rows[page_indexes[neighbor]]["word"] if 0 <= neighbor < page_size else "")
            context[row_index] = (*words, page_size, text_counts[rows[row_index]["text"]])
    return context

def pack_values(values: Sequence[float], global_mean: float, global_positive: float) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    positive = array[array > 0]
    count = len(array)
    return (
        float((np.sum(array) + 20.0 * global_mean) / (count + 20.0)),
        float(np.median(array)),
        float(np.mean(array == 0.0)),
        math.log1p(count),
        float(np.mean(positive)) if len(positive) else global_positive,
    )

def build_stats(rows: Sequence[dict[str, str]], targets: np.ndarray) -> dict:
    global_mean = float(np.mean(targets))
    positive = targets[targets > 0]
    global_positive = float(np.mean(positive)) if len(positive) else global_mean
    buckets: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in ("word", "prefix", "suffix", "family")}
    for row, target in zip(rows, targets, strict=True):
        lower = clean_word(row["word"]).lower()
        for name, key in (("word", lower), ("prefix", lower[:3]), ("suffix", lower[-3:]), ("family", family(row["text"]))):
            buckets[name][key].append(float(target))
    packed = {"global": (global_mean, global_positive)}
    packed.update({name: {key: pack_values(values, global_mean, global_positive) for key, values in bucket.items()} for name, bucket in buckets.items()})
    return packed

def sparse_features(row: dict[str, str], context: tuple[str, str, str, str, int, int]) -> dict[str, float]:
    word = row["word"]
    lower = word.lower()
    _, page, index = parse_word_id(row["word_id"])
    alpha_count = sum(char.isalpha() for char in word)
    digit_count = sum(char.isdigit() for char in word)
    return {
        "bias": 1.0, "word=" + lower: 1.0, "family=" + family(row["text"]): 1.0, "length": float(len(word)),
        "log_length": math.log1p(len(word)), "alpha_count": float(alpha_count), "digit_count": float(digit_count),
        "page": float(page), "index": float(index), "log_index": math.log1p(index),
        "is_url": float(word.startswith("http") or "www." in word), "is_capitalized": float(word[:1].isupper()),
        "is_upper": float(word.isupper()), "is_punctuation_only": float(bool(word) and all(not char.isalnum() for char in word)),
        "has_mark": float(any(char in ROMANIAN_MARKS for char in word)),
    }

def numeric_features(row: dict[str, str], context: tuple[str, str, str, str, int, int], stats: dict) -> list[float]:
    word = row["word"]
    lower = clean_word(word).lower()
    source, page, index = parse_word_id(row["word_id"])
    page_size, text_size = context[4], context[5]
    values = [
        1.0, float(len(word)), math.log1p(len(word)), float(sum(char.isalpha() for char in word)),
        float(sum(char.isdigit() for char in word)), float(sum(char in VOWELS for char in word)),
        float(sum(char in ROMANIAN_MARKS for char in word)), float(word[:1].isupper()), float(word.isupper()),
        float(word.startswith("http") or "www." in word), float("-" in word), float(source), math.log1p(source),
        float(page), float(index), math.log1p(index), float(page_size), float(text_size),
        index / (page_size + 1.0), index / (text_size + 1.0),
        float(FAMILIES.index(family(row["text"])) if family(row["text"]) in FAMILIES else -1),
    ]
    values.extend(float(len(clean_word(item))) for item in context[:4])
    default = (stats["global"][0], stats["global"][0], 0.0, 0.0, stats["global"][1])
    for name, key in (("word", lower), ("prefix", lower[:3]), ("suffix", lower[-3:]), ("family", family(row["text"]))):
        values.extend(stats[name].get(key, default))
    return values

def predict_parts(train_rows: list[dict[str, str]], targets: np.ndarray, test_rows: list[dict[str, str]]) -> np.ndarray:
    train_context, test_context = build_context(train_rows), build_context(test_rows)
    ridge = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=100.0, random_state=0))
    ridge.fit([sparse_features(row, ctx) for row, ctx in zip(train_rows, train_context, strict=True)], targets)
    ridge_pred = ridge.predict([sparse_features(row, ctx) for row, ctx in zip(test_rows, test_context, strict=True)])
    if len(train_rows) < 40:
        clipped = np.clip(ridge_pred, 0.0, 10000.0)
        return np.column_stack([clipped, clipped, clipped])
    stats = build_stats(train_rows, targets)
    numeric_train = np.asarray([numeric_features(row, ctx, stats) for row, ctx in zip(train_rows, train_context, strict=True)])
    numeric_test = np.asarray([numeric_features(row, ctx, stats) for row, ctx in zip(test_rows, test_context, strict=True)])
    regressor = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.1, random_state=11)
    regressor.fit(numeric_train, targets)
    raw_pred = regressor.predict(numeric_test)
    read_targets = (targets > 0.0).astype(int)
    if len(set(read_targets)) == 2:
        classifier = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.1, random_state=12)
        classifier.fit(numeric_train, read_targets)
        read_prob = classifier.predict_proba(numeric_test)[:, 1]
    else:
        read_prob = np.full(len(test_rows), float(read_targets[0]))
    positive_mask = targets > 0.0
    if not np.any(positive_mask):
        return np.column_stack([np.clip(ridge_pred, 0.0, 10000.0), np.clip(raw_pred, 0.0, 10000.0), np.zeros(len(test_rows))])
    positive_regressor = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.1, random_state=13)
    positive_regressor.fit(numeric_train[positive_mask], targets[positive_mask])
    hurdle_pred = read_prob * positive_regressor.predict(numeric_test)
    return np.column_stack([np.clip(ridge_pred, 0.0, 10000.0), np.clip(raw_pred, 0.0, 10000.0), np.clip(hurdle_pred, 0.0, 10000.0)])

def metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = max(0.0, 1.0 - float(np.sum((truth - prediction) ** 2)) / variance) if variance else 0.0
    pearson = 0.0 if np.std(prediction) == 0.0 else abs(float(np.corrcoef(truth, prediction)[0, 1]))
    return 50.0 * (r2 + (0.0 if math.isnan(pearson) else pearson))

def predict(train_rows: list[dict[str, str]], targets: np.ndarray, test_rows: list[dict[str, str]]) -> np.ndarray:
    texts = [row["text"] for row in train_rows]
    if len(set(texts)) < 2 or len(train_rows) < 80:
        return np.average(predict_parts(train_rows, targets, test_rows), axis=1, weights=[0.8, 0.1, 0.1])
    oof = np.zeros((len(train_rows), 3), dtype=float)
    for train_index, valid_index in GroupKFold(n_splits=min(3, len(set(texts)))).split(train_rows, targets, groups=texts):
        fold_train = [train_rows[index] for index in train_index]
        fold_valid = [train_rows[index] for index in valid_index]
        oof[valid_index] = predict_parts(fold_train, targets[train_index], fold_valid)
    stacker = Ridge(alpha=10.0, positive=True).fit(oof, targets)
    stacked_oof = np.clip(stacker.predict(oof), 0.0, 10000.0)
    best_index = int(np.argmax([metric(targets, oof[:, index]) for index in range(3)] + [metric(targets, stacked_oof)]))
    final_parts = predict_parts(train_rows, targets, test_rows)
    final_prediction = final_parts[:, best_index] if best_index < 3 else stacker.predict(final_parts)
    return np.clip(final_prediction, 0.0, 10000.0)

def write_output(rows: Sequence[dict[str, str]], predictions: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{max(0.0, float(prediction)):.6f}"})

def generate(train_path: Path, test_path: Path, output_path: Path) -> None:
    train_rows, test_rows = read_rows(train_path), read_rows(test_path)
    targets = np.asarray([float(row["answer"]) for row in train_rows], dtype=float)
    write_output(test_rows, predict(train_rows, targets, test_rows), output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Safe V6 local-score candidate.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.train, args.test, args.output)


if __name__ == "__main__":
    main()
