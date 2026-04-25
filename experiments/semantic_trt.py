from __future__ import annotations

import argparse
import csv
import math
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

try:
    from experiments.semantic_features import build_stats, numeric_row, page_context, parse_word_id, read_rows, sparse_row
except ModuleNotFoundError:
    from semantic_features import build_stats, numeric_row, page_context, parse_word_id, read_rows, sparse_row


def has_accent(word: str) -> bool:
    return any(unicodedata.category(char) == "Mn" for char in unicodedata.normalize("NFD", word))


def stable_sparse_row(row: dict[str, str]) -> dict[str, float]:
    word = row["word"]
    _, page, index = parse_word_id(row["word_id"])
    return {"bias": 1.0, "word=" + word.lower(): 1.0, "family=" + row["text"].split("_", 1)[0]: 1.0,
            "length": float(len(word)), "log_length": math.log1p(len(word)), "alpha_count": float(sum(c.isalpha() for c in word)),
            "digit_count": float(sum(c.isdigit() for c in word)), "page": float(page), "index": float(index),
            "log_index": math.log1p(index), "is_url": float(word.startswith("http")), "is_capitalized": float(word[:1].isupper()),
            "is_upper": float(word.isupper()), "is_punctuation_only": float(bool(word) and all(not c.isalnum() for c in word)),
            "has_accent": float(has_accent(word))}


def quantile_map(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    probs = np.empty(len(scores), dtype=float)
    probs[order] = np.linspace(0.0, 1.0, len(scores))
    return np.quantile(targets, probs)


def predict_views(train_rows: list[dict[str, str]], y: np.ndarray, test_rows: list[dict[str, str]]) -> np.ndarray:
    train_ctx, test_ctx = page_context(train_rows), page_context(test_rows)
    stable = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=100.0, random_state=0)).fit([stable_sparse_row(row) for row in train_rows], y)
    stable_pred = np.clip(stable.predict([stable_sparse_row(row) for row in test_rows]), 0.0, 10000.0)
    sparse_train = [sparse_row(row, ctx) for row, ctx in zip(train_rows, train_ctx, strict=True)]
    sparse_test = [sparse_row(row, ctx) for row, ctx in zip(test_rows, test_ctx, strict=True)]
    ridge = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=120.0, random_state=3)).fit(sparse_train, y)
    ridge_pred = np.clip(ridge.predict(sparse_test), 0.0, 10000.0)
    if len(train_rows) < 60:
        return np.column_stack([stable_pred, ridge_pred, ridge_pred, ridge_pred, ridge_pred])
    stats = build_stats(train_rows, y)
    x_train = np.asarray([numeric_row(row, ctx, stats) for row, ctx in zip(train_rows, train_ctx, strict=True)])
    x_test = np.asarray([numeric_row(row, ctx, stats) for row, ctx in zip(test_rows, test_ctx, strict=True)])
    raw = HistGradientBoostingRegressor(max_iter=420, learning_rate=0.032, max_leaf_nodes=31, l2_regularization=0.08, random_state=21).fit(x_train, y).predict(x_test)
    mask = y > 0
    classifier = HistGradientBoostingClassifier(max_iter=320, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.08, random_state=22).fit(x_train, mask.astype(int))
    positive = HistGradientBoostingRegressor(max_iter=420, learning_rate=0.032, max_leaf_nodes=31, l2_regularization=0.08, random_state=23).fit(x_train[mask], y[mask]).predict(x_test)
    hurdle = classifier.predict_proba(x_test)[:, 1] * positive
    rank_y = np.argsort(np.argsort(y)).astype(float) / max(1, len(y) - 1)
    rank_scores = HistGradientBoostingRegressor(max_iter=260, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.2, random_state=24).fit(x_train, rank_y).predict(x_test)
    rank = quantile_map(rank_scores, y)
    return np.column_stack([stable_pred, ridge_pred, np.clip(raw, 0, 10000), np.clip(hurdle, 0, 10000), np.clip(rank, 0, 10000)])


def metric(y: np.ndarray, pred: np.ndarray) -> float:
    variance = float(np.sum((y - y.mean()) ** 2))
    r2 = max(0.0, 1.0 - float(np.sum((y - pred) ** 2)) / variance) if variance else 0.0
    pearson = 0.0 if np.std(pred) == 0.0 else abs(float(np.corrcoef(y, pred)[0, 1]))
    return 50.0 * (r2 + (0.0 if math.isnan(pearson) else pearson))


def predict(train_rows: list[dict[str, str]], y: np.ndarray, test_rows: list[dict[str, str]]) -> np.ndarray:
    texts = [row["text"] for row in train_rows]
    if len(set(texts)) < 2 or len(train_rows) < 100:
        return np.mean(predict_views(train_rows, y, test_rows), axis=1)
    oof = np.zeros((len(train_rows), 5), dtype=float)
    for train_index, valid_index in GroupKFold(n_splits=min(3, len(set(texts)))).split(train_rows, y, groups=texts):
        fold_train = [train_rows[index] for index in train_index]
        fold_valid = [train_rows[index] for index in valid_index]
        oof[valid_index] = predict_views(fold_train, y[train_index], fold_valid)
    stacker = Ridge(alpha=20.0, positive=True).fit(oof, y)
    stacked = np.clip(stacker.predict(oof), 0.0, 10000.0)
    scores = [metric(y, oof[:, index]) for index in range(oof.shape[1])] + [metric(y, stacked)]
    final_views = predict_views(train_rows, y, test_rows)
    best_index = int(np.argmax(scores))
    prediction = final_views[:, best_index] if best_index < final_views.shape[1] else stacker.predict(final_views)
    return np.clip(prediction, 0.0, 10000.0)


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
    parser = argparse.ArgumentParser(description="Run the CSV-only semantic TRT candidate.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.train, args.test, args.output)


if __name__ == "__main__":
    main()
