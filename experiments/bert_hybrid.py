from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bert_two_head import predict_chunks, train_model
from experiments.bert_two_head_data import chunk_rows, read_rows, write_submission

POSITION_RE = re.compile(r"_page_(\d+)_(\d+)$")
TRANSFORMER_GATE = 38.99673098959958


def parse_position(word_id: str) -> tuple[int, int]:
    match = POSITION_RE.search(word_id)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def has_accent(word: str) -> bool:
    return any(unicodedata.category(char) == "Mn" for char in unicodedata.normalize("NFD", word))


def surface_features(row: dict[str, str]) -> dict[str, float]:
    word = row["word"]
    page, index = parse_position(row["word_id"])
    return {"bias": 1.0, "word=" + word.lower(): 1.0, "family=" + row["text"].split("_", 1)[0]: 1.0,
            "length": float(len(word)), "log_length": math.log1p(len(word)), "alpha_count": float(sum(char.isalpha() for char in word)),
            "digit_count": float(sum(char.isdigit() for char in word)), "page": float(page), "index": float(index),
            "log_index": math.log1p(index), "is_url": float(word.startswith("http")), "is_capitalized": float(word[:1].isupper()),
            "is_upper": float(word.isupper()), "is_punctuation_only": float(bool(word) and all(not char.isalnum() for char in word)),
            "has_accent": float(has_accent(word))}


def ridge_predict(train_rows: list[dict[str, str]], targets: np.ndarray, test_rows: list[dict[str, str]]) -> np.ndarray:
    model = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=100.0, random_state=0))
    model.fit([surface_features(row) for row in train_rows], targets)
    return np.clip(model.predict([surface_features(row) for row in test_rows]), 0.0, 10000.0)


def bert_predict(args: argparse.Namespace, train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> np.ndarray:
    train_chunks = chunk_rows(train_rows, True, args.chunk_words, args.max_train_chunks)
    test_chunks = chunk_rows(test_rows, False, args.chunk_words)
    raw = predict_chunks(args, train_model(args, train_chunks), test_chunks, len(test_rows))
    return np.clip(raw * args.bert_scale, 0.0, 10000.0)


def metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.sum((truth - truth.mean()) ** 2))
    r2 = max(0.0, 1.0 - float(np.sum((truth - prediction) ** 2)) / variance) if variance else 0.0
    pearson = 0.0 if np.std(prediction) == 0.0 else abs(float(np.corrcoef(truth, prediction)[0, 1]))
    return 50.0 * (r2 + (0.0 if math.isnan(pearson) else pearson))


def oof_predictions(args: argparse.Namespace, rows: list[dict[str, str]], targets: np.ndarray) -> np.ndarray:
    groups = [row["text"] for row in rows]
    folds = min(args.inner_folds, len(set(groups)))
    if folds < 2 or len(rows) < 20:
        base = ridge_predict(rows, targets, rows)
        return np.column_stack([base, base, np.zeros(len(rows))])
    oof = np.zeros((len(rows), 3), dtype=float)
    for train_index, valid_index in GroupKFold(n_splits=folds).split(rows, targets, groups=groups):
        fold_train = [rows[index] for index in train_index]
        fold_valid = [rows[index] for index in valid_index]
        base = ridge_predict(fold_train, targets[train_index], fold_valid)
        bert = bert_predict(args, fold_train, fold_valid)
        oof[valid_index] = np.column_stack([base, bert, bert - base])
    return oof


def fit_hybrid(args: argparse.Namespace, rows: list[dict[str, str]], targets: np.ndarray):
    oof = oof_predictions(args, rows, targets)
    stacker = Ridge(alpha=args.stack_alpha, positive=True).fit(oof, targets)
    stacked = np.clip(stacker.predict(oof), 0.0, 10000.0)
    base_score = metric(targets, oof[:, 0])
    stacked_score = metric(targets, stacked)
    use_stacker = stacked_score > base_score + args.min_gain
    return stacker, use_stacker, {"base_score": base_score, "stacked_score": stacked_score, "use_stacker": use_stacker}


def predict(args: argparse.Namespace, train_rows: list[dict[str, str]], targets: np.ndarray, test_rows: list[dict[str, str]]) -> tuple[np.ndarray, dict]:
    stacker, use_stacker, report = fit_hybrid(args, train_rows, targets)
    base = ridge_predict(train_rows, targets, test_rows)
    if not use_stacker:
        return base, report
    bert = bert_predict(args, train_rows, test_rows)
    features = np.column_stack([base, bert, bert - base])
    return np.clip(stacker.predict(features), 0.0, 10000.0), report


def write_json(path: Path | None, payload: dict) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    train_rows = read_rows(args.train)
    test_rows = read_rows(args.test)
    targets = np.asarray([float(row["answer"]) for row in train_rows], dtype=float)
    predictions, fit_report = predict(args, train_rows, targets, test_rows)
    write_submission(test_rows, predictions, args.output)
    report = {"mode": "bert_hybrid_predict", "score_type": "local_estimate", "row_count": len(test_rows),
              "promotion_gate": TRANSFORMER_GATE, "promoted": False, **fit_report}
    write_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fold-safe Ridge plus BERT hybrid experiment.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--backend", choices=["hf", "tiny"], default="hf")
    parser.add_argument("--model", default="dumitrescustefan/bert-base-romanian-cased-v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    parser.add_argument("--chunk-words", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-chunks", type=int)
    parser.add_argument("--skip-weight", type=float, default=0.5)
    parser.add_argument("--time-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prediction-scale", type=float, default=1.0)
    parser.add_argument("--zero-rate", type=float, default=0.0)
    parser.add_argument("--bert-scale", type=float, default=1.2)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--stack-alpha", type=float, default=10.0)
    parser.add_argument("--min-gain", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(__import__("json").dumps(result, indent=2, sort_keys=True))
