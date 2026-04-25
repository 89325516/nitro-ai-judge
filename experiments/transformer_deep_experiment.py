from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import solution
from experiments.transformer_model import DeterministicWordEncoder, HuggingFaceWordEncoder, compute_row_vectors, vector_feature_dicts
from experiments.transformer_surprisal import (
    DeterministicSurprisalScorer,
    HuggingFaceMaskedLMScorer,
    compute_surprisal_features,
    read_feature_cache,
    write_feature_cache,
)

RAW_RIDGE_SCORE = 38.388844
PROMOTION_SCORE = 39.388844
DEFAULT_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def make_surprisal_scorer(args: argparse.Namespace):
    if args.scorer == "deterministic":
        return DeterministicSurprisalScorer()
    return HuggingFaceMaskedLMScorer(args.model, device=args.device, max_words=args.max_words)


def make_vector_encoder(args: argparse.Namespace):
    if not args.use_hidden:
        return None
    if args.scorer == "deterministic":
        return DeterministicWordEncoder(width=args.raw_width)
    return HuggingFaceWordEncoder(args.model, device=args.device, chunk_size=args.chunk_size, overlap=args.overlap)


def deep_features(rows: list[dict[str, str]], args: argparse.Namespace, cache_path: Path | None = None) -> list[dict[str, float]]:
    base = [solution.build_features(row) for row in rows]
    if cache_path and cache_path.exists():
        surprisal = read_feature_cache(cache_path, len(rows))
    else:
        surprisal = compute_surprisal_features(rows, make_surprisal_scorer(args))
        if cache_path:
            write_feature_cache(cache_path, surprisal)
    if args.use_hidden:
        encoder = make_vector_encoder(args)
        hidden = vector_feature_dicts(compute_row_vectors(rows, encoder, args.feature_width), prefix="tf")
    else:
        hidden = [{} for _ in rows]
    return [{**left, **middle, **right} for left, middle, right in zip(base, surprisal, hidden, strict=True)]


def write_submission(test_rows: list[dict[str, str]], predictions: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, prediction in zip(test_rows, predictions, strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{max(0.0, float(prediction)):.6f}"})


def run_smoke(args: argparse.Namespace) -> dict:
    rows = read_rows(args.train, args.limit_rows)
    features = compute_surprisal_features(rows, make_surprisal_scorer(args))
    if args.cache:
        write_feature_cache(args.cache, features)
    promoted = False
    report = {
        "mode": "transformer_surprisal_smoke",
        "score_type": "local_estimate",
        "model": args.model if args.scorer == "hf" else "deterministic",
        "row_count": len(rows),
        "feature_keys": sorted(features[0]) if features else [],
        "baseline_score": RAW_RIDGE_SCORE,
        "promotion_score": PROMOTION_SCORE,
        "promoted": promoted,
    }
    write_json(args.report, report)
    return report


def run_predict(args: argparse.Namespace) -> dict:
    train_rows = read_rows(args.train)
    test_rows = read_rows(args.test)
    train_features = deep_features(train_rows, args, args.train_cache)
    test_features = deep_features(test_rows, args, args.test_cache)
    targets = np.asarray([float(row["answer"]) for row in train_rows], dtype=float)
    model = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=args.alpha, random_state=0))
    predictions = np.clip(model.fit(train_features, targets).predict(test_features), 0.0, 10000.0)
    write_submission(test_rows, predictions, args.output)
    return {"mode": "transformer_deep_predict", "score_type": "local_estimate", "row_count": len(test_rows), "promoted": False}


def write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scorer", choices=["hf", "deterministic"], default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--use-hidden", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--feature-width", type=int, default=13)
    parser.add_argument("--raw-width", type=int, default=16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment-only Transformer surprisal checks.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--train", type=Path, required=True)
    smoke.add_argument("--limit-rows", type=int, default=8)
    smoke.add_argument("--cache", type=Path)
    smoke.add_argument("--report", type=Path, default=Path("reports/transformer_surprisal_smoke.json"))
    add_common(smoke)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--train", type=Path, required=True)
    predict.add_argument("--test", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--train-cache", type=Path)
    predict.add_argument("--test-cache", type=Path)
    predict.add_argument("--alpha", type=float, default=100.0)
    add_common(predict)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke(args) if args.mode == "smoke" else run_predict(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
