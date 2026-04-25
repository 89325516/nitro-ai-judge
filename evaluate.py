from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

METRIC_SCALE = 100.0
FOLD_TEST_COLUMNS = ["word_id", "word", "participant_id", "text", "datapointID"]
TRAIN_COLUMNS = ["word_id", "word", "answer", "participant_id", "text"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_metric(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) != len(pred) or len(true) == 0:
        raise ValueError("truth and prediction lengths must match and be non-empty")
    raw_r2 = float(r2_score(true, pred))
    r2 = max(0.0, raw_r2)
    if np.std(true) == 0.0 or np.std(pred) == 0.0:
        pearson = 0.0
    else:
        pearson = float(pearsonr(true, pred)[0])
        if math.isnan(pearson):
            pearson = 0.0
    pearson_abs = abs(pearson)
    return {"score": METRIC_SCALE * (r2 + pearson_abs) / 2.0, "r2": r2, "pearson": pearson_abs}


def parse_truth(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    truth: dict[str, float] = {}
    for row in rows:
        datapoint_id = row.get("datapointID")
        if datapoint_id is None or datapoint_id == "":
            raise ValueError("truth row is missing datapointID")
        if datapoint_id in truth:
            raise ValueError(f"duplicate truth datapointID: {datapoint_id}")
        truth[datapoint_id] = parse_number(row.get("answer"), f"truth answer for {datapoint_id}")
    return truth


def parse_predictions(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    predictions: dict[str, float] = {}
    for row in rows:
        datapoint_id = row.get("datapointID")
        if datapoint_id is None or datapoint_id == "":
            raise ValueError("prediction row is missing datapointID")
        if datapoint_id in predictions:
            raise ValueError(f"duplicate prediction datapointID: {datapoint_id}")
        predictions[datapoint_id] = parse_number(row.get("answer"), f"prediction answer for {datapoint_id}")
    return predictions


def parse_number(value: str | None, label: str) -> float:
    try:
        number = float(value) if value is not None else math.nan
    except ValueError as exc:
        raise ValueError(f"non-numeric {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}")
    return number


def score_rows(truth_rows: Sequence[dict[str, str]], prediction_rows: Sequence[dict[str, str]]) -> dict:
    truth = parse_truth(truth_rows)
    predictions = parse_predictions(prediction_rows)
    missing = sorted(set(truth) - set(predictions))
    unexpected = sorted(set(predictions) - set(truth))
    if missing:
        raise ValueError("missing prediction IDs: " + ", ".join(missing[:5]))
    if unexpected:
        raise ValueError("unexpected prediction IDs: " + ", ".join(unexpected[:5]))
    ordered_ids = list(truth.keys())
    metrics = compute_metric([truth[key] for key in ordered_ids], [predictions[key] for key in ordered_ids])
    return {"row_count": len(ordered_ids), **metrics}


def score_files(truth_path: Path, predictions_path: Path, report_path: Path | None) -> dict:
    result = {"mode": "score", "score_type": "exact_with_truth", **score_rows(read_csv(truth_path), read_csv(predictions_path))}
    if report_path:
        write_json(report_path, result)
    return result


def add_datapoint(row: dict[str, str], datapoint_id: int) -> dict[str, str]:
    return {"word_id": row["word_id"], "word": row["word"], "participant_id": row["participant_id"], "text": row["text"], "datapointID": str(datapoint_id)}


def run_candidate(command: str, train_path: Path, test_path: Path, output_path: Path) -> None:
    rendered = command.format(
        train=shlex.quote(str(train_path)),
        test=shlex.quote(str(test_path)),
        output=shlex.quote(str(output_path)),
    )
    subprocess.run(rendered, shell=True, check=True)


def cross_validate(train_path: Path, folds: int, command: str, report_path: Path | None) -> dict:
    rows = read_csv(train_path)
    groups = [row["text"] for row in rows]
    if len(set(groups)) < folds:
        raise ValueError("fold count cannot exceed distinct text groups")
    splitter = GroupKFold(n_splits=folds)
    fold_reports = []
    with tempfile.TemporaryDirectory() as temp_root:
        temp_dir = Path(temp_root)
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(rows, groups=groups), start=1):
            fold_train = [rows[index] for index in train_idx]
            fold_test = [add_datapoint(rows[index], index) for index in valid_idx]
            fold_truth = [{"datapointID": str(index), "answer": rows[index]["answer"]} for index in valid_idx]
            train_fold_path = temp_dir / f"train_fold_{fold_index}.csv"
            test_fold_path = temp_dir / f"test_fold_{fold_index}.csv"
            output_fold_path = temp_dir / f"prediction_fold_{fold_index}.csv"
            write_csv(train_fold_path, fold_train, TRAIN_COLUMNS)
            write_csv(test_fold_path, fold_test, FOLD_TEST_COLUMNS)
            run_candidate(command, train_fold_path, test_fold_path, output_fold_path)
            scored = score_rows(fold_truth, read_csv(output_fold_path))
            fold_reports.append({"fold": fold_index, "heldout_texts": sorted({rows[index]["text"] for index in valid_idx}), **scored})
    result = summarize_folds(fold_reports)
    if report_path:
        write_json(report_path, result)
    return result


def summarize_folds(folds: Sequence[dict]) -> dict:
    scores = np.asarray([fold["score"] for fold in folds], dtype=float)
    r2_values = np.asarray([fold["r2"] for fold in folds], dtype=float)
    pearson_values = np.asarray([fold["pearson"] for fold in folds], dtype=float)
    return {
        "mode": "cross_validate",
        "score_type": "local_estimate",
        "strategy": "text",
        "fold_count": len(folds),
        "row_count": int(sum(fold["row_count"] for fold in folds)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_r2": float(np.mean(r2_values)),
        "mean_pearson": float(np.mean(pearson_values)),
        "folds": list(folds),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Nitro AI Judge predictions.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--truth", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--report", type=Path)
    cv_parser = subparsers.add_parser("cross-validate")
    cv_parser.add_argument("--train", type=Path, required=True)
    cv_parser.add_argument("--strategy", choices=["text"], default="text")
    cv_parser.add_argument("--folds", type=int, default=3)
    cv_parser.add_argument("--command", required=True)
    cv_parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "score":
        result = score_files(args.truth, args.predictions, args.report)
    else:
        result = cross_validate(args.train, args.folds, args.command, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
