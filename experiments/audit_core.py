from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    r2 = max(0.0, float(r2_score(true, pred)))
    if np.std(true) == 0.0 or np.std(pred) == 0.0:
        pearson = 0.0
    else:
        pearson = float(pearsonr(true, pred)[0])
        if math.isnan(pearson):
            pearson = 0.0
    pearson = abs(pearson)
    return {"score": 100.0 * (r2 + pearson) / 2.0, "r2": r2, "pearson": pearson}


def score_from_groups(rows: list[dict[str, str]], group_key: str, value_key: str) -> dict[str, float]:
    groups = np.asarray([row[group_key] for row in rows])
    y = np.asarray([float(row["answer"]) for row in rows], dtype=float)
    splits = min(5, len(set(groups)))
    preds: list[float] = []
    truth: list[float] = []
    for train_idx, valid_idx in GroupKFold(n_splits=splits).split(rows, groups=groups):
        means: dict[str, list[float]] = defaultdict(list)
        for index in train_idx:
            means[rows[index][value_key]].append(y[index])
        global_mean = float(np.mean(y[train_idx]))
        lookup = {key: float(np.mean(values)) for key, values in means.items()}
        for index in valid_idx:
            truth.append(float(y[index]))
            preds.append(lookup.get(rows[index][value_key], global_mean))
    return metric(truth, preds)


def same_item_oracle(rows: list[dict[str, str]]) -> dict[str, float]:
    groups = np.asarray([row["participant_id"] for row in rows])
    y = np.asarray([float(row["answer"]) for row in rows], dtype=float)
    preds: list[float] = []
    truth: list[float] = []
    splits = min(5, len(set(groups)), len(rows))
    for train_idx, valid_idx in GroupKFold(n_splits=splits).split(rows, groups=groups):
        means: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for index in train_idx:
            row = rows[index]
            means[(row["text"], row["word_id"], row["word"])].append(y[index])
        lookup = {key: float(np.mean(values)) for key, values in means.items()}
        global_mean = float(np.mean(y[train_idx]))
        for index in valid_idx:
            row = rows[index]
            truth.append(float(y[index]))
            preds.append(lookup.get((row["text"], row["word_id"], row["word"]), global_mean))
    return metric(truth, preds)


def sample_output_diagnostic(sample_rows: list[dict[str, str]]) -> dict:
    answers = np.asarray([float(row["answer"]) for row in sample_rows], dtype=float)
    row_index = np.arange(len(answers), dtype=float)
    corr = 0.0 if np.std(answers) == 0 else float(abs(pearsonr(row_index, answers)[0]))
    return {
        "evidence_class": "leakage_signal",
        "row_count": len(sample_rows),
        "min_answer": float(np.min(answers)),
        "max_answer": float(np.max(answers)),
        "mean_answer": float(np.mean(answers)),
        "row_order_abs_pearson": corr,
        "looks_like_probability_scale": bool(np.min(answers) >= 0.0 and np.max(answers) <= 1.0),
    }


def leakage_summary(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> dict:
    train_texts = {row["text"] for row in train_rows}
    test_texts = {row["text"] for row in test_rows}
    train_participants = {row["participant_id"] for row in train_rows}
    test_participants = {row["participant_id"] for row in test_rows}
    train_word_ids = {row["word_id"] for row in train_rows}
    test_word_ids = {row["word_id"] for row in test_rows}
    train_words = {row["word"].lower() for row in train_rows}
    return {
        "evidence_class": "leakage_signal",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "text_overlap_count": len(train_texts & test_texts),
        "participant_overlap_count": len(train_participants & test_participants),
        "word_id_overlap_count": len(train_word_ids & test_word_ids),
        "test_word_seen_rate": sum(row["word"].lower() in train_words for row in test_rows) / len(test_rows),
    }
