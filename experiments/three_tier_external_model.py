from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Sequence

import numpy as np

try:
    from experiments.three_tier_external_data import SUBJECTS, TEST_PARTICIPANTS, parse_word_id, text_key
except ModuleNotFoundError:
    from three_tier_external_data import SUBJECTS, TEST_PARTICIPANTS, parse_word_id, text_key


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def ridge(features: list[list[float]], targets: list[float], alpha: float = 500.0) -> tuple[np.ndarray, float]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0.0] = 1.0
    z = (x - means) / scales
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    slopes = coef[1:] / scales
    intercept = coef[0] - float(np.sum(coef[1:] * means / scales))
    return slopes.astype(float), float(intercept)


def surface_features(row: dict[str, str]) -> list[float]:
    word = row["word"]
    _, source, page, index = parse_word_id(row["word_id"])
    alpha = sum(char.isalpha() for char in word)
    digit = sum(char.isdigit() for char in word)
    punct = float(bool(word) and all(not char.isalnum() for char in word))
    return [1.0, len(word), math.log1p(len(word)), alpha, digit, punct, float(word[:1].isupper()), float(word.isupper()), source, page, index, math.log1p(index)]


def safe_component(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], base: list[float]) -> list[float]:
    slopes, intercept = ridge([surface_features(row) for row in train_rows], [float(row["answer"]) for row in train_rows], 800.0)
    surface = [float(intercept + np.dot(slopes, surface_features(row))) for row in test_rows]
    return [max(0.0, 0.65 * b + 0.35 * s) for b, s in zip(base, surface, strict=True)]


def item_values(matrix: dict[str, dict[str, float]], word_id: str) -> list[float]:
    return [values[word_id] for values in matrix.values() if word_id in values]


def medium_features(row: dict[str, str], matrix: dict[str, dict[str, float]], text_stats: dict[str, float]) -> list[float]:
    values = item_values(matrix, row["word_id"])
    if not values:
        values = [0.0]
    array = np.asarray(values, dtype=float)
    return [float(np.mean(array)), median(values), float(np.std(array)), float(np.mean(array == 0.0)), len(row["word"]), math.log1p(len(row["word"])), text_stats.get(text_key(row), 0.0)]


def text_public_stats(public: Sequence[dict[str, str]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in public:
        buckets[text_key(row)].append(float(row["fixations_TRT"]))
    return {key: float(np.mean(values)) for key, values in buckets.items() if values}


def medium_component(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], public: Sequence[dict[str, str]], matrix: dict[str, dict[str, float]]) -> list[float]:
    text_stats = text_public_stats(public)
    slopes, intercept = ridge([medium_features(row, matrix, text_stats) for row in train_rows], [float(row["answer"]) for row in train_rows], 500.0)
    return [max(0.0, float(intercept + np.dot(slopes, medium_features(row, matrix, text_stats)))) for row in test_rows]


def direct_calibration(train_rows: list[dict[str, str]], matrix: dict[str, dict[str, float]]) -> tuple[np.ndarray, float]:
    features, targets = [], []
    for row in train_rows:
        values = item_values(matrix, row["word_id"])
        for value in values:
            features.append([float(value), len(row["word"]), math.log1p(len(row["word"]))])
            targets.append(float(row["answer"]))
    return ridge(features, targets, 500.0)


def direct_value(row: dict[str, str], matrix: dict[str, dict[str, float]], mapping: dict[str, str], cal: tuple[np.ndarray, float] | None) -> float:
    values = item_values(matrix, row["word_id"])
    fallback = float(np.mean(values)) if values else 0.0
    raw = matrix.get(mapping[row["participant_id"]], {}).get(row["word_id"], fallback)
    if cal is None:
        return max(0.0, float(raw))
    slopes, intercept = cal
    features = [float(raw), len(row["word"]), math.log1p(len(row["word"]))]
    return max(0.0, float(intercept + np.dot(slopes, features)))


def distribution_distance(values: list[float], target: dict[str, float]) -> float:
    array = np.asarray(values, dtype=float)
    stats = {"mean": float(np.mean(array)), "std": float(np.std(array)), "zero_rate": float(np.mean(array == 0.0))}
    return abs(stats["mean"] - target["mean"]) / target["mean"] + abs(stats["std"] - target["std"]) / target["std"] + 2.0 * abs(stats["zero_rate"] - target["zero_rate"])


def rank_mappings(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], matrix: dict[str, dict[str, float]], calibrated: bool) -> list[dict]:
    cal = direct_calibration(train_rows, matrix) if calibrated else None
    target_values = np.asarray([float(row["answer"]) for row in train_rows], dtype=float)
    target = {"mean": float(np.mean(target_values)), "std": float(np.std(target_values)), "zero_rate": float(np.mean(target_values == 0.0))}
    rows = []
    for permutation in itertools.permutations(SUBJECTS):
        mapping = dict(zip(TEST_PARTICIPANTS, permutation, strict=True))
        values = [direct_value(row, matrix, mapping, cal) for row in test_rows]
        rows.append({"mapping": mapping, "permutation": list(permutation), "distance": float(distribution_distance(values, target))})
    return sorted(rows, key=lambda row: (row["distance"], row["permutation"]))


def high_component(test_rows: list[dict[str, str]], matrix: dict[str, dict[str, float]], mapping: dict[str, str], mode: str, high_base: list[float] | None, cal: tuple[np.ndarray, float]) -> list[float]:
    if mode == "base" and high_base is not None:
        return [max(0.0, float(value)) for value in high_base]
    active_cal = cal if mode in {"cal", "base"} else None
    return [direct_value(row, matrix, mapping, active_cal) for row in test_rows]


def fuse_values(config: dict, safe: list[float], medium: list[float], high: list[float]) -> list[float]:
    weights = config["weights"]
    values = []
    for s, m, h in zip(safe, medium, high, strict=True):
        value = weights["safe"] * s + weights["medium"] * m + weights["high"] * h
        values.append(min(10000.0, max(0.0, float(value * config.get("scale", 1.0)))))
    return values
