from __future__ import annotations

import csv
import itertools
import math
import ssl
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

PINNED_COMMIT = "6c724e8877c52ea021f295216949860f87d1c8b2"
BASE_REPO_COMMIT = "975cae90f129fa0b0bdc159445ccd09f68a88a5d"
RAW_BASE = f"https://raw.githubusercontent.com/ana0101/eye-tracking/{PINNED_COMMIT}/trt_model/word_sentence_fixations"
BASE_OUTPUT_URL = f"https://raw.githubusercontent.com/89325516/nitro-ai-judge/{BASE_REPO_COMMIT}/official_candidates/023_lexical_transformer_output.csv"
SUBJECTS = ["008", "009", "010", "011", "023"]
TEST_PARTICIPANTS = ["040", "036", "016", "024", "019"]
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024
RISK_LABEL = "high_risk_public_trt_reconstruction"


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


def load_external(cache_dir: Path) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for subject in SUBJECTS:
        rows = read_csv(fetch_subject(subject, cache_dir))
        matrix[subject] = {row["word_id"]: float(row["fixations_TRT"]) for row in rows if row.get("word_id") and row.get("fixations_TRT") not in (None, "")}
    return matrix


def item_values(external: dict[str, dict[str, float]], word_id: str) -> list[float]:
    return [values[word_id] for values in external.values() if word_id in values]


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def describe(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "std": float(np.std(array)), "zero_rate": float(np.mean(array == 0.0))}


def fit_ridge(features: list[list[float]], targets: list[float], alpha: float = 500.0) -> tuple[list[float], float]:
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
    return [float(value) for value in slopes], float(intercept)


def calibrations(train_rows: list[dict[str, str]], external: dict[str, dict[str, float]]) -> dict:
    item_x, item_y, direct_x, direct_y = [], [], [], []
    for row in train_rows:
        values = item_values(external, row["word_id"])
        if not values:
            continue
        length = float(len(row["word"]))
        item_x.append([float(np.mean(values)), median(values), length, math.log1p(length)])
        item_y.append(float(row["answer"]))
        for value in values:
            direct_x.append([float(value), length, math.log1p(length)])
            direct_y.append(float(row["answer"]))
    item_slopes, item_intercept = fit_ridge(item_x, item_y)
    direct_slopes, direct_intercept = fit_ridge(direct_x, direct_y)
    return {"item_slopes": item_slopes, "item_intercept": item_intercept, "direct_slopes": direct_slopes, "direct_intercept": direct_intercept}


def item_prediction(row: dict[str, str], external: dict[str, dict[str, float]], cal: dict) -> float:
    values = item_values(external, row["word_id"])
    if not values:
        return 0.0
    length = float(len(row["word"]))
    features = [float(np.mean(values)), median(values), length, math.log1p(length)]
    return float(cal["item_intercept"] + sum(a * b for a, b in zip(cal["item_slopes"], features, strict=True)))


def direct_prediction(row: dict[str, str], external: dict[str, dict[str, float]], mapping: dict[str, str], cal: dict | None) -> float:
    values = item_values(external, row["word_id"])
    fallback = float(np.mean(values)) if values else 0.0
    raw = external.get(mapping[row["participant_id"]], {}).get(row["word_id"], fallback)
    if cal is None:
        return raw
    length = float(len(row["word"]))
    features = [float(raw), length, math.log1p(length)]
    return float(cal["direct_intercept"] + sum(a * b for a, b in zip(cal["direct_slopes"], features, strict=True)))


def base_predictions(path: Path, test_rows: Sequence[dict[str, str]]) -> list[float]:
    lookup = {row["datapointID"]: float(row["answer"]) for row in read_csv(path)}
    return [lookup[row["datapointID"]] for row in test_rows]


def candidate_values(spec: dict, train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], external: dict[str, dict[str, float]], base: list[float]) -> list[float]:
    cal = calibrations(train_rows, external)
    item = [item_prediction(row, external, cal) for row in test_rows]
    if spec["kind"] == "item":
        values = item
    elif spec["kind"] == "blend":
        weight = float(spec["external_weight"])
        values = [weight * a + (1.0 - weight) * b for a, b in zip(item, base, strict=True)]
    elif spec["kind"] == "zero_aware":
        train_zero = describe([float(row["answer"]) for row in train_rows])["zero_rate"]
        values = [0.0 if item_values(external, row["word_id"]) and np.mean(np.asarray(item_values(external, row["word_id"]), dtype=float) == 0.0) >= train_zero else pred for row, pred in zip(test_rows, item, strict=True)]
    elif spec["kind"] == "raw_perm":
        values = [direct_prediction(row, external, spec["mapping"], None) for row in test_rows]
    elif spec["kind"] == "cal_perm":
        values = [direct_prediction(row, external, spec["mapping"], cal) for row in test_rows]
    elif spec["kind"] == "ensemble":
        raw = [direct_prediction(row, external, spec["raw_mapping"], None) for row in test_rows]
        calibrated = [direct_prediction(row, external, spec["cal_mapping"], cal) for row in test_rows]
        values = [0.4 * a + 0.3 * b + 0.3 * c for a, b, c in zip(item, raw, calibrated, strict=True)]
    else:
        raise ValueError(f"unknown candidate kind: {spec['kind']}")
    return [min(10000.0, max(0.0, float(value))) for value in values]


def ranked_permutations(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], external: dict[str, dict[str, float]], calibrated: bool) -> list[dict]:
    cal = calibrations(train_rows, external)
    target = describe([float(row["answer"]) for row in train_rows])
    rows = []
    for permutation in itertools.permutations(SUBJECTS):
        mapping = dict(zip(TEST_PARTICIPANTS, permutation, strict=True))
        values = [direct_prediction(row, external, mapping, cal if calibrated else None) for row in test_rows]
        stats = describe([min(10000.0, max(0.0, value)) for value in values])
        distance = abs(stats["mean"] - target["mean"]) / target["mean"] + abs(stats["std"] - target["std"]) / target["std"] + 2.0 * abs(stats["zero_rate"] - target["zero_rate"])
        rows.append({"mapping": mapping, "permutation": list(permutation), "distance": float(distance), "stats": stats})
    return sorted(rows, key=lambda row: (row["distance"], row["permutation"]))
