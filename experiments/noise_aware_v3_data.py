from __future__ import annotations

import csv
import re
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from experiments.three_tier_external_data import RAW_BASE
except ModuleNotFoundError:
    from three_tier_external_data import RAW_BASE

WORD_RE = re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$")
SUBJECTS = ("008", "009", "010", "011", "023")
FAMS = ("arg", "enc", "ins", "lit", "popsci")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fetch(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if not path.exists():
        with urllib.request.urlopen(f"{RAW_BASE}/{name}", context=ssl._create_unverified_context(), timeout=90) as response:
            path.write_bytes(response.read())
    return path


def parse_word_id(word_id: str) -> tuple[str, int, int, int]:
    match = WORD_RE.match(word_id)
    return (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))) if match else ("", 0, 0, 0)


def family(text: str) -> str:
    return text.split("_", 1)[0] if text else ""


def output_values(path: Path, test_rows: list[dict[str, str]]) -> np.ndarray:
    rows = {row["datapointID"]: float(row["answer"]) for row in read_csv(path)}
    return np.asarray([rows[row["datapointID"]] for row in test_rows], dtype=float)


def load_external(cache_dir: Path) -> dict:
    subject_maps: dict[str, dict[str, float]] = {}
    complexity: dict[str, float] = {}
    merged: dict[str, float] = {}
    for subject in SUBJECTS:
        rows = read_csv(fetch(cache_dir, f"words_dict_romanian_{int(subject):03d}.csv"))
        subject_maps[subject] = {row["word_id"]: float(row["fixations_TRT"]) for row in rows if row.get("word_id") and row.get("fixations_TRT") not in (None, "")}
    for name in ("words_dict_romanian_merged.csv", "words_dict_romanian_merged_008_009_010_011.csv"):
        for row in read_csv(fetch(cache_dir, name)):
            if row.get("word_id") and row.get("average_TRT") not in (None, ""):
                merged[row["word_id"]] = float(row["average_TRT"])
                try:
                    complexity[row["word_id"]] = float(row.get("complexity") or 0.0)
                except ValueError:
                    complexity[row["word_id"]] = 0.0
    return {"subjects": subject_maps, "merged": merged, "complexity": complexity}


def consensus(external: dict, word_id: str, default: float = 0.0) -> dict[str, float]:
    values = [mapping[word_id] for mapping in external["subjects"].values() if word_id in mapping]
    if word_id in external["merged"]:
        values.append(external["merged"][word_id])
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return {"value": default, "count": 0.0, "std": 999.0, "zero_rate": 0.0}
    return {"value": float(np.median(arr)), "count": float(len(arr)), "std": float(arr.std()), "zero_rate": float(np.mean(arr == 0.0))}


def grouped_stats(train_rows: list[dict[str, str]], external: dict) -> dict:
    by_word: dict[str, list[float]] = defaultdict(list); by_text: dict[str, list[float]] = defaultdict(list); by_page: dict[tuple[str, int], list[float]] = defaultdict(list); by_family: dict[str, list[float]] = defaultdict(list); by_participant: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        value = float(row["answer"]); _, _, page, _ = parse_word_id(row["word_id"])
        by_word[row["word_id"]].append(value); by_text[row["text"]].append(value); by_page[(row["text"], page)].append(value); by_family[family(row["text"])].append(value); by_participant[row["participant_id"]].append(value)
    all_values = np.asarray([float(row["answer"]) for row in train_rows], dtype=float)
    item_mean = {key: float(np.mean(vals)) for key, vals in by_word.items()}
    participant = {key: {"mean": float(np.mean(vals)), "zero_rate": float(np.mean(np.asarray(vals) == 0.0))} for key, vals in by_participant.items()}
    ext_coverage = sum(1 for word_id in by_word if consensus(external, word_id)["count"] > 0) / max(1, len(by_word))
    return {"by_word": by_word, "item_mean": item_mean, "text_mean": {k: float(np.mean(v)) for k, v in by_text.items()}, "page_mean": {k: float(np.mean(v)) for k, v in by_page.items()}, "family_mean": {k: float(np.mean(v)) for k, v in by_family.items()}, "participant": participant, "global_mean": float(all_values.mean()), "global_std": float(all_values.std()), "zero_rate": float(np.mean(all_values == 0.0)), "external_train_item_coverage": ext_coverage}


def target_profile(train_rows: list[dict[str, str]], grouped: dict, external: dict, test_rows: list[dict[str, str]]) -> dict:
    item_stds = [float(np.std(vals)) for vals in grouped["by_word"].values()]
    test_cover = sum(consensus(external, row["word_id"])["count"] > 0 for row in test_rows) / max(1, len(test_rows))
    return {"train_rows": len(train_rows), "train_items": len(grouped["by_word"]), "mean_within_item_std": float(np.mean(item_stds)), "global_zero_rate": grouped["zero_rate"], "external_train_item_coverage": grouped["external_train_item_coverage"], "external_test_row_coverage": float(test_cover)}
