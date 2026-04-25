from __future__ import annotations

import csv
import math
import re
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor
except Exception:  # pragma: no cover - source scripts fail loudly if sklearn is unavailable.
    ExtraTreesRegressor = HistGradientBoostingRegressor = HuberRegressor = None

try:
    from experiments.three_tier_external_data import RAW_BASE
except ModuleNotFoundError:
    from three_tier_external_data import RAW_BASE

WORD_RE = re.compile(r"^(.*)_([0-9]+)_page_([0-9]+)_([0-9]+)$")
FAMS = ("arg", "enc", "ins", "lit", "popsci")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def output_values(path: Path, test_rows: list[dict[str, str]]) -> np.ndarray:
    rows = {row["datapointID"]: float(row["answer"]) for row in read_csv(path)}
    return np.asarray([rows[row["datapointID"]] for row in test_rows], dtype=float)


def external_maps(cache_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    values: dict[str, float] = {}
    complexity: dict[str, float] = {}
    for name in ("words_dict_romanian_merged.csv", "words_dict_romanian_merged_008_009_010_011.csv"):
        path = cache_dir / name
        if not path.exists():
            with urllib.request.urlopen(f"{RAW_BASE}/{name}", context=ssl._create_unverified_context(), timeout=90) as response:
                path.write_bytes(response.read())
        for row in read_csv(path):
            if row.get("word_id") and row.get("average_TRT") not in (None, ""):
                values[row["word_id"]] = float(row["average_TRT"])
                try:
                    complexity[row["word_id"]] = float(row.get("complexity") or 0.0)
                except ValueError:
                    complexity[row["word_id"]] = 0.0
    return values, complexity


def parse_word_id(word_id: str) -> tuple[str, int, int, int]:
    match = WORD_RE.match(word_id)
    return (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))) if match else ("", 0, 0, 0)


def family(text: str) -> str:
    return text.split("_", 1)[0] if text else ""


def grouped_stats(train_rows: list[dict[str, str]]) -> dict:
    by_word: dict[str, list[float]] = defaultdict(list)
    by_text: dict[str, list[float]] = defaultdict(list)
    by_page: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    values = []
    for row in train_rows:
        value = float(row["answer"])
        _, _, page, _ = parse_word_id(row["word_id"])
        values.append(value)
        by_word[row["word_id"]].append(value)
        by_text[row["text"]].append(value)
        by_page[(row["text"], page)].append(value)
        by_family[family(row["text"])].append(value)
    arr = np.asarray(values, dtype=float)
    return {"by_word": by_word, "text_mean": {k: float(np.mean(v)) for k, v in by_text.items()}, "page_mean": {k: float(np.mean(v)) for k, v in by_page.items()}, "family_mean": {k: float(np.mean(v)) for k, v in by_family.items()}, "global_mean": float(arr.mean()), "zero_rate": float(np.mean(arr == 0.0))}


def aggregate(values: list[float], kind: str) -> float:
    arr = np.asarray(values, dtype=float)
    pos = arr[arr > 0]
    if kind == "median": return float(np.median(arr))
    if kind in {"positive", "log"}: return float(pos.mean()) if len(pos) else 0.0
    if kind == "zeroaware": return (float(pos.mean()) if len(pos) else float(arr.mean())) * (1.0 - float(np.mean(arr == 0.0)))
    if kind.startswith("trim"):
        q = float(kind[4:]) / 100.0; lo, hi = np.quantile(arr, [q, 1 - q]); cut = arr[(arr >= lo) & (arr <= hi)]
        return float(cut.mean()) if len(cut) else float(arr.mean())
    if kind.startswith("winsor"):
        q = float(kind[6:]) / 100.0; lo, hi = np.quantile(arr, [q, 1 - q])
        return float(np.clip(arr, lo, hi).mean())
    return float(arr.mean())
