from __future__ import annotations
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import numpy as np
WORD_ID_RE = re.compile(r"^(.*)_(\d+)_page_(\d+)_(\d+)$")
STRIP = ".,;:!?()[]{}\"'-–—«»„“”"
VOWELS = set("aeiouăâîAEIOUĂÂÎ")
MARKS = set("ăâîșțĂÂÎȘȚ")
FAMILIES = ["arg", "enc", "ins", "lit", "popsci"]
def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
def parse_word_id(word_id: str) -> tuple[int, int, int]:
    match = WORD_ID_RE.match(word_id)
    return (int(match.group(2)), int(match.group(3)), int(match.group(4))) if match else (0, 0, 0)
def clean(word: str) -> str:
    return word.strip(STRIP).lower()
def family(text: str) -> str:
    return text.split("_", 1)[0]
def shape(word: str) -> str:
    values = []
    for char in word[:16]:
        values.append("A" if char.isupper() else "a" if char.islower() else "0" if char.isdigit() else ".")
    return "".join(values)
def page_context(rows: Sequence[dict[str, str]]) -> list[dict[str, float | str]]:
    page_words: dict[tuple[str, int, int], dict[int, str]] = defaultdict(dict)
    text_positions: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    for row in rows:
        source, page, index = parse_word_id(row["word_id"])
        page_words[(row["text"], source, page)].setdefault(index, row["word"])
        text_positions[row["text"]].add((source, page, index))
    meta_by_key: dict[tuple[str, int, int, int], dict[str, float | str]] = {}
    for key, words_by_index in page_words.items():
        ordered = sorted(words_by_index.items())
        sentence_start = 0
        sentence_id = 0
        for offset, (index, word) in enumerate(ordered):
            if offset and ordered[offset - 1][1].rstrip().endswith((".", "!", "?", ":")):
                sentence_start = offset
                sentence_id += 1
            next_break = len(ordered)
            for probe in range(offset + 1, len(ordered)):
                if ordered[probe - 1][1].rstrip().endswith((".", "!", "?", ":")):
                    next_break = probe
                    break
            prev_word = ordered[offset - 1][1] if offset else ""
            next_word = ordered[offset + 1][1] if offset + 1 < len(ordered) else ""
            meta_by_key[(*key, index)] = {
                "prev": prev_word, "next": next_word, "prev_len": len(clean(prev_word)), "next_len": len(clean(next_word)),
                "page_len": len(ordered), "page_pos": offset / max(1, len(ordered) - 1), "sent_id": sentence_id,
                "sent_pos": (offset - sentence_start) / max(1, next_break - sentence_start - 1), "sent_len": next_break - sentence_start,
            }
    output = []
    for row in rows:
        source, page, index = parse_word_id(row["word_id"])
        output.append({**meta_by_key.get((row["text"], source, page, index), {}), "text_len": len(text_positions[row["text"]])})
    return output
def packed(values: list[float], global_mean: float, global_pos: float) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    positive = array[array > 0]
    return (float((array.sum() + 25 * global_mean) / (len(array) + 25)), float(np.mean(array == 0)), math.log1p(len(array)), float(positive.mean()) if len(positive) else global_pos)
def build_stats(rows: Sequence[dict[str, str]], y: np.ndarray) -> dict:
    global_mean = float(y.mean())
    global_pos = float(y[y > 0].mean()) if np.any(y > 0) else global_mean
    buckets = {name: defaultdict(list) for name in ("word", "prefix", "suffix", "family")}
    for row, target in zip(rows, y, strict=True):
        word = clean(row["word"])
        for name, key in (("word", word), ("prefix", word[:4]), ("suffix", word[-4:]), ("family", family(row["text"]))):
            buckets[name][key].append(float(target))
    stats = {"global": (global_mean, global_pos)}
    stats.update({name: {key: packed(vals, global_mean, global_pos) for key, vals in bucket.items()} for name, bucket in buckets.items()})
    return stats
def sparse_row(row: dict[str, str], ctx: dict[str, float | str]) -> dict[str, float]:
    word = row["word"]
    low = clean(word)
    source, page, index = parse_word_id(row["word_id"])
    return {"bias": 1.0, "word=" + word.lower(): 1.0, "clean_word=" + low: 1.0, "shape=" + shape(word): 1.0,
            "family=" + family(row["text"]): 1.0, "prefix=" + low[:3]: 1.0, "suffix=" + low[-3:]: 1.0,
            "prev=" + clean(str(ctx.get("prev", ""))): 1.0, "next=" + clean(str(ctx.get("next", ""))): 1.0,
            "length": float(len(word)), "log_length": math.log1p(len(word)), "alpha_count": float(sum(char.isalpha() for char in word)),
            "digit_count": float(sum(char.isdigit() for char in word)), "source": float(source), "page": float(page),
            "index": float(index), "log_index": math.log1p(index), "page_pos": float(ctx.get("page_pos", 0.0)),
            "sent_pos": float(ctx.get("sent_pos", 0.0)), "is_url": float(word.startswith("http") or "www." in word),
            "is_capitalized": float(word[:1].isupper()), "is_upper": float(word.isupper()),
            "is_punct": float(bool(word) and all(not char.isalnum() for char in word)), "has_mark": float(any(char in MARKS for char in word))}
def numeric_row(row: dict[str, str], ctx: dict[str, float | str], stats: dict) -> list[float]:
    word = row["word"]
    low = clean(word)
    source, page, index = parse_word_id(row["word_id"])
    values = [1.0, len(word), math.log1p(len(word)), sum(c.isalpha() for c in word), sum(c.isdigit() for c in word),
              sum(c in VOWELS for c in word), sum(c in MARKS for c in word), float(word[:1].isupper()), float(word.isupper()),
              float(word.startswith("http") or "www." in word), source, math.log1p(source), page, index, math.log1p(index),
              float(ctx.get("prev_len", 0.0)), float(ctx.get("next_len", 0.0)), float(ctx.get("page_len", 1.0)),
              float(ctx.get("text_len", 1.0)), float(ctx.get("sent_len", 1.0)), float(ctx.get("page_pos", 0.0)),
              float(ctx.get("sent_pos", 0.0)), float(FAMILIES.index(family(row["text"]))) if family(row["text"]) in FAMILIES else -1.0]
    default = (stats["global"][0], 0.0, 0.0, stats["global"][1])
    for name, key in (("word", low), ("prefix", low[:4]), ("suffix", low[-4:]), ("family", family(row["text"]))):
        values.extend(stats[name].get(key, default))
    return [float(value) for value in values]
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
