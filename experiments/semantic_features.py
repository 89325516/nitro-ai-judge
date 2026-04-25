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
