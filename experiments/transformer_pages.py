from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

POSITION_PATTERN = re.compile(r"_page_(\d+)_(\d+)$")


@dataclass(frozen=True)
class PageKey:
    text: str
    page: int


@dataclass(frozen=True)
class PageWords:
    key: PageKey
    words: tuple[str, ...]
    indices: tuple[int, ...]


def parse_position(word_id: str) -> tuple[int, int]:
    match = POSITION_PATTERN.search(word_id)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def page_key(row: dict[str, str]) -> PageKey:
    page, _ = parse_position(row["word_id"])
    return PageKey(text=row["text"], page=page)


def build_page_words(rows: Iterable[dict[str, str]]) -> list[PageWords]:
    grouped: OrderedDict[PageKey, dict[int, str]] = OrderedDict()
    for row in rows:
        key = page_key(row)
        _, index = parse_position(row["word_id"])
        grouped.setdefault(key, OrderedDict()).setdefault(index, row["word"])
    pages = []
    for key, index_to_word in grouped.items():
        ordered = sorted(index_to_word.items())
        pages.append(PageWords(key=key, indices=tuple(i for i, _ in ordered), words=tuple(w for _, w in ordered)))
    return pages


def average_subword_vectors(hidden_states: np.ndarray, word_ids: list[int | None], word_count: int) -> np.ndarray:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape token_count x hidden_size")
    sums = np.zeros((word_count, hidden_states.shape[1]), dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        sums[word_id] += hidden_states[token_index]
        counts[word_id] += 1.0
    present = counts > 0
    sums[present] /= counts[present, None]
    return sums


def reduce_vectors(vectors: np.ndarray, width: int) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape row_count x hidden_size")
    stats = np.column_stack(
        [
            vectors.mean(axis=1),
            vectors.std(axis=1),
            np.linalg.norm(vectors, axis=1),
            vectors.min(axis=1),
            vectors.max(axis=1),
        ]
    )
    prefix = vectors[:, : max(0, width - stats.shape[1])]
    return np.column_stack([stats, prefix]).astype(float)
