from __future__ import annotations

import csv
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

WORD_ID_RE = re.compile(r"^(.*)_(\d+)_page_(\d+)_(\d+)$")


@dataclass(frozen=True)
class WordChunk:
    words: tuple[str, ...]
    row_indices: tuple[int, ...]
    targets: tuple[float, ...] | None
    datapoint_ids: tuple[str, ...]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_word_id(word_id: str) -> tuple[int, int, int]:
    match = WORD_ID_RE.match(word_id)
    return (int(match.group(2)), int(match.group(3)), int(match.group(4))) if match else (0, 0, 0)


def chunk_rows(rows: Sequence[dict[str, str]], labeled: bool, chunk_words: int, limit_chunks: int | None = None) -> list[WordChunk]:
    groups: OrderedDict[tuple[str, str, int, int], list[tuple[int, dict[str, str]]]] = OrderedDict()
    for row_index, row in enumerate(rows):
        source, page, word_index = parse_word_id(row["word_id"])
        key = (row["participant_id"], row["text"], source, page)
        groups.setdefault(key, []).append((word_index, {**row, "_row_index": str(row_index)}))
    chunks: list[WordChunk] = []
    for values in groups.values():
        ordered = sorted(values, key=lambda item: item[0])
        for start in range(0, len(ordered), chunk_words):
            part = [row for _, row in ordered[start : start + chunk_words]]
            targets = tuple(float(row["answer"]) for row in part) if labeled else None
            ids = tuple(row.get("datapointID", row["_row_index"]) for row in part)
            chunks.append(WordChunk(tuple(row["word"] for row in part), tuple(int(row["_row_index"]) for row in part), targets, ids))
            if limit_chunks is not None and len(chunks) >= limit_chunks:
                return chunks
    return chunks


def targets_from_chunks(chunks: Sequence[WordChunk]) -> np.ndarray:
    values = []
    for chunk in chunks:
        if chunk.targets is None:
            raise ValueError("labeled chunks are required")
        values.extend(chunk.targets)
    return np.asarray(values, dtype=float)


def write_submission(rows: Sequence[dict[str, str]], predictions: np.ndarray, output_path: Path) -> None:
    if len(rows) != len(predictions):
        raise ValueError("prediction count must match row count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{max(0.0, float(prediction)):.6f}"})
