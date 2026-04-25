from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def profile(rows: Sequence[dict[str, str]]) -> dict:
    values = np.asarray([float(row["answer"]) for row in rows], dtype=float)
    by_word: dict[str, list[float]] = defaultdict(list)
    by_part: dict[str, list[float]] = defaultdict(list)
    by_text: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["answer"])
        by_word[row["word_id"]].append(value)
        by_part[row["participant_id"]].append(value)
        by_text[row["text"]].append(value)
    word_stds = np.asarray([np.std(v) for v in by_word.values()], dtype=float)
    word_means = np.asarray([np.mean(v) for v in by_word.values()], dtype=float)
    participant = {}
    for key, vals in sorted(by_part.items()):
        arr = np.asarray(vals, dtype=float)
        participant[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "zero_rate": float(np.mean(arr == 0.0))}
    text = {key: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "zero_rate": float(np.mean(np.asarray(vals) == 0.0))} for key, vals in sorted(by_text.items())}
    return {
        "row_count": len(rows),
        "global_mean": float(np.mean(values)),
        "global_std": float(np.std(values)),
        "global_zero_rate": float(np.mean(values == 0.0)),
        "unique_word_count": len(by_word),
        "mean_within_word_std": float(np.mean(word_stds)),
        "median_within_word_std": float(np.median(word_stds)),
        "between_word_mean_std": float(np.std(word_means)),
        "within_to_between_ratio": float(np.mean(word_stds) / max(np.std(word_means), 1e-9)),
        "participant_reliability_inputs": participant,
        "text_noise_stats": text,
    }
