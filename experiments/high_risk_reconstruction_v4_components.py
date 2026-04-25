from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from experiments.high_risk_reconstruction_v4_contracts import RuntimeContext
from experiments.high_risk_reconstruction_v4_data import first_seen_participants


def describe(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()), "zero_rate": float(np.mean(array == 0.0))}


def safe_corr(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def zmatch(values: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float); r = np.asarray(reference, dtype=float)
    return (a - a.mean()) / (a.std() or 1.0) * (r.std() or 1.0) + r.mean()


def item_default(ctx: RuntimeContext, word_id: str) -> float:
    values = ctx.provider.item_values(word_id)
    return float(np.median(values)) if values else float(np.mean(ctx.anchor))


def hard_values(ctx: RuntimeContext, mapping: dict[str, str]) -> np.ndarray:
    output = []
    for row in ctx.test_rows:
        default = item_default(ctx, row["word_id"])
        output.append(ctx.provider.subject_value(mapping[row["participant_id"]], row["word_id"], default))
    return np.asarray(output, dtype=float)


def soft_values(ctx: RuntimeContext, weights: dict[str, dict[str, float]]) -> np.ndarray:
    output = []
    for row in ctx.test_rows:
        default = item_default(ctx, row["word_id"])
        total = 0.0
        for subject_id, weight in weights[row["participant_id"]].items():
            total += float(weight) * ctx.provider.subject_value(subject_id, row["word_id"], default)
        output.append(total)
    return np.asarray(output, dtype=float)


def base_values(ctx: RuntimeContext, config: dict) -> np.ndarray:
    if config["source"] == "hard":
        return hard_values(ctx, config["mapping"])
    if config["source"] == "soft":
        return soft_values(ctx, config["weights"])
    return np.asarray(ctx.anchor, dtype=float)


def zero_mask(ctx: RuntimeContext, config: dict, raw: np.ndarray) -> np.ndarray:
    mode = config.get("zero_strategy", "none")
    if mode == "none":
        return np.zeros(len(ctx.test_rows), dtype=bool)
    masks = []
    for row, value in zip(ctx.test_rows, raw, strict=True):
        item = np.asarray(ctx.provider.item_values(row["word_id"]), dtype=float)
        rate = float(np.mean(item == 0.0)) if len(item) else 0.0
        threshold = float(config.get("zero_threshold", 0.6))
        masks.append((mode == "mapped" and value == 0.0) or (mode == "consensus" and rate >= threshold) or (mode == "hybrid" and (value == 0.0 or rate >= threshold)))
    return np.asarray(masks, dtype=bool)


def restore_scale(values: np.ndarray, target_std: float | None) -> np.ndarray:
    if target_std is None:
        return values
    mean = float(values.mean()); std = float(values.std()) or 1.0
    return (values - mean) / std * float(target_std) + mean


def tail_restore(ctx: RuntimeContext, values: np.ndarray, raw: np.ndarray, config: dict) -> np.ndarray:
    boost = float(config.get("tail_boost", 0.0))
    if boost <= 0.0:
        return values
    threshold = float(np.quantile(raw, float(config.get("tail_quantile", 0.90))))
    excess = np.maximum(0.0, raw - threshold)
    return values + boost * excess


def finalize(ctx: RuntimeContext, raw: np.ndarray, config: dict) -> np.ndarray:
    anchor = np.asarray(ctx.anchor, dtype=float)
    shaped = zmatch(raw, anchor)
    weight = float(config.get("movement_weight", 1.0))
    values = anchor + weight * (shaped - anchor)
    values = restore_scale(values, config.get("target_std"))
    values = tail_restore(ctx, values, raw, config)
    mask = zero_mask(ctx, config, raw)
    values[mask] = 0.0 if config.get("zero_floor", 0.0) == 0.0 else float(config["zero_floor"])
    return np.clip(values, 0.0, 10000.0)


def candidate_values(ctx: RuntimeContext, config: dict) -> np.ndarray:
    if config["mode"] == "fusion":
        parts = [candidate_values(ctx, part) for part in config["parts"]]
        weights = np.asarray(config["weights"], dtype=float)
        values = np.average(np.vstack(parts), axis=0, weights=weights)
        values = restore_scale(values, config.get("target_std"))
        return np.clip(values, 0.0, 10000.0)
    return finalize(ctx, base_values(ctx, config), config)


def rank_mappings(ctx: RuntimeContext) -> list[dict]:
    import itertools
    participants = first_seen_participants(ctx.test_rows)
    direction = np.asarray(ctx.anchor, dtype=float) - np.asarray(ctx.previous_anchor, dtype=float)
    ranked = []
    for permutation in itertools.permutations(ctx.provider.subject_ids()):
        mapping = dict(zip(participants, permutation, strict=True))
        raw = hard_values(ctx, mapping)
        shaped = zmatch(raw, ctx.anchor)
        stats = describe(shaped)
        score = abs(stats["std"] - np.std(ctx.anchor)) / max(1.0, np.std(ctx.anchor)) - 0.35 * safe_corr(shaped - np.asarray(ctx.anchor), direction)
        ranked.append({"mapping": mapping, "score": float(score), "stats": stats})
    return sorted(ranked, key=lambda row: row["score"])


def mixture_from_ranked(ranked: list[dict], participants: Sequence[str], limit: int, strength: float) -> dict[str, dict[str, float]]:
    subject_ids = sorted({subject for row in ranked[:limit] for subject in row["mapping"].values()})
    weights: dict[str, dict[str, float]] = {}
    for participant in participants:
        counts = {subject: 1.0 - strength for subject in subject_ids}
        for rank, row in enumerate(ranked[:limit], start=1):
            counts[row["mapping"][participant]] += strength / rank
        total = sum(counts.values()) or 1.0
        weights[participant] = {subject: value / total for subject, value in counts.items()}
    return weights
