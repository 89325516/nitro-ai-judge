from __future__ import annotations

import math

import numpy as np

try:
    from experiments.noise_aware_v2_components import FAMS, aggregate, family, parse_word_id
except ModuleNotFoundError:
    from noise_aware_v2_components import FAMS, aggregate, family, parse_word_id


def features(row: dict[str, str], external: dict[str, float], complexity: dict[str, float]) -> list[float]:
    word = row["word"]
    _, src, page, idx = parse_word_id(row["word_id"])
    fam = family(row.get("text", ""))
    ext = external.get(row["word_id"], 0.0)
    comp = complexity.get(row["word_id"], 0.0)
    return [1.0, len(word), math.log1p(len(word)), sum(ch.isalpha() for ch in word), sum(ch.isdigit() for ch in word), float(bool(word) and all(not ch.isalnum() for ch in word)), float(word[:1].isupper()), float(word.isupper()), src, page, idx, math.log1p(idx), ext, math.log1p(max(ext, 0.0)), float(ext == 0.0), comp] + [float(fam == item) for item in FAMS]


def zmatch(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return (values - values.mean()) / (values.std() or 1.0) * (reference.std() or 1.0) + reference.mean()


def shaped_direction(direction: np.ndarray, style: str) -> np.ndarray:
    out = np.asarray(direction, dtype=float).copy()
    if style == "winsor":
        lo, hi = np.quantile(out, [0.03, 0.97]); out = np.clip(out, lo, hi)
    elif style == "shrink":
        lo, hi = np.quantile(np.abs(out), 0.90), np.quantile(np.abs(out), 0.99)
        out = np.sign(out) * np.minimum(np.abs(out), lo + 0.35 * np.maximum(0.0, np.abs(out) - lo))
        out = np.clip(out, -hi, hi)
    elif style == "zboost":
        out = zmatch(out, direction) * 1.08
    return out


def predict_model(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], target: list[float], model: str, external: dict[str, float], complexity: dict[str, float], scale_param: int = 0) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    x = np.asarray([features(row, external, complexity) for row in train_rows], dtype=float)
    z = np.asarray([features(row, external, complexity) for row in test_rows], dtype=float)
    y = np.asarray(target, dtype=float)
    if model == "trees":
        reg = ExtraTreesRegressor(n_estimators=int(scale_param or 256), min_samples_leaf=4, max_features=.90, random_state=53, n_jobs=1)
    elif model == "hgb":
        reg = HistGradientBoostingRegressor(max_iter=int(scale_param or 260), learning_rate=.035, l2_regularization=.06, max_leaf_nodes=31, random_state=53)
    elif model == "huber":
        reg = HuberRegressor(alpha=.0005, epsilon=1.45, max_iter=220)
    else:
        reg = Ridge(alpha=180.0)
    return np.asarray(reg.fit(x, y).predict(z), dtype=float)


def denoised_target(train_rows: list[dict[str, str]], grouped: dict, kind: str) -> list[float]:
    return [aggregate(grouped["by_word"][row["word_id"]], kind) for row in train_rows]


def eb_target(train_rows: list[dict[str, str]], grouped: dict, external: dict[str, float], prior: str, strength: float) -> list[float]:
    out = []
    for row in train_rows:
        _, _, page, _ = parse_word_id(row["word_id"])
        vals = grouped["by_word"][row["word_id"]]
        base = float(np.mean(vals))
        ext = external.get(row["word_id"], base)
        if prior == "external": p = ext
        elif prior == "text": p = grouped["text_mean"].get(row["text"], grouped["global_mean"])
        elif prior == "page": p = grouped["page_mean"].get((row["text"], page), grouped["global_mean"])
        elif prior == "family": p = grouped["family_mean"].get(family(row["text"]), grouped["global_mean"])
        else: p = .5 * ext + .25 * grouped["text_mean"].get(row["text"], grouped["global_mean"]) + .25 * grouped["page_mean"].get((row["text"], page), grouped["global_mean"])
        out.append((len(vals) * base + strength * p) / (len(vals) + strength))
    return out
