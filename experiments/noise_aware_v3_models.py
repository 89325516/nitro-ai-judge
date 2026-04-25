from __future__ import annotations

import math

import numpy as np

try:
    from experiments.noise_aware_v3_data import FAMS, consensus, family, parse_word_id
except ModuleNotFoundError:
    from noise_aware_v3_data import FAMS, consensus, family, parse_word_id


def aggregate(values: list[float], kind: str) -> float:
    arr = np.asarray(values, dtype=float); pos = arr[arr > 0]
    if kind == "median_of_means":
        parts = [chunk for chunk in np.array_split(np.sort(arr), min(5, len(arr))) if len(chunk)]
        return float(np.median([part.mean() for part in parts]))
    if kind == "huber":
        med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)) or arr.std() or 1.0)
        return float(np.clip(arr, med - 1.5 * mad, med + 1.5 * mad).mean())
    if kind == "winsor_positive":
        if not len(pos): return 0.0
        lo, hi = np.quantile(pos, [.05, .95]); return float(np.clip(pos, lo, hi).mean())
    if kind == "zero_hurdle":
        base = float(pos.mean()) if len(pos) else float(arr.mean())
        return base * (1.0 - float(np.mean(arr == 0.0)))
    return float(np.median(arr))


def clean_target(rows: list[dict[str, str]], grouped: dict, external: dict, kind: str) -> list[float]:
    out = []
    for row in rows:
        vals = grouped["by_word"][row["word_id"]]
        local = aggregate(vals, kind)
        ext = consensus(external, row["word_id"], local)
        strength = 4.0 + ext["count"] * 2.0
        if kind == "posterior":
            local = aggregate(vals, "huber")
        out.append((len(vals) * local + strength * ext["value"]) / (len(vals) + strength))
    return out


def features(row: dict[str, str], grouped: dict, external: dict) -> list[float]:
    word = row["word"]; text, src, page, idx = parse_word_id(row["word_id"]); fam = family(row.get("text", text)); ext = consensus(external, row["word_id"], grouped["global_mean"]); participant = grouped["participant"].get(row.get("participant_id", ""), {"mean": grouped["global_mean"], "zero_rate": grouped["zero_rate"]})
    return [1.0, len(word), math.log1p(len(word)), sum(ch.isalpha() for ch in word), sum(ch.isdigit() for ch in word), float(bool(word) and all(not ch.isalnum() for ch in word)), float(word[:1].isupper()), float(word.isupper()), src, page, idx, math.log1p(idx), ext["value"], math.log1p(max(ext["value"], 0.0)), ext["count"], ext["std"], ext["zero_rate"], external["complexity"].get(row["word_id"], 0.0), grouped["text_mean"].get(row.get("text", ""), grouped["global_mean"]), grouped["page_mean"].get((row.get("text", ""), page), grouped["global_mean"]), grouped["family_mean"].get(fam, grouped["global_mean"]), participant["mean"], participant["zero_rate"]] + [float(fam == item) for item in FAMS]


def zmatch(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float); reference = np.asarray(reference, dtype=float)
    return (values - values.mean()) / (values.std() or 1.0) * (reference.std() or 1.0) + reference.mean()


def fit_predict(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], target: list[float], model: str, capacity: int, grouped: dict, external: dict, small_loss: float | None = None) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.asarray([features(row, grouped, external) for row in train_rows], dtype=float); z = np.asarray([features(row, grouped, external) for row in test_rows], dtype=float); y = np.asarray(target, dtype=float)
    if small_loss is not None:
        raw = np.asarray([float(row["answer"]) for row in train_rows], dtype=float); keep = np.abs(raw - y) <= np.quantile(np.abs(raw - y), small_loss); x, y = x[keep], y[keep]
    if model == "trees": reg = ExtraTreesRegressor(n_estimators=int(capacity or 512), min_samples_leaf=4, max_features=.92, random_state=71, n_jobs=1)
    elif model == "hgb": reg = HistGradientBoostingRegressor(max_iter=int(capacity or 520), learning_rate=.028, l2_regularization=.05, max_leaf_nodes=39, random_state=71)
    elif model == "huber": reg = make_pipeline(StandardScaler(), HuberRegressor(alpha=.0004, epsilon=1.45, max_iter=320))
    else: reg = make_pipeline(StandardScaler(), Ridge(alpha=140.0))
    return np.asarray(reg.fit(x, y).predict(z), dtype=float)


def uncertainty_shrink(test_rows: list[dict[str, str]], residual: np.ndarray, grouped: dict, external: dict, strength: float) -> np.ndarray:
    factors = []
    for row in test_rows:
        ext = consensus(external, row["word_id"], grouped["global_mean"])
        factors.append(1.0 / (1.0 + strength * (ext["std"] / max(1.0, grouped["global_std"])) / max(1.0, ext["count"])))
    return np.asarray(residual, dtype=float) * np.asarray(factors, dtype=float)


def gate_residual(best: np.ndarray, residual: np.ndarray, target_mae: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=float) - float(np.mean(residual))
    mae = float(np.mean(np.abs(residual))) or 1.0
    residual = residual * (target_mae / mae)
    values = best + residual
    if abs(float(values.std() - best.std())) > 12.0:
        residual *= 12.0 / abs(float(values.std() - best.std()))
    return residual
