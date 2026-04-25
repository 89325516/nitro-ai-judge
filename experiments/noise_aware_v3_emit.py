from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

try:
    from experiments.noise_aware_v3_data import grouped_stats, load_external, output_values, read_csv, target_profile
    from experiments.noise_aware_v3_models import clean_target, fit_predict, gate_residual, uncertainty_shrink, zmatch
    from experiments.three_tier_external_data import OUTPUT_LIMIT, SOURCE_LIMIT, describe
except ModuleNotFoundError:
    from noise_aware_v3_data import grouped_stats, load_external, output_values, read_csv, target_profile
    from noise_aware_v3_models import clean_target, fit_predict, gate_residual, uncertainty_shrink, zmatch
    from three_tier_external_data import OUTPUT_LIMIT, SOURCE_LIMIT, describe


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_context(train_path: Path, test_path: Path, best_path: Path, neutral_path: Path) -> dict:
    train_rows, test_rows = read_csv(train_path), read_csv(test_path)
    external = load_external(Path(".cache/noise_aware_v3_external"))
    grouped = grouped_stats(train_rows, external)
    return {"train": train_rows, "test": test_rows, "external": external, "grouped": grouped, "best": output_values(best_path, test_rows), "neutral": output_values(neutral_path, test_rows), "pred_cache": {}, "target_profile": target_profile(train_rows, grouped, external, test_rows)}


def prediction(ctx: dict, target: str, model: str, capacity: int, method: str = "", small_loss: float | None = None) -> np.ndarray:
    key = json.dumps([target, model, capacity, method, small_loss])
    if key not in ctx["pred_cache"]:
        y = clean_target(ctx["train"], ctx["grouped"], ctx["external"], target)
        if method == "ngboost_like":
            y = list(np.log1p(np.maximum(0.0, np.asarray(y, dtype=float))))
            pred = np.expm1(fit_predict(ctx["train"], ctx["test"], y, model, capacity, ctx["grouped"], ctx["external"], small_loss))
        else:
            pred = fit_predict(ctx["train"], ctx["test"], y, model, capacity, ctx["grouped"], ctx["external"], small_loss)
        ctx["pred_cache"][key] = np.maximum(0.0, pred)
    return ctx["pred_cache"][key]


def residual(ctx: dict, pred: np.ndarray, strength: float, target_mae: float, neutral_avoid: float) -> np.ndarray:
    best = ctx["best"]; raw = zmatch(pred, best) - best
    raw = uncertainty_shrink(ctx["test"], raw, ctx["grouped"], ctx["external"], strength)
    neutral = ctx["neutral"] - best
    denom = float(np.dot(neutral, neutral)) or 1.0
    raw = raw - neutral_avoid * (float(np.dot(raw, neutral)) / denom) * neutral
    return gate_residual(best, raw, target_mae)


def candidate_values(ctx: dict, config: dict) -> np.ndarray:
    mode = config["mode"]; best = ctx["best"]
    if mode == "clean":
        pred = prediction(ctx, config["target"], config["model"], int(config.get("capacity", 0)), "clean")
        return best + residual(ctx, pred, .75, float(config["target_mae"]), .25)
    if mode == "capacity":
        pred = prediction(ctx, config["target"], config["model"], int(config["capacity"]), "capacity")
        return best + residual(ctx, pred, .90, float(config["target_mae"]), .35)
    if mode == "robust":
        return best + robust_residual(ctx, config)
    if mode == "fusion":
        return best + fusion_residual(ctx, config)
    return best + champion_residual(ctx, config)


def robust_residual(ctx: dict, config: dict) -> np.ndarray:
    method = config["method"]
    if method == "superlearner":
        preds = [prediction(ctx, "posterior", "hgb", 520, method), prediction(ctx, "huber", "trees", 512, method), prediction(ctx, "zero_hurdle", "ridge", 0, method)]
        pred = .45 * preds[0] + .35 * preds[1] + .20 * preds[2]
    else:
        small = .70 if method == "co_teach" else None
        pred = prediction(ctx, config["target"], config.get("model", "hgb"), int(config.get("capacity", 520)), method, small)
    strength = 1.20 if method in {"distributional", "ngboost_like"} else .85
    return residual(ctx, pred, strength, float(config["target_mae"]), .45)


def fusion_residual(ctx: dict, config: dict) -> np.ndarray:
    clean = residual(ctx, prediction(ctx, "posterior", "hgb", 520, "fusion"), .90, 7.2, .40)
    cap = residual(ctx, prediction(ctx, "huber", "trees", 640, "fusion"), 1.05, 7.6, .45)
    robust = robust_residual(ctx, {"method": "superlearner", "target": "posterior", "model": "blend", "capacity": 0, "target_mae": 7.4})
    weights = {"clean_heavy": (.50, .30, .20), "model_heavy": (.25, .50, .25), "uncertainty_heavy": (.30, .25, .45)}[config["style"]]
    return gate_residual(ctx["best"], weights[0] * clean + weights[1] * cap + weights[2] * robust, float(config["target_mae"]))


def champion_residual(ctx: dict, config: dict) -> np.ndarray:
    clean = residual(ctx, prediction(ctx, "posterior", "hgb", 650, "champion"), 1.0, 7.4, .50)
    cap = residual(ctx, prediction(ctx, "huber", "trees", 768, "champion"), 1.1, 7.8, .55)
    robust = robust_residual(ctx, {"method": "superlearner", "target": "posterior", "model": "blend", "capacity": 0, "target_mae": 7.2})
    return gate_residual(ctx["best"], .38 * clean + .37 * cap + .25 * robust, float(config["target_mae"]))


def write_output(path: Path, test_rows: list[dict[str, str]], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, value in zip(test_rows, np.clip(values, 0.0, 10000.0), strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{float(value):.6f}"})


def source_text(config: dict) -> str:
    return f'''from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={config!r}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--neutral-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.noise_aware_v3_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.neutral_output,a.output,CONFIG)
if __name__=='__main__': main()
'''


def runtime_write(train_path: Path, test_path: Path, best_path: Path, neutral_path: Path, output_path: Path, config: dict) -> None:
    ctx = build_context(train_path, test_path, best_path, neutral_path)
    write_output(output_path, ctx["test"], candidate_values(ctx, config))


def materialize(root: Path, spec: dict, args, ctx: dict) -> dict:
    source = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_source.py"; output = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_output.csv"
    source.write_text(source_text(spec["config"]), encoding="utf-8")
    write_output(output, ctx["test"], candidate_values(ctx, spec["config"]))
    cmd = ["python3", str(source.relative_to(root)), "--train", str(args.train), "--test", str(args.test), "--best-output", str(args.best_output), "--neutral-output", str(args.neutral_output), "--output", str(output.relative_to(root))]
    return {key: value for key, value in spec.items() if key != "config"} | {"source_file": str(source.relative_to(root)), "output_file": str(output.relative_to(root)), "reproduction_command": " ".join(cmd)}


def validate_candidate(root: Path, candidate: dict, test_ids: list[str], best_path: Path) -> dict:
    output = read_csv(root / candidate["output_file"]); values = np.asarray([float(row["answer"]) for row in output], dtype=float); best = np.asarray([float(row["answer"]) for row in read_csv(best_path)], dtype=float)
    row = {**candidate, "mae_vs_199": float(np.mean(np.abs(values - best))), "mean_delta_vs_199": float(np.mean(values) - np.mean(best)), "std_delta_vs_199": float(np.std(values) - np.std(best)), "source_bytes": (root / candidate["source_file"]).stat().st_size, "output_bytes": (root / candidate["output_file"]).stat().st_size, "row_count": len(output), "columns": list(output[0].keys()) if output else [], "ids_match_test_order": [row["datapointID"] for row in output] == test_ids, "answers_finite_non_negative": bool(np.all(np.isfinite(values)) and np.all(values >= 0.0)), **{f"answer_{key}": value for key, value in describe(values).items()}}
    row["upload_ready"] = row["source_bytes"] < SOURCE_LIMIT and row["output_bytes"] < OUTPUT_LIMIT and row["row_count"] == len(test_ids) and row["columns"] == ["subtaskID", "datapointID", "answer"] and row["ids_match_test_order"] and row["answers_finite_non_negative"]
    return row
