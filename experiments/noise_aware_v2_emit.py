from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

try:
    from experiments.noise_aware_v2_components import external_maps, grouped_stats, output_values, read_csv
    from experiments.noise_aware_v2_models import denoised_target, eb_target, predict_model, shaped_direction, zmatch
    from experiments.three_tier_external_data import OUTPUT_LIMIT, SOURCE_LIMIT, describe
except ModuleNotFoundError:
    from noise_aware_v2_components import external_maps, grouped_stats, output_values, read_csv
    from noise_aware_v2_models import denoised_target, eb_target, predict_model, shaped_direction, zmatch
    from three_tier_external_data import OUTPUT_LIMIT, SOURCE_LIMIT, describe


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def component_values(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], anchor: np.ndarray, best: np.ndarray, config: dict, external: dict[str, float], complexity: dict[str, float], grouped: dict) -> np.ndarray:
    mode = config.get("mode")
    direction = best - anchor
    if mode == "anchor":
        return best.copy()
    if mode == "direction":
        return best + float(config.get("gamma", 0.0)) * shaped_direction(direction, config.get("style", "raw"))
    if mode == "scaled":
        pred = predict_model(train_rows, test_rows, denoised_target(train_rows, grouped, config.get("target", "median")), config.get("model", "ridge"), external, complexity, config.get("scale_param", 0))
        return best + float(config.get("gamma", .35)) * direction + float(config.get("weight", .08)) * (zmatch(np.maximum(0.0, pred), best) - best)
    if mode == "external":
        pred = predict_model(train_rows, test_rows, eb_target(train_rows, grouped, external, config.get("prior", "hybrid"), float(config.get("strength", 18.0))), config.get("model", "ridge"), external, complexity, 256)
        return best + float(config.get("gamma", .35)) * direction + float(config.get("weight", .08)) * (zmatch(np.maximum(0.0, pred), best) - best)
    return fused_values(train_rows, test_rows, anchor, best, config, external, complexity, grouped)


def fused_values(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], anchor: np.ndarray, best: np.ndarray, config: dict, external: dict[str, float], complexity: dict[str, float], grouped: dict) -> np.ndarray:
    direction = shaped_direction(best - anchor, config.get("style", "raw"))
    scaled = predict_model(train_rows, test_rows, denoised_target(train_rows, grouped, "trim10"), "hgb", external, complexity, 300)
    eb = predict_model(train_rows, test_rows, eb_target(train_rows, grouped, external, "hybrid", 24.0), "hgb", external, complexity, 280)
    hurdle_y = [external.get(row["word_id"], 0.0) * .45 + denoised_target([row], grouped, "zeroaware")[0] * .55 for row in train_rows]
    hurdle = predict_model(train_rows, test_rows, hurdle_y, "hgb", external, complexity, 260)
    raw_ext = np.asarray([external.get(row["word_id"], grouped["global_mean"]) for row in test_rows], dtype=float)
    out = best + float(config.get("gamma", .65)) * direction
    out += float(config.get("scaled_weight", .045)) * (zmatch(np.maximum(0.0, scaled), best) - best)
    out += float(config.get("external_weight", .04)) * (zmatch(raw_ext, best) - best)
    out += float(config.get("eb_weight", .035)) * (zmatch(np.maximum(0.0, eb), best) - best)
    out += float(config.get("hurdle_weight", .025)) * (zmatch(np.maximum(0.0, hurdle), best) - best)
    return out


def champion_values(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], anchor: np.ndarray, best: np.ndarray, config: dict, external: dict[str, float], complexity: dict[str, float], grouped: dict) -> tuple[np.ndarray, float]:
    for gamma in config.get("fallback_gammas", [.65, .50, .80, 1.00, .35]):
        candidate = fused_values(train_rows, test_rows, anchor, best, {**config, "gamma": gamma}, external, complexity, grouped)
        metrics = movement(candidate, best)
        if 8.0 <= metrics["mae_vs_199"] <= 18.0 and abs(metrics["mean_delta_vs_199"]) <= 10.0 and abs(metrics["std_delta_vs_199"]) <= 14.0:
            return candidate, float(gamma)
    return fused_values(train_rows, test_rows, anchor, best, {**config, "gamma": .50}, external, complexity, grouped), .50


def movement(values: np.ndarray, best: np.ndarray) -> dict[str, float]:
    return {"mae_vs_199": float(np.mean(np.abs(values - best))), "mean_delta_vs_199": float(np.mean(values) - np.mean(best)), "std_delta_vs_199": float(np.std(values) - np.std(best))}


def source_text(config: dict) -> str:
    return f'''from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={config!r}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--anchor-output',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.noise_aware_v2_emit import runtime_write
    runtime_write(a.train,a.test,a.anchor_output,a.best_output,a.output,CONFIG)
if __name__=='__main__': main()
'''


def runtime_write(train_path: Path, test_path: Path, anchor_path: Path, best_path: Path, output_path: Path, config: dict) -> None:
    train_rows, test_rows = read_csv(train_path), read_csv(test_path)
    external, complexity = external_maps(Path(".cache/noise_aware_v2_external"))
    grouped = grouped_stats(train_rows)
    anchor = output_values(anchor_path, test_rows)
    best = output_values(best_path, test_rows)
    values = champion_values(train_rows, test_rows, anchor, best, config, external, complexity, grouped)[0] if config.get("mode") == "champion" else component_values(train_rows, test_rows, anchor, best, config, external, complexity, grouped)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, value in zip(test_rows, np.clip(values, 0.0, 10000.0), strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{float(value):.6f}"})


def materialize(root: Path, spec: dict, args) -> dict:
    source = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_source.py"
    output = root / "official_candidates" / f"{spec['candidate_id']}_{spec['name']}_output.csv"
    source.write_text(source_text(spec["config"]), encoding="utf-8")
    cmd = ["python3", str(source), "--train", str(args.train), "--test", str(args.test), "--anchor-output", str(args.anchor_output), "--best-output", str(args.best_output), "--output", str(output)]
    subprocess.run(cmd, cwd=root, check=True)
    row = {key: value for key, value in spec.items() if key != "config"}
    row.update({"source_file": str(source.relative_to(root)), "output_file": str(output.relative_to(root)), "reproduction_command": " ".join(cmd).replace(str(root) + "/", "")})
    return row


def validate_candidate(root: Path, candidate: dict, test_ids: list[str], best_path: Path) -> dict:
    output = read_csv(root / candidate["output_file"])
    values = np.asarray([float(row["answer"]) for row in output], dtype=float)
    best = np.asarray([float(row["answer"]) for row in read_csv(best_path)], dtype=float)
    metrics = movement(values, best)
    row = {**candidate, **metrics, "source_bytes": (root / candidate["source_file"]).stat().st_size, "output_bytes": (root / candidate["output_file"]).stat().st_size, "row_count": len(output), "columns": list(output[0].keys()) if output else [], "ids_match_test_order": [row["datapointID"] for row in output] == test_ids, "answers_finite_non_negative": bool(np.all(np.isfinite(values)) and np.all(values >= 0.0)), **{f"answer_{key}": value for key, value in describe(values).items()}}
    row["upload_ready"] = row["source_bytes"] < SOURCE_LIMIT and row["output_bytes"] < OUTPUT_LIMIT and row["row_count"] == len(test_ids) and row["columns"] == ["subtaskID", "datapointID", "answer"] and row["ids_match_test_order"] and row["answers_finite_non_negative"]
    return row
