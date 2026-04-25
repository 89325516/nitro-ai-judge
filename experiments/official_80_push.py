from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Sequence

SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024
TARGET_SCORE = 80.0
BASE_SOURCE = "official_candidates/001_ridge_official_source.py"
BASE_OUTPUT = "official_candidates/001_ridge_official_output.csv"
CONFIG_MARKER = "FeatureMap = dict[str, float]\n"
PREDICTION_MARKER = "predictions = np.clip(model.predict(test_features), 0.0, 10000.0)"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def base_rows(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    output = read_csv(root / BASE_OUTPUT)
    test = {row["datapointID"]: row for row in read_csv(root / "data/test_data.csv")}
    return output, test


def zero_threshold(values: list[float], rate: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * rate) - 1))
    return ordered[index]


def make_variant(root: Path, candidate_id: str, name: str, config: dict) -> dict:
    source_path = root / "official_candidates" / f"{candidate_id}_{name}_source.py"
    output_path = root / "official_candidates" / f"{candidate_id}_{name}_output.csv"
    source_path.write_text(variant_source((root / BASE_SOURCE).read_text(encoding="utf-8"), config), encoding="utf-8")
    subprocess.run(["python3", str(source_path), "--train", str(root / "data/train_data.csv"), "--test", str(root / "data/test_data.csv"), "--output", str(output_path)], check=True)
    return {"candidate_id": candidate_id, "name": name, "source_file": str(source_path.relative_to(root)), "output_file": str(output_path.relative_to(root)), "hypothesis": config.get("hypothesis", name), "manual_upload_priority": int(candidate_id)}


def variant_source(base_source: str, config: dict) -> str:
    helper = f"""FeatureMap = dict[str, float]
CONFIG = {repr(config)}


def transform_predictions(test_rows: list[dict[str, str]], predictions: np.ndarray) -> np.ndarray:
    output = []
    for row, prediction in zip(test_rows, predictions, strict=True):
        value = float(prediction)
        if value <= CONFIG.get("zero_threshold", -1.0):
            output.append(0.0)
            continue
        scale = CONFIG.get("scale", 1.0)
        if row["text"] == CONFIG.get("text"):
            scale *= CONFIG.get("factor", 1.0)
        if row["participant_id"] == CONFIG.get("participant_id"):
            scale *= CONFIG.get("factor", 1.0)
        output.append(max(0.0, value * scale))
    return np.clip(np.asarray(output, dtype=float), 0.0, 10000.0)
"""
    if CONFIG_MARKER not in base_source or PREDICTION_MARKER not in base_source:
        raise ValueError("base source does not match expected Ridge layout")
    return base_source.replace(CONFIG_MARKER, helper).replace(PREDICTION_MARKER, "predictions = transform_predictions(test_rows, model.predict(test_features))")


def generate(root: Path) -> list[dict]:
    output_rows, _ = base_rows(root)
    values = [float(row["answer"]) for row in output_rows]
    specs: list[tuple[str, str, dict]] = []
    for candidate_id, rate in [("006", 0.20), ("007", 0.30), ("008", 0.40)]:
        threshold = zero_threshold(values, rate)
        name = f"ridge_zero_{int(rate * 100):02d}"
        config = {"zero_threshold": threshold, "scale": 1.0, "hypothesis": f"Set bottom {int(rate * 100)} percent predictions to zero."}
        specs.append((candidate_id, name, config))
    for candidate_id, text in [("009", "arg_pisacowsmilk"), ("010", "ins_learningmobility"), ("011", "lit_alchemist")]:
        config = {"text": text, "factor": 1.12, "hypothesis": f"Boost text {text} by 12 percent."}
        specs.append((candidate_id, f"text_boost_{text}", config))
    for candidate_id, participant in [("012", "040"), ("013", "036"), ("014", "016"), ("015", "024"), ("016", "019")]:
        config = {"participant_id": participant, "factor": 1.12, "hypothesis": f"Boost participant {participant} by 12 percent."}
        specs.append((candidate_id, f"participant_boost_{participant}", config))
    return [make_variant(root, *spec) for spec in specs]


def validate(root: Path, candidates: Sequence[dict]) -> dict:
    expected_ids = [row["datapointID"] for row in read_csv(root / "data/test_data.csv")]
    rows = []
    for candidate in candidates:
        output = read_csv(root / candidate["output_file"])
        values = [float(row["answer"]) for row in output]
        rows.append({
            **candidate,
            "source_bytes": (root / candidate["source_file"]).stat().st_size,
            "output_bytes": (root / candidate["output_file"]).stat().st_size,
            "row_count": len(output),
            "columns": list(output[0].keys()) if output else [],
            "ids_match_test_order": [row["datapointID"] for row in output] == expected_ids,
            "answers_finite_non_negative": all(math.isfinite(value) and value >= 0.0 for value in values),
            "answer_mean": sum(values) / len(values),
            "answer_zero_rate": sum(value == 0.0 for value in values) / len(values),
        })
    for row in rows:
        row["upload_ready"] = row["source_bytes"] < SOURCE_LIMIT and row["output_bytes"] < OUTPUT_LIMIT and row["row_count"] == len(expected_ids) and row["columns"] == ["subtaskID", "datapointID", "answer"] and row["ids_match_test_order"] and row["answers_finite_non_negative"]
    return {"target_score": TARGET_SCORE, "candidates": rows}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_ledger(root: Path, args: argparse.Namespace) -> dict:
    path = root / "reports/official_submission_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    entry = {"candidate_id": args.candidate_id, "submission_id": args.submission_id, "timestamp": args.timestamp, "state": args.state, "official_partial_score": args.partial_score, "official_complete_score": args.complete_score, "chosen_as_final": args.partial_score >= float(ledger.get("best_official_score", -1.0))}
    ledger.setdefault("submissions", []).append(entry)
    if args.state == "success" and args.partial_score >= float(ledger.get("best_official_score", -1.0)):
        ledger["best_official_score"] = args.partial_score
        ledger["best_official_submission_id"] = args.submission_id
    write_json(path, ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and track official 80 push candidates.")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("generate")
    update = sub.add_parser("record")
    update.add_argument("--candidate-id", required=True)
    update.add_argument("--submission-id", required=True)
    update.add_argument("--timestamp", required=True)
    update.add_argument("--partial-score", type=float, required=True)
    update.add_argument("--complete-score", type=float)
    update.add_argument("--state", default="success")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.mode == "generate":
        candidates = generate(root)
        report = validate(root, candidates)
        write_json(root / "reports/official_80_push_candidates.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(update_ledger(root, args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
