from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.three_tier_external_data import FUSION_LABEL, HIGH_LABEL, MEDIUM_LABEL, SAFE_LABEL, public_rows, read_csv
    from experiments.three_tier_external_emit import materialize, validate_candidate, write_json
except ModuleNotFoundError:
    from three_tier_external_data import FUSION_LABEL, HIGH_LABEL, MEDIUM_LABEL, SAFE_LABEL, public_rows, read_csv
    from three_tier_external_emit import materialize, validate_candidate, write_json

ANCHOR_ID = "087"
ANCHOR_SCORE = 37.35945
ANCHOR_MODE = "base"
ANCHOR_MAPPING = {"040": "010", "036": "011", "016": "008", "024": "023", "019": "009"}
RISK_LABELS = [SAFE_LABEL, MEDIUM_LABEL, HIGH_LABEL]


def config(weights: tuple[float, float, float], scale: float, **extra: float) -> dict:
    payload = {"mode": ANCHOR_MODE, "weights": {"safe": weights[0], "medium": weights[1], "high": weights[2]}, "scale": scale, "mapping": ANCHOR_MAPPING, "risk_labels": RISK_LABELS}
    payload.update(extra)
    return payload


def specs(start: int) -> list[dict]:
    rows = [
        ("anchor_clone", (0.20, 0.45, 0.35), 1.04, {}, "Reproduce official Candidate 087 exactly as the feedback anchor."),
        ("safe25_medium45_high30", (0.25, 0.45, 0.30), 1.04, {}, "Shift five points from high-risk reconstruction into safe language signal."),
        ("safe25_medium50_high25", (0.25, 0.50, 0.25), 1.04, {}, "Raise medium public behavior weight while reducing high-risk reconstruction."),
        ("safe30_medium45_high25", (0.30, 0.45, 0.25), 1.04, {}, "Test stronger safe language generalization at the same medium weight."),
        ("safe20_medium55_high25", (0.20, 0.55, 0.25), 1.04, {}, "Push the strongest medium public behavior weighting near the anchor."),
        ("safe30_medium50_high20", (0.30, 0.50, 0.20), 1.04, {}, "Reduce reconstruction most while keeping all three tiers active."),
        ("anchor_scale098", (0.20, 0.45, 0.35), 0.98, {}, "Lower anchor scale to test official distribution sensitivity."),
        ("anchor_scale100", (0.20, 0.45, 0.35), 1.00, {}, "Reset anchor scale to neutral calibration."),
        ("anchor_scale102", (0.20, 0.45, 0.35), 1.02, {}, "Slightly lower scale than Candidate 087 without changing weights."),
        ("zero_tail02", (0.25, 0.50, 0.25), 1.02, {"zero_quantile": 0.02}, "Zero the lowest two percent after stable three-tier fusion."),
        ("floor_tail02", (0.25, 0.50, 0.25), 1.02, {"floor_quantile": 0.02}, "Clip the lowest two percent upward after stable three-tier fusion."),
        ("champion_safe25_medium50_high25", (0.25, 0.50, 0.25), 1.02, {}, "Champion upload target centered on stronger safe and medium signals."),
    ]
    output = []
    for offset, (name, weights, scale, extra, hypothesis) in enumerate(rows):
        candidate_id = f"{start + offset:03d}"
        cfg = config(weights, scale, **extra)
        output.append({"candidate_id": candidate_id, "name": f"feedback_{name}", "config": cfg, "family": "feedback_champion", "mode": ANCHOR_MODE, "component_weights": cfg["weights"], "scale": scale, "risk_label": FUSION_LABEL, "risk_labels": RISK_LABELS, "anchor_candidate_id": ANCHOR_ID, "anchor_official_score": ANCHOR_SCORE, "next_manual_upload_target": candidate_id == f"{start + len(rows) - 1:03d}", "hypothesis": hypothesis})
    return output


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    marker = "## Batch 10: Feedback Champion Around Candidate 087"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    text += "This batch treats Candidate `087` as the official feedback anchor at `37.35945 / 100`. The largest ID is the manual upload champion, so upload Candidate `099` next.\n\n"
    text += "| Candidate | Weights safe/medium/high | Scale | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        weights = candidate["component_weights"]
        triplet = f"{weights['safe']:.2f}/{weights['medium']:.2f}/{weights['high']:.2f}"
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{triplet}` | `{candidate['scale']:.2f}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += "\nNext manual upload target: `099_feedback_champion_safe25_medium50_high25`.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    existing = {row.get("candidate_id"): row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids}
    for candidate in candidates:
        existing[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/feedback_champion_report.json", "local_score": None, "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "component_weights": candidate["component_weights"], "scale": candidate["scale"], "anchor_candidate_id": ANCHOR_ID, "next_manual_upload_target": candidate["next_manual_upload_target"]}
    ledger["pending_candidates"] = sorted(existing.values(), key=lambda row: int(row.get("manual_upload_priority", 9999)))
    ledger["next_manual_upload_target"] = "099"
    ledger["next_manual_upload_target_source"] = "official_candidates/099_feedback_champion_safe25_medium50_high25_source.py"
    ledger["next_manual_upload_target_output"] = "official_candidates/099_feedback_champion_safe25_medium50_high25_output.csv"
    write_json(path, ledger)


def generate(root: Path, args: argparse.Namespace) -> dict:
    public_rows(root / ".cache/three_tier_external")
    candidate_specs = specs(args.candidate_start)
    materialized = [materialize(root, spec, args) for spec in candidate_specs]
    test_ids = [row["datapointID"] for row in read_csv(args.test)]
    candidates = [validate_candidate(root, candidate, test_ids) for candidate in materialized]
    report = {"mode": "generate", "risk_label": FUSION_LABEL, "anchor_candidate_id": ANCHOR_ID, "anchor_official_score": ANCHOR_SCORE, "candidate_count": len(candidates), "next_manual_upload_target": "099", "candidates": candidates}
    write_json(args.report, report)
    update_readme(root, candidates)
    update_ledger(root, candidates)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feedback-driven champion candidates around Candidate 087.")
    parser.add_argument("generate", nargs="?")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--high-risk-base-output", type=Path, required=True)
    parser.add_argument("--candidate-start", type=int, default=88)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(Path(__file__).resolve().parents[1], args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
