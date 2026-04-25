from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.noise_aware_v3_emit import build_context, materialize, validate_candidate, write_json
    from experiments.noise_aware_v3_specs import BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FAMILY, NEUTRAL_ID, NEUTRAL_SCORE, NEUTRAL_SUBMISSION_ID, RISK_LABEL, specs
except ModuleNotFoundError:
    from noise_aware_v3_emit import build_context, materialize, validate_candidate, write_json
    from noise_aware_v3_specs import BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FAMILY, NEUTRAL_ID, NEUTRAL_SCORE, NEUTRAL_SUBMISSION_ID, RISK_LABEL, specs


def record_feedback(ledger: dict) -> dict:
    ledger.update({"best_official_score": BEST_SCORE, "best_official_candidate_id": BEST_ID, "best_official_submission_id": BEST_SUBMISSION_ID, "latest_official_feedback_candidate_id": NEUTRAL_ID, "latest_official_feedback_submission_id": NEUTRAL_SUBMISSION_ID, "latest_official_feedback_submission_id_missing": True, "latest_official_feedback_score": NEUTRAL_SCORE, "latest_manual_submission_policy": "largest_numeric_candidate_id_at_manual_upload_time"})
    ledger["noise_aware_v3_anchor"] = {"official_best_candidate_id": BEST_ID, "official_best_score": BEST_SCORE, "neutral_candidate_id": NEUTRAL_ID, "neutral_score": NEUTRAL_SCORE, "next_search_direction": "low-noise targets, higher-capacity tabular models, robust noisy-label stacking"}
    exists = any(row.get("candidate_id") == NEUTRAL_ID and row.get("submission_id") == NEUTRAL_SUBMISSION_ID for row in ledger.get("submissions", []))
    if not exists:
        ledger.setdefault("submissions", []).append({"candidate_id": NEUTRAL_ID, "submission_id": NEUTRAL_SUBMISSION_ID, "submission_id_missing": True, "timestamp": "2026-04-25 22:45", "state": "success", "chosen_as_final": False, "official_partial_score": NEUTRAL_SCORE, "official_complete_score": None, "assignment_policy": "largest_numeric_candidate_id_at_manual_upload_time", "source_file": "official_candidates/249_noise_v2_champion_directional_denoising_source.py", "output_file": "official_candidates/249_noise_v2_champion_directional_denoising_output.csv", "local_report": "reports/noise_aware_v2_push_report.json", "hypothesis": "Candidate 249 matched Candidate 199, so it anchors the no-gain direction for V3."})
    for row in ledger.get("pending_candidates", []):
        if row.get("candidate_id") == NEUTRAL_ID:
            row.update({"status": "official_scored", "official_partial_score": NEUTRAL_SCORE, "official_complete_score": None, "official_submission_id": NEUTRAL_SUBMISSION_ID, "submission_id_missing": True, "next_manual_upload_target": False})
        elif row.get("status") == "pending" and 200 <= int(row.get("candidate_id", 0)) < 250:
            row.update({"status": "superseded_unsubmitted", "next_manual_upload_target": False})
    return ledger


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = record_feedback(json.loads(path.read_text(encoding="utf-8")))
    for candidate in candidates:
        ledger.setdefault("pending_candidates", [])[:] = [row for row in ledger["pending_candidates"] if row.get("candidate_id") != candidate["candidate_id"]]
        ledger["pending_candidates"].append({"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/noise_aware_v3_push_report.json", "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "method_labels": candidate["method_labels"], "component_weights": candidate["component_weights"], "next_manual_upload_target": candidate["next_manual_upload_target"]})
    ledger["pending_candidates"] = sorted(ledger["pending_candidates"], key=lambda row: int(row.get("manual_upload_priority", row.get("candidate_id", 9999))))
    champion = candidates[-1]
    ledger["next_manual_upload_target"] = champion["candidate_id"]; ledger["next_manual_upload_target_source"] = champion["source_file"]; ledger["next_manual_upload_target_output"] = champion["output_file"]
    write_json(path, ledger)


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"; marker = "## Current Batch: Noise-Aware V3 Low-Noise Push"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip()
    text = text.replace("Current next manual upload target: `249_noise_v2_champion_directional_denoising`. Candidate `199` remains the official best until a higher judge score is recorded.", "Historical next manual upload target for this batch: `249_noise_v2_champion_directional_denoising`. Candidate `249` later matched Candidate `199` at `53.8099243 / 100`.")
    text += "\n\n" + marker + "\n\nCandidate `249` matched Candidate `199` at `53.8099243 / 100`, so Candidate `199` remains the official best. This smaller batch uses low-noise targets, higher-capacity models, and robust training; Candidate `269` is the next manual upload target.\n\n"
    text += "| Candidate | Family | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{candidate['family']}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += "\nCurrent next manual upload target: `269_noise_v3_champion_low_noise_stack`. Candidate `199` remains the official best until a higher judge score is recorded.\n"
    path.write_text(text, encoding="utf-8")


def generate(root: Path, args: argparse.Namespace) -> dict:
    ctx = build_context(args.train, args.test, args.best_output, args.neutral_output)
    test_ids = [row["datapointID"] for row in ctx["test"]]
    rows = [validate_candidate(root, materialize(root, spec, args, ctx), test_ids, args.best_output) for spec in specs(args.candidate_start)]
    champion = rows[-1]
    gate = {"mae_vs_199_min": 5.0, "mae_vs_199_max": 14.0, "mean_delta_abs_max": 8.0, "std_delta_abs_max": 12.0, "passed": 5.0 <= champion["mae_vs_199"] <= 14.0 and abs(champion["mean_delta_vs_199"]) <= 8.0 and abs(champion["std_delta_vs_199"]) <= 12.0}
    report = {"mode": "generate", "family": FAMILY, "risk_label": RISK_LABEL, "official_feedback": {"candidate_id": NEUTRAL_ID, "official_partial_score": NEUTRAL_SCORE, "submission_id": NEUTRAL_SUBMISSION_ID, "submission_id_missing": True, "best_remains_candidate_id": BEST_ID}, "low_noise_target_profile": ctx["target_profile"], "external_consensus_coverage": {"train_item": ctx["target_profile"]["external_train_item_coverage"], "test_row": ctx["target_profile"]["external_test_row_coverage"]}, "model_capacity_summary": {"hgb_max_iter": 700, "extra_trees_max_estimators": 768, "candidate_count": len(rows)}, "robust_training_method_labels": sorted({label for row in rows for label in row["method_labels"]}), "candidate_count": len(rows), "next_manual_upload_target": champion["candidate_id"], "movement_gate": gate, "candidates": rows}
    write_json(args.report, report); update_readme(root, rows); update_ledger(root, rows)
    return report
