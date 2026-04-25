from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.noise_aware_emit import materialize, validate_candidate, write_json
    from experiments.noise_aware_profile import profile
    from experiments.noise_aware_specs import BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FUSION_LABEL, PREVIOUS_BEST_ID, PREVIOUS_BEST_SCORE, specs
    from experiments.three_tier_external_data import read_csv
except ModuleNotFoundError:
    from noise_aware_emit import materialize, validate_candidate, write_json
    from noise_aware_profile import profile
    from noise_aware_specs import BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FUSION_LABEL, PREVIOUS_BEST_ID, PREVIOUS_BEST_SCORE, specs
    from three_tier_external_data import read_csv


def record_best(ledger: dict) -> dict:
    ledger["best_official_score"] = BEST_SCORE
    ledger["best_official_candidate_id"] = BEST_ID
    ledger["best_official_submission_id"] = BEST_SUBMISSION_ID
    ledger["latest_official_feedback_candidate_id"] = BEST_ID
    ledger["latest_official_feedback_submission_id"] = BEST_SUBMISSION_ID
    ledger["latest_official_feedback_submission_id_missing"] = True
    ledger["latest_official_feedback_score"] = BEST_SCORE
    ledger["noise_aware_anchor"] = {"candidate_id": BEST_ID, "official_partial_score": BEST_SCORE, "previous_best_candidate_id": PREVIOUS_BEST_ID, "previous_best_official_score": PREVIOUS_BEST_SCORE, "next_search_direction": "train-label denoising and participant reliability modeling"}
    exists = any(row.get("candidate_id") == BEST_ID and row.get("submission_id") == BEST_SUBMISSION_ID for row in ledger.get("submissions", []))
    if not exists:
        ledger.setdefault("submissions", []).append({"candidate_id": BEST_ID, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "timestamp": "2026-04-25 21:10", "state": "success", "chosen_as_final": True, "official_partial_score": BEST_SCORE, "official_complete_score": None, "assignment_policy": "largest_numeric_candidate_id_at_manual_upload_time", "source_file": "official_candidates/159_rebound_champion_counter_prior_stack_source.py", "output_file": "official_candidates/159_rebound_champion_counter_prior_stack_output.csv", "local_report": "reports/failure_rebound_report.json", "hypothesis": "Candidate 159 improved by controlled rebound and becomes the noise-aware training anchor."})
    for row in ledger.get("pending_candidates", []):
        if row.get("candidate_id") == BEST_ID:
            row.update({"status": "official_scored", "official_partial_score": BEST_SCORE, "official_complete_score": None, "official_submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "next_manual_upload_target": False})
    return ledger


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    marker = "## Batch 13: Noise-Aware Denoising Push From Candidate 159"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    text += "Candidate `159` scored `38.87076 / 100` and is now the official best. This batch pivots to train-label denoising and makes Candidate `199` the next manual upload target.\n\n"
    text += "| Candidate | Family | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{candidate['family']}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += "\nNext manual upload target: `199_noise_champion_scaled_external_denoising`. Candidate `159` remains the official best until a higher judge score is recorded.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = record_best(json.loads(path.read_text(encoding="utf-8")))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    existing = {row.get("candidate_id"): row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids}
    for candidate in candidates:
        existing[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/noise_aware_push_report.json", "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "noise_evidence_groups": candidate["noise_evidence_groups"], "next_manual_upload_target": candidate["next_manual_upload_target"]}
    ledger["pending_candidates"] = sorted(existing.values(), key=lambda row: int(row.get("manual_upload_priority", 9999)))
    champion = candidates[-1]
    ledger["next_manual_upload_target"] = champion["candidate_id"]
    ledger["next_manual_upload_target_source"] = champion["source_file"]
    ledger["next_manual_upload_target_output"] = champion["output_file"]
    write_json(path, ledger)


def generate(root: Path, args: argparse.Namespace) -> dict:
    train_rows, test_rows = read_csv(args.train), read_csv(args.test)
    test_ids = [row["datapointID"] for row in test_rows]
    rows = [validate_candidate(root, materialize(root, spec, args), test_ids, args.best_output) for spec in specs(args.candidate_start)]
    champion = rows[-1]
    gate = {"mae_vs_159_min": 6.0, "mae_vs_159_max": 18.0, "mean_delta_abs_max": 12.0, "std_delta_abs_max": 18.0, "passed": 6.0 <= champion["mae_vs_159"] <= 18.0 and abs(champion["mean_delta_vs_159"]) <= 12.0 and abs(champion["std_delta_vs_159"]) <= 18.0}
    report = {"mode": "generate", "risk_label": FUSION_LABEL, "official_feedback": {"candidate_id": BEST_ID, "official_partial_score": BEST_SCORE, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True}, "noise_profile": profile(train_rows), "candidate_count": len(rows), "next_manual_upload_target": champion["candidate_id"], "movement_gate": gate, "candidates": rows}
    write_json(args.report, report)
    update_readme(root, rows)
    update_ledger(root, rows)
    return report
