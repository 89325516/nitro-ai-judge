from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.failure_rebound_emit import materialize, validate_candidate, write_json
    from experiments.failure_rebound_specs import BEST_ID, BEST_SCORE, FAILED_ID, FAILED_SCORE, FAILED_SUBMISSION_ID, FUSION_LABEL, specs
    from experiments.three_tier_external_data import read_csv
except ModuleNotFoundError:
    from failure_rebound_emit import materialize, validate_candidate, write_json
    from failure_rebound_specs import BEST_ID, BEST_SCORE, FAILED_ID, FAILED_SCORE, FAILED_SUBMISSION_ID, FUSION_LABEL, specs
    from three_tier_external_data import read_csv


def record_failure(ledger: dict) -> dict:
    ledger["best_official_score"] = BEST_SCORE
    ledger["best_official_candidate_id"] = BEST_ID
    ledger["latest_official_feedback_candidate_id"] = FAILED_ID
    ledger["latest_official_feedback_submission_id"] = FAILED_SUBMISSION_ID
    ledger["latest_official_feedback_submission_id_missing"] = True
    ledger["latest_official_feedback_score"] = FAILED_SCORE
    ledger["feedback_failure_anchor"] = {"candidate_id": FAILED_ID, "official_partial_score": FAILED_SCORE, "best_candidate_id": BEST_ID, "best_official_score": BEST_SCORE, "next_search_direction": "damp or reverse the failed 099-to-129 movement"}
    exists = any(row.get("candidate_id") == FAILED_ID and row.get("submission_id") == FAILED_SUBMISSION_ID for row in ledger.get("submissions", []))
    if not exists:
        ledger.setdefault("submissions", []).append({"candidate_id": FAILED_ID, "submission_id": FAILED_SUBMISSION_ID, "submission_id_missing": True, "timestamp": "2026-04-25 20:45", "state": "success", "chosen_as_final": False, "official_partial_score": FAILED_SCORE, "official_complete_score": None, "assignment_policy": "largest_numeric_candidate_id_at_manual_upload_time", "source_file": "official_candidates/129_aggressive_champion_trained_direction_risk_source.py", "output_file": "official_candidates/129_aggressive_champion_trained_direction_risk_output.csv", "local_report": "reports/aggressive_feedback_push_report.json", "hypothesis": "Candidate 129 moved too far from Candidate 099 and became a negative feedback anchor."})
    for row in ledger.get("pending_candidates", []):
        if row.get("candidate_id") == FAILED_ID:
            row.update({"status": "official_scored", "official_partial_score": FAILED_SCORE, "official_complete_score": None, "official_submission_id": FAILED_SUBMISSION_ID, "submission_id_missing": True, "next_manual_upload_target": False})
    return ledger


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    marker = "## Batch 12: Failure Rebound From Candidate 129"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    text += "Candidate `129` scored `37.89482 / 100`, below Candidate `099` at `38.20618 / 100`. This batch records `129` as a negative anchor and makes Candidate `159` the next manual upload target.\n\n"
    text += "| Candidate | Family | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{candidate['family']}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += "\nNext manual upload target: `159_rebound_champion_counter_prior_stack`. Candidate `099` remains the official best until a higher judge score is recorded.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = record_failure(json.loads(path.read_text(encoding="utf-8")))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    existing = {row.get("candidate_id"): row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids}
    for candidate in candidates:
        existing[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/failure_rebound_report.json", "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "fused_evidence_groups": candidate["fused_evidence_groups"], "next_manual_upload_target": candidate["next_manual_upload_target"]}
    ledger["pending_candidates"] = sorted(existing.values(), key=lambda row: int(row.get("manual_upload_priority", 9999)))
    champion = candidates[-1]
    ledger["next_manual_upload_target"] = champion["candidate_id"]
    ledger["next_manual_upload_target_source"] = champion["source_file"]
    ledger["next_manual_upload_target_output"] = champion["output_file"]
    write_json(path, ledger)


def generate(root: Path, args: argparse.Namespace) -> dict:
    test_ids = [row["datapointID"] for row in read_csv(args.test)]
    rows = [validate_candidate(root, materialize(root, spec, args), test_ids, args.best_output, args.failed_output) for spec in specs(args.candidate_start)]
    champion = rows[-1]
    failed_mae = next(row["mae_vs_099"] for row in rows if row["candidate_id"] == FAILED_ID) if any(row["candidate_id"] == FAILED_ID for row in rows) else 29.54
    gate = {"mae_vs_099_min": 8.0, "mae_vs_099_max": failed_mae, "passed": 8.0 <= champion["mae_vs_099"] < failed_mae}
    report = {"mode": "generate", "risk_label": FUSION_LABEL, "official_feedback": {"best_candidate_id": BEST_ID, "best_official_score": BEST_SCORE, "failed_candidate_id": FAILED_ID, "failed_official_score": FAILED_SCORE, "failed_submission_id": FAILED_SUBMISSION_ID, "submission_id_missing": True}, "candidate_count": len(rows), "next_manual_upload_target": champion["candidate_id"], "rebound_gate": gate, "candidates": rows}
    write_json(args.report, report)
    update_readme(root, rows)
    update_ledger(root, rows)
    return report
