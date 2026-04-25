from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.aggressive_feedback_emit import compare_outputs, materialize, validate_candidate, write_json
    from experiments.aggressive_feedback_specs import BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FUSION_LABEL, specs
    from experiments.three_tier_external_data import read_csv
except ModuleNotFoundError:
    from aggressive_feedback_emit import compare_outputs, materialize, validate_candidate, write_json
    from aggressive_feedback_specs import BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FUSION_LABEL, specs
    from three_tier_external_data import read_csv


def record_score(ledger: dict) -> dict:
    ledger["best_official_score"] = BEST_SCORE
    ledger["best_official_candidate_id"] = BEST_ID
    ledger["best_official_submission_id"] = BEST_SUBMISSION_ID
    ledger["latest_official_feedback_candidate_id"] = BEST_ID
    ledger["latest_official_feedback_submission_id"] = BEST_SUBMISSION_ID
    ledger["latest_official_feedback_submission_id_missing"] = True
    ledger["feedback_driven_next_candidate_anchor"] = {"candidate_id": BEST_ID, "official_partial_score": BEST_SCORE, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "next_search_direction": "aggressive trained three-tier fusion with official-direction extrapolation"}
    exists = any(row.get("candidate_id") == BEST_ID and row.get("submission_id") == BEST_SUBMISSION_ID for row in ledger.get("submissions", []))
    if not exists:
        ledger.setdefault("submissions", []).append({"candidate_id": BEST_ID, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "timestamp": "2026-04-25 20:10", "state": "success", "chosen_as_final": True, "official_partial_score": BEST_SCORE, "official_complete_score": None, "assignment_policy": "largest_numeric_candidate_id_at_manual_upload_time", "source_file": "official_candidates/099_feedback_champion_safe25_medium50_high25_source.py", "output_file": "official_candidates/099_feedback_champion_safe25_medium50_high25_output.csv", "local_report": "reports/feedback_champion_report.json", "hypothesis": "Candidate 099 made the first verified improvement over Candidate 087 but stayed too correlated, so it anchors an aggressive trained search."})
    for row in ledger.get("pending_candidates", []):
        if row.get("candidate_id") == BEST_ID:
            row.update({"status": "official_scored", "official_partial_score": BEST_SCORE, "official_complete_score": None, "official_submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "next_manual_upload_target": False})
    return ledger


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    marker = "## Batch 11: Aggressive Feedback Push From Candidate 099"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    text += "Candidate `099` scored `38.20618 / 100` and is now the official best. This batch stops small correlated movement by generating trained, three-tier fused candidates `100-129`. The next manual upload target is Candidate `129`.\n\n"
    text += "| Candidate | Family | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{candidate['family']}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += "\nNext manual upload target: `129_aggressive_aggressive_champion_trained_direction_risk`. Do not claim an official gain until judge feedback is recorded.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = record_score(json.loads(path.read_text(encoding="utf-8")))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    existing = {row.get("candidate_id"): row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids}
    for candidate in candidates:
        existing[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/aggressive_feedback_push_report.json", "local_score": None, "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "trained_feature_groups": candidate["trained_feature_groups"], "next_manual_upload_target": candidate["next_manual_upload_target"]}
    ledger["pending_candidates"] = sorted(existing.values(), key=lambda row: int(row.get("manual_upload_priority", 9999)))
    champion = candidates[-1]
    ledger["next_manual_upload_target"] = champion["candidate_id"]
    ledger["next_manual_upload_target_source"] = champion["source_file"]
    ledger["next_manual_upload_target_output"] = champion["output_file"]
    write_json(path, ledger)


def generate(root: Path, args: argparse.Namespace) -> dict:
    candidate_specs = specs(args.candidate_start)
    test_ids = [row["datapointID"] for row in read_csv(args.test)]
    rows = [validate_candidate(root, materialize(root, spec, args), test_ids) for spec in candidate_specs]
    for row in rows:
        row.update(compare_outputs(root, row, args.best_output))
    champion = rows[-1]
    report = {"mode": "generate", "risk_label": FUSION_LABEL, "official_feedback": {"candidate_id": BEST_ID, "official_partial_score": BEST_SCORE, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True}, "candidate_count": len(rows), "next_manual_upload_target": champion["candidate_id"], "champion_difference_gate": {"corr_vs_099_max": 0.995, "mae_vs_099_min": 15.0, "passed": champion["corr_vs_099"] <= 0.995 and champion["mae_vs_099"] >= 15.0}, "candidates": rows}
    write_json(args.report, report)
    update_readme(root, rows)
    update_ledger(root, rows)
    return report
