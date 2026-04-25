from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

try:
    from experiments.noise_aware_profile import profile
    from experiments.noise_aware_v2_emit import materialize, validate_candidate, write_json
    from experiments.noise_aware_v2_specs import ANCHOR_ID, ANCHOR_SCORE, BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FAMILY, RISK_LABEL, specs
    from experiments.three_tier_external_data import read_csv
except ModuleNotFoundError:
    from noise_aware_profile import profile
    from noise_aware_v2_emit import materialize, validate_candidate, write_json
    from noise_aware_v2_specs import ANCHOR_ID, ANCHOR_SCORE, BEST_ID, BEST_SCORE, BEST_SUBMISSION_ID, FAMILY, RISK_LABEL, specs
    from three_tier_external_data import read_csv

KEEP_IDS = {"001", "002", "023", "045", "087", "099", "129", "159", "199"}


def record_best(ledger: dict) -> dict:
    ledger.update({"best_official_score": BEST_SCORE, "best_official_candidate_id": BEST_ID, "best_official_submission_id": BEST_SUBMISSION_ID, "latest_official_feedback_candidate_id": BEST_ID, "latest_official_feedback_submission_id": BEST_SUBMISSION_ID, "latest_official_feedback_submission_id_missing": True, "latest_official_feedback_score": BEST_SCORE, "latest_manual_submission_policy": "largest_numeric_candidate_id_at_manual_upload_time"})
    ledger["noise_aware_v2_anchor"] = {"candidate_id": BEST_ID, "official_partial_score": BEST_SCORE, "previous_noise_anchor_candidate_id": ANCHOR_ID, "previous_noise_anchor_score": ANCHOR_SCORE, "next_search_direction": "extend the officially validated 159-to-199 denoising direction"}
    exists = any(row.get("candidate_id") == BEST_ID and row.get("submission_id") == BEST_SUBMISSION_ID for row in ledger.get("submissions", []))
    if not exists:
        ledger.setdefault("submissions", []).append({"candidate_id": BEST_ID, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "timestamp": "2026-04-25 22:05", "state": "success", "chosen_as_final": True, "official_partial_score": BEST_SCORE, "official_complete_score": None, "assignment_policy": "largest_numeric_candidate_id_at_manual_upload_time", "source_file": "official_candidates/199_noise_champion_scaled_external_denoising_source.py", "output_file": "official_candidates/199_noise_champion_scaled_external_denoising_output.csv", "local_report": "reports/noise_aware_push_report.json", "hypothesis": "Candidate 199 validates the noise-aware denoising direction and anchors V2."})
    for row in ledger.get("pending_candidates", []):
        if row.get("candidate_id") == BEST_ID:
            row.update({"status": "official_scored", "official_partial_score": BEST_SCORE, "official_complete_score": None, "official_submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True, "next_manual_upload_target": False})
    return ledger


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    marker = "## Batch 14: Noise-Aware V2 Directional Push From Candidate 199"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip()
    text = text.replace("Current next manual upload target: `199_noise_champion_scaled_external_denoising`. Candidate `159` remains the official best until a higher judge score is recorded.", "Historical next manual upload target for this batch: `199_noise_champion_scaled_external_denoising`. Candidate `199` later scored `53.8099243 / 100`.")
    text += "\n\n" + marker + "\n\n"
    text += "Candidate `199` scored `53.8099243 / 100` and is now the official best. This batch continues the same noise-aware direction and makes Candidate `249` the next manual upload target. Older unscored exploration files are pruned; official anchors remain tracked.\n\n"
    text += "| Candidate | Family | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{candidate['family']}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += "\nCurrent next manual upload target: `249_noise_v2_champion_directional_denoising`. Candidate `199` remains the official best until a higher judge score is recorded.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = record_best(json.loads(path.read_text(encoding="utf-8")))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    kept = {}
    for row in ledger.get("pending_candidates", []):
        cid = str(row.get("candidate_id", "")).zfill(3)
        if cid in KEEP_IDS or row.get("status") == "official_scored":
            row = {**row, "next_manual_upload_target": False}
            if cid in {"023", "045"} and row.get("status") == "pending":
                row["status"] = "retained_input"
            kept[row["candidate_id"]] = row
    for candidate in candidates:
        kept[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/noise_aware_v2_push_report.json", "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "noise_evidence_groups": candidate["noise_evidence_groups"], "component_weights": candidate.get("component_weights"), "next_manual_upload_target": candidate["next_manual_upload_target"]}
    ledger["pending_candidates"] = sorted(kept.values(), key=lambda row: int(row.get("manual_upload_priority", row.get("candidate_id", 9999))))
    champion = candidates[-1]
    ledger["next_manual_upload_target"] = champion["candidate_id"]
    ledger["next_manual_upload_target_source"] = champion["source_file"]
    ledger["next_manual_upload_target_output"] = champion["output_file"]
    ledger["pruned_candidate_file_policy"] = "Keep official anchors and current V2 candidates; remove unscored historical exploration source/output files."
    write_json(path, ledger)


def delete_pruned_candidate_files(root: Path, keep_ids: set[str]) -> list[str]:
    removed: set[str] = set()
    tracked = subprocess.run(["git", "ls-files", "official_candidates"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    for rel in tracked:
        name = Path(rel).name
        if len(name) >= 3 and name[:3].isdigit() and name[:3] not in keep_ids and int(name[:3]) < 200:
            path = root / rel
            if path.exists():
                path.unlink()
            removed.add(rel)
    for path in (root / "official_candidates").glob("[0-9][0-9][0-9]_*_*.*"):
        cid = path.name[:3]
        if cid not in keep_ids and int(cid) < 200:
            path.unlink()
            removed.add(str(path.relative_to(root)))
    return sorted(removed)


def generate(root: Path, args: argparse.Namespace) -> dict:
    train_rows, test_rows = read_csv(args.train), read_csv(args.test)
    test_ids = [row["datapointID"] for row in test_rows]
    rows = [validate_candidate(root, materialize(root, spec, args), test_ids, args.best_output) for spec in specs(args.candidate_start)]
    champion = rows[-1]
    gate = {"mae_vs_199_min": 8.0, "mae_vs_199_max": 18.0, "mean_delta_abs_max": 10.0, "std_delta_abs_max": 14.0, "passed": 8.0 <= champion["mae_vs_199"] <= 18.0 and abs(champion["mean_delta_vs_199"]) <= 10.0 and abs(champion["std_delta_vs_199"]) <= 14.0}
    keep_ids = KEEP_IDS | {row["candidate_id"] for row in rows}
    pruned = delete_pruned_candidate_files(root, keep_ids)
    report = {"mode": "generate", "risk_label": RISK_LABEL, "family": FAMILY, "official_feedback": {"candidate_id": BEST_ID, "official_partial_score": BEST_SCORE, "submission_id": BEST_SUBMISSION_ID, "submission_id_missing": True}, "official_direction": {"from_candidate_id": ANCHOR_ID, "to_candidate_id": BEST_ID, "from_score": ANCHOR_SCORE, "to_score": BEST_SCORE}, "noise_profile": profile(train_rows), "candidate_count": len(rows), "next_manual_upload_target": champion["candidate_id"], "movement_gate": gate, "pruned_candidate_files": pruned, "candidates": rows}
    write_json(args.report, report)
    update_readme(root, rows)
    update_ledger(root, rows)
    return report
