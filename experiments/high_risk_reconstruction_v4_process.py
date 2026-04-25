from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.high_risk_reconstruction_v4_components import describe
from experiments.high_risk_reconstruction_v4_data import RISK_LABEL
from experiments.high_risk_reconstruction_v4_emit import build_context, materialize, validate_candidate, write_json
from experiments.high_risk_reconstruction_v4_specs import FAMILY, build_specs


def record_269_feedback(ledger: dict, args: argparse.Namespace) -> dict:
    score = float(args.official_score)
    submission_id = f"manual_269_score_{str(score).replace('.', '_')}_approx" if args.score_approximate else f"manual_269_score_{str(score).replace('.', '_')}"
    ledger.update({"best_official_score": score, "best_official_candidate_id": "269", "best_official_submission_id": submission_id, "best_official_score_approximate": bool(args.score_approximate), "latest_official_feedback_candidate_id": "269", "latest_official_feedback_submission_id": submission_id, "latest_official_feedback_submission_id_missing": True, "latest_official_feedback_score": score, "latest_official_feedback_score_approximate": bool(args.score_approximate)})
    ledger["high_risk_reconstruction_v4_anchor"] = {"official_best_candidate_id": "269", "official_best_score": score, "score_approximate": bool(args.score_approximate), "previous_best_candidate_id": "199", "next_search_direction": "participant TRT reconstruction, zero behavior, scale restoration"}
    exists = any(row.get("candidate_id") == "269" and row.get("submission_id") == submission_id for row in ledger.get("submissions", []))
    if not exists:
        ledger.setdefault("submissions", []).append({"candidate_id": "269", "submission_id": submission_id, "submission_id_missing": True, "timestamp": "2026-04-25 23:30", "state": "success", "chosen_as_final": True, "official_partial_score": score, "official_score_approximate": bool(args.score_approximate), "official_complete_score": None, "assignment_policy": "largest_numeric_candidate_id_at_manual_upload_time", "source_file": "official_candidates/269_noise_v3_champion_low_noise_stack_source.py", "output_file": "official_candidates/269_noise_v3_champion_low_noise_stack_output.csv", "local_report": "reports/noise_aware_v3_push_report.json", "hypothesis": "Candidate 269 improved over Candidate 199 but still requires high-risk row-level reconstruction for a larger jump."})
    for row in ledger.get("pending_candidates", []):
        cid = row.get("candidate_id")
        if cid == "269":
            row.update({"status": "official_scored", "official_partial_score": score, "official_score_approximate": bool(args.score_approximate), "official_submission_id": submission_id, "submission_id_missing": True, "next_manual_upload_target": False})
        elif cid and row.get("status") == "pending" and 250 <= int(cid) < 269:
            row.update({"status": "superseded_unsubmitted", "next_manual_upload_target": False})
    return ledger


def update_ledger(root: Path, candidates: Sequence[dict], args: argparse.Namespace) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = record_269_feedback(json.loads(path.read_text(encoding="utf-8")), args)
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    ledger["pending_candidates"] = [row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids]
    for candidate in candidates:
        ledger["pending_candidates"].append({"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/high_risk_reconstruction_v4_report.json", "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "component_labels": candidate["component_labels"], "participant_mapping_mode": candidate["participant_mapping_mode"], "zero_strategy": candidate["zero_strategy"], "scale_strategy": candidate["scale_strategy"], "next_manual_upload_target": candidate["next_manual_upload_target"]})
    ledger["pending_candidates"] = sorted(ledger["pending_candidates"], key=lambda row: int(row.get("manual_upload_priority", row.get("candidate_id", 9999))))
    champion = candidates[-1]
    ledger["next_manual_upload_target"] = champion["candidate_id"]; ledger["next_manual_upload_target_source"] = champion["source_file"]; ledger["next_manual_upload_target_output"] = champion["output_file"]
    write_json(path, ledger)


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("| `199_noise_champion_scaled_external_denoising` | current official best | `53.8099243` | Validated noise-aware scaled external denoising path. |", "| `199_noise_champion_scaled_external_denoising` | official history | `53.8099243` | Validated noise-aware scaled external denoising path. |\n| `269_noise_v3_champion_low_noise_stack` | current approximate official best | `55.0 approx` | Positive V3 anchor for V4 reconstruction. |")
    text = text.replace("Current next manual upload target: `269_noise_v3_champion_low_noise_stack`. Candidate `199` remains the official best until a higher judge score is recorded.", "Historical next manual upload target: `269_noise_v3_champion_low_noise_stack`. Candidate `269` later received approximate `55 / 100` feedback.")
    marker = "## Current Batch: High-Risk Participant Reconstruction V4"
    text = text.split(marker)[0].rstrip() + "\n\n" + marker + "\n\nCandidate `269` is treated as the approximate official best at `55 / 100`. This batch uses high-risk participant TRT reconstruction, zero behavior, and scale restoration. Candidate `349` is the next manual upload target.\n\n"
    text += "| Candidate | Family | Source | Output | Hypothesis |\n| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{key}` | `{FAMILY}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += f"\nRisk label for all V4 rows: `{RISK_LABEL}`. Current next manual upload target: `349_v4_champion_fusion_10`.\n"
    path.write_text(text, encoding="utf-8")


def generate(root: Path, args: argparse.Namespace) -> dict:
    ctx = build_context(args.train, args.test, args.best_output, args.previous_best_output)
    specs = build_specs(ctx, args.start_id, args.limit)
    test_ids = [row["datapointID"] for row in ctx.test_rows]
    candidates = [validate_candidate(root, materialize(root, spec, ctx, args), test_ids, ctx.anchor) for spec in specs]
    champion = candidates[-1]
    report = {"mode": "generate", "family": FAMILY, "risk_label": RISK_LABEL, "official_feedback": {"candidate_id": "269", "official_partial_score": float(args.official_score), "score_approximate": bool(args.score_approximate), "submission_id": "manual_269_score_55_0_approx"}, "anchor_summary": {"candidate_id": "269", **describe(ctx.anchor)}, "candidate_count": len(candidates), "next_manual_upload_target": champion["candidate_id"], "quality_gate": {"mae_vs_269_min": 25.0, "champion_passed": champion["mae_vs_269"] >= 25.0}, "official_80_target_met": False, "final_99_target_met": False, "candidates": candidates}
    write_json(args.report, report); update_readme(root, candidates); update_ledger(root, candidates, args)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-risk participant reconstruction V4 candidates.")
    parser.add_argument("--train", type=Path, required=True); parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--best-output", type=Path, required=True); parser.add_argument("--previous-best-output", type=Path, required=True)
    parser.add_argument("--start-id", type=int, default=270); parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--official-score", type=float, default=55.0); parser.add_argument("--score-approximate", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("reports/high_risk_reconstruction_v4_report.json"))
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = generate(root, parse_args())
    print(json.dumps({"candidate_count": result["candidate_count"], "next_manual_upload_target": result["next_manual_upload_target"], "champion_passed": result["quality_gate"]["champion_passed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
