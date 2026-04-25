from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.three_tier_external_data import FUSION_LABEL, public_rows, read_csv
    from experiments.three_tier_external_emit import candidate_specs, materialize, probe_report, validate_candidate, write_json
except ModuleNotFoundError:
    from three_tier_external_data import FUSION_LABEL, public_rows, read_csv
    from three_tier_external_emit import candidate_specs, materialize, probe_report, validate_candidate, write_json


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    marker = "## Batch 9: Three-Tier Fusion External Data"
    text = path.read_text(encoding="utf-8").split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    text += "All Batch 9 candidates fuse safe language features, medium public behavior aggregates, and high-risk reconstruction signals into one output. Recommended order: highest high-risk weight first, then calibrated/base variants that survive official feedback.\n\n"
    text += "| Candidate | Weights safe/medium/high | Source | Output | Hypothesis |\n"
    text += "| --- | --- | --- | --- | --- |\n"
    for candidate in candidates:
        key = f"{candidate['candidate_id']}_{candidate['name']}"
        weights = candidate["component_weights"]
        triplet = f"{weights['safe']:.2f}/{weights['medium']:.2f}/{weights['high']:.2f}"
        text += f"| `{key}` | `{triplet}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += f"\nRisk label for all Batch 9 rows: `{FUSION_LABEL}` with safe, medium, and high-risk components.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "reports/official_submission_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    existing = {row.get("candidate_id"): row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids}
    for candidate in candidates:
        existing[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/three_tier_external_report.json", "local_score": None, "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": candidate["risk_label"], "risk_labels": candidate["risk_labels"], "component_weights": candidate["component_weights"]}
    ledger["pending_candidates"] = sorted(existing.values(), key=lambda row: int(row.get("manual_upload_priority", 9999)))
    write_json(path, ledger)


def generate(root: Path, args: argparse.Namespace) -> dict:
    train_rows, test_rows = read_csv(args.train), read_csv(args.test)
    rows = public_rows(root / ".cache/three_tier_external")
    specs, rankings = candidate_specs(train_rows, test_rows, rows, args.candidate_start)
    materialized = [materialize(root, spec, args) for spec in specs]
    test_ids = [row["datapointID"] for row in test_rows]
    candidates = [validate_candidate(root, candidate, test_ids) for candidate in materialized]
    report = {"mode": "generate", "risk_label": FUSION_LABEL, "candidate_count": len(candidates), "base_candidate": "023_lexical_transformer", "high_risk_base_candidate": "045_public_trt_ensemble_fallback", "candidates": candidates, "coverage": probe_report(args.train, args.test, None), **rankings}
    write_json(args.report, report)
    update_readme(root, candidates)
    update_ledger(root, candidates)
    return report
