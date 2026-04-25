from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from experiments.high_risk_public_trt_data import RISK_LABEL, base_predictions, load_external, read_csv
    from experiments.high_risk_public_trt_emit import candidate_specs, materialize, probe, validate_candidate, write_json
except ModuleNotFoundError:
    from high_risk_public_trt_data import RISK_LABEL, base_predictions, load_external, read_csv
    from high_risk_public_trt_emit import candidate_specs, materialize, probe, validate_candidate, write_json


def update_readme(root: Path, candidates: Sequence[dict]) -> None:
    path = root / "official_candidates/README.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Batch 8: High-Risk Public TRT Recovery"
    text = text.split(marker)[0].rstrip() + "\n\n" + marker + "\n\n"
    text += "These candidates use public fixation-derived TRT data from `ana0101/eye-tracking` and are labeled high-risk reconstruction candidates. Upload them only when intentionally testing the public-data recovery route. Recommended order: `045`, `037`, `029`, then `024-028`.\n\n"
    text += "| Candidate | Source | Output | Hypothesis |\n"
    text += "| --- | --- | --- | --- |\n"
    for candidate in candidates:
        candidate_key = f"{candidate['candidate_id']}_{candidate['name']}"
        text += f"| `{candidate_key}` | `{candidate['source_file']}` | `{candidate['output_file']}` | {candidate['hypothesis']} |\n"
    text += f"\nRisk label for all Batch 8 rows: `{RISK_LABEL}`.\n"
    path.write_text(text, encoding="utf-8")


def update_ledger(root: Path, candidates: Sequence[dict]) -> None:
    import json

    path = root / "reports/official_submission_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    new_ids = {candidate["candidate_id"] for candidate in candidates}
    existing = {row.get("candidate_id"): row for row in ledger.get("pending_candidates", []) if row.get("candidate_id") not in new_ids}
    for candidate in candidates:
        existing[candidate["candidate_id"]] = {"candidate_id": candidate["candidate_id"], "name": candidate["name"], "hypothesis": candidate["hypothesis"], "source_file": candidate["source_file"], "output_file": candidate["output_file"], "local_report": "reports/high_risk_public_trt_report.json", "local_score": None, "manual_upload_priority": int(candidate["candidate_id"]), "status": "pending", "upload_ready": candidate["upload_ready"], "risk_label": RISK_LABEL}
    ledger["pending_candidates"] = sorted(existing.values(), key=lambda row: int(row.get("manual_upload_priority", 9999)))
    write_json(path, ledger)


def generate(root: Path, args: argparse.Namespace) -> dict:
    train_rows, test_rows = read_csv(args.train), read_csv(args.test)
    external = load_external(root / ".cache/high_risk_public_trt")
    base = base_predictions(args.base_output, test_rows)
    specs, rankings = candidate_specs(train_rows, test_rows, external, args.candidate_start, args.materialize_limit)
    test_ids = [row["datapointID"] for row in test_rows]
    materialized = [materialize(root, spec, train_rows, test_rows, external, base) for spec in specs]
    candidates = [validate_candidate(root, candidate, test_ids) for candidate in materialized]
    report = {"mode": "generate", "risk_label": RISK_LABEL, "base_candidate": "023_lexical_transformer", "candidate_count": len(candidates), "candidates": candidates, "coverage": probe(args.test, None), **rankings}
    write_json(args.report, report)
    update_readme(root, candidates)
    update_ledger(root, candidates)
    return report
