from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.high_risk_reconstruction_v4_components import candidate_values, describe, safe_corr
from experiments.high_risk_reconstruction_v4_contracts import CandidateSpec, RuntimeContext
from experiments.high_risk_reconstruction_v4_data import OUTPUT_LIMIT, RISK_LABEL, SOURCE_LIMIT, load_provider, output_values, read_csv, write_output


def build_context(train_path: Path, test_path: Path, best_path: Path, previous_path: Path) -> RuntimeContext:
    test_rows = read_csv(test_path)
    return RuntimeContext(read_csv(train_path), test_rows, load_provider(Path(".cache/high_risk_reconstruction_v4")), output_values(best_path, test_rows), output_values(previous_path, test_rows))


def source_text(config: dict) -> str:
    return f'''from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={config!r}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
'''


def runtime_write(train_path: Path, test_path: Path, best_path: Path, previous_path: Path, output_path: Path, config: dict) -> None:
    ctx = build_context(train_path, test_path, best_path, previous_path)
    write_output(output_path, ctx.test_rows, candidate_values(ctx, config))


def materialize(root: Path, spec: CandidateSpec, ctx: RuntimeContext, args: argparse.Namespace) -> dict:
    source = root / "official_candidates" / f"{spec.candidate_id}_{spec.name}_source.py"
    output = root / "official_candidates" / f"{spec.candidate_id}_{spec.name}_output.csv"
    source.write_text(source_text(spec.config), encoding="utf-8")
    write_output(output, ctx.test_rows, candidate_values(ctx, spec.config))
    cmd = f"python3 {source.relative_to(root)} --train {args.train} --test {args.test} --best-output {args.best_output} --previous-best-output {args.previous_best_output} --output {output.relative_to(root)}"
    return {"candidate_id": spec.candidate_id, "name": spec.name, "family": spec.family, "hypothesis": spec.hypothesis, "component_labels": list(spec.component_labels), "participant_mapping_mode": spec.participant_mapping_mode, "zero_strategy": spec.zero_strategy, "scale_strategy": spec.scale_strategy, "next_manual_upload_target": spec.next_manual_upload_target, "risk_label": RISK_LABEL, "source_file": str(source.relative_to(root)), "output_file": str(output.relative_to(root)), "reproduction_command": cmd}


def validate_candidate(root: Path, candidate: dict, test_ids: list[str], anchor: list[float]) -> dict:
    rows = read_csv(root / candidate["output_file"])
    values = np.asarray([float(row["answer"]) for row in rows], dtype=float)
    ref = np.asarray(anchor, dtype=float)
    stats = describe(values)
    row = {**candidate, "mae_vs_269": float(np.mean(np.abs(values - ref))), "corr_vs_269": safe_corr(values, ref), "source_bytes": (root / candidate["source_file"]).stat().st_size, "output_bytes": (root / candidate["output_file"]).stat().st_size, "row_count": len(rows), "columns": list(rows[0].keys()) if rows else [], "ids_match_test_order": [item["datapointID"] for item in rows] == test_ids, "answers_finite_non_negative": bool(np.all(np.isfinite(values)) and np.all(values >= 0.0)), "answer_mean": stats["mean"], "answer_std": stats["std"], "answer_zero_rate": stats["zero_rate"]}
    row["upload_ready"] = row["source_bytes"] < SOURCE_LIMIT and row["output_bytes"] < OUTPUT_LIMIT and row["row_count"] == len(test_ids) and row["columns"] == ["subtaskID", "datapointID", "answer"] and row["ids_match_test_order"] and row["answers_finite_non_negative"]
    return row


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
