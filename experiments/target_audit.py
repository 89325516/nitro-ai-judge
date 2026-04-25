from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_core import (
    leakage_summary,
    read_csv,
    sample_output_diagnostic,
    same_item_oracle,
    score_from_groups,
    write_json,
)


def build_reports(train_path: Path, test_path: Path, sample_path: Path, output_dir: Path) -> dict[str, dict]:
    train_rows = read_csv(train_path)
    test_rows = read_csv(test_path)
    sample_rows = read_csv(sample_path)
    ceiling = {
        "evidence_class": "oracle_ceiling",
        "same_item_unseen_participant": same_item_oracle(train_rows),
        "word_mean_unseen_text": score_from_groups(train_rows, "text", "word"),
        "family_word_mean_unseen_text": score_from_groups(
            [{**row, "family_word": row["text"].split("_", 1)[0] + "|" + row["word"].lower()} for row in train_rows],
            "text",
            "family_word",
        ),
    }
    leakage = leakage_summary(train_rows, test_rows)
    sample = sample_output_diagnostic(sample_rows)
    reports = {
        "score_ceiling_report": ceiling,
        "leakage_audit_report": leakage,
        "sample_output_diagnostic": sample,
    }
    for name, payload in reports.items():
        write_json(output_dir / f"{name}.json", payload)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reachability and leakage signals for the 99+ target.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = build_reports(args.train, args.test, args.sample, args.output_dir)
    for name, payload in reports.items():
        print(name, payload.get("evidence_class"))


if __name__ == "__main__":
    main()
