from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from experiments.official_80_push import update_ledger


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class Official80PushTest(unittest.TestCase):
    def test_report_keeps_target_unmet_until_official_80(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(80.0, ledger["official_80_target_score"])
        self.assertFalse(ledger["official_80_target_met"])
        self.assertLess(ledger["best_official_score"], 80.0)

    def test_generated_candidates_are_upload_ready(self) -> None:
        report = json.loads((ROOT / "reports/official_80_push_candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(80.0, report["target_score"])
        self.assertGreaterEqual(len(report["candidates"]), 10)
        for candidate in report["candidates"]:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertTrue(candidate["upload_ready"])
                self.assertEqual(9425, candidate["row_count"])
                self.assertEqual(["subtaskID", "datapointID", "answer"], candidate["columns"])
                self.assertTrue(candidate["ids_match_test_order"])
                self.assertTrue(candidate["answers_finite_non_negative"])

    def test_variant_source_reproduces_output(self) -> None:
        expected = read_csv(ROOT / "official_candidates/006_ridge_zero_20_output.csv")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.csv"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "official_candidates/006_ridge_zero_20_source.py"),
                    "--train",
                    str(ROOT / "data/train_data.csv"),
                    "--test",
                    str(ROOT / "data/test_data.csv"),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            self.assertEqual(expected, read_csv(output))

    def test_ledger_update_preserves_previous_official_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            ledger = {"best_official_score": 36.11668, "best_official_submission_id": "old", "submissions": []}
            (reports / "official_submission_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
            args = SimpleNamespace(candidate_id="006", submission_id="new", timestamp="2026-04-25 15:30", state="success", partial_score=35.0, complete_score=None)

            updated = update_ledger(root, args)

            self.assertEqual(36.11668, updated["best_official_score"])
            self.assertEqual("old", updated["best_official_submission_id"])
            self.assertEqual(35.0, updated["submissions"][0]["official_partial_score"])


if __name__ == "__main__":
    unittest.main()
