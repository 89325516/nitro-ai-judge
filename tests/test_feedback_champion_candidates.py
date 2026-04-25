from __future__ import annotations

import csv
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class FeedbackChampionCandidateTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/feedback_champion_report.json").read_text(encoding="utf-8"))

    def test_report_records_anchor_and_champion(self) -> None:
        report = self.report()
        self.assertEqual("087", report["anchor_candidate_id"])
        self.assertAlmostEqual(37.35945, float(report["anchor_official_score"]))
        self.assertEqual("099", report["next_manual_upload_target"])
        self.assertEqual(12, report["candidate_count"])
        self.assertEqual("088", report["candidates"][0]["candidate_id"])
        self.assertEqual("099", report["candidates"][-1]["candidate_id"])
        self.assertTrue(report["candidates"][-1]["next_manual_upload_target"])

    def test_anchor_clone_matches_candidate_087(self) -> None:
        self.assertEqual(
            read_csv(ROOT / "official_candidates/087_three_tier_fusion_base_14_output.csv"),
            read_csv(ROOT / "official_candidates/088_feedback_anchor_clone_output.csv"),
        )

    def test_candidate_099_batch_output_is_upload_ready(self) -> None:
        candidate = self.report()["candidates"][-1]
        rows = read_csv(ROOT / candidate["output_file"])
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        self.assertEqual("099", candidate["candidate_id"])
        self.assertTrue(candidate["upload_ready"])
        self.assertLess(candidate["source_bytes"], SOURCE_LIMIT)
        self.assertLess(candidate["output_bytes"], OUTPUT_LIMIT)
        self.assertEqual(9425, candidate["row_count"])
        self.assertEqual(["subtaskID", "datapointID", "answer"], candidate["columns"])
        self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
        self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))

    def test_ledger_keeps_099_as_best_after_later_feedback(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(38.20618, ledger["best_official_score"])
        self.assertEqual("129", ledger["latest_official_feedback_candidate_id"])
        self.assertEqual("159", ledger["next_manual_upload_target"])
        candidate = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "099")
        self.assertFalse(candidate["next_manual_upload_target"])
        self.assertEqual("official_scored", candidate["status"])
        self.assertAlmostEqual(38.20618, float(candidate["official_partial_score"]))

    def test_champion_source_reproduces_output(self) -> None:
        candidate = self.report()["candidates"][-1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.csv"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / candidate["source_file"]),
                    "--train",
                    str(ROOT / "data/train_data.csv"),
                    "--test",
                    str(ROOT / "data/test_data.csv"),
                    "--base-output",
                    str(ROOT / "official_candidates/023_lexical_transformer_output.csv"),
                    "--high-risk-base-output",
                    str(ROOT / "official_candidates/045_public_trt_ensemble_fallback_output.csv"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
