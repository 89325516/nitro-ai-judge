from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024


class OfficialCandidateTest(unittest.TestCase):
    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def test_ledger_preserves_first_official_score(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        first = ledger["submissions"][0]
        self.assertEqual("4e1670481a8f", first["submission_id"])
        self.assertEqual("success", first["state"])
        self.assertAlmostEqual(36.11668, float(first["official_partial_score"]))
        self.assertGreaterEqual(float(ledger["best_official_score"]), float(first["official_partial_score"]))
        self.assertTrue(ledger["should_choose_best_as_final_now"])

    def test_latest_largest_id_feedback_is_current_best(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(37.35945, float(ledger["best_official_score"]))
        self.assertEqual("07ead7cdad22", ledger["best_official_submission_id"])
        self.assertEqual("largest_numeric_candidate_id_at_manual_upload_time", ledger["latest_manual_submission_policy"])
        self.assertEqual("087", ledger["latest_official_feedback_candidate_id"])
        latest = ledger["submissions"][-1]
        self.assertEqual("087", latest["candidate_id"])
        self.assertEqual("success", latest["state"])
        self.assertAlmostEqual(37.35945, float(latest["official_partial_score"]))
        self.assertTrue(latest["chosen_as_final"])

    def test_candidate_087_is_scored_and_still_upload_ready(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        candidate = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "087")
        self.assertEqual("official_scored", candidate["status"])
        self.assertEqual("07ead7cdad22", candidate["official_submission_id"])
        self.assertAlmostEqual(37.35945, float(candidate["official_partial_score"]))
        self.assertTrue(candidate["upload_ready"])
        self.assertLess((ROOT / candidate["source_file"]).stat().st_size, SOURCE_LIMIT)
        self.assertLess((ROOT / candidate["output_file"]).stat().st_size, OUTPUT_LIMIT)

    def test_pending_candidates_are_upload_ready(self) -> None:
        validation = json.loads((ROOT / "reports/official_candidate_validation.json").read_text(encoding="utf-8"))
        for candidate in validation["candidates"]:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertLess(candidate["source_bytes"], SOURCE_LIMIT)
                self.assertLess(candidate["output_bytes"], OUTPUT_LIMIT)
                self.assertEqual(9425, candidate["row_count"])
                self.assertEqual(["subtaskID", "datapointID", "answer"], candidate["columns"])
                self.assertTrue(candidate["ids_match_test_order"])
                self.assertTrue(candidate["answers_finite_non_negative"])
                self.assertTrue(candidate["upload_ready"])

    def test_candidate_outputs_match_test_contract(self) -> None:
        test_rows = self.read_csv(ROOT / "data/test_data.csv")
        expected_ids = [row["datapointID"] for row in test_rows]
        for output_path in sorted((ROOT / "official_candidates").glob("*_output.csv")):
            with self.subTest(output=output_path.name):
                rows = self.read_csv(output_path)
                self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
                for row in rows:
                    value = float(row["answer"])
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
