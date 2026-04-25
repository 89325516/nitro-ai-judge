from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class OfficialCandidateTest(unittest.TestCase):
    def ledger(self) -> dict:
        return json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))

    def test_ledger_preserves_official_score_history(self) -> None:
        ledger = self.ledger()
        first = ledger["submissions"][0]
        self.assertEqual("4e1670481a8f", first["submission_id"])
        self.assertAlmostEqual(36.11668, float(first["official_partial_score"]))
        self.assertAlmostEqual(53.8099243, float(ledger["best_official_score"]))
        self.assertEqual("199", ledger["best_official_candidate_id"])
        self.assertEqual("manual_199_score_53_8099243", ledger["best_official_submission_id"])
        self.assertEqual("249", ledger["next_manual_upload_target"])
        self.assertTrue(ledger["latest_official_feedback_submission_id_missing"])

    def test_scored_anchor_candidates_remain_upload_ready(self) -> None:
        ledger = self.ledger()
        expected = {"087": 37.35945, "099": 38.20618, "129": 37.89482, "159": 38.87076, "199": 53.8099243}
        for cid, score in expected.items():
            with self.subTest(candidate=cid):
                candidate = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == cid)
                self.assertEqual("official_scored", candidate["status"])
                self.assertAlmostEqual(score, float(candidate["official_partial_score"]))
                self.assertFalse(candidate["next_manual_upload_target"])
                self.assertLess((ROOT / candidate["source_file"]).stat().st_size, SOURCE_LIMIT)
                self.assertLess((ROOT / candidate["output_file"]).stat().st_size, OUTPUT_LIMIT)

    def test_only_current_batch_has_pending_upload_targets(self) -> None:
        ledger = self.ledger()
        pending = [row for row in ledger["pending_candidates"] if row["status"] == "pending"]
        self.assertEqual(50, len(pending))
        self.assertEqual({str(i) for i in range(200, 250)}, {row["candidate_id"] for row in pending})
        targets = [row for row in pending if row["next_manual_upload_target"]]
        self.assertEqual(["249"], [row["candidate_id"] for row in targets])

    def test_existing_candidate_outputs_match_test_contract(self) -> None:
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        for output_path in sorted((ROOT / "official_candidates").glob("*_output.csv")):
            with self.subTest(output=output_path.name):
                rows = read_csv(output_path)
                self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
                self.assertEqual(["subtaskID", "datapointID", "answer"], list(rows[0].keys()))
                for row in rows:
                    value = float(row["answer"])
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
