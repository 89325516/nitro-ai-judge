from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class AggressiveFeedbackHistoryTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/aggressive_feedback_push_report.json").read_text(encoding="utf-8"))

    def test_report_preserves_failed_129_batch_evidence(self) -> None:
        report = self.report()
        self.assertEqual("129", report["next_manual_upload_target"])
        self.assertEqual(30, report["candidate_count"])
        champion = report["candidates"][-1]
        self.assertEqual("129", champion["candidate_id"])
        self.assertTrue(report["champion_difference_gate"]["passed"])
        self.assertGreaterEqual(champion["mae_vs_099"], 15.0)

    def test_candidate_129_anchor_is_retained_and_scored(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "129")
        self.assertEqual("official_scored", scored["status"])
        self.assertAlmostEqual(37.89482, float(scored["official_partial_score"]))
        self.assertFalse(scored["next_manual_upload_target"])
        rows = read_csv(ROOT / scored["output_file"])
        self.assertEqual(9425, len(rows))
        self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
