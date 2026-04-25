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


class FailureReboundHistoryTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/failure_rebound_report.json").read_text(encoding="utf-8"))

    def test_report_preserves_159_rebound_champion_evidence(self) -> None:
        report = self.report()
        self.assertEqual("129", report["official_feedback"]["failed_candidate_id"])
        self.assertEqual("159", report["next_manual_upload_target"])
        self.assertEqual(30, report["candidate_count"])
        champion = report["candidates"][-1]
        self.assertEqual("159", champion["candidate_id"])
        self.assertTrue(report["rebound_gate"]["passed"])

    def test_candidate_159_anchor_is_retained_and_scored(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "159")
        self.assertEqual("official_scored", scored["status"])
        self.assertAlmostEqual(38.87076, float(scored["official_partial_score"]))
        self.assertFalse(scored["next_manual_upload_target"])
        rows = read_csv(ROOT / scored["output_file"])
        self.assertEqual(9425, len(rows))
        self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
