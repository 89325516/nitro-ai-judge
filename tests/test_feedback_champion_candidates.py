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


class FeedbackChampionHistoryTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/feedback_champion_report.json").read_text(encoding="utf-8"))

    def test_report_records_099_historical_champion(self) -> None:
        report = self.report()
        self.assertEqual("087", report["anchor_candidate_id"])
        self.assertEqual("099", report["next_manual_upload_target"])
        self.assertEqual(12, report["candidate_count"])
        self.assertEqual("099", report["candidates"][-1]["candidate_id"])

    def test_candidate_099_anchor_is_retained_and_scored(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        candidate = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "099")
        self.assertEqual("official_scored", candidate["status"])
        self.assertAlmostEqual(38.20618, float(candidate["official_partial_score"]))
        rows = read_csv(ROOT / candidate["output_file"])
        self.assertEqual(9425, len(rows))
        self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
