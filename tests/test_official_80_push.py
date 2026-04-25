from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.official_80_push import update_ledger

ROOT = Path(__file__).resolve().parents[1]


class Official80PushTest(unittest.TestCase):
    def test_report_keeps_target_unmet_until_official_80(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(80.0, ledger["official_80_target_score"])
        self.assertFalse(ledger["official_80_target_met"])
        self.assertLess(ledger["best_official_score"], 80.0)

    def test_historical_report_preserves_candidate_metadata_after_pruning(self) -> None:
        report = json.loads((ROOT / "reports/official_80_push_candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(80.0, report["target_score"])
        self.assertGreaterEqual(len(report["candidates"]), 10)
        self.assertTrue(all(candidate["upload_ready"] for candidate in report["candidates"]))

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
