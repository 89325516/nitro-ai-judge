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


class NoiseAwarePushHistoryTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/noise_aware_push_report.json").read_text(encoding="utf-8"))

    def test_candidate_199_history_is_recorded_as_scored_best(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(53.8099243, float(ledger["best_official_score"]))
        self.assertEqual("199", ledger["best_official_candidate_id"])
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "199")
        self.assertEqual("official_scored", scored["status"])
        self.assertFalse(scored["next_manual_upload_target"])

    def test_noise_aware_v1_report_preserves_199_champion_evidence(self) -> None:
        report = self.report()
        self.assertEqual(40, report["candidate_count"])
        self.assertEqual("199", report["next_manual_upload_target"])
        champion = report["candidates"][-1]
        self.assertEqual("199", champion["candidate_id"])
        self.assertTrue(report["movement_gate"]["passed"])
        self.assertIn("clean_external_aggregate_prior", champion["risk_labels"])
        self.assertIn("mac_scaled_model", champion["noise_evidence_groups"])

    def test_candidate_199_output_is_retained_and_upload_ready(self) -> None:
        candidate = self.report()["candidates"][-1]
        rows = read_csv(ROOT / candidate["output_file"])
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        self.assertLess((ROOT / candidate["source_file"]).stat().st_size, SOURCE_LIMIT)
        self.assertLess((ROOT / candidate["output_file"]).stat().st_size, OUTPUT_LIMIT)
        self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
        self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))

    def test_candidate_199_source_reproduces_output(self) -> None:
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
                    "--best-output",
                    str(ROOT / "official_candidates/159_rebound_champion_counter_prior_stack_output.csv"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
