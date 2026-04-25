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
RISK_LABELS = {"safe_external_generalization", "medium_public_behavior_mapping", "high_risk_public_reconstruction"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class FailureReboundFeedbackTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/failure_rebound_report.json").read_text(encoding="utf-8"))

    def test_records_129_as_negative_feedback(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual("099", ledger["best_official_candidate_id"])
        self.assertAlmostEqual(38.20618, float(ledger["best_official_score"]))
        self.assertEqual("129", ledger["latest_official_feedback_candidate_id"])
        self.assertAlmostEqual(37.89482, float(ledger["latest_official_feedback_score"]))
        row = next(item for item in ledger["pending_candidates"] if item["candidate_id"] == "129")
        self.assertEqual("official_scored", row["status"])
        self.assertFalse(row["next_manual_upload_target"])

    def test_candidate_159_is_current_upload_target(self) -> None:
        ids = [int(path.name.split("_", 1)[0]) for path in (ROOT / "official_candidates").glob("*_source.py")]
        self.assertEqual(159, max(ids))
        report = self.report()
        champion = report["candidates"][-1]
        self.assertEqual("159", report["next_manual_upload_target"])
        self.assertEqual("159", champion["candidate_id"])
        self.assertTrue(champion["next_manual_upload_target"])
        self.assertTrue(report["rebound_gate"]["passed"])
        self.assertGreaterEqual(champion["mae_vs_099"], report["rebound_gate"]["mae_vs_099_min"])
        self.assertLess(champion["mae_vs_099"], report["rebound_gate"]["mae_vs_099_max"])

    def test_rebound_candidates_are_upload_ready_fused_outputs(self) -> None:
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        report = self.report()
        self.assertEqual(30, report["candidate_count"])
        for candidate in report["candidates"]:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertTrue(candidate["upload_ready"])
                self.assertLess(candidate["source_bytes"], SOURCE_LIMIT)
                self.assertLess(candidate["output_bytes"], OUTPUT_LIMIT)
                self.assertEqual(9425, candidate["row_count"])
                self.assertEqual(["subtaskID", "datapointID", "answer"], candidate["columns"])
                rows = read_csv(ROOT / candidate["output_file"])
                self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
                self.assertEqual(RISK_LABELS, set(candidate["risk_labels"]))
                groups = set(candidate["fused_evidence_groups"])
                self.assertIn("safe_surface_via_best_fusion", groups)
                self.assertIn("medium_public_behavior_via_best_fusion", groups)
                self.assertIn("high_public_reconstruction_via_best_fusion", groups)
                self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))

    def test_selected_rebound_sources_reproduce_outputs(self) -> None:
        selected_ids = {"133", "151", "159"}
        selected = [row for row in self.report()["candidates"] if row["candidate_id"] in selected_ids]
        with tempfile.TemporaryDirectory() as directory:
            for candidate in selected:
                output = Path(directory) / f"{candidate['candidate_id']}.csv"
                subprocess.run(
                    [
                        "python3",
                        str(ROOT / candidate["source_file"]),
                        "--train",
                        str(ROOT / "data/train_data.csv"),
                        "--test",
                        str(ROOT / "data/test_data.csv"),
                        "--best-output",
                        str(ROOT / "official_candidates/099_feedback_champion_safe25_medium50_high25_output.csv"),
                        "--failed-output",
                        str(ROOT / "official_candidates/129_aggressive_champion_trained_direction_risk_output.csv"),
                        "--anchor-output",
                        str(ROOT / "official_candidates/087_three_tier_fusion_base_14_output.csv"),
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
