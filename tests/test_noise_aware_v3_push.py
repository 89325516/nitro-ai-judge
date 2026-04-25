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


class NoiseAwareV3PushTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/noise_aware_v3_push_report.json").read_text(encoding="utf-8"))

    def test_records_v3_history_before_v4_anchor(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual("269", ledger["best_official_candidate_id"])
        self.assertAlmostEqual(55.0, float(ledger["best_official_score"]))
        self.assertTrue(ledger["best_official_score_approximate"])
        self.assertEqual("manual_269_score_55_0_approx", ledger["best_official_submission_id"])
        self.assertEqual("269", ledger["latest_official_feedback_candidate_id"])
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "249")
        self.assertEqual("official_scored", scored["status"])
        self.assertEqual("manual_249_score_53_8099243", scored["official_submission_id"])
        self.assertFalse(scored["next_manual_upload_target"])
        anchor = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "269")
        self.assertEqual("official_scored", anchor["status"])
        self.assertAlmostEqual(55.0, float(anchor["official_partial_score"]))

    def test_candidate_269_is_next_upload_target(self) -> None:
        report = self.report()
        self.assertEqual(20, report["candidate_count"])
        self.assertEqual("269", report["next_manual_upload_target"])
        champion = report["candidates"][-1]
        self.assertEqual("269", champion["candidate_id"])
        self.assertTrue(champion["next_manual_upload_target"])
        self.assertTrue(report["movement_gate"]["passed"])
        self.assertGreaterEqual(champion["mae_vs_199"], report["movement_gate"]["mae_vs_199_min"])
        self.assertLessEqual(champion["mae_vs_199"], report["movement_gate"]["mae_vs_199_max"])
        self.assertLessEqual(abs(champion["mean_delta_vs_199"]), report["movement_gate"]["mean_delta_abs_max"])
        self.assertLessEqual(abs(champion["std_delta_vs_199"]), report["movement_gate"]["std_delta_abs_max"])

    def test_report_contains_low_noise_and_robust_evidence(self) -> None:
        report = self.report()
        self.assertEqual(1.0, report["external_consensus_coverage"]["test_row"])
        self.assertEqual(1.0, report["external_consensus_coverage"]["train_item"])
        self.assertGreater(report["low_noise_target_profile"]["mean_within_item_std"], 0.0)
        self.assertEqual(700, report["model_capacity_summary"]["hgb_max_iter"])
        self.assertEqual(768, report["model_capacity_summary"]["extra_trees_max_estimators"])
        labels = set(report["robust_training_method_labels"])
        self.assertIn("co_teach", labels)
        self.assertIn("superlearner", labels)
        self.assertIn("ngboost_like", labels)

    def test_v3_outputs_are_upload_ready(self) -> None:
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        for candidate in self.report()["candidates"]:
            rows = read_csv(ROOT / candidate["output_file"])
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertTrue(candidate["upload_ready"])
                self.assertLess(candidate["source_bytes"], SOURCE_LIMIT)
                self.assertLess(candidate["output_bytes"], OUTPUT_LIMIT)
                self.assertEqual(9425, candidate["row_count"])
                self.assertEqual(["subtaskID", "datapointID", "answer"], candidate["columns"])
                self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
                self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))

    def test_selected_sources_reproduce_outputs(self) -> None:
        selected_ids = {"250", "256", "263", "269"}
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
                        str(ROOT / "official_candidates/199_noise_champion_scaled_external_denoising_output.csv"),
                        "--neutral-output",
                        str(ROOT / "official_candidates/249_noise_v2_champion_directional_denoising_output.csv"),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
