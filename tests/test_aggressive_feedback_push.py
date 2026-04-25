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


class AggressiveFeedbackPushTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/aggressive_feedback_push_report.json").read_text(encoding="utf-8"))

    def test_ledger_records_candidate_099_score(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(38.20618, float(ledger["best_official_score"]))
        self.assertEqual("099", ledger["best_official_candidate_id"])
        self.assertEqual("manual_099_score_38_20618", ledger["best_official_submission_id"])
        self.assertTrue(ledger["latest_official_feedback_submission_id_missing"])
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "099")
        self.assertEqual("official_scored", scored["status"])
        self.assertAlmostEqual(38.20618, float(scored["official_partial_score"]))

    def test_candidate_129_was_batch_champion_and_is_now_scored(self) -> None:
        report = self.report()
        champion = report["candidates"][-1]
        self.assertEqual("129", report["next_manual_upload_target"])
        self.assertEqual("129", champion["candidate_id"])
        self.assertTrue(champion["next_manual_upload_target"])
        self.assertTrue(report["champion_difference_gate"]["passed"])
        self.assertLessEqual(champion["corr_vs_099"], 0.995)
        self.assertGreaterEqual(champion["mae_vs_099"], 15.0)
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "129")
        self.assertEqual("official_scored", scored["status"])
        self.assertFalse(scored["next_manual_upload_target"])

    def test_new_candidates_are_fused_upload_ready_outputs(self) -> None:
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
                self.assertEqual(expected_ids, [row["datapointID"] for row in read_csv(ROOT / candidate["output_file"])])
                self.assertEqual(RISK_LABELS, set(candidate["risk_labels"]))
                groups = set(candidate["trained_feature_groups"])
                self.assertIn("safe_surface", groups)
                self.assertIn("medium_public_behavior", groups)
                self.assertIn("high_public_reconstruction", groups)
                values = [float(row["answer"]) for row in read_csv(ROOT / candidate["output_file"])]
                self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in values))

    def test_trained_candidates_report_training_groups(self) -> None:
        trained = [row for row in self.report()["candidates"] if 110 <= int(row["candidate_id"]) <= 119 or row["candidate_id"] == "129"]
        self.assertTrue(trained)
        for candidate in trained:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertIn("trained_stacker", candidate["trained_feature_groups"])

    def test_selected_sources_reproduce_outputs(self) -> None:
        selected_ids = {"102", "110", "129"}
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
                        "--base-output",
                        str(ROOT / "official_candidates/023_lexical_transformer_output.csv"),
                        "--high-risk-base-output",
                        str(ROOT / "official_candidates/045_public_trt_ensemble_fallback_output.csv"),
                        "--anchor-output",
                        str(ROOT / "official_candidates/087_three_tier_fusion_base_14_output.csv"),
                        "--best-output",
                        str(ROOT / "official_candidates/099_feedback_champion_safe25_medium50_high25_output.csv"),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
