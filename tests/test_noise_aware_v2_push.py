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


class NoiseAwareV2PushTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/noise_aware_v2_push_report.json").read_text(encoding="utf-8"))

    def test_records_199_as_historical_best_anchor(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(55.0, float(ledger["best_official_score"]))
        self.assertEqual("269", ledger["best_official_candidate_id"])
        self.assertEqual("manual_269_score_55_0_approx", ledger["best_official_submission_id"])
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "199")
        self.assertEqual("official_scored", scored["status"])
        self.assertFalse(scored["next_manual_upload_target"])

    def test_candidate_249_is_current_upload_target(self) -> None:
        report = self.report()
        ids = [int(row["candidate_id"]) for row in report["candidates"]]
        self.assertEqual(249, max(ids))
        self.assertEqual(50, report["candidate_count"])
        self.assertEqual("249", report["next_manual_upload_target"])
        champion = report["candidates"][-1]
        self.assertEqual("249", champion["candidate_id"])
        self.assertTrue(champion["next_manual_upload_target"])
        self.assertTrue(report["movement_gate"]["passed"])
        self.assertGreaterEqual(champion["mae_vs_199"], report["movement_gate"]["mae_vs_199_min"])
        self.assertLessEqual(champion["mae_vs_199"], report["movement_gate"]["mae_vs_199_max"])
        self.assertLessEqual(abs(champion["mean_delta_vs_199"]), report["movement_gate"]["mean_delta_abs_max"])
        self.assertLessEqual(abs(champion["std_delta_vs_199"]), report["movement_gate"]["std_delta_abs_max"])

    def test_v2_outputs_are_upload_ready(self) -> None:
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
                self.assertIn("component_weights", candidate)
                self.assertIn("official_winning_direction", candidate["noise_evidence_groups"])

    def test_pruned_old_exploration_files_but_kept_anchors(self) -> None:
        report = self.report()
        self.assertIn("pruned_candidate_files", report)
        for cid in ["001", "002", "023", "045", "087", "099", "129", "159", "199", "249"]:
            matches = list((ROOT / "official_candidates").glob(f"{cid}_*_output.csv"))
            self.assertTrue(matches, cid)
        self.assertFalse(list((ROOT / "official_candidates").glob("160_*_output.csv")))
        self.assertFalse(list((ROOT / "official_candidates").glob("198_*_source.py")))

    def test_selected_sources_reproduce_outputs(self) -> None:
        selected_ids = {"200", "213", "225", "237", "249"}
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
                        "--anchor-output",
                        str(ROOT / "official_candidates/159_rebound_champion_counter_prior_stack_output.csv"),
                        "--best-output",
                        str(ROOT / "official_candidates/199_noise_champion_scaled_external_denoising_output.csv"),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
