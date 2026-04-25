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
RISK_LABELS = {"clean_external_aggregate_prior", "scaled_mac_model", "train_label_denoising"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class NoiseAwarePushTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/noise_aware_push_report.json").read_text(encoding="utf-8"))

    def test_records_159_as_current_best(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(38.87076, float(ledger["best_official_score"]))
        self.assertEqual("159", ledger["best_official_candidate_id"])
        self.assertEqual("manual_159_score_38_87076", ledger["best_official_submission_id"])
        scored = next(row for row in ledger["pending_candidates"] if row["candidate_id"] == "159")
        self.assertEqual("official_scored", scored["status"])
        self.assertAlmostEqual(38.87076, float(scored["official_partial_score"]))

    def test_candidate_199_is_current_upload_target(self) -> None:
        ids = [int(path.name.split("_", 1)[0]) for path in (ROOT / "official_candidates").glob("*_source.py")]
        self.assertEqual(199, max(ids))
        report = self.report()
        champion = report["candidates"][-1]
        self.assertEqual("199", report["next_manual_upload_target"])
        self.assertEqual("199", champion["candidate_id"])
        self.assertTrue(champion["next_manual_upload_target"])
        self.assertTrue(report["movement_gate"]["passed"])
        self.assertGreaterEqual(champion["mae_vs_159"], report["movement_gate"]["mae_vs_159_min"])
        self.assertLessEqual(champion["mae_vs_159"], report["movement_gate"]["mae_vs_159_max"])
        self.assertLessEqual(abs(champion["mean_delta_vs_159"]), report["movement_gate"]["mean_delta_abs_max"])
        self.assertLessEqual(abs(champion["std_delta_vs_159"]), report["movement_gate"]["std_delta_abs_max"])

    def test_noise_profile_records_core_bottleneck(self) -> None:
        profile = self.report()["noise_profile"]
        self.assertEqual(135210, profile["row_count"])
        self.assertEqual(4507, profile["unique_word_count"])
        self.assertGreater(profile["mean_within_word_std"], profile["between_word_mean_std"])
        self.assertGreater(profile["global_zero_rate"], 0.30)
        self.assertIn("participant_reliability_inputs", profile)
        self.assertIn("text_noise_stats", profile)

    def test_noise_candidates_are_upload_ready_outputs(self) -> None:
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        report = self.report()
        self.assertEqual(40, report["candidate_count"])
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
                groups = set(candidate["noise_evidence_groups"])
                self.assertIn("official_best_base", groups)
                self.assertIn("train_label_noise_profile", groups)
                self.assertIn("clean_external_aggregate_prior", groups)
                self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))

    def test_selected_noise_sources_reproduce_outputs(self) -> None:
        selected_ids = {"160", "180", "190", "199"}
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
