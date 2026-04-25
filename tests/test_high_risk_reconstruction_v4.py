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
RISK_LABEL = "high_risk_participant_trt_reconstruction_v4"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class HighRiskReconstructionV4Test(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/high_risk_reconstruction_v4_report.json").read_text(encoding="utf-8"))

    def test_ledger_records_269_approximate_anchor(self) -> None:
        ledger = json.loads((ROOT / "reports/official_submission_ledger.json").read_text(encoding="utf-8"))
        self.assertEqual("269", ledger["best_official_candidate_id"])
        self.assertAlmostEqual(55.0, float(ledger["best_official_score"]))
        self.assertTrue(ledger["best_official_score_approximate"])
        self.assertEqual("manual_269_score_55_0_approx", ledger["best_official_submission_id"])
        self.assertEqual("349", ledger["next_manual_upload_target"])

    def test_report_contains_full_v4_batch(self) -> None:
        report = self.report()
        self.assertEqual(RISK_LABEL, report["risk_label"])
        self.assertEqual(80, report["candidate_count"])
        ids = [int(row["candidate_id"]) for row in report["candidates"]]
        self.assertEqual(list(range(270, 350)), ids)
        self.assertEqual("349", report["next_manual_upload_target"])
        self.assertFalse(report["official_80_target_met"])
        self.assertFalse(report["final_99_target_met"])

    def test_v4_outputs_are_upload_ready(self) -> None:
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
                self.assertEqual(RISK_LABEL, candidate["risk_label"])
                self.assertIn("participant_mapping_mode", candidate)
                self.assertIn("zero_strategy", candidate)
                self.assertIn("scale_strategy", candidate)

    def test_batch_contains_required_strategy_variants(self) -> None:
        candidates = self.report()["candidates"]
        anchor_std = self.report()["anchor_summary"]["std"]
        self.assertTrue(any(row["answer_zero_rate"] > 0.0 for row in candidates))
        self.assertTrue(any(row["answer_std"] > anchor_std + 25.0 for row in candidates))
        self.assertTrue(any(row["participant_mapping_mode"] == "soft" for row in candidates))
        champion = candidates[-1]
        self.assertEqual("349", champion["candidate_id"])
        self.assertGreaterEqual(champion["mae_vs_269"], 25.0)
        self.assertTrue(self.report()["quality_gate"]["champion_passed"])

    def test_selected_sources_reproduce_outputs(self) -> None:
        selected_ids = {"270", "290", "310", "325", "349"}
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
                        str(ROOT / "official_candidates/269_noise_v3_champion_low_noise_stack_output.csv"),
                        "--previous-best-output",
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
