from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIMIT = 35 * 1024
OUTPUT_LIMIT = 50 * 1024 * 1024
RISK_LABEL = "high_risk_public_trt_reconstruction"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class HighRiskPublicTRTTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/high_risk_public_trt_report.json").read_text(encoding="utf-8"))

    def test_probe_records_public_trt_coverage(self) -> None:
        probe = json.loads((ROOT / "reports/high_risk_public_trt_probe.json").read_text(encoding="utf-8"))
        self.assertEqual(RISK_LABEL, probe["risk_label"])
        self.assertEqual(1885, probe["test_unique_word_ids"])
        self.assertEqual(1885, probe["item_coverage"])
        self.assertEqual(1.0, probe["item_coverage_rate"])
        self.assertGreaterEqual(probe["subject_coverage"]["008"], 1884)

    def test_generated_candidates_are_upload_ready(self) -> None:
        report = self.report()
        self.assertEqual(RISK_LABEL, report["risk_label"])
        self.assertEqual(22, report["candidate_count"])
        self.assertEqual(120, len(report["raw_permutation_rankings"]))
        self.assertEqual(120, len(report["calibrated_permutation_rankings"]))
        for candidate in report["candidates"]:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertEqual(RISK_LABEL, candidate["risk_label"])
                self.assertTrue(candidate["upload_ready"])
                self.assertLess(candidate["source_bytes"], SOURCE_LIMIT)
                self.assertLess(candidate["output_bytes"], OUTPUT_LIMIT)
                self.assertEqual(9425, candidate["row_count"])
                self.assertEqual(["subtaskID", "datapointID", "answer"], candidate["columns"])
                self.assertTrue(candidate["ids_match_test_order"])
                self.assertTrue(candidate["answers_finite_non_negative"])

    def test_high_risk_outputs_match_test_contract(self) -> None:
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        for candidate in self.report()["candidates"]:
            rows = read_csv(ROOT / candidate["output_file"])
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
                self.assertEqual({"subtaskID", "datapointID", "answer"}, set(rows[0]))
                self.assertTrue(all(float(row["answer"]) >= 0.0 for row in rows))

    def test_selected_sources_reproduce_outputs(self) -> None:
        report = self.report()
        selected = [report["candidates"][0]]
        selected.append(next(candidate for candidate in report["candidates"] if candidate["family"] == "raw_permutation" and candidate["permutation_rank"] == 1))
        selected.append(next(candidate for candidate in report["candidates"] if candidate["family"] == "calibrated_permutation" and candidate["permutation_rank"] == 1))
        selected.append(next(candidate for candidate in report["candidates"] if candidate["family"] == "ensemble"))
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
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
