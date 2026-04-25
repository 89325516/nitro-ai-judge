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
FUSION_LABEL = "three_tier_fusion_with_high_risk_reconstruction"
RISK_LABELS = {"safe_external_generalization", "medium_public_behavior_mapping", "high_risk_public_reconstruction"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class ThreeTierExternalCandidateTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/three_tier_external_report.json").read_text(encoding="utf-8"))

    def test_probe_records_fusion_coverage(self) -> None:
        probe = json.loads((ROOT / "reports/three_tier_external_probe.json").read_text(encoding="utf-8"))
        self.assertEqual(FUSION_LABEL, probe["risk_label"])
        self.assertEqual(RISK_LABELS, set(probe["risk_labels"]))
        self.assertEqual(1885, probe["test_unique_word_ids"])
        self.assertEqual(1885, probe["test_item_coverage"])
        self.assertTrue(probe["exact_row_reconstruction_risk"])
        self.assertEqual(["arg_pisacowsmilk", "ins_learningmobility", "lit_alchemist"], probe["test_text_overlap"])

    def test_every_candidate_is_one_fused_individual(self) -> None:
        report = self.report()
        self.assertEqual(42, report["candidate_count"])
        self.assertEqual({"raw", "cal", "base"}, {candidate["mode"] for candidate in report["candidates"]})
        for candidate in report["candidates"]:
            with self.subTest(candidate=candidate["candidate_id"]):
                self.assertEqual(FUSION_LABEL, candidate["risk_label"])
                self.assertEqual(RISK_LABELS, set(candidate["risk_labels"]))
                self.assertFalse(candidate["safe_component_uses_public_trt"])
                self.assertFalse(candidate["medium_exact_lookup"])
                self.assertTrue(candidate["high_reconstruction"])
                weights = candidate["component_weights"]
                self.assertGreater(weights["safe"], 0.0)
                self.assertGreater(weights["medium"], 0.0)
                self.assertGreater(weights["high"], 0.0)
                self.assertAlmostEqual(1.0, sum(weights.values()), places=6)

    def test_outputs_are_upload_ready(self) -> None:
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
                self.assertTrue(all(float(row["answer"]) >= 0.0 for row in rows))

    def test_selected_sources_reproduce_outputs(self) -> None:
        report = self.report()
        selected = []
        for mode in ("raw", "cal", "base"):
            selected.extend([candidate for candidate in report["candidates"] if candidate["mode"] == mode][:2])
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
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                self.assertEqual(read_csv(ROOT / candidate["output_file"]), read_csv(output))


if __name__ == "__main__":
    unittest.main()
