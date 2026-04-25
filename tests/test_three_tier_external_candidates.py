from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUSION_LABEL = "three_tier_fusion_with_high_risk_reconstruction"
RISK_LABELS = {"safe_external_generalization", "medium_public_behavior_mapping", "high_risk_public_reconstruction"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class ThreeTierExternalHistoryTest(unittest.TestCase):
    def report(self) -> dict:
        return json.loads((ROOT / "reports/three_tier_external_report.json").read_text(encoding="utf-8"))

    def test_probe_records_fusion_coverage(self) -> None:
        probe = json.loads((ROOT / "reports/three_tier_external_probe.json").read_text(encoding="utf-8"))
        self.assertEqual(FUSION_LABEL, probe["risk_label"])
        self.assertEqual(RISK_LABELS, set(probe["risk_labels"]))
        self.assertEqual(1885, probe["test_item_coverage"])
        self.assertTrue(probe["exact_row_reconstruction_risk"])

    def test_candidate_087_anchor_is_retained(self) -> None:
        report = self.report()
        self.assertEqual(42, report["candidate_count"])
        self.assertEqual("087", report["official_feedback"]["candidate_id"])
        anchor = next(row for row in report["candidates"] if row["candidate_id"] == "087")
        rows = read_csv(ROOT / anchor["output_file"])
        expected_ids = [row["datapointID"] for row in read_csv(ROOT / "data/test_data.csv")]
        self.assertEqual(expected_ids, [row["datapointID"] for row in rows])
        self.assertTrue(all(math.isfinite(float(row["answer"])) and float(row["answer"]) >= 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
