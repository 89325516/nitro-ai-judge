from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK_LABEL = "high_risk_public_trt_reconstruction"


class HighRiskPublicTRTHistoryTest(unittest.TestCase):
    def test_probe_records_public_trt_coverage(self) -> None:
        probe = json.loads((ROOT / "reports/high_risk_public_trt_probe.json").read_text(encoding="utf-8"))
        self.assertEqual(RISK_LABEL, probe["risk_label"])
        self.assertEqual(1885, probe["test_unique_word_ids"])
        self.assertEqual(1885, probe["item_coverage"])
        self.assertEqual(1.0, probe["item_coverage_rate"])

    def test_historical_report_is_metadata_after_pruning(self) -> None:
        report = json.loads((ROOT / "reports/high_risk_public_trt_report.json").read_text(encoding="utf-8"))
        self.assertEqual(RISK_LABEL, report["risk_label"])
        self.assertEqual(22, report["candidate_count"])
        self.assertEqual(120, len(report["raw_permutation_rankings"]))
        self.assertEqual(120, len(report["calibrated_permutation_rankings"]))
        self.assertTrue((ROOT / "official_candidates/023_lexical_transformer_output.csv").exists())
        self.assertTrue((ROOT / "official_candidates/045_public_trt_ensemble_fallback_output.csv").exists())


if __name__ == "__main__":
    unittest.main()
