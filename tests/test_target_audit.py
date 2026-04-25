from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_core import leakage_summary, sample_output_diagnostic, same_item_oracle
from experiments.candidate_compare import PROMOTION_SCORE, compare_candidates
from experiments.target_audit import build_reports


class TargetAuditTest(unittest.TestCase):
    def test_leakage_summary_reports_overlap_counts(self) -> None:
        train = [self.row("a", "w1", "p1", "word"), self.row("b", "w2", "p2", "other")]
        test = [self.row("c", "w3", "p3", "word"), self.row("a", "w1", "p4", "missing")]
        report = leakage_summary(train, test)
        self.assertEqual(report["evidence_class"], "leakage_signal")
        self.assertEqual(report["text_overlap_count"], 1)
        self.assertEqual(report["participant_overlap_count"], 0)
        self.assertEqual(report["word_id_overlap_count"], 1)

    def test_sample_output_diagnostic_identifies_probability_scale(self) -> None:
        report = sample_output_diagnostic([
            {"datapointID": "0", "answer": "0.1"},
            {"datapointID": "1", "answer": "0.9"},
        ])
        self.assertEqual(report["evidence_class"], "leakage_signal")
        self.assertTrue(report["looks_like_probability_scale"])

    def test_candidate_comparison_enforces_promotion_threshold(self) -> None:
        result = compare_candidates([
            {"name": "base", "score": 38.4, "score_type": "local_estimate", "mode": "cross_validate", "row_count": 10},
            {"name": "better", "score": PROMOTION_SCORE, "score_type": "local_estimate", "mode": "cross_validate", "row_count": 10},
        ])
        self.assertTrue(result["promoted"])
        self.assertEqual(result["best_candidate"]["name"], "better")

    def test_build_reports_writes_expected_evidence_classes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train = root_path / "train.csv"
            test = root_path / "test.csv"
            sample = root_path / "sample.csv"
            train.write_text(
                "word_id,word,answer,participant_id,text\n"
                "a_page_1_0,A,1,p1,t1\n"
                "a_page_1_0,A,2,p2,t1\n"
                "b_page_1_0,B,3,p1,t2\n"
                "b_page_1_0,B,4,p2,t2\n",
                encoding="utf-8",
            )
            test.write_text("word_id,word,participant_id,text,datapointID\nc_page_1_0,C,p3,t3,0\n", encoding="utf-8")
            sample.write_text("subtaskID,datapointID,answer\n1,0,0.5\n", encoding="utf-8")
            reports = build_reports(train, test, sample, root_path)
            self.assertEqual(reports["score_ceiling_report"]["evidence_class"], "oracle_ceiling")
            self.assertEqual(reports["leakage_audit_report"]["evidence_class"], "leakage_signal")
            self.assertEqual(reports["sample_output_diagnostic"]["evidence_class"], "leakage_signal")
            self.assertTrue((root_path / "score_ceiling_report.json").exists())

    @staticmethod
    def row(word_id: str, text: str, participant_id: str, word: str) -> dict[str, str]:
        return {"word_id": word_id, "text": text, "participant_id": participant_id, "word": word, "answer": "1"}


if __name__ == "__main__":
    unittest.main()
