from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.semantic_features import page_context, parse_word_id
from experiments.semantic_trt import generate

TRAIN_ROWS = [
    {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "answer": "0", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "answer": "210", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_2", "word": "repede.", "answer": "180", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "arg_beta_2_page_1_0", "word": "Laptele", "answer": "260", "participant_id": "002", "text": "arg_beta"},
    {"word_id": "arg_beta_2_page_1_1", "word": "este", "answer": "0", "participant_id": "002", "text": "arg_beta"},
    {"word_id": "arg_beta_2_page_1_2", "word": "bun", "answer": "160", "participant_id": "002", "text": "arg_beta"},
]

TEST_ROWS = [
    {"word_id": "ins_gamma_3_page_1_0", "word": "Cuvant", "participant_id": "999", "text": "ins_gamma", "datapointID": "a"},
    {"word_id": "ins_gamma_3_page_1_1", "word": "nou", "participant_id": "999", "text": "ins_gamma", "datapointID": "b"},
]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SemanticTrtTest(unittest.TestCase):
    def test_parse_word_id_reads_source_page_and_index(self) -> None:
        self.assertEqual((10, 3, 42), parse_word_id("arg_pisacowsmilk_10_page_3_42"))

    def test_page_context_uses_unique_word_order_not_participant_duplicates(self) -> None:
        rows = [
            {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "participant_id": "001", "text": "lit_alpha"},
            {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "participant_id": "002", "text": "lit_alpha"},
            {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "participant_id": "001", "text": "lit_alpha"},
        ]
        contexts = page_context(rows)
        self.assertEqual("citeste", contexts[0]["next"])
        self.assertEqual("citeste", contexts[1]["next"])

    def test_cli_contract_for_unseen_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            output_path = root / "submission.csv"
            write_csv(train_path, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test_path, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            generate(train_path, test_path, output_path)
            rows = read_csv(output_path)
            self.assertEqual(["subtaskID", "datapointID", "answer"], list(rows[0].keys()))
            self.assertEqual(["a", "b"], [row["datapointID"] for row in rows])
            for row in rows:
                self.assertEqual("1", row["subtaskID"])
                value = float(row["answer"])
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)

    def test_test_answers_do_not_change_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            changed_path = root / "changed.csv"
            output_path = root / "out.csv"
            changed_output_path = root / "changed_out.csv"
            changed_rows = [dict(row, answer=str(index * 1111)) for index, row in enumerate(TEST_ROWS, start=1)]
            write_csv(train_path, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test_path, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            write_csv(changed_path, changed_rows, ["word_id", "word", "participant_id", "text", "datapointID", "answer"])
            generate(train_path, test_path, output_path)
            generate(train_path, changed_path, changed_output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), changed_output_path.read_text(encoding="utf-8"))

    def test_candidate_report_is_local_estimate_and_not_promoted(self) -> None:
        report_path = Path(__file__).resolve().parents[1] / "reports/semantic_trt_candidate_validation.json"
        if not report_path.exists():
            self.skipTest("semantic candidate report has not been generated")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual("local_estimate", report["score_type"])
        self.assertFalse(report["promoted"])
        self.assertLess(report["semantic_local_score"], report["transformer_local_score"])


if __name__ == "__main__":
    unittest.main()
