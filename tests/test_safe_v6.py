from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from evaluate import summarize_folds
from experiments.safe_v6 import generate


TRAIN_ROWS = [
    {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "answer": "0", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "answer": "210", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_2", "word": "repede", "answer": "180", "participant_id": "001", "text": "lit_alpha"},
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


class SafeV6Test(unittest.TestCase):
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
                answer = float(row["answer"])
                self.assertTrue(math.isfinite(answer))
                self.assertGreaterEqual(answer, 0.0)

    def test_test_answers_do_not_change_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            changed_test_path = root / "changed_test.csv"
            output_path = root / "submission.csv"
            changed_output_path = root / "changed_submission.csv"
            changed_rows = [dict(row, answer=str(index * 9999)) for index, row in enumerate(TEST_ROWS, start=1)]
            write_csv(train_path, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test_path, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            write_csv(changed_test_path, changed_rows, ["word_id", "word", "participant_id", "text", "datapointID", "answer"])

            generate(train_path, test_path, output_path)
            generate(train_path, changed_test_path, changed_output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), changed_output_path.read_text(encoding="utf-8"))

    def test_local_estimate_report_is_not_exact_evidence(self) -> None:
        report = summarize_folds([{"score": 1.0, "r2": 0.0, "pearson": 0.02, "row_count": 1}])
        self.assertEqual("local_estimate", report["score_type"])
        self.assertNotEqual("exact_with_truth", report["score_type"])


if __name__ == "__main__":
    unittest.main()
