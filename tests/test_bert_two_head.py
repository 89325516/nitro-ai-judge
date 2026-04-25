from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.bert_two_head import run_smoke, run_train_predict
from experiments.bert_two_head_data import chunk_rows, write_submission
from experiments.bert_two_head_model import combine_predictions, first_token_positions

ROWS = [
    {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "answer": "0", "participant_id": "001", "text": "lit_alpha", "datapointID": "a"},
    {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "answer": "210", "participant_id": "001", "text": "lit_alpha", "datapointID": "b"},
    {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "answer": "120", "participant_id": "002", "text": "lit_alpha", "datapointID": "c"},
    {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "answer": "230", "participant_id": "002", "text": "lit_alpha", "datapointID": "d"},
]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class BertTwoHeadTest(unittest.TestCase):
    def test_chunks_keep_participant_page_labels(self) -> None:
        chunks = chunk_rows(ROWS, labeled=True, chunk_words=8)
        self.assertEqual(2, len(chunks))
        self.assertEqual(("Ana", "citeste"), chunks[0].words)
        self.assertEqual((0.0, 210.0), chunks[0].targets)
        self.assertEqual((2, 3), chunks[1].row_indices)

    def test_first_token_positions_preserve_one_word_mapping(self) -> None:
        self.assertEqual([1, 3], first_token_positions([None, 0, 0, 1, None], 2))
        with self.assertRaises(ValueError):
            first_token_positions([None, 0, None], 2)

    def test_prediction_combination_is_finite_non_negative(self) -> None:
        values = combine_predictions(np.asarray([-10.0, 0.0, 10.0]), np.asarray([0.0, math.log1p(200.0), math.log1p(300.0)]))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertGreater(values[2], values[1])

    def test_write_submission_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.csv"
            write_submission(ROWS[:2], np.asarray([0.0, 123.4]), output)
            with output.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(["subtaskID", "datapointID", "answer"], list(written[0].keys()))
            self.assertEqual(["a", "b"], [row["datapointID"] for row in written])

    def test_tiny_smoke_report_is_local_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            report = root / "report.json"
            write_csv(train, ROWS, ["word_id", "word", "answer", "participant_id", "text", "datapointID"])
            args = SimpleNamespace(train=train, backend="tiny", model="unused", device="cpu", epochs=1, learning_rate=0.01,
                                   dropout=0.0, unfreeze_layers=0, chunk_words=8, max_length=32, max_train_chunks=2,
                                   skip_weight=0.5, time_weight=1.0, seed=0, report=report)
            result = run_smoke(args)
            stored = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(result, stored)
        self.assertEqual("local_estimate", result["score_type"])
        self.assertTrue(result["finite_non_negative"])
        self.assertFalse(result["promoted"])

    def test_tiny_train_predict_writes_one_answer_per_test_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            test = root / "test.csv"
            output = root / "submission.csv"
            report = root / "report.json"
            train_columns = ["word_id", "word", "answer", "participant_id", "text", "datapointID"]
            test_rows = [{key: row[key] for key in ["word_id", "word", "participant_id", "text", "datapointID"]} for row in ROWS[:2]]
            write_csv(train, ROWS, train_columns)
            write_csv(test, test_rows, ["word_id", "word", "participant_id", "text", "datapointID"])
            args = SimpleNamespace(train=train, test=test, output=output, backend="tiny", model="unused", device="cpu", epochs=1,
                                   learning_rate=0.01, dropout=0.0, unfreeze_layers=0, chunk_words=8, max_length=32,
                                   max_train_chunks=2, skip_weight=0.5, time_weight=1.0, seed=0, report=report)
            result = run_train_predict(args)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual("local_estimate", result["score_type"])
        self.assertEqual(["a", "b"], [row["datapointID"] for row in rows])
        self.assertTrue(all(float(row["answer"]) >= 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
