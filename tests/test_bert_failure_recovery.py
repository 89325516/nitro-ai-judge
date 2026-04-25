from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.bert_hybrid import run

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


class BertFailureRecoveryTest(unittest.TestCase):
    def test_022_rejection_report_is_not_promoted(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/bert_two_head_022_rejection.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("rejected", report["decision"])
        self.assertLess(report["mean_score"], report["fallback_candidates"]["transformer_text_holdout"])

    def test_hybrid_tiny_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            test = root / "test.csv"
            output = root / "submission.csv"
            report = root / "report.json"
            write_csv(train, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            args = type("Args", (), {"train": train, "test": test, "output": output, "report": report, "backend": "tiny", "model": "unused",
                                      "device": "cpu", "epochs": 1, "learning_rate": 0.01, "dropout": 0.0, "unfreeze_layers": 0,
                                      "chunk_words": 8, "max_length": 32, "max_train_chunks": 2, "skip_weight": 0.5,
                                      "time_weight": 1.0, "seed": 0, "prediction_scale": 1.0, "zero_rate": 0.0,
                                      "bert_scale": 1.0, "inner_folds": 2, "stack_alpha": 10.0, "min_gain": 0.05})()
            result = run(args)
            rows = read_csv(output)
        self.assertEqual("local_estimate", result["score_type"])
        self.assertEqual(["subtaskID", "datapointID", "answer"], list(rows[0].keys()))
        self.assertEqual(["a", "b"], [row["datapointID"] for row in rows])
        for row in rows:
            value = float(row["answer"])
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_test_answers_do_not_change_hybrid_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            test = root / "test.csv"
            changed = root / "changed.csv"
            out = root / "out.csv"
            changed_out = root / "changed_out.csv"
            changed_rows = [dict(row, answer="9999") for row in TEST_ROWS]
            write_csv(train, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            write_csv(changed, changed_rows, ["word_id", "word", "participant_id", "text", "datapointID", "answer"])
            base = {"train": train, "backend": "tiny", "model": "unused", "device": "cpu", "epochs": 1, "learning_rate": 0.01,
                    "dropout": 0.0, "unfreeze_layers": 0, "chunk_words": 8, "max_length": 32, "max_train_chunks": 2,
                    "skip_weight": 0.5, "time_weight": 1.0, "seed": 0, "prediction_scale": 1.0, "zero_rate": 0.0,
                    "bert_scale": 1.0, "inner_folds": 2, "stack_alpha": 10.0, "min_gain": 0.05, "report": None}
            run(type("Args", (), {**base, "test": test, "output": out})())
            run(type("Args", (), {**base, "test": changed, "output": changed_out})())
            self.assertEqual(out.read_text(encoding="utf-8"), changed_out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
