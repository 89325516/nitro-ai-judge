from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import evaluate

ROOT = Path(__file__).resolve().parents[1]


class MetricTest(unittest.TestCase):
    def test_negative_r2_is_clamped_and_pearson_is_absolute(self) -> None:
        result = evaluate.compute_metric([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertEqual(result["r2"], 0.0)
        self.assertAlmostEqual(result["pearson"], 1.0)
        self.assertAlmostEqual(result["score"], 50.0)

    def test_exact_score_requires_matching_ids(self) -> None:
        truth = [{"datapointID": "a", "answer": "1"}, {"datapointID": "b", "answer": "2"}]
        predictions = [{"datapointID": "a", "answer": "1.5"}, {"datapointID": "b", "answer": "2.5"}]
        result = evaluate.score_rows(truth, predictions)
        self.assertEqual(result["row_count"], 2)
        self.assertIn("score", result)

    def test_exact_score_rejects_missing_duplicate_and_non_numeric_rows(self) -> None:
        truth = [{"datapointID": "a", "answer": "1"}, {"datapointID": "b", "answer": "2"}]
        with self.assertRaisesRegex(ValueError, "missing prediction IDs"):
            evaluate.score_rows(truth, [{"datapointID": "a", "answer": "1"}])
        with self.assertRaisesRegex(ValueError, "duplicate prediction datapointID"):
            evaluate.score_rows(truth, [{"datapointID": "a", "answer": "1"}, {"datapointID": "a", "answer": "2"}])
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            evaluate.score_rows(truth, [{"datapointID": "a", "answer": "bad"}, {"datapointID": "b", "answer": "2"}])


class CrossValidationTest(unittest.TestCase):
    def test_cross_validation_runs_candidate_through_csv_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_dir = Path(temp_root)
            train_path = temp_dir / "train.csv"
            report_path = temp_dir / "report.json"
            predictor_path = temp_dir / "dummy_predictor.py"
            self.write_train_fixture(train_path)
            predictor_path.write_text(self.dummy_predictor_source(), encoding="utf-8")

            result = evaluate.cross_validate(
                train_path=train_path,
                folds=2,
                command=f"{sys.executable} {predictor_path} --train {{train}} --test {{test}} --output {{output}}",
                report_path=report_path,
            )

            self.assertEqual(result["mode"], "cross_validate")
            self.assertEqual(result["score_type"], "local_estimate")
            self.assertEqual(result["strategy"], "text")
            self.assertEqual(result["fold_count"], 2)
            self.assertEqual(result["row_count"], 4)
            self.assertEqual(len(result["folds"]), 2)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)

    @staticmethod
    def write_train_fixture(path: Path) -> None:
        rows = [
            {"word_id": "text_a_page_1_0", "word": "alpha", "answer": "10", "participant_id": "001", "text": "text_a"},
            {"word_id": "text_a_page_1_1", "word": "beta", "answer": "20", "participant_id": "001", "text": "text_a"},
            {"word_id": "text_b_page_1_0", "word": "gamma", "answer": "30", "participant_id": "002", "text": "text_b"},
            {"word_id": "text_b_page_1_1", "word": "delta", "answer": "40", "participant_id": "002", "text": "text_b"},
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=evaluate.TRAIN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def dummy_predictor_source() -> str:
        return '''
import argparse
import csv
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=Path, required=True)
parser.add_argument("--test", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with args.test.open(newline="", encoding="utf-8-sig") as source, args.output.open("w", newline="", encoding="utf-8") as target:
    rows = list(csv.DictReader(source))
    writer = csv.DictWriter(target, fieldnames=["subtaskID", "datapointID", "answer"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": len(row["word"])})
'''


if __name__ == "__main__":
    unittest.main()
