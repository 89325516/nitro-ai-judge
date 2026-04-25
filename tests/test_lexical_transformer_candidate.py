from __future__ import annotations

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "official_candidates/023_lexical_transformer_source.py"


def load_candidate():
    spec = importlib.util.spec_from_file_location("lexical_transformer_candidate", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAIN_ROWS = [
    {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "answer": "0", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "answer": "210", "participant_id": "001", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_0", "word": "Ana", "answer": "120", "participant_id": "002", "text": "lit_alpha"},
    {"word_id": "lit_alpha_1_page_1_1", "word": "citeste", "answer": "230", "participant_id": "002", "text": "lit_alpha"},
]
TEST_ROWS = [
    {"word_id": "arg_beta_2_page_1_0", "word": "Laptele", "participant_id": "999", "text": "arg_beta", "datapointID": "a"},
    {"word_id": "arg_beta_2_page_1_1", "word": "bun", "participant_id": "999", "text": "arg_beta", "datapointID": "b"},
]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class LexicalTransformerCandidateTest(unittest.TestCase):
    def test_base_features_include_finite_frequency_values(self) -> None:
        candidate = load_candidate()
        features = candidate.base_features(TEST_ROWS[0])
        self.assertIn("zipf_frequency", features)
        self.assertTrue(math.isfinite(features["zipf_frequency"]))
        self.assertTrue(math.isfinite(features["zipf_x_length"]))

    def test_deterministic_encoder_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            test = root / "test.csv"
            output = root / "out.csv"
            write_csv(train, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            args = SimpleNamespace(encoder="deterministic", model="unused", device="cpu", chunk_size=4, overlap=1, feature_width=7, raw_width=8, alpha=10.0)

            load_candidate().generate(train, test, output, args)

            rows = read_csv(output)
        self.assertEqual(["subtaskID", "datapointID", "answer"], list(rows[0].keys()))
        self.assertEqual(["a", "b"], [row["datapointID"] for row in rows])
        for row in rows:
            self.assertEqual("1", row["subtaskID"])
            self.assertGreaterEqual(float(row["answer"]), 0.0)

    def test_test_answers_do_not_change_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            test = root / "test.csv"
            changed = root / "changed.csv"
            output = root / "out.csv"
            changed_output = root / "changed_out.csv"
            changed_rows = [dict(row, answer="9999") for row in TEST_ROWS]
            args = SimpleNamespace(encoder="deterministic", model="unused", device="cpu", chunk_size=4, overlap=1, feature_width=7, raw_width=8, alpha=10.0)
            write_csv(train, TRAIN_ROWS, ["word_id", "word", "answer", "participant_id", "text"])
            write_csv(test, TEST_ROWS, ["word_id", "word", "participant_id", "text", "datapointID"])
            write_csv(changed, changed_rows, ["word_id", "word", "participant_id", "text", "datapointID", "answer"])

            candidate = load_candidate()
            candidate.generate(train, test, output, args)
            candidate.generate(train, changed, changed_output, args)

            self.assertEqual(output.read_text(encoding="utf-8"), changed_output.read_text(encoding="utf-8"))

    def test_validation_report_recommends_only_local_estimate(self) -> None:
        report = json.loads((ROOT / "reports/lexical_transformer_candidate_validation.json").read_text(encoding="utf-8"))
        self.assertEqual("local_estimate", report["score_type"])
        self.assertTrue(report["recommended_for_official_upload"])
        self.assertGreater(report["mean_score"], report["frozen_transformer_local_score"])


if __name__ == "__main__":
    unittest.main()
