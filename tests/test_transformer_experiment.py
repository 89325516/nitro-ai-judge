from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.transformer_model import DeterministicWordEncoder, compute_row_vectors, vector_feature_dicts
from experiments.transformer_pages import average_subword_vectors, build_page_words, parse_position
from experiments.transformer_experiment import run_smoke


class TransformerPageMappingTest(unittest.TestCase):
    def test_parse_position_reads_page_and_word_index(self) -> None:
        self.assertEqual(parse_position("arg_text_10_page_3_42"), (3, 42))
        self.assertEqual(parse_position("missing-pattern"), (0, 0))

    def test_page_words_deduplicate_repeated_participants(self) -> None:
        rows = [
            self.row("text_a_page_1_0", "Alpha", "001", "text_a"),
            self.row("text_a_page_1_0", "Alpha", "002", "text_a"),
            self.row("text_a_page_1_1", "Beta", "001", "text_a"),
        ]
        pages = build_page_words(rows)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].indices, (0, 1))
        self.assertEqual(pages[0].words, ("Alpha", "Beta"))

    def test_average_subword_vectors_returns_one_vector_per_word(self) -> None:
        hidden = np.asarray([[1, 1], [3, 3], [10, 10], [20, 20]], dtype=np.float32)
        vectors = average_subword_vectors(hidden, [None, 0, 0, 1], word_count=2)
        self.assertTrue(np.allclose(vectors, [[6.5, 6.5], [20.0, 20.0]]))

    def test_compute_row_vectors_preserves_input_row_count(self) -> None:
        rows = [
            self.row("text_a_page_1_0", "Alpha", "001", "text_a"),
            self.row("text_a_page_1_1", "Beta", "001", "text_a"),
            self.row("text_a_page_1_0", "Alpha", "002", "text_a"),
        ]
        vectors = compute_row_vectors(rows, DeterministicWordEncoder(width=8), width=7)
        self.assertEqual(vectors.shape, (3, 7))
        self.assertTrue(np.allclose(vectors[0], vectors[2]))
        self.assertEqual(len(vector_feature_dicts(vectors)), 3)

    @staticmethod
    def row(word_id: str, word: str, participant_id: str, text: str) -> dict[str, str]:
        return {"word_id": word_id, "word": word, "participant_id": participant_id, "text": text, "answer": "1"}


class TransformerSmokeReportTest(unittest.TestCase):
    def test_smoke_report_is_local_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            report_path = Path(temp_root) / "smoke.json"
            args = type(
                "Args",
                (),
                {
                    "train": Path("data/train_data.csv"),
                    "limit_rows": 6,
                    "report": report_path,
                    "encoder": "deterministic",
                    "model": "unused",
                    "device": "cpu",
                    "chunk_size": 4,
                    "overlap": 1,
                    "feature_width": 7,
                    "raw_width": 8,
                },
            )()
            result = run_smoke(args)
            stored = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result, stored)
        self.assertEqual(result["score_type"], "local_estimate")
        self.assertEqual(result["row_count"], 6)
        self.assertFalse(result["promoted"])


if __name__ == "__main__":
    unittest.main()
