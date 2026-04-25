from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.transformer_deep_experiment import PROMOTION_SCORE, run_smoke
from experiments.transformer_surprisal import (
    DeterministicSurprisalScorer,
    compute_surprisal_features,
    make_feature_dict,
    read_feature_cache,
    token_spans,
    write_feature_cache,
)


class SurprisalFeatureTest(unittest.TestCase):
    def test_token_spans_preserve_original_word_mapping(self) -> None:
        spans = token_spans([None, 0, 0, 1, None, 2], word_count=3)
        self.assertEqual(spans, [[1, 2], [3], [5]])

    def test_feature_dict_values_are_finite(self) -> None:
        features = make_feature_dict([1.0, 3.0, float("nan")], token_count=2, missing_count=0)
        self.assertEqual(features["surprisal_mean"], 2.0)
        self.assertEqual(features["surprisal_max"], 3.0)
        self.assertEqual(features["surprisal_token_count"], 2.0)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))

    def test_compute_surprisal_features_returns_one_row_per_input(self) -> None:
        rows = [
            self.row("text_a_page_1_0", "Alpha", "001", "text_a"),
            self.row("text_a_page_1_1", "Beta", "001", "text_a"),
            self.row("text_a_page_1_0", "Alpha", "002", "text_a"),
        ]
        features = compute_surprisal_features(rows, DeterministicSurprisalScorer())
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0], features[2])
        self.assertIn("surprisal_mean", features[0])

    def test_feature_cache_rejects_row_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            path = Path(temp_root) / "features.json"
            write_feature_cache(path, [{"surprisal_mean": 1.0}])
            self.assertEqual(read_feature_cache(path, expected_rows=1), [{"surprisal_mean": 1.0}])
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                read_feature_cache(path, expected_rows=2)

    @staticmethod
    def row(word_id: str, word: str, participant_id: str, text: str) -> dict[str, str]:
        return {"word_id": word_id, "word": word, "participant_id": participant_id, "text": text, "answer": "1"}


class SurprisalSmokeReportTest(unittest.TestCase):
    def test_smoke_report_stays_unpromoted_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            report_path = Path(temp_root) / "report.json"
            args = type(
                "Args",
                (),
                {
                    "train": Path("data/train_data.csv"),
                    "limit_rows": 5,
                    "cache": None,
                    "report": report_path,
                    "scorer": "deterministic",
                    "model": "unused",
                    "device": "cpu",
                    "max_words": 8,
                    "use_hidden": False,
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
        self.assertEqual(result["promotion_score"], PROMOTION_SCORE)
        self.assertFalse(result["promoted"])


if __name__ == "__main__":
    unittest.main()
