from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["subtaskID", "datapointID", "answer"]
SOURCE_UPLOAD_LIMIT_BYTES = 35 * 1024


class SubmissionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_path = Path(cls.temp_dir.name) / "submission.csv"
        subprocess.run(
            [
                sys.executable,
                "solution.py",
                "--train",
                "data/train_data.csv",
                "--test",
                "data/test_data.csv",
                "--output",
                str(cls.output_path),
            ],
            cwd=ROOT,
            check=True,
        )
        cls.test_rows = cls.read_csv(ROOT / "data/test_data.csv")
        cls.output_rows = cls.read_csv(cls.output_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def test_output_has_required_shape(self) -> None:
        with self.output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            self.assertEqual(next(reader), REQUIRED_COLUMNS)
        self.assertEqual(len(self.output_rows), len(self.test_rows))
        self.assertEqual(len(self.output_rows), 9425)

    def test_datapoint_ids_match_test_file(self) -> None:
        test_ids = [row["datapointID"] for row in self.test_rows]
        output_ids = [row["datapointID"] for row in self.output_rows]
        self.assertEqual(output_ids, test_ids)
        self.assertEqual(len(output_ids), len(set(output_ids)))

    def test_answers_are_finite_non_negative_numbers(self) -> None:
        for row in self.output_rows:
            self.assertEqual(row["subtaskID"], "1")
            answer = float(row["answer"])
            self.assertTrue(math.isfinite(answer))
            self.assertGreaterEqual(answer, 0.0)

    def test_source_file_fits_judge_upload_limit(self) -> None:
        self.assertLess((ROOT / "solution.py").stat().st_size, SOURCE_UPLOAD_LIMIT_BYTES)

    def test_project_files_do_not_contain_chinese_characters(self) -> None:
        violations = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
