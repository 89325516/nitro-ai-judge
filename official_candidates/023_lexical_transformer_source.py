from __future__ import annotations
import argparse
import csv
import math
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None
POSITION_PATTERN = re.compile(r"_page_(\d+)_(\d+)$")
DEFAULT_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
FAMILIES = ("arg", "enc", "ins", "lit", "popsci")
def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
def parse_position(word_id: str) -> tuple[int, int]:
    match = POSITION_PATTERN.search(word_id)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)
def family(text: str) -> str:
    return text.split("_", 1)[0]
def has_accent(word: str) -> bool:
    return any(unicodedata.category(char) == "Mn" for char in unicodedata.normalize("NFD", word))
def word_frequency(word: str) -> float:
    return float(zipf_frequency(word.lower(), "ro")) if zipf_frequency else 0.0
def base_features(row: dict[str, str]) -> dict[str, float]:
    word = row["word"]
    lower = word.lower()
    page, index = parse_position(row["word_id"])
    freq = word_frequency(lower)
    text_family = family(row["text"])
    features = {
        "bias": 1.0,
        "word=" + lower: 1.0,
        "family=" + text_family: 1.0,
        "length": float(len(word)),
        "log_length": math.log1p(len(word)),
        "alpha_count": float(sum(char.isalpha() for char in word)),
        "digit_count": float(sum(char.isdigit() for char in word)),
        "page": float(page),
        "index": float(index),
        "log_index": math.log1p(index),
        "is_url": float(word.startswith("http")),
        "is_capitalized": float(word[:1].isupper()),
        "is_upper": float(word.isupper()),
        "is_punctuation_only": float(bool(word) and all(not char.isalnum() for char in word)),
        "has_accent": float(has_accent(word)),
        "zipf_frequency": freq,
        "zipf_x_length": freq * math.log1p(len(word)),
    }
    for item in FAMILIES:
        features[f"zipf_family={item}"] = freq if text_family == item else 0.0
    return features
def page_key(row: dict[str, str]) -> tuple[str, int]:
    page, _ = parse_position(row["word_id"])
    return row["text"], page
def build_page_words(rows: list[dict[str, str]]) -> list[tuple[tuple[str, int], tuple[int, ...], tuple[str, ...]]]:
    grouped: OrderedDict[tuple[str, int], OrderedDict[int, str]] = OrderedDict()
    for row in rows:
        _, index = parse_position(row["word_id"])
        grouped.setdefault(page_key(row), OrderedDict()).setdefault(index, row["word"])
    pages = []
    for key, indexed_words in grouped.items():
        ordered = sorted(indexed_words.items())
        pages.append((key, tuple(index for index, _ in ordered), tuple(word for _, word in ordered)))
    return pages
def average_subword_vectors(hidden_states: np.ndarray, word_ids: list[int | None], word_count: int) -> np.ndarray:
    sums = np.zeros((word_count, hidden_states.shape[1]), dtype=np.float32)
    counts = np.zeros(word_count, dtype=np.float32)
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and 0 <= word_id < word_count:
            sums[word_id] += hidden_states[token_index]
            counts[word_id] += 1.0
    present = counts > 0
    sums[present] /= counts[present, None]
    return sums
def reduce_vectors(vectors: np.ndarray, width: int) -> np.ndarray:
    stats = np.column_stack([vectors.mean(axis=1), vectors.std(axis=1), np.linalg.norm(vectors, axis=1), vectors.min(axis=1), vectors.max(axis=1)])
    prefix = vectors[:, : max(0, width - stats.shape[1])]
    return np.column_stack([stats, prefix]).astype(float)
class DeterministicEncoder:
    def __init__(self, width: int):
        self.width = width
    def encode_words(self, words: tuple[str, ...]) -> np.ndarray:
        output = np.zeros((len(words), self.width), dtype=np.float32)
        for row_index, word in enumerate(words):
            for col_index, char in enumerate(word[: self.width]):
                output[row_index, col_index] = (ord(char) % 97) / 97.0
        return output
class WordEncoder:
    def __init__(self, model_name: str, device: str, chunk_size: int, overlap: int):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = self.select_device(device)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    def select_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.torch.backends.mps.is_available():
            return "mps"
        if self.torch.cuda.is_available():
            return "cuda"
        return "cpu"
    def encode_words(self, words: tuple[str, ...]) -> np.ndarray:
        if not words:
            return np.zeros((0, 1), dtype=np.float32)
        sums = None
        counts = np.zeros(len(words), dtype=np.float32)
        for start in range(0, len(words), max(1, self.chunk_size - self.overlap)):
            chunk = words[start : start + self.chunk_size]
            encoded = self.tokenizer(list(chunk), is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            tensors = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                hidden = self.model(**tensors).last_hidden_state[0].detach().cpu().numpy()
            chunk_vectors = average_subword_vectors(hidden, encoded.word_ids(), len(chunk))
            if sums is None:
                sums = np.zeros((len(words), chunk_vectors.shape[1]), dtype=np.float32)
            for local_index, vector in enumerate(chunk_vectors):
                global_index = start + local_index
                if global_index < len(words) and np.any(vector):
                    sums[global_index] += vector
                    counts[global_index] += 1.0
        present = counts > 0
        sums[present] /= counts[present, None]
        return sums
def make_encoder(args: argparse.Namespace):
    if args.encoder == "deterministic":
        return DeterministicEncoder(args.raw_width)
    return WordEncoder(args.model, args.device, args.chunk_size, args.overlap)
def row_vectors(rows: list[dict[str, str]], encoder, width: int) -> np.ndarray:
    page_vectors = {}
    for key, indices, words in build_page_words(rows):
        reduced = reduce_vectors(encoder.encode_words(words), width)
        page_vectors[key] = {index: reduced[position] for position, index in enumerate(indices)}
    return np.asarray([page_vectors[page_key(row)][parse_position(row["word_id"])[1]] for row in rows], dtype=float)
def combined_features(rows: list[dict[str, str]], encoder, width: int) -> list[dict[str, float]]:
    vectors = row_vectors(rows, encoder, width)
    output = []
    for row, vector in zip(rows, vectors, strict=True):
        features = base_features(row)
        features.update({f"tf_{index}": float(value) for index, value in enumerate(vector)})
        output.append(features)
    return output
def write_submission(rows: list[dict[str, str]], predictions: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtaskID", "datapointID", "answer"], lineterminator="\n")
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({"subtaskID": "1", "datapointID": row["datapointID"], "answer": f"{max(0.0, float(prediction)):.6f}"})
def generate(train_path: Path, test_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    train_rows, test_rows = read_rows(train_path), read_rows(test_path)
    encoder = make_encoder(args)
    targets = np.asarray([float(row["answer"]) for row in train_rows], dtype=float)
    model = make_pipeline(DictVectorizer(sparse=True), Ridge(alpha=args.alpha, random_state=0))
    predictions = model.fit(combined_features(train_rows, encoder, args.feature_width), targets).predict(combined_features(test_rows, encoder, args.feature_width))
    write_submission(test_rows, np.clip(predictions, 0.0, 10000.0), output_path)
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lexical-frequency frozen Transformer candidate output.")
    parser.add_argument("--train", type=Path, default=Path("data/train_data.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--encoder", choices=["hf", "deterministic"], default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--feature-width", type=int, default=13)
    parser.add_argument("--raw-width", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=100.0)
    args = parser.parse_args()
    generate(args.train, args.test, args.output, args)
if __name__ == "__main__":
    main()
