from __future__ import annotations

from typing import Protocol

import numpy as np

from experiments.transformer_pages import PageKey, average_subword_vectors, build_page_words, page_key, parse_position, reduce_vectors


class WordVectorEncoder(Protocol):
    def encode_words(self, words: tuple[str, ...]) -> np.ndarray:
        """Return one contextual vector per input word."""


class DeterministicWordEncoder:
    def __init__(self, width: int = 8):
        self.width = width

    def encode_words(self, words: tuple[str, ...]) -> np.ndarray:
        rows = []
        for word in words:
            code_sum = sum(ord(char) for char in word)
            rows.append([(code_sum + index * len(word)) % 97 / 97.0 for index in range(self.width)])
        return np.asarray(rows, dtype=np.float32)


class HuggingFaceWordEncoder:
    def __init__(self, model_name: str, device: str = "auto", chunk_size: int = 96, overlap: int = 8):
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
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + self.chunk_size]
            encoded = self.tokenizer(
                list(chunk),
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            word_ids = encoded.word_ids()
            tensors = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**tensors).last_hidden_state[0].detach().cpu().numpy()
            chunk_vectors = average_subword_vectors(output, word_ids, len(chunk))
            if sums is None:
                sums = np.zeros((len(words), chunk_vectors.shape[1]), dtype=np.float32)
            for local_index, vector in enumerate(chunk_vectors):
                global_index = start + local_index
                if global_index >= len(words):
                    continue
                if np.any(vector):
                    sums[global_index] += vector
                    counts[global_index] += 1.0
        if sums is None:
            return np.zeros((len(words), 1), dtype=np.float32)
        present = counts > 0
        sums[present] /= counts[present, None]
        return sums


def compute_row_vectors(rows: list[dict[str, str]], encoder: WordVectorEncoder, width: int) -> np.ndarray:
    page_vectors: dict[PageKey, dict[int, np.ndarray]] = {}
    for page in build_page_words(rows):
        encoded = encoder.encode_words(page.words)
        reduced = reduce_vectors(encoded, width)
        page_vectors[page.key] = {index: reduced[position] for position, index in enumerate(page.indices)}
    features = []
    for row in rows:
        _, index = parse_position(row["word_id"])
        features.append(page_vectors[page_key(row)][index])
    return np.asarray(features, dtype=float)


def vector_feature_dicts(vectors: np.ndarray, prefix: str = "tf") -> list[dict[str, float]]:
    return [{f"{prefix}_{index}": float(value) for index, value in enumerate(row)} for row in vectors]
