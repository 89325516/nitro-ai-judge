from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from experiments.transformer_pages import build_page_words, page_key, parse_position


class WordSurprisalScorer(Protocol):
    def score_words(self, words: tuple[str, ...]) -> list[dict[str, float]]:
        """Return one surprisal feature dictionary per input word."""


class DeterministicSurprisalScorer:
    def score_words(self, words: tuple[str, ...]) -> list[dict[str, float]]:
        features = []
        for word in words:
            token_count = max(1, math.ceil(len(word) / 4))
            char_score = sum(ord(char) for char in word) % 113
            mean = math.log1p(len(word) + char_score / 17.0)
            features.append(make_feature_dict([mean] * token_count, token_count, 0))
        return features


class HuggingFaceMaskedLMScorer:
    def __init__(self, model_name: str, device: str = "auto", max_words: int = 80):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = self.select_device(device)
        self.max_words = max_words
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.mask_token_id is None:
            raise ValueError("tokenizer does not define a mask token")

    def select_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.torch.backends.mps.is_available():
            return "mps"
        if self.torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def score_words(self, words: tuple[str, ...]) -> list[dict[str, float]]:
        if not words:
            return []
        encoded = self.tokenizer(
            list(words[: self.max_words]),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = encoded.word_ids()
        input_ids = encoded["input_ids"][0]
        spans = token_spans(word_ids, len(words))
        features = []
        for word_index in range(len(words)):
            span = spans[word_index]
            if word_index >= self.max_words or not span:
                features.append(make_feature_dict([], 0, 1))
                continue
            surprisals = []
            for token_position in span:
                masked = input_ids.clone()
                masked[token_position] = self.tokenizer.mask_token_id
                batch = {key: value.clone().to(self.device) for key, value in encoded.items()}
                batch["input_ids"] = masked.unsqueeze(0).to(self.device)
                with self.torch.no_grad():
                    logits = self.model(**batch).logits[0, token_position]
                log_probs = self.torch.log_softmax(logits, dim=-1)
                token_id = int(input_ids[token_position])
                surprisals.append(float(-log_probs[token_id].detach().cpu()))
            features.append(make_feature_dict(surprisals, len(span), 0))
        return features


def token_spans(word_ids: list[int | None], word_count: int) -> list[list[int]]:
    spans = [[] for _ in range(word_count)]
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and 0 <= word_id < word_count:
            spans[word_id].append(token_index)
    return spans


def make_feature_dict(values: list[float], token_count: int, missing_count: int) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        finite = [0.0]
    return {
        "surprisal_mean": float(np.mean(finite)),
        "surprisal_max": float(np.max(finite)),
        "surprisal_sum": float(np.sum(finite)),
        "surprisal_token_count": float(token_count),
        "surprisal_split_complexity": float(max(0, token_count - 1)),
        "surprisal_missing_count": float(missing_count),
    }


def compute_surprisal_features(rows: list[dict[str, str]], scorer: WordSurprisalScorer) -> list[dict[str, float]]:
    page_features = {}
    for page in build_page_words(rows):
        scored = scorer.score_words(page.words)
        page_features[page.key] = {index: scored[position] for position, index in enumerate(page.indices)}
    features = []
    for row in rows:
        _, index = parse_position(row["word_id"])
        features.append(page_features[page_key(row)][index])
    return features


def write_feature_cache(path: Path, features: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, sort_keys=True) + "\n", encoding="utf-8")


def read_feature_cache(path: Path, expected_rows: int) -> list[dict[str, float]]:
    features = json.loads(path.read_text(encoding="utf-8"))
    if len(features) != expected_rows:
        raise ValueError(f"feature row count mismatch: expected {expected_rows}, got {len(features)}")
    return [{str(key): float(value) for key, value in row.items()} for row in features]
