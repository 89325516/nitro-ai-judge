from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EncodedWords(Protocol):
    def word_ids(self) -> list[int | None]: ...


def first_token_positions(word_ids: list[int | None], word_count: int) -> list[int]:
    positions = [-1] * word_count
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and 0 <= word_id < word_count and positions[word_id] < 0:
            positions[word_id] = token_index
    missing = [index for index, value in enumerate(positions) if value < 0]
    if missing:
        raise ValueError("tokenizer did not preserve every input word")
    return positions


def combine_predictions(read_logits: np.ndarray, log_times: np.ndarray) -> np.ndarray:
    read_prob = 1.0 / (1.0 + np.exp(-read_logits.astype(float)))
    positive_ms = np.expm1(np.clip(log_times.astype(float), 0.0, 10.0))
    return np.clip(read_prob * positive_ms, 0.0, 10000.0)


@dataclass(frozen=True)
class BatchPrediction:
    row_indices: tuple[int, ...]
    values: np.ndarray



class TinyEncoding(dict):
    def __init__(self, ids: list[int], word_ids: list[int | None]):
        super().__init__({"input_ids": ids, "attention_mask": [1] * len(ids)})
        self._word_ids = word_ids

    def word_ids(self) -> list[int | None]:
        return self._word_ids


class TinyTokenizer:
    def __call__(self, words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128):
        ids = [1]
        word_ids: list[int | None] = [None]
        for index, word in enumerate(words[: max_length - 2]):
            ids.append(2 + sum(ord(char) for char in word) % 251)
            word_ids.append(index)
        ids.append(0)
        word_ids.append(None)
        return TinyEncoding(ids, word_ids)


def make_tiny_model():
    import torch

    class TinyTwoHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(260, 16)
            self.read_head = torch.nn.Linear(16, 1)
            self.time_head = torch.nn.Linear(16, 1)

        def forward(self, input_ids, attention_mask):
            hidden = self.embedding(input_ids)
            return self.read_head(hidden).squeeze(-1), torch.nn.functional.softplus(self.time_head(hidden).squeeze(-1))

    return TinyTokenizer(), TinyTwoHead()


def make_two_head_model(model_name: str, dropout: float, unfreeze_layers: int):
    import torch
    from transformers import AutoModel

    class BertTwoHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(dropout)
            self.read_head = torch.nn.Linear(hidden, 1)
            self.time_head = torch.nn.Linear(hidden, 1)
            self.configure_trainable_layers(unfreeze_layers)

        def configure_trainable_layers(self, layer_count: int) -> None:
            for param in self.encoder.parameters():
                param.requires_grad = False
            layers = getattr(getattr(self.encoder, "encoder", None), "layer", [])
            for layer in list(layers)[-max(0, layer_count) :]:
                for param in layer.parameters():
                    param.requires_grad = True

        def forward(self, input_ids, attention_mask, **unused):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            hidden = self.dropout(hidden)
            return self.read_head(hidden).squeeze(-1), torch.nn.functional.softplus(self.time_head(hidden).squeeze(-1))

    return BertTwoHead()
