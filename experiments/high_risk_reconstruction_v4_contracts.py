from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class ExternalTRTProvider(Protocol):
    def subject_ids(self) -> tuple[str, ...]: ...
    def item_values(self, word_id: str) -> list[float]: ...
    def subject_value(self, subject_id: str, word_id: str, default: float) -> float: ...


class CandidateComponent(Protocol):
    def values(self, rows: Sequence[dict[str, str]]) -> list[float]: ...


class OutputValidator(Protocol):
    def validate(self, rows: Sequence[dict[str, str]], values: Sequence[float]) -> dict: ...


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    name: str
    config: dict
    family: str
    hypothesis: str
    component_labels: tuple[str, ...]
    participant_mapping_mode: str
    zero_strategy: str
    scale_strategy: str
    next_manual_upload_target: bool = False


@dataclass(frozen=True)
class RuntimeContext:
    train_rows: list[dict[str, str]]
    test_rows: list[dict[str, str]]
    provider: ExternalTRTProvider
    anchor: list[float]
    previous_anchor: list[float]
