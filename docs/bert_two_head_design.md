# Fine-Tuned Romanian BERT Two-Head Design

## Purpose

This experiment adds a token-level Romanian BERT candidate for Total Reading Time prediction. It stays outside the stable submission path until official rules and local evidence support promotion.

## Model Behavior

The model predicts two observable quantities for each word:

- read probability: whether the word receives positive reading time;
- positive reading time: the expected milliseconds if the word is read.

The final word prediction is the product of read probability and positive reading time. This matches the data fact that many words are skipped while read words have continuous millisecond values.

## Data Flow

Rows are grouped by participant, text, source, and page parsed from `word_id`. Each group is split into bounded word chunks so that tokenizer truncation does not drop rows. During training, every chunk carries the original word labels. During prediction, chunk outputs are written back to the original test row order.

The production backend loads `dumitrescustefan/bert-base-romanian-cased-v1` through Hugging Face. A tiny backend exists only for deterministic smoke tests and does not represent the real candidate.

## Training Objective

The read head uses binary cross entropy against `answer > 0`. The time head predicts `log1p(answer)` for positive rows only. Prediction converts the time head back to milliseconds and multiplies by the read probability.

The default training setup freezes most encoder layers and fine-tunes only the last encoder layers plus the two heads. This keeps runtime bounded while still adapting Romanian BERT to this task.

## Promotion Policy

This track is experimental. It must not replace `solution.py` or an official candidate unless it beats the current official-best direction under valid evidence and remains reproducible under source-size and output-size limits.

## Primitive Acceptance Criteria

- A command can train a two-head word-level model from a labeled train CSV.
- A command can write one finite non-negative answer for each test datapoint.
- The output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Test-file answer values do not affect predictions.
- Smoke reports identify themselves as local estimates, not official scores.
- Token-to-word mapping preserves one prediction per input row.
- Model artifacts and downloaded weights are not tracked in git.
