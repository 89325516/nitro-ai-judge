# Submission Pipeline Design

## Goal

Generate a reproducible CSV submission for Nitro AI Judge reading-time prediction. The product boundary is a local command that reads the provided data files and writes one numeric prediction for each test datapoint.

## Mechanism

The pipeline trains a compact regression model on observable word and position signals that can generalize to unseen participants and unseen texts. It does not use concrete participant identifiers because all test participants are absent from the training set.

The model uses these input signals:

- lowercased word identity;
- text family prefix, such as `arg`, `ins`, `lit`, `enc`, or `popsci`;
- word length and logarithmic word length;
- page number and word index parsed from `word_id`;
- URL, capitalization, uppercase, punctuation-only, digit, and accent flags.

The current stable baseline trains Ridge regression directly on raw Total Reading Time milliseconds. Predictions are clipped to non-negative reading-time values before the output file is written.

## Interfaces

- Input train file: CSV with `word_id`, `word`, `answer`, `participant_id`, and `text`.
- Input test file: CSV with `word_id`, `word`, `participant_id`, `text`, and `datapointID`.
- Output file: CSV with exactly `subtaskID`, `datapointID`, and `answer`.
- Command: `python3 solution.py --train data/train_data.csv --test data/test_data.csv --output submission.csv`.

## Primitive Acceptance Criteria

- The command exits successfully for the provided files.
- The output contains one row for every test datapoint.
- Every output row uses subtask ID `1`.
- Every output answer is finite and non-negative.
- Output datapoint IDs match the test datapoint IDs exactly.
- The pipeline can be rerun from the repository contents without reading the PDF.
