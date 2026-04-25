# Lexical Frequency Transformer Candidate Design

## Purpose

The next candidate stops tuning old output scales and adds a new signal: public lexical frequency. Romanian reading time should decrease for common words and increase for rare words, so this candidate combines frozen contextual Transformer features with word-frequency features before the Ridge regressor.

## Mechanism

- Read the provided train and test CSV files.
- Build the same observable surface features used by the frozen Transformer candidate.
- Add Romanian Zipf frequency values when the optional `wordfreq` package is available.
- Add frequency interactions with word length and text family.
- Encode each page with a frozen Romanian BERT model and reduce word-level vectors to compact numeric features.
- Train a direct raw-target Ridge regressor on milliseconds.
- Write a submission with exactly `subtaskID`, `datapointID`, and `answer`.

## Evidence Boundary

This candidate is still a local-estimate candidate until Nitro Judge returns an official score. The local three-fold unseen-text estimate is higher than the previous frozen Transformer local estimate, but official performance can still differ because the official participants and texts are hidden.

## Primitive Acceptance Criteria

- The candidate command writes one output row for every input test row.
- The output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Every answer is finite and non-negative.
- Test-file `answer` values, if accidentally present, do not change generated predictions.
- The source file stays below the official source upload limit.
- The generated local report is labeled as a local estimate.
- The candidate is recommended for upload only if its local unseen-text estimate beats the previous frozen Transformer local estimate.
