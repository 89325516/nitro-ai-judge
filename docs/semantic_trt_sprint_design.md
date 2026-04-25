# Semantic TRT Sprint Design

## Purpose

This sprint adds a conservative CSV-only candidate path for improving Nitro reading-time predictions without changing the stable submission path. The candidate treats Total Reading Time as the combination of skipped words, positive reading time for words that are read, contextual ranking signals, and distribution calibration.

## Evidence Policy

Official judge feedback is the strongest available evidence. Local text-holdout scores remain useful for rejecting weak candidates, but they are not hidden-test claims. The current official best remains the selected fallback until a new successful official score is higher.

The final 100 target can only be satisfied by an official score or exact hidden-label score. No local estimate, smoke test, or calibration hypothesis can satisfy that target.

## CSV-Only Candidate

The candidate uses only the provided train and test CSV fields. It ignores any accidental `answer` column in a test file. It reconstructs page and sentence context from `word_id`, text names, row order, and punctuation. It derives observable lexical features such as word shape, local position, neighboring word shape, repeated words, URL flags, capitalization, diacritics, prefixes, and suffixes.

The model has three separate prediction views:

- skip view: estimates whether a word receives positive reading time;
- positive-time view: estimates milliseconds conditional on positive reading time;
- rank view: preserves word-level difficulty ordering for Pearson correlation.

A fold-safe local blender chooses or combines those views using only training-fold predictions. The upload output remains a plain CSV with the required three columns.

## Calibration Strategy

Low-dimensional calibration remains allowed after official feedback: global scale, zero-rate threshold, text-family multiplier, and test-text multiplier. These transformations are used to correct distribution mismatch, not to infer hidden labels.

## External Resource Track

External public resources and pretrained Romanian models remain isolated experiments until competition rules are confirmed. They may be used for analysis, but the conservative upload candidate must be reproducible from the provided CSVs and normal Python dependencies.

## Primitive Acceptance Criteria

- A command reads train and test CSV files and writes one answer row for every test datapoint.
- The output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Every answer is finite and non-negative.
- Test-file answer values do not affect predictions.
- Candidate reports identify local estimates separately from official results.
- Official scores remain the promotion authority when available.
- No candidate is described as reaching 100 without official or exact-truth evidence.
