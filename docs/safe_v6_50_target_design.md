# Safe V6 50 Target Design

## Purpose

Safe V6 is a short-term candidate for improving the local Nitro AI Judge score while preserving a leakage-safe validation process. The target is a local three-fold unseen-text estimate of at least `50 / 100`. This target is not an official leaderboard claim.

## Score Evidence

The primary gate is the existing evaluator command with text-grouped cross-validation. Each validation fold contains texts that are absent from the fold training file. The report must be labeled `local_estimate`.

Official hidden-test performance can only be claimed from Nitro Judge feedback or from an exact truth file. A local report, train score, smoke test, or model console score cannot satisfy an official-score claim.

## Candidate Mechanism

Safe V6 treats prediction as a CSV-to-CSV command. It reads train and test CSV files, computes features from the training file and observable test fields, and writes the required submission columns.

The model combines four leak-safe signals:

- a sparse Ridge baseline over word and surface features;
- a numeric gradient-boosting regressor over word shape, context window, position, and smoothed word statistics;
- a hurdle estimate that predicts whether a word is read and then predicts positive reading time;
- a compact Ridge stacker trained from inner text-holdout predictions inside the candidate training file.

Concrete participant IDs are not used in promoted features because public test participants are unseen. Target-derived statistics are computed from the candidate training file only. If a test file accidentally includes an `answer` column, that column is ignored.

## Promotion Rule

Safe V6 remains experimental unless `reports/safe_v6_text_holdout.json` reports `mean_score >= 50.0`. If the gate is not met, the stable submission remains unchanged and the report records the best local estimate.

If the gate is met, the compact promoted path may replace the stable submission path, provided it preserves the output contract and source-size limit.

## Primitive Acceptance Criteria

- A command can generate one output row for each test datapoint.
- The output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Each output answer is finite and non-negative.
- The primary report is labeled as a local estimate.
- The primary report has a mean score of at least `50 / 100` before promotion.
- Changing answers in a test file does not change generated predictions.
- Missing hidden-test labels do not produce official-score claims.
