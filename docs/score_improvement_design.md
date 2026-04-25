# Score Improvement Design

## Goal

Improve the local estimate above the corrected official baseline while preserving the Nitro submission contract.

## Official Baseline Correction

The competition announcement clarified that the baseline was accidentally displayed as `0.36 / 100` and should be interpreted as `36 / 100`. The local evaluator already reports scores on the corrected 0 to 100 scale, so the metric formula does not change.

## Immediate Model Change

The previous baseline optimized `log1p(answer)` and converted predictions back with `expm1`. That compressed millisecond predictions and reduced R2. The improved baseline trains Ridge regression directly on raw Total Reading Time milliseconds with `alpha=100.0`.

The model keeps the same generalizable inputs:

- lowercased word identity;
- text family prefix;
- word length and logarithmic word length;
- page number and word index parsed from `word_id`;
- URL, capitalization, uppercase, punctuation-only, digit, and accent flags.

Concrete participant IDs remain excluded because public test participants are unseen.

## Transformer Track

Transformer fine-tuning can be useful for future contextual features or token-level regression, but it is not part of the current judge upload path. A Transformer-based model must first beat the raw-Ridge local text-holdout score and remain compatible with runtime and source-size constraints before it replaces the deterministic baseline.

## Primitive Acceptance Criteria

- The local text-holdout mean score is above `38.0 / 100`.
- The score report is labeled as a local estimate, not a hidden-test score.
- The generated submission keeps exactly one prediction per test datapoint.
- The generated submission preserves the required `subtaskID`, `datapointID`, and `answer` columns.
- The source file remains below the judge upload limit.
