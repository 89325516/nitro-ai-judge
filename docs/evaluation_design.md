# Evaluation Design

## Goal

Measure candidate submission quality with the same public metric described by Nitro AI Judge, while keeping local estimates separate from exact hidden-test scores.

## Modes

- Exact scoring compares a prediction CSV against a truth CSV that contains `datapointID` and `answer`.
- Cross-validation estimates performance by holding out labeled training rows, running a candidate command, and scoring the produced fold predictions.

## Default Estimate

The default validation strategy is text holdout. This matches the current data shape where public test texts are absent from the training texts. The report must describe this score as a local estimate, not as a hidden-test leaderboard score.

## Interfaces

- Exact score command: `python3 evaluate.py score --truth truth.csv --predictions submission.csv --report reports/exact_score.json`.
- Cross-validation command: `python3 evaluate.py cross-validate --train data/train_data.csv --strategy text --folds 3 --command "python3 solution.py --train {train} --test {test} --output {output}" --report reports/baseline_text_holdout.json`.
- Report output: JSON with aggregate score, R2, Pearson correlation, row count, mode, and fold details when applicable.

## Primitive Acceptance Criteria

- The metric clamps negative R2 to zero.
- The metric uses the absolute Pearson correlation.
- Exact scoring rejects missing, duplicate, non-numeric, and non-finite prediction answers.
- Cross-validation runs the candidate command only through the documented CSV interface.
- Cross-validation reports fold-level scores and aggregate scores.
- Local reports do not claim to know hidden test performance without truth labels.
