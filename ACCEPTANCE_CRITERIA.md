# Acceptance Criteria

## Repository Bootstrap

- The public repository is discoverable as `nitro-ai-judge`.
- The repository contains the three provided CSV files under `data/`.
- The repository does not contain the source PDF file.
- Project documentation and tracked project text are written in English.
- The default branch contains a clean, reproducible starting point for future work.

## Data Availability

- A reader can open `data/train_data.csv` and observe labeled examples.
- A reader can open `data/test_data.csv` and observe records that require answers.
- A reader can open `data/sample_output.csv` and observe the required answer format.
- The sample output has one answer row for each test `datapointID`.

## Future Product Behavior

- For every accepted test record, the product produces exactly one visible answer.
- Each produced answer is associated with the correct `datapointID`.
- Missing, duplicated, or unparseable output rows are not acceptable.
- Users can distinguish source data, generated outputs, and documentation without inspecting internal code.

## Submission Pipeline

- A local command can produce a submission CSV from the provided train and test CSV files.
- The generated submission has exactly the columns `subtaskID`, `datapointID`, and `answer`.
- The generated submission contains exactly one row for each test datapoint.
- Each generated answer is a finite, non-negative number.
- The generated datapoint IDs match the test datapoint IDs without omissions or duplicates.
- The generation process does not require the source PDF.
- The source file used for submission remains below the judge source upload size limit.

## Evaluation Program

- The evaluator computes the Nitro metric from observable truth and prediction values.
- Negative R2 contributes zero to the final score.
- Pearson correlation contributes by absolute value.
- Exact scoring rejects missing, duplicate, non-numeric, or non-finite prediction rows.
- Cross-validation scores are labeled as local estimates, not hidden-test results.
- Candidate solutions are evaluated through CSV files and command execution, not private model internals.

## Model Improvement

- A replacement submission model must beat `36 / 100` on the local text-holdout estimate before it is treated as a viable candidate.
- A model report must identify whether the score is a local estimate or an exact score with truth labels.
- Model changes must preserve one generated answer for every test datapoint.
- Model changes must preserve the public submission CSV columns.
- Hidden-test performance is not claimed without judge feedback or truth labels.

## Transformer Experiment

- Transformer work remains experimental until its local estimate beats the stable raw-Ridge estimate.
- Transformer reports identify themselves as local estimates.
- Each evaluated row receives exactly one contextual feature vector.
- The stable submission path is not replaced by a Transformer path without documented score improvement.
- Generated Transformer caches and model weights are not tracked in git.

## Transformer Surprisal

- Surprisal experiments preserve the stable raw-Ridge submission unless the local estimate exceeds `39.388844 / 100`.
- Each surprisal feature row contains finite numeric values.
- Surprisal feature loading rejects row-count mismatches.
- Generated feature caches are excluded from git.
- Fine-tuning is not introduced without a separate documented mechanism.

## Safe V6 50 Target

- Safe V6 is promoted only when the local three-fold unseen-text estimate is at least `50 / 100`.
- Safe V6 reports are labeled as local estimates unless exact truth labels or official judge feedback are available.
- Safe V6 output contains exactly one row for each input test datapoint.
- Safe V6 output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Safe V6 answers are finite and non-negative.
- Test-file answer values do not affect Safe V6 predictions.
- Target-derived statistics are not computed from held-out prediction rows.

## Final 99+ Target

- The final product target is an official or exact-truth score of at least `99 / 100`.
- Local estimates, smoke tests, and oracle ceilings do not satisfy the final `99+` target.
- Every score report identifies its evidence class.
- Leakage and ceiling checks run before a high-cost model is promoted.
- The stable fallback submission remains available until a candidate passes the local promotion threshold.

## Change Discipline

- Any new mechanism is documented before implementation.
- Any changed behavior updates this file in the same change set.
- Acceptance checks describe externally observable outcomes, not private implementation details.
- Dead paths introduced by replaced requirements are removed rather than preserved as comments.
