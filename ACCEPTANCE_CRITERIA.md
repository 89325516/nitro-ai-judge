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

## Semantic TRT Sprint

- Semantic TRT candidates use only observable train and test CSV fields unless an external-resource track is explicitly allowed.
- Semantic TRT output contains exactly one row for each input test datapoint.
- Semantic TRT output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Semantic TRT answers are finite and non-negative.
- Test-file answer values do not affect Semantic TRT predictions.
- Semantic TRT local reports are labeled as local estimates, not official or exact scores.
- The final `100 / 100` target is satisfied only by official judge feedback or exact hidden truth labels.

## BERT Two-Head Experiment

- BERT two-head outputs contain exactly one row for each input test datapoint.
- BERT two-head output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- BERT two-head answers are finite and non-negative.
- Test-file answer values do not affect BERT two-head predictions.
- BERT two-head smoke reports are labeled as local estimates, not official scores.
- Token-to-word mapping preserves one prediction per input row.
- BERT model weights and generated model artifacts are not tracked in git.

## BERT Failure Recovery

- Candidate `022_bert_two_head_scale120` is not recommended for official upload after a local text-holdout score below the active fallback candidates.
- Future BERT official candidates require a full local text-holdout report before packaging as upload-ready.
- BERT hybrid outputs contain exactly one row for each input test datapoint.
- BERT hybrid answers are finite and non-negative.
- BERT hybrid reports are labeled as local estimates unless official judge feedback is available.
- BERT hybrid promotion requires beating the frozen Transformer local estimate before official packaging.

## Lexical Frequency Transformer Candidate

- Lexical frequency Transformer outputs contain exactly one row for each input test datapoint.
- Lexical frequency Transformer output columns are exactly `subtaskID`, `datapointID`, and `answer`.
- Lexical frequency Transformer answers are finite and non-negative.
- Test-file answer values do not affect lexical frequency Transformer predictions.
- Lexical frequency Transformer reports are labeled as local estimates unless official judge feedback is available.
- Lexical frequency Transformer upload recommendation requires beating the frozen Transformer local estimate.

## Official Feedback

- Official judge scores outrank local estimates when choosing submission candidates.
- Each official submission record preserves the reported submission ID and score.
- The highest successful official score is identifiable from the project records.
- Pending official candidates have uploadable source and output files.
- Each pending official candidate output has exactly one row per test datapoint.
- Each pending official candidate output uses exactly `subtaskID`, `datapointID`, and `answer`.
- Candidate source and output files stay within the official upload size limits.

## Official 80 Push

- The official `80 / 100` target is satisfied only by an official or exact-truth score of at least `80`.
- Official 80 push candidates have deterministic source and output pairs.
- Official 80 push candidates preserve test datapoint ID order.
- Official 80 push candidate answers are finite and non-negative.
- Official 80 push reports identify upload priority and hypothesis for each pending candidate.
- The official ledger can be updated without changing previously recorded official scores.
- The current best successful official score is identifiable after each ledger update.

## External Data Branching

- The conservative branch does not directly reconstruct test labels from public row-level matches.
- The high-risk branch labels any direct test-row reconstruction before a candidate is promoted.
- External-data candidates preserve exactly one answer for each test datapoint.
- External-data candidates preserve the required public submission columns.
- Each external-data candidate report states its evidence class.
- Branch score targets are not treated as met without official feedback or exact truth labels.

## High-Risk Public TRT Recovery

- Public TRT recovery candidates state their pinned external data source.
- Public TRT recovery candidates label direct row-level matching as high risk.
- Public TRT recovery outputs contain exactly one row for each test datapoint.
- Public TRT recovery outputs use exactly `subtaskID`, `datapointID`, and `answer` columns.
- Public TRT recovery answers are finite and non-negative.
- Public TRT recovery source files can reproduce their paired output files from public URLs.
- Public TRT recovery reports do not mark `80+` or `99+` complete without official or exact-truth evidence.

## Three-Tier External Data Fusion Sprint

- Three-tier fusion candidates combine safe, medium, and high-risk components in one output.
- Three-tier fusion candidates state component weights and risk labels.
- The safe component does not read public fixation or TRT rows.
- The medium component does not use exact test participant plus text plus word lookup.
- The high-risk component labels lookup or reconstruction behavior.
- Three-tier fusion outputs contain exactly one row for each test datapoint.
- Three-tier fusion outputs use exactly `subtaskID`, `datapointID`, and `answer` columns.
- Three-tier fusion answers are finite and non-negative.
- Three-tier fusion reports do not mark `80+` or `99+` complete without official or exact-truth evidence.

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
