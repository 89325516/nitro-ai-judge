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

## Change Discipline

- Any new mechanism is documented before implementation.
- Any changed behavior updates this file in the same change set.
- Acceptance checks describe externally observable outcomes, not private implementation details.
- Dead paths introduced by replaced requirements are removed rather than preserved as comments.
