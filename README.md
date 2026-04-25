# Nitro AI Judge

Nitro AI Judge is a public project repository for the provided training, test, and sample output data. The current repository state is intentionally limited to data publication and acceptance criteria; model code, training scripts, and evaluation logic will be added only after their requirements are explicitly defined.

## Repository Contents

- `data/train_data.csv` contains labeled training records with `word_id`, `word`, `answer`, `participant_id`, and `text` columns.
- `data/test_data.csv` contains unlabeled test records with `word_id`, `word`, `participant_id`, `text`, and `datapointID` columns.
- `data/sample_output.csv` contains the expected output shape with `subtaskID`, `datapointID`, and `answer` columns.
- `ACCEPTANCE_CRITERIA.md` defines the observable product criteria that future changes must preserve or intentionally update.

## Current Boundary

This bootstrap does not include a model, feature pipeline, scoring logic, deployment process, or generated predictions. Future mechanisms must be documented before implementation, and each mechanism must include primitive acceptance criteria before code is changed.

## Data Contract

The public data files are expected to remain readable as CSV files, preserve their current headers, and support producing one answer for each requested `datapointID` when prediction functionality is introduced.

## Development Discipline

Changes should keep module boundaries small, avoid hidden state, prefer injected dependencies, and test observable behavior rather than implementation details.
