# Nitro AI Judge

Nitro AI Judge is a public project repository for the provided reading-time data and a compact reproducible baseline submission pipeline.

## Repository Contents

- `data/train_data.csv` contains labeled training records with `word_id`, `word`, `answer`, `participant_id`, and `text` columns.
- `data/test_data.csv` contains unlabeled test records with `word_id`, `word`, `participant_id`, `text`, and `datapointID` columns.
- `data/sample_output.csv` contains the expected output shape with `subtaskID`, `datapointID`, and `answer` columns.
- `solution.py` generates a baseline prediction file from the train and test CSV files.
- `submission.csv` is the generated Nitro-compatible output file.
- `docs/submission_pipeline_design.md` documents the baseline mechanism and primitive acceptance criteria.
- `ACCEPTANCE_CRITERIA.md` defines the observable product criteria that future changes must preserve or intentionally update.

## Current Boundary

This repository includes a local baseline model and generated submission artifact. It does not include a deployed service, automated judge upload, or leaderboard-specific tuning workflow. Future mechanisms must be documented before implementation, and each mechanism must include primitive acceptance criteria before code is changed.

## Data Contract

The public data files are expected to remain readable as CSV files, preserve their current headers, and support producing one answer for each requested `datapointID`.

## Generate A Submission

Run the local baseline pipeline with:

```bash
python3 solution.py --train data/train_data.csv --test data/test_data.csv --output submission.csv
```

The command writes a Nitro-compatible CSV with `subtaskID`, `datapointID`, and `answer` columns. The solution intentionally uses general word and position features instead of concrete participant IDs because the test participants are unseen.

## Development Discipline

Changes should keep module boundaries small, avoid hidden state, prefer injected dependencies, and test observable behavior rather than implementation details.
