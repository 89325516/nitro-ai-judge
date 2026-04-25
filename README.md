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

The command writes a Nitro-compatible CSV with `subtaskID`, `datapointID`, and `answer` columns. The solution uses raw-target Ridge regression with general word and position features instead of concrete participant IDs because the test participants are unseen.

The competition baseline correction means `36 / 100` is the practical baseline floor. The current local estimate should be read on the same 0 to 100 scale.

## Evaluate A Solution

Run the default local estimate with:

```bash
python3 evaluate.py cross-validate --train data/train_data.csv --strategy text --folds 3 --command "python3 solution.py --train {train} --test {test} --output {output}" --report reports/baseline_text_holdout.json
```

This cross-validation score is an estimate based on held-out training texts. It is not the hidden Nitro leaderboard score. If a truth file is available, run exact scoring with:

```bash
python3 evaluate.py score --truth truth.csv --predictions submission.csv --report reports/exact_score.json
```

## Transformer Experiments

The `codex/transformer-experiment` branch contains optional Transformer experiments. These experiments add contextual word features from a pretrained language model, but they do not replace the stable raw-Ridge submission unless the local text-holdout score improves beyond `38.388844 / 100`.

Run a tiny smoke test with:

```bash
python3 experiments/transformer_experiment.py smoke --train data/train_data.csv --limit-rows 12 --report reports/transformer_smoke.json
```

For the deeper surprisal experiment, run a deterministic smoke test with:

```bash
python3 experiments/transformer_deep_experiment.py smoke --train data/train_data.csv --limit-rows 8 --scorer deterministic --report reports/transformer_surprisal_smoke.json
```

A real masked-language-model smoke test can be run with:

```bash
python3 experiments/transformer_deep_experiment.py smoke --train data/train_data.csv --limit-rows 4 --scorer hf --model dumitrescustefan/bert-base-romanian-cased-v1 --device auto --report reports/transformer_surprisal_hf_smoke.json
```

## 99+ Target Checks

The final target is `99 / 100`, but that target can only be accepted from exact truth labels or Nitro Judge feedback. Run the local reachability audit with:

```bash
python3 experiments/target_audit.py --train data/train_data.csv --test data/test_data.csv --sample data/sample_output.csv --output-dir reports
```

The audit separates local estimates, oracle ceilings, leakage signals, and exact scores so model progress is not confused with official completion.

## Development Discipline

Changes should keep module boundaries small, avoid hidden state, prefer injected dependencies, and test observable behavior rather than implementation details.
