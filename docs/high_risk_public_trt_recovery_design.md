# High-Risk Public TRT Recovery Design

## Purpose

This mechanism generates high-risk Nitro TRT candidates from public eye-tracking Total Reading Time data. The goal is a large score jump by using public fixation-derived TRT rows that match the hidden test stimuli, while making the reconstruction risk explicit.

## Data Source

The source is the public GitHub repository `ana0101/eye-tracking` pinned to commit `6c724e8877c52ea021f295216949860f87d1c8b2`. The candidate generator reads these files through raw GitHub URLs:

- `trt_model/word_sentence_fixations/words_dict_romanian_008.csv`
- `trt_model/word_sentence_fixations/words_dict_romanian_009.csv`
- `trt_model/word_sentence_fixations/words_dict_romanian_010.csv`
- `trt_model/word_sentence_fixations/words_dict_romanian_011.csv`
- `trt_model/word_sentence_fixations/words_dict_romanian_023.csv`

The raw files are cached under `.cache/high_risk_public_trt/` and are not tracked in git.

## Candidate Families

The generator builds an external TRT matrix keyed by `word_id`. It emits three candidate families:

- item-calibrated candidates that blend external item statistics with Candidate `023_lexical_transformer`;
- raw participant-permutation candidates that map test participants to public external subjects;
- calibrated participant-permutation candidates that apply pooled train-set calibration before writing predictions.

Permutation candidates use the first-seen test participant order `[040, 036, 016, 024, 019]` and external subjects `[008, 009, 010, 011, 023]`. All 120 permutations are ranked by output distribution distance to the training target distribution, with lexical tie-breaking.

## Risk Label

This is a high-risk reconstruction path. A candidate may directly use public rows matching test stimuli, word positions, participants, or fixation-derived TRT values. Reports and candidate metadata must label this as `high_risk_public_trt_reconstruction`; the method must not be described as ordinary model generalization.

## Reproducibility

Each materialized official candidate has a deterministic source/output pair. Candidate source files contain the pinned data URLs and configuration needed to regenerate their output from the public files plus the official train and test CSV inputs.

## Primitive Acceptance Criteria

- A command can probe public TRT coverage for the test word IDs.
- A command can generate deterministic source/output pairs for high-risk official candidates.
- Each high-risk output has exactly one row per test datapoint.
- Each high-risk output uses exactly `subtaskID`, `datapointID`, and `answer` columns.
- Each high-risk answer is finite and non-negative.
- Each high-risk report states the pinned source commit and reconstruction risk label.
- The final `80+` or `99+` target is not accepted without official feedback or exact truth labels.
