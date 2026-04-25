# External Data Branching Design

## Purpose

This document defines the repository split between conservative external-data modeling and high-risk public-data recovery. The split keeps score-chasing work explicit while preserving the current submission contract and evidence discipline.

## Shared Baseline

Both branches start from Candidate `023_lexical_transformer`, the strongest recorded local candidate at `40.25002577495926 / 100` on three-fold unseen-text validation. Candidate `023` remains local evidence until official Nitro feedback or exact truth labels prove hidden-test performance.

The shared output contract stays unchanged: each submission writes exactly one finite non-negative `answer` for every test `datapointID` with columns `subtaskID`, `datapointID`, and `answer`.

## Conservative Branch Boundary

Branch `codex/conservative` may use external public resources that improve generalization without directly reconstructing hidden test labels. Allowed resources include public Romanian corpora, lexical frequency lists, pretrained language models, surprisal features, linguistic annotations, and public eye-tracking data that does not expose row-level answers for the test records.

The conservative branch must not directly look up or reconstruct true test TRT values by matching test participant, text, page, word index, or fixation records. Its score path is `40+` from Candidate `023`, then `50-65` through better external features and domain adaptation. It does not claim an `80+` path without official evidence.

## High-Risk Branch Boundary

Branch `codex/high-risk` may pursue public eye-tracking/raw-fixation or derived-TRT recovery when the data appears to match test stimuli, participants, or rows. This branch optimizes for the highest possible official score while labeling the work as high-risk because it may be interpreted as hidden-label reconstruction.

The high-risk score path is `40+` from Candidate `023`, `50-70` from partial external mapping, `80+` from broad true-TRT reconstruction, and `99+` only from official judge feedback or exact truth labels.

## Evidence and Reproducibility

Every external-data candidate must state its evidence class, source data origin, mapping rule, and reproducibility command. High-risk candidates must additionally state whether the output depends on direct test-row reconstruction and whether the uploaded source can reproduce the submitted output under the competition runtime constraints.

## Primitive Acceptance Criteria

- Branch purpose is visible without reading private implementation code.
- Conservative candidates do not use direct test-label reconstruction.
- High-risk candidates label reconstruction risk before promotion.
- Each branch preserves one visible answer per test datapoint.
- Each branch preserves the required output columns.
- Each candidate report states whether its score is local, official, exact, or risk-labeled reconstruction evidence.
