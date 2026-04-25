# Three-Tier Fusion External Data Sprint Design

## Purpose

This mechanism creates fused high-risk-branch candidates. Each candidate combines safe language generalization, medium-risk public behavior mapping, and high-risk public reconstruction into one prediction vector. The three tiers are components of the same score-pushing individual, not separate candidate lanes.

## Fusion Components

- Safe component: uses provided CSV fields, Candidate `023_lexical_transformer`, lexical complexity, word shape, page position, and train-calibrated surface priors. It does not read public fixation or TRT rows by itself.
- Medium component: uses pinned public eye-tracking data only as aggregated item, text, page, zero-rate, and variability statistics. It does not use exact test participant plus text plus word lookup.
- High-risk component: may use public rows that match test stimuli, subject-permutation mappings, fixation-derived TRT, or Candidate `045` as reconstruction evidence. It is labeled as reconstruction risk, not ordinary model generalization.

## Candidate Plan

Candidates `046-087` are all three-tier fusion candidates. They vary fusion weights, calibration scale, zero handling, and high-risk mapping source, but every materialized candidate carries all three component predictions in its recipe and report.

The pinned public source is `ana0101/eye-tracking` commit `6c724e8877c52ea021f295216949860f87d1c8b2`. Raw public CSV files are cached under `.cache/three_tier_external/` and are not tracked in git.

## Reproducibility

The generator writes a source/output pair for every candidate. Each source stores the pinned public URLs, risk labels, component weights, and deterministic config. Reproduction uses train/test CSV files, Candidate `023`, and optionally Candidate `045`; if Candidate `045` is unavailable, the high-risk component is recomputed from public rows.

## Primitive Acceptance Criteria

- A probe report states public source coverage and exact reconstruction risk.
- Every generated candidate reports safe, medium, and high-risk component weights.
- The safe component does not read public fixation or TRT rows.
- The medium component does not use exact test participant plus text plus word lookup.
- The high-risk component labels lookup or reconstruction behavior.
- Every generated output has one row per test datapoint.
- Every generated output uses exactly `subtaskID`, `datapointID`, and `answer` columns.
- Every generated answer is finite and non-negative.
- No `80+` or `99+` milestone is accepted without official or exact-truth evidence.
