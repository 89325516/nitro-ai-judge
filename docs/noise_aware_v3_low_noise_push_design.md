# Noise-Aware V3 Low-Noise Push Design

## Purpose

Candidate `249` repeated Candidate `199`'s official score at `53.8099243 / 100`. V3 keeps Candidate `199` as the official best and treats Candidate `249` as a neutral/no-gain direction anchor. The next push should not blindly extrapolate from `199` to `249`; it should improve the data target and model training quality around the proven `199` base.

## Strategy

V3 generates a smaller batch of 20 high-quality candidates. Each candidate starts from Candidate `199`, then adds a controlled residual from cleaner external consensus targets, denoised train labels, larger tabular models, robust noisy-label filtering, or uncertainty-aware stacking. Candidate `269` is the only next manual upload target.

## Method Families

- `250-254`: low-noise clean targets from external consensus and empirical-Bayes shrinkage.
- `255-260`: MacBook-bounded high-capacity HGB, ExtraTrees, Ridge, and Huber residual models.
- `261-265`: paper-inspired robust training labels: ordered target encodings, small-loss filtering, distributional shrinkage, and Super Learner-style blending.
- `266-268`: compact fusions of clean target, high-capacity model, and uncertainty-aware residuals.
- `269`: champion using Candidate `199` plus clean target residual, high-capacity denoised stacker, and uncertainty shrinkage while avoiding the no-gain `249 - 199` direction.

## Primitive Acceptance Criteria

- Candidate `249` is recorded as scored without replacing Candidate `199` as best.
- Candidate `269` is the largest numeric candidate after generation.
- Candidate `269` is the only next manual upload target.
- Every V3 candidate has one answer for every test datapoint.
- Every V3 candidate preserves the required submission columns.
- Every V3 answer is finite and non-negative.
- Every V3 source can reproduce its paired output.
- V3 reports include low-noise target evidence and model capacity evidence.
- Candidate `269` moves from Candidate `199` within the documented movement gate.
- No pruned historical exploration files are restored.
