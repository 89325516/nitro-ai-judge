# Noise-Aware V2 Directional Push Design

## Purpose

Candidate `199_noise_champion_scaled_external_denoising` scored `53.8099243 / 100`, so the next push should not change strategy. V2 treats the `159 -> 199` output delta as official evidence and continues the same noise-aware path with controlled directional extrapolation.

## Strategy

The batch keeps one goal: improve the hidden score by extending the officially validated denoising direction without over-shooting. Every candidate starts from Candidate `199`, then adds some combination of the official-winning direction, denoised Mac-scaled model residuals, low-noise public aggregate TRT consensus, empirical-Bayes shrinkage, and zero-hurdle correction.

The largest candidate is the only next manual upload target. Earlier candidates are diagnostics that keep the same strategy but test different movement and denoising assumptions.

## Candidate Families

- `200`: reproduces the current official champion output.
- `201-212`: moves along `output_199 - output_159` with fixed gamma and residual-shaping variants.
- `213-224`: uses Mac-scaled denoised stackers with larger but still local-device-safe HGB, ExtraTrees, Ridge, and Huber models.
- `225-236`: strengthens low-noise external aggregate TRT consensus and empirical-Bayes shrinkage.
- `237-248`: fuses direction, denoised stacker, external consensus, EB, and hurdle components in one output.
- `249`: champion selected by a fixed movement gate from the same V2 components.

## Primitive Acceptance Criteria

- Candidate `199` is recorded as the current official best before V2 candidates are generated.
- Candidate `249` is the largest numeric candidate after generation.
- Candidate `249` is marked as the next manual upload target.
- Every V2 candidate has one answer for every test datapoint.
- Every V2 candidate preserves the required submission columns.
- Every V2 answer is finite and non-negative.
- Every V2 source can reproduce its paired output from the tracked inputs and pinned public data.
- Candidate `249` moves from Candidate `199` within the documented movement gate.
- V2 reports do not claim another official improvement before judge feedback is recorded.
