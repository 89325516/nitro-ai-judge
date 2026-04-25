# Noise-Aware Push Design

## Purpose

Candidate `159_rebound_champion_counter_prior_stack` scored `38.87076 / 100` and becomes the new official best. The next bottleneck is not more external feature plumbing; the train labels are noisy. Every training word has 30 participant observations, while within-word variation is large enough to overwhelm item difficulty. This mechanism makes label denoising the primary optimization route.

## Noise Evidence

The training set has a global zero rate around `0.306`, mean within-word standard deviation around `210ms`, and between-word mean standard deviation around `194ms`. Participant means and zero rates vary widely, so a raw row label is a noisy observation of item difficulty plus participant behavior rather than a clean target.

## Candidate Families

Candidates `160-199` remain upload-ready official candidates. They use Candidate `159` as the official-best base and add controlled denoised residuals.

- `160-169`: item-level denoising targets, including median, trimmed mean, winsorized mean, and zero-aware mean.
- `170-179`: participant reliability weighting based on participant scale, zero rate, and within-item deviation.
- `180-189`: hierarchical empirical-Bayes targets that shrink item/text/page estimates toward stable priors.
- `190-198`: zero-inflated hurdle outputs that separate skip probability from positive TRT intensity.
- `199`: champion that blends the official best with hierarchical, reliability, and hurdle residuals.

## Primitive Acceptance Criteria

- Candidate `159` is recorded as an official successful score.
- Candidate `159` is the current best official candidate.
- Candidate `199` is the largest numeric candidate after generation.
- Candidate `199` is the next manual upload target.
- Noise-aware candidates preserve one output row per test datapoint.
- Noise-aware candidates preserve the required submission columns.
- Noise-aware candidate answers are finite and non-negative.
- Noise-aware reports include observable noise profile evidence.
- The champion moves materially from Candidate `159` without repeating the failed Candidate `129` overcorrection.
