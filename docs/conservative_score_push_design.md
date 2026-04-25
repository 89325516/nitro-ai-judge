# Conservative Score Push Design

## Purpose

This branch improves Nitro TRT predictions inside the medium-risk boundary. It may use public external resources for generalization, but it must not directly reconstruct hidden test answers through row-level matches.

## Evidence Position

The branch starts from Candidate `023_lexical_transformer`, which has the best recorded local text-holdout estimate: `40.25002577495926 / 100`. This score is local evidence only. Official judge feedback or exact truth labels remain the only evidence that can prove hidden-test score targets.

## Allowed Data Use

Allowed inputs are public resources that describe language, reading difficulty, or population-level eye-tracking behavior without exposing the exact test answer rows. Examples include Romanian corpora, word frequency lists, pretrained language models, surprisal estimates, morphological or syntactic annotations, and public eye-tracking datasets used as additional training data when test-row labels are not directly copied.

The branch may learn features from external eye-tracking data, but it must not submit values obtained by matching public records to the exact test participant, text, page, word index, or fixation-derived TRT row.

## Score Route

The `40+` route is Candidate `023` as the stable local baseline. The `50-65` route should come from stronger external frequency, surprisal, morphology, and domain-adaptation features that improve local unseen-text validation and official score feedback. The branch does not claim an `80+` route unless official feedback proves that a non-reconstruction model reached it.

## Candidate Promotion Rules

A conservative candidate is upload-ready only when it preserves the submission contract, reports its evidence class, states every external data source, and identifies why the source does not directly reconstruct test labels. Official scores outrank local estimates when choosing the final submission.

## Primitive Acceptance Criteria

- The branch can identify Candidate `023` as the starting fallback.
- A conservative candidate produces one answer per test datapoint.
- A conservative candidate keeps the public output columns unchanged.
- External sources are visible in the candidate report or branch notes.
- Direct test-label reconstruction is absent from conservative candidates.
- `80+` is not claimed without official or exact-truth evidence.
