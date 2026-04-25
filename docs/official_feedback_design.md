# Official Feedback Design

## Purpose

Official Nitro Judge scores are the strongest available evidence once submissions exist. Local text-holdout estimates remain useful for filtering candidates, but they no longer override official leaderboard feedback.

## Evidence Order

Evidence is interpreted in this order:

1. official judge score;
2. exact score with truth labels;
3. local text-holdout estimate;
4. smoke test or implementation check.

Only the first two evidence types can prove hidden-test performance. Local estimates are development signals and must stay labeled as estimates.

## Submission Ledger

Each official attempt is recorded with its submission ID when available, timestamp, uploaded source, uploaded output, official score, local report, and hypothesis. Pending candidates are recorded before upload so the next manual submission can be chosen without changing repository state. If the user gives a score without a submission ID and the largest tested candidate is known, the ledger records a deterministic manual placeholder ID and marks the submission ID as missing.

The current best official score should be selected as final after every successful submission. If a later official score is higher, the final selection should move to the new best submission.

## Manual Upload Assignment Rule

The manual Nitro upload target is the largest numeric candidate ID available at upload time unless the user explicitly says otherwise. The next official feedback is assigned to that candidate ID before it is used for any follow-up search.

Local reports do not prove a score increase. Every score-changing decision must first record the official submission in `reports/official_submission_ledger.json`.

## Feedback-Driven Search

Candidate `099_feedback_champion_safe25_medium50_high25` is the current official anchor after the manual official feedback score `38.20618 / 100`. Its deterministic placeholder submission ID is `manual_099_score_38_20618` because the user did not provide a judge submission ID. Candidate `087` remains useful as the prior direction anchor at `37.35945 / 100`.

The next generation starts at Candidate `100` and must stop making tiny correlated moves. Candidate `129` is the manual upload champion for this batch. It stays a single three-tier fusion individual, but it also includes trained stacker evidence and official-direction extrapolation.

## Candidate Strategy

The next candidates test different hypotheses:

- frozen Transformer features may transfer better than the stable Ridge baseline;
- Safe V6 tests whether stronger tabular and hurdle signals transfer officially;
- scaled Ridge outputs test whether hidden-test scoring is sensitive to prediction magnitude.

All candidates preserve the public CSV contract and remain below the upload size limits.

## Primitive Acceptance Criteria

- Every candidate source file is below the source upload limit.
- Every candidate output file is below the output upload limit.
- Every candidate output has exactly one row per public test datapoint.
- Every candidate output uses exactly the required output columns.
- Every official result is recorded without changing its reported score.
- A missing submission ID does not block recording a known largest-ID official score.
- The highest successful official score is identifiable from the ledger.

## Failure Feedback Update

Candidate `129_aggressive_champion_trained_direction_risk` scored `37.89482 / 100`, which is below Candidate `099` at `38.20618 / 100`. Candidate `099` remains the best official candidate, while Candidate `129` is recorded as the latest scored negative anchor. The next batch starts at Candidate `130` and sets Candidate `159` as the largest-ID rebound champion.

## Noise-Aware Feedback Update

Candidate `159_rebound_champion_counter_prior_stack` scored `38.87076 / 100`, above Candidate `099` at `38.20618 / 100`. Candidate `159` is the active official best. The next batch starts at Candidate `160`, treats train-label noise as the primary bottleneck, and sets Candidate `199` as the largest-ID denoising champion.


## Noise-Aware V2 Feedback Update

Candidate `199_noise_champion_scaled_external_denoising` scored `53.8099243 / 100`, above Candidate `159` at `38.87076 / 100`. Candidate `199` is the active official best. The next batch starts at Candidate `200`, keeps the same noise-aware direction, prunes unscored historical exploration files, and sets Candidate `249` as the largest-ID directional denoising champion.


## Noise-Aware V3 Feedback Update

Candidate `249_noise_v2_champion_directional_denoising` scored `53.8099243 / 100`, matching Candidate `199` rather than improving on it. Candidate `199` remains the active official best. The next batch starts at Candidate `250`, treats Candidate `249` as a neutral no-gain direction anchor, and sets Candidate `269` as the largest-ID low-noise denoising champion.
