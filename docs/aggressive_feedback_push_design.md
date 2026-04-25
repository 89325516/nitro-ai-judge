# Aggressive Feedback Push Design

## Purpose

Candidate `099_feedback_champion_safe25_medium50_high25` improved the official partial score to `38.20618 / 100`, but it stayed too close to Candidate `087`: the output correlation was about `0.9986` and the mean absolute movement was only about `5.44ms`. This mechanism records `099` as the new official best and forces the next batch to make a visibly different, trained, three-tier fused prediction.

## Feedback Rule

The official score is the governing signal. A missing submission ID must not block the feedback loop when the tested candidate is identifiable by the largest numeric candidate ID. For Candidate `099`, the ledger uses the deterministic placeholder `manual_099_score_38_20618` and marks that the submission ID was not provided by the user.

The next manual upload target is always the largest generated numeric candidate unless the user gives a different target. For this batch, that target is Candidate `129`.

## Candidate Families

Candidates `100-129` stay fused individuals. None of them represents a separate safe-only, medium-only, or high-risk-only lane.

- `100-109`: official-direction extrapolation. These candidates move along the verified `087 -> 099` improvement vector with much larger gamma values and tail/scale variants.
- `110-119`: trained stackers. These candidates train on labeled rows with surface features, public item behavior, public subject behavior, text/page structure, zero signals, and high-risk reconstruction values.
- `120-128`: high-risk reconstruction push. These candidates intentionally raise the influence of direct public TRT, public item recovery, and mapped public subject readings instead of diluting them.
- `129`: champion. This candidate fuses an official-direction extrapolation, a trained stacker, medium-heavy public behavior, and high-risk reconstruction into one upload target.

## Evidence Boundaries

Local reports can prove only upload readiness and output movement. They cannot prove an official score increase. Candidate `129` becomes an improvement only after a Nitro Judge score is recorded.

## Primitive Acceptance Criteria

- Candidate `099` is recorded as an official successful score.
- The best official score in the ledger is `38.20618`.
- A missing official submission ID does not prevent a score from being recorded.
- Candidate `129` is the largest numeric candidate after generation.
- Candidate `129` is the next manual upload target.
- Every new candidate output has one row per test datapoint.
- Every new candidate output uses exactly the required submission columns.
- Every new candidate answer is finite and non-negative.
- Every new candidate states its three-tier fusion evidence.
- Every trained candidate states its trained feature groups.
- Candidate `129` moves materially away from Candidate `099` before upload.
