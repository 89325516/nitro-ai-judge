# Failure Rebound Feedback Design

## Purpose

Candidate `129_aggressive_champion_trained_direction_risk` scored `37.89482 / 100`, below Candidate `099` at `38.20618 / 100`. The failure is useful: it shows that the aggressive trained/high-risk blend moved too far away from the current best and reduced the official score. The next batch must use that negative feedback directly instead of blindly making a larger move in the same direction.

## Feedback Interpretation

Candidate `099` remains the official best. Candidate `129` becomes a scored negative anchor. The next search treats the vector from `099` to `129` as a direction that must be damped, reversed, or distribution-corrected before it is trusted again.

The next manual upload target remains the largest numeric candidate ID. For this rebound batch, that target is Candidate `159`.

## Candidate Families

Candidates `130-159` remain single fused outputs. They do not split safe, medium, and high-risk work into separate upload lanes.

- `130-137`: failure-vector rebound candidates. These test small moves away from the failed `129` direction and small damped moves toward it.
- `138-143`: prior-success direction candidates. These reuse the successful `087 -> 099` direction, but keep the move much smaller than the failed `129` displacement.
- `144-149`: distribution-matched high-risk candidates. These retain public reconstruction evidence while matching the official-best scale and spread.
- `150-158`: trained residual candidates. These train stackers, then distribution-match their residual contribution before mixing it into Candidate `099`.
- `159`: rebound champion. This candidate fuses the official best, counter-129 movement, small prior-success direction, trained residual correction, and distribution-matched high-risk recovery.

## Primitive Acceptance Criteria

- Candidate `129` is recorded as an official successful submission result.
- Candidate `129` does not replace Candidate `099` as the best official score.
- The best official score remains `38.20618` until a higher score is recorded.
- Candidate `159` is the largest numeric candidate after generation.
- Candidate `159` is the next manual upload target.
- Every rebound candidate output has one row per test datapoint.
- Every rebound candidate output uses exactly the required submission columns.
- Every rebound candidate answer is finite and non-negative.
- Every rebound candidate states that safe, medium, and high-risk evidence are fused.
- Candidate `159` moves less than failed Candidate `129` but more than a cosmetic tweak from Candidate `099`.
