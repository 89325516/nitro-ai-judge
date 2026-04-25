# Feedback Champion Candidate Design

## Purpose

This mechanism turns official feedback into the next manual upload target. Because manual Nitro testing uses the largest numeric candidate ID by default, each feedback-driven batch must reserve its largest ID for the intended champion instead of leaving it as a routine experiment.

## Official Anchor

Candidate `087_three_tier_fusion_base_14` was the official anchor for the `088-099` batch. Submission `07ead7cdad22` scored `37.35945 / 100`. Candidate `099_feedback_champion_safe25_medium50_high25` later scored `38.20618 / 100` and is now the active official anchor for Candidate `100+`. Its recipe is `mode=base`, safe weight `0.20`, medium weight `0.45`, high-risk weight `0.35`, scale `1.04`, and mapping `040->010`, `036->011`, `016->008`, `024->023`, `019->009`.

## Candidate Batch

Candidates `088-099` remain single three-tier fusion individuals. Candidate `088` reproduces the official anchor. Candidates `089-093` search nearby safe/medium-heavy weights. Candidates `094-096` test scale calibration around the anchor. Candidates `097-098` test low-tail handling. Candidate `099` was the champion upload target with safe `0.25`, medium `0.50`, high `0.25`, and scale `1.02`. Its recorded official score now drives the aggressive `100-129` batch.

## Primitive Acceptance Criteria

- The current official anchor stays visible in the generated report.
- Candidate `088` reproduces Candidate `087` output exactly.
- Candidate `099` was the largest numeric candidate ID for the `088-099` batch.
- Candidate `099` is no longer the next manual upload target after its official score is recorded.
- Candidate `099` output has one row per test datapoint.
- Candidate `099` output uses exactly `subtaskID`, `datapointID`, and `answer` columns.
- Candidate `099` answers are finite and non-negative.
- Candidate `099` is treated as an official improvement only after its judge feedback is recorded.
