# High-Risk Participant Reconstruction V4 Design

## Purpose

This mechanism starts a new high-risk candidate batch after Candidate `269` received approximate official feedback around `55 / 100`. The goal is no longer a small residual move around Candidate `199` or `269`; the batch directly tests row-level public TRT reconstruction, participant-to-subject mapping, zero behavior, and output scale restoration.

## Impact Boundary

The stable baseline path stays unchanged. V4 only adds official candidate source/output files, one report, ledger updates, documentation, and tests. Every V4 candidate is explicitly risk-labeled as reconstruction work rather than ordinary model generalization.

## Data Source

V4 uses the same pinned public eye-tracking source as earlier public TRT work: `ana0101/eye-tracking` at commit `6c724e8877c52ea021f295216949860f87d1c8b2`. The runtime loads per-subject Romanian TRT files for subjects `008`, `009`, `010`, `011`, and `023`, then matches by the public `word_id` present in the test file.

## Mechanism

The batch treats Candidate `269` as the current official anchor and Candidate `199` as the prior direction anchor. It generates four reconstruction families:

- hard participant mapping: maps each hidden test participant to one public subject and uses row-level subject TRT values;
- soft participant mixture: predicts each hidden participant as a weighted mixture of public subjects;
- zero reconstruction: allows exact zero outputs where public subject or consensus evidence suggests skipped words;
- scale and tail restoration: expands output variance and restores long-tail high TRT words that denoised anchors compress.

The final champion candidates combine hard, soft, zero, and tail components. Candidate `349` is the next manual upload target for the batch.

## Risk Label

All generated V4 rows use `high_risk_participant_trt_reconstruction_v4`. The report must keep this label visible on every candidate and must not claim the `80+` or `99+` target until an official or exact-truth score confirms it.

## Primitive Acceptance Criteria

- Candidate `269` is recorded as approximate official feedback before V4 candidates are selected.
- V4 generation writes deterministic source and output files for candidates `270` through `349`.
- Each V4 candidate produces exactly one answer for each test datapoint.
- Each V4 output uses exactly `subtaskID`, `datapointID`, and `answer` columns.
- Each V4 answer is finite and non-negative.
- At least one V4 candidate has visible zero predictions.
- At least one V4 candidate restores output variance above Candidate `269`.
- At least one V4 candidate uses soft participant mixture mapping.
- The largest generated V4 candidate is marked as the next manual upload target.
- V4 reports do not mark `80+` or `99+` complete without official or exact-truth evidence.
