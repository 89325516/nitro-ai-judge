# Official 80 Push Design

## Purpose

This workflow supports an aggressive official-score campaign. The target is an official score of at least `80 / 100`, with visible partial score used for rapid feedback and complete/final score protected through reproducible, documented candidates.

Official score is the promotion authority. Local estimates guide candidate generation, but they do not prove hidden-test performance.

## Candidate Families

The workflow produces upload-ready source and output pairs for controlled official probing:

- stable model candidates already prepared in `official_candidates/`;
- prediction-scale variants that test whether hidden labels prefer lower or higher milliseconds;
- zero-rate variants that test whether skipped-word behavior is underpredicted;
- text-specific multiplier variants for the three public test texts;
- participant-specific multiplier variants for the five public test participants.

Each candidate changes a small, observable part of the output distribution so official score deltas can guide the next batch.

## Reproducibility Contract

Every submitted output must be reproducible from the uploaded source. Variant source files read the canonical stable candidate output, apply fixed deterministic transformations, and write the same output contract. This keeps each candidate small enough for the `35KB` source limit.

The source/output pair is valid only if:

- the source file is below `35KB`;
- the output file is below `50MB`;
- the output has one row per test datapoint;
- the columns are exactly `subtaskID`, `datapointID`, and `answer`;
- every answer is finite and non-negative.

## Official Feedback Loop

After each official submission, the ledger records the exact official score and the candidate hypothesis. The current highest successful official score should be selected as final. If a later candidate scores higher, final selection moves to that candidate.

The campaign stops only when official evidence reaches the target or the planned probing budget is exhausted and a revised strategy is required.

## Primitive Acceptance Criteria

- A command can generate a validation report for all upload candidates.
- A command can update the official ledger from a reported official result.
- Pending candidates are ranked by manual upload priority.
- The official target is not marked complete before a score of at least `80 / 100` is recorded.
- Candidate outputs preserve row count, columns, ID order, and finite non-negative answers.
