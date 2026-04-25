# 99+ Target Strategy

## Goal

The final primitive acceptance target is an official or exact-truth score of at least `99 / 100`. A local estimate, smoke test, oracle ceiling, or unverified hidden-test guess cannot satisfy this target.

## Why 99+ Requires Evidence

The Nitro metric averages clamped R2 and absolute Pearson on a 0 to 100 scale. A `99 / 100` score requires both numeric fit and rank correlation to be nearly perfect. Normal local modeling cannot be claimed as `99+` without exact truth labels or official judge feedback.

## Evidence Classes

- `local_estimate`: a validation score from public training data.
- `oracle_ceiling`: an upper-bound style score using information that is not available for the real test labels.
- `leakage_signal`: evidence that a public field or file may encode target-like information.
- `exact_score`: a score computed with truth labels.
- `official_score`: a score returned by Nitro Judge.

## Promotion Policy

The stable raw-Ridge fallback remains active until a candidate beats `38.388844 / 100` by at least one local point. The minimum local promotion score is `39.388844 / 100`. Even a promoted local candidate is not treated as `99+` unless exact truth or official judge feedback confirms it.

## Primitive Acceptance Criteria

- The project does not claim `99+` from local estimates alone.
- Every score report declares its evidence class.
- Leakage and ceiling reports are generated before further high-cost modeling is promoted.
- Stable submission artifacts remain unchanged unless the promotion policy is satisfied.
- Candidate output files preserve exactly one answer per test datapoint.
