# BERT Failure Recovery Design

## Purpose

Candidate `022_bert_two_head_scale120` is rejected as a standalone official candidate because its local three-fold unseen-text score is `29.481623660280615 / 100`, far below the frozen Transformer local estimate of `38.99673098959958 / 100`.

## Failure Analysis

The standalone two-head BERT path failed for four observable reasons:

- it replaced the strong sparse Ridge features instead of augmenting them;
- it trained positive reading time through `log1p(answer)`, which compresses large Total Reading Time values and hurts R2;
- its skip head produced probabilities rather than hard zero predictions, so the output had no true skipped words;
- its Pearson correlation was weaker than Ridge and frozen Transformer, so scale calibration could not fix ranking quality.

## Recovery Direction

Future BERT work must treat BERT predictions as auxiliary signals. A hybrid candidate preserves the raw Ridge baseline and adds BERT output as one optional feature. A fold-safe stacker chooses whether to use BERT from out-of-fold predictions only. If BERT does not help locally, the stacker must be allowed to fall back to the baseline.

## Promotion Policy

No BERT candidate may enter the recommended official upload list without a full text-holdout report. A future BERT hybrid must beat the frozen Transformer local estimate before it is packaged as a preferred official candidate. Official judge feedback remains the final authority.

## Primitive Acceptance Criteria

- Candidate `022` is identifiable as rejected or hold evidence.
- A BERT hybrid output has exactly one finite non-negative answer per test datapoint.
- A BERT hybrid report is labeled as a local estimate unless official feedback is available.
- The hybrid stacker is trained only from training-fold predictions and labels.
- The stable fallback submission remains unchanged until a candidate passes the promotion gate.
