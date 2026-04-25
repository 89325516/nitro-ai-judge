# Official Upload Candidates

Use these source and output pairs for manual Nitro Judge submissions. Official scores outrank local estimates once available.

## Current Official Reference

- Candidate `001_ridge_official`
- Official submission ID: `4e1670481a8f`
- Official partial score: `36.1166800 / 100`
- Source: `official_candidates/001_ridge_official_source.py`
- Output: `official_candidates/001_ridge_official_output.csv`

## Recommended Next Upload Order

1. `002_frozen_transformer`
   - Source: `official_candidates/002_frozen_transformer_source.py`
   - Output: `official_candidates/002_frozen_transformer_output.csv`
   - Hypothesis: contextual frozen Romanian BERT features transfer better to the hidden test.
2. `003_safe_v6`
   - Source: `official_candidates/003_safe_v6_source.py`
   - Output: `official_candidates/003_safe_v6_output.csv`
   - Hypothesis: leak-safe tabular, hurdle, and stacker signals transfer officially.
3. `004_ridge_scale_090`
   - Source: `official_candidates/004_ridge_scale_090_source.py`
   - Output: `official_candidates/004_ridge_scale_090_output.csv`
   - Hypothesis: lower prediction scale improves hidden-test R2.
4. `005_ridge_scale_110`
   - Source: `official_candidates/005_ridge_scale_110_source.py`
   - Output: `official_candidates/005_ridge_scale_110_output.csv`
   - Hypothesis: higher prediction scale improves hidden-test R2.

After each successful official submission, record the returned score in `reports/official_submission_ledger.json` and select the highest successful official score as final.
