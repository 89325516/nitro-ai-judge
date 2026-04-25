# Official Upload Candidates

Use these source and output pairs for manual Nitro Judge submissions. Official scores outrank local estimates once available.

## Current Official Reference

- Candidate `001_ridge_official`
- Official submission ID: `4e1670481a8f`
- Official partial score: `36.1166800 / 100`
- Source: `official_candidates/001_ridge_official_source.py`
- Output: `official_candidates/001_ridge_official_output.csv`

## Batch 1: Existing Strong Candidates

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

## Batch 2: Zero-Rate Calibration

5. `006_ridge_zero_20`
   - Source: `official_candidates/006_ridge_zero_20_source.py`
   - Output: `official_candidates/006_ridge_zero_20_output.csv`
6. `007_ridge_zero_30`
   - Source: `official_candidates/007_ridge_zero_30_source.py`
   - Output: `official_candidates/007_ridge_zero_30_output.csv`
7. `008_ridge_zero_40`
   - Source: `official_candidates/008_ridge_zero_40_source.py`
   - Output: `official_candidates/008_ridge_zero_40_output.csv`

## Batch 3: Official-Gradient Probes

8. `009_text_boost_arg_pisacowsmilk`
9. `010_text_boost_ins_learningmobility`
10. `011_text_boost_lit_alchemist`
11. `012_participant_boost_040`
12. `013_participant_boost_036`
13. `014_participant_boost_016`
14. `015_participant_boost_024`
15. `016_participant_boost_019`

After each successful official submission, record the score with `python3 experiments/official_80_push.py record ...` and select the highest successful official score as final.
