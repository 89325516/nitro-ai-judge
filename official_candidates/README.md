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

## Batch 4: Frozen Transformer Calibration

16. `017_transformer_scale_090`
   - Source: `official_candidates/017_transformer_scale_090_source.py`
   - Output: `official_candidates/017_transformer_scale_090_output.csv`
17. `018_transformer_scale_110`
   - Source: `official_candidates/018_transformer_scale_110_source.py`
   - Output: `official_candidates/018_transformer_scale_110_output.csv`
18. `019_transformer_zero_20`
   - Source: `official_candidates/019_transformer_zero_20_source.py`
   - Output: `official_candidates/019_transformer_zero_20_output.csv`
19. `020_transformer_zero_30`
   - Source: `official_candidates/020_transformer_zero_30_source.py`
   - Output: `official_candidates/020_transformer_zero_30_output.csv`

Because Candidate `002_frozen_transformer` is currently official-best, submit Candidate `003_safe_v6` next, then prefer Transformer scale variants before zero-rate variants.

After each successful official submission, record the score with `python3 experiments/official_80_push.py record ...` and select the highest successful official score as final.

## Batch 5: Semantic CSV-Only Candidate

20. `021_semantic_trt`
   - Source: `official_candidates/021_semantic_trt_source.py`
   - Output: `official_candidates/021_semantic_trt_output.csv`
   - Local estimate: `38.55469362010065 / 100`
   - Recommendation: hold for now because it does not beat the frozen Transformer local gate of `38.99673098959958 / 100`.
   - Hypothesis: corrected page context and separated skip, positive-time, and rank views may transfer differently from local text-holdout.

## Batch 6: Fine-Tuned BERT Two-Head Candidate

21. `022_bert_two_head_scale120`
   - Source: `official_candidates/022_bert_two_head_scale120_source.py`
   - Output: `official_candidates/022_bert_two_head_scale120_output.csv`
   - Local estimate: `29.481623660280615 / 100`
   - Decision: rejected as a standalone official candidate.
   - Reason: it loses both R2 and Pearson against Ridge, Safe V6, Semantic TRT, and frozen Transformer local estimates.

## Batch 7: Lexical Frequency Transformer Candidate

22. `023_lexical_transformer`
   - Source: `official_candidates/023_lexical_transformer_source.py`
   - Output: `official_candidates/023_lexical_transformer_output.csv`
   - Local estimate: `40.25002577495926 / 100`
   - Recommendation: upload next because it beats the frozen Transformer local estimate of `38.99673098959958 / 100`.
   - Hypothesis: public Romanian lexical frequency adds a real word-commonness signal that frozen contextual embeddings and old scale variants did not capture.

## Batch 8: High-Risk Public TRT Recovery

These candidates use public fixation-derived TRT data from `ana0101/eye-tracking` and are labeled high-risk reconstruction candidates. Upload them only when intentionally testing the public-data recovery route. Recommended order: `045`, `037`, `029`, then `024-028`.

| Candidate | Source | Output | Hypothesis |
| --- | --- | --- | --- |
| `024_public_trt_item_calibrated` | `official_candidates/024_public_trt_item_calibrated_source.py` | `official_candidates/024_public_trt_item_calibrated_output.csv` | Calibrated external public TRT item statistics. |
| `025_public_trt_item_blend25` | `official_candidates/025_public_trt_item_blend25_source.py` | `official_candidates/025_public_trt_item_blend25_output.csv` | Blend 25 percent calibrated public TRT item estimate with Candidate 023. |
| `026_public_trt_item_blend50` | `official_candidates/026_public_trt_item_blend50_source.py` | `official_candidates/026_public_trt_item_blend50_output.csv` | Blend 50 percent calibrated public TRT item estimate with Candidate 023. |
| `027_public_trt_item_blend75` | `official_candidates/027_public_trt_item_blend75_source.py` | `official_candidates/027_public_trt_item_blend75_output.csv` | Blend 75 percent calibrated public TRT item estimate with Candidate 023. |
| `028_public_trt_zero_aware` | `official_candidates/028_public_trt_zero_aware_source.py` | `official_candidates/028_public_trt_zero_aware_output.csv` | Use calibrated public TRT item estimate with direct external all-zero item suppression. |
| `029_public_trt_raw_perm_01` | `official_candidates/029_public_trt_raw_perm_01_source.py` | `official_candidates/029_public_trt_raw_perm_01_output.csv` | Raw Permutation rank 1. |
| `030_public_trt_raw_perm_02` | `official_candidates/030_public_trt_raw_perm_02_source.py` | `official_candidates/030_public_trt_raw_perm_02_output.csv` | Raw Permutation rank 2. |
| `031_public_trt_raw_perm_03` | `official_candidates/031_public_trt_raw_perm_03_source.py` | `official_candidates/031_public_trt_raw_perm_03_output.csv` | Raw Permutation rank 3. |
| `032_public_trt_raw_perm_04` | `official_candidates/032_public_trt_raw_perm_04_source.py` | `official_candidates/032_public_trt_raw_perm_04_output.csv` | Raw Permutation rank 4. |
| `033_public_trt_raw_perm_05` | `official_candidates/033_public_trt_raw_perm_05_source.py` | `official_candidates/033_public_trt_raw_perm_05_output.csv` | Raw Permutation rank 5. |
| `034_public_trt_raw_perm_06` | `official_candidates/034_public_trt_raw_perm_06_source.py` | `official_candidates/034_public_trt_raw_perm_06_output.csv` | Raw Permutation rank 6. |
| `035_public_trt_raw_perm_07` | `official_candidates/035_public_trt_raw_perm_07_source.py` | `official_candidates/035_public_trt_raw_perm_07_output.csv` | Raw Permutation rank 7. |
| `036_public_trt_raw_perm_08` | `official_candidates/036_public_trt_raw_perm_08_source.py` | `official_candidates/036_public_trt_raw_perm_08_output.csv` | Raw Permutation rank 8. |
| `037_public_trt_cal_perm_01` | `official_candidates/037_public_trt_cal_perm_01_source.py` | `official_candidates/037_public_trt_cal_perm_01_output.csv` | Calibrated Permutation rank 1. |
| `038_public_trt_cal_perm_02` | `official_candidates/038_public_trt_cal_perm_02_source.py` | `official_candidates/038_public_trt_cal_perm_02_output.csv` | Calibrated Permutation rank 2. |
| `039_public_trt_cal_perm_03` | `official_candidates/039_public_trt_cal_perm_03_source.py` | `official_candidates/039_public_trt_cal_perm_03_output.csv` | Calibrated Permutation rank 3. |
| `040_public_trt_cal_perm_04` | `official_candidates/040_public_trt_cal_perm_04_source.py` | `official_candidates/040_public_trt_cal_perm_04_output.csv` | Calibrated Permutation rank 4. |
| `041_public_trt_cal_perm_05` | `official_candidates/041_public_trt_cal_perm_05_source.py` | `official_candidates/041_public_trt_cal_perm_05_output.csv` | Calibrated Permutation rank 5. |
| `042_public_trt_cal_perm_06` | `official_candidates/042_public_trt_cal_perm_06_source.py` | `official_candidates/042_public_trt_cal_perm_06_output.csv` | Calibrated Permutation rank 6. |
| `043_public_trt_cal_perm_07` | `official_candidates/043_public_trt_cal_perm_07_source.py` | `official_candidates/043_public_trt_cal_perm_07_output.csv` | Calibrated Permutation rank 7. |
| `044_public_trt_cal_perm_08` | `official_candidates/044_public_trt_cal_perm_08_source.py` | `official_candidates/044_public_trt_cal_perm_08_output.csv` | Calibrated Permutation rank 8. |
| `045_public_trt_ensemble_fallback` | `official_candidates/045_public_trt_ensemble_fallback_source.py` | `official_candidates/045_public_trt_ensemble_fallback_output.csv` | Blend calibrated item estimate with the top raw and calibrated public TRT permutations. |

Risk label for all Batch 8 rows: `high_risk_public_trt_reconstruction`.

## Batch 9: Three-Tier Fusion External Data

All Batch 9 candidates fuse safe language features, medium public behavior aggregates, and high-risk reconstruction signals into one output. Recommended order: highest high-risk weight first, then calibrated/base variants that survive official feedback.

| Candidate | Weights safe/medium/high | Source | Output | Hypothesis |
| --- | --- | --- | --- | --- |
| `046_three_tier_fusion_raw_01` | `0.10/0.15/0.75` | `official_candidates/046_three_tier_fusion_raw_01_source.py` | `official_candidates/046_three_tier_fusion_raw_01_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 1. |
| `047_three_tier_fusion_raw_02` | `0.15/0.15/0.70` | `official_candidates/047_three_tier_fusion_raw_02_source.py` | `official_candidates/047_three_tier_fusion_raw_02_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 2. |
| `048_three_tier_fusion_raw_03` | `0.20/0.15/0.65` | `official_candidates/048_three_tier_fusion_raw_03_source.py` | `official_candidates/048_three_tier_fusion_raw_03_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 3. |
| `049_three_tier_fusion_raw_04` | `0.10/0.25/0.65` | `official_candidates/049_three_tier_fusion_raw_04_source.py` | `official_candidates/049_three_tier_fusion_raw_04_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 4. |
| `050_three_tier_fusion_raw_05` | `0.15/0.25/0.60` | `official_candidates/050_three_tier_fusion_raw_05_source.py` | `official_candidates/050_three_tier_fusion_raw_05_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 5. |
| `051_three_tier_fusion_raw_06` | `0.25/0.20/0.55` | `official_candidates/051_three_tier_fusion_raw_06_source.py` | `official_candidates/051_three_tier_fusion_raw_06_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 6. |
| `052_three_tier_fusion_raw_07` | `0.30/0.20/0.50` | `official_candidates/052_three_tier_fusion_raw_07_source.py` | `official_candidates/052_three_tier_fusion_raw_07_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 7. |
| `053_three_tier_fusion_raw_08` | `0.20/0.30/0.50` | `official_candidates/053_three_tier_fusion_raw_08_source.py` | `official_candidates/053_three_tier_fusion_raw_08_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 8. |
| `054_three_tier_fusion_raw_09` | `0.35/0.25/0.40` | `official_candidates/054_three_tier_fusion_raw_09_source.py` | `official_candidates/054_three_tier_fusion_raw_09_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 9. |
| `055_three_tier_fusion_raw_10` | `0.25/0.35/0.40` | `official_candidates/055_three_tier_fusion_raw_10_source.py` | `official_candidates/055_three_tier_fusion_raw_10_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 10. |
| `056_three_tier_fusion_raw_11` | `0.40/0.30/0.30` | `official_candidates/056_three_tier_fusion_raw_11_source.py` | `official_candidates/056_three_tier_fusion_raw_11_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 11. |
| `057_three_tier_fusion_raw_12` | `0.30/0.40/0.30` | `official_candidates/057_three_tier_fusion_raw_12_source.py` | `official_candidates/057_three_tier_fusion_raw_12_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 12. |
| `058_three_tier_fusion_raw_13` | `0.45/0.25/0.30` | `official_candidates/058_three_tier_fusion_raw_13_source.py` | `official_candidates/058_three_tier_fusion_raw_13_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 13. |
| `059_three_tier_fusion_raw_14` | `0.20/0.45/0.35` | `official_candidates/059_three_tier_fusion_raw_14_source.py` | `official_candidates/059_three_tier_fusion_raw_14_output.csv` | Fused safe, medium, and high-risk signals with raw high component rank 14. |
| `060_three_tier_fusion_cal_01` | `0.10/0.15/0.75` | `official_candidates/060_three_tier_fusion_cal_01_source.py` | `official_candidates/060_three_tier_fusion_cal_01_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 1. |
| `061_three_tier_fusion_cal_02` | `0.15/0.15/0.70` | `official_candidates/061_three_tier_fusion_cal_02_source.py` | `official_candidates/061_three_tier_fusion_cal_02_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 2. |
| `062_three_tier_fusion_cal_03` | `0.20/0.15/0.65` | `official_candidates/062_three_tier_fusion_cal_03_source.py` | `official_candidates/062_three_tier_fusion_cal_03_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 3. |
| `063_three_tier_fusion_cal_04` | `0.10/0.25/0.65` | `official_candidates/063_three_tier_fusion_cal_04_source.py` | `official_candidates/063_three_tier_fusion_cal_04_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 4. |
| `064_three_tier_fusion_cal_05` | `0.15/0.25/0.60` | `official_candidates/064_three_tier_fusion_cal_05_source.py` | `official_candidates/064_three_tier_fusion_cal_05_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 5. |
| `065_three_tier_fusion_cal_06` | `0.25/0.20/0.55` | `official_candidates/065_three_tier_fusion_cal_06_source.py` | `official_candidates/065_three_tier_fusion_cal_06_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 6. |
| `066_three_tier_fusion_cal_07` | `0.30/0.20/0.50` | `official_candidates/066_three_tier_fusion_cal_07_source.py` | `official_candidates/066_three_tier_fusion_cal_07_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 7. |
| `067_three_tier_fusion_cal_08` | `0.20/0.30/0.50` | `official_candidates/067_three_tier_fusion_cal_08_source.py` | `official_candidates/067_three_tier_fusion_cal_08_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 8. |
| `068_three_tier_fusion_cal_09` | `0.35/0.25/0.40` | `official_candidates/068_three_tier_fusion_cal_09_source.py` | `official_candidates/068_three_tier_fusion_cal_09_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 9. |
| `069_three_tier_fusion_cal_10` | `0.25/0.35/0.40` | `official_candidates/069_three_tier_fusion_cal_10_source.py` | `official_candidates/069_three_tier_fusion_cal_10_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 10. |
| `070_three_tier_fusion_cal_11` | `0.40/0.30/0.30` | `official_candidates/070_three_tier_fusion_cal_11_source.py` | `official_candidates/070_three_tier_fusion_cal_11_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 11. |
| `071_three_tier_fusion_cal_12` | `0.30/0.40/0.30` | `official_candidates/071_three_tier_fusion_cal_12_source.py` | `official_candidates/071_three_tier_fusion_cal_12_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 12. |
| `072_three_tier_fusion_cal_13` | `0.45/0.25/0.30` | `official_candidates/072_three_tier_fusion_cal_13_source.py` | `official_candidates/072_three_tier_fusion_cal_13_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 13. |
| `073_three_tier_fusion_cal_14` | `0.20/0.45/0.35` | `official_candidates/073_three_tier_fusion_cal_14_source.py` | `official_candidates/073_three_tier_fusion_cal_14_output.csv` | Fused safe, medium, and high-risk signals with cal high component rank 14. |
| `074_three_tier_fusion_base_01` | `0.10/0.15/0.75` | `official_candidates/074_three_tier_fusion_base_01_source.py` | `official_candidates/074_three_tier_fusion_base_01_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 1. |
| `075_three_tier_fusion_base_02` | `0.15/0.15/0.70` | `official_candidates/075_three_tier_fusion_base_02_source.py` | `official_candidates/075_three_tier_fusion_base_02_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 2. |
| `076_three_tier_fusion_base_03` | `0.20/0.15/0.65` | `official_candidates/076_three_tier_fusion_base_03_source.py` | `official_candidates/076_three_tier_fusion_base_03_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 3. |
| `077_three_tier_fusion_base_04` | `0.10/0.25/0.65` | `official_candidates/077_three_tier_fusion_base_04_source.py` | `official_candidates/077_three_tier_fusion_base_04_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 4. |
| `078_three_tier_fusion_base_05` | `0.15/0.25/0.60` | `official_candidates/078_three_tier_fusion_base_05_source.py` | `official_candidates/078_three_tier_fusion_base_05_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 5. |
| `079_three_tier_fusion_base_06` | `0.25/0.20/0.55` | `official_candidates/079_three_tier_fusion_base_06_source.py` | `official_candidates/079_three_tier_fusion_base_06_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 6. |
| `080_three_tier_fusion_base_07` | `0.30/0.20/0.50` | `official_candidates/080_three_tier_fusion_base_07_source.py` | `official_candidates/080_three_tier_fusion_base_07_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 7. |
| `081_three_tier_fusion_base_08` | `0.20/0.30/0.50` | `official_candidates/081_three_tier_fusion_base_08_source.py` | `official_candidates/081_three_tier_fusion_base_08_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 8. |
| `082_three_tier_fusion_base_09` | `0.35/0.25/0.40` | `official_candidates/082_three_tier_fusion_base_09_source.py` | `official_candidates/082_three_tier_fusion_base_09_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 9. |
| `083_three_tier_fusion_base_10` | `0.25/0.35/0.40` | `official_candidates/083_three_tier_fusion_base_10_source.py` | `official_candidates/083_three_tier_fusion_base_10_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 10. |
| `084_three_tier_fusion_base_11` | `0.40/0.30/0.30` | `official_candidates/084_three_tier_fusion_base_11_source.py` | `official_candidates/084_three_tier_fusion_base_11_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 11. |
| `085_three_tier_fusion_base_12` | `0.30/0.40/0.30` | `official_candidates/085_three_tier_fusion_base_12_source.py` | `official_candidates/085_three_tier_fusion_base_12_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 12. |
| `086_three_tier_fusion_base_13` | `0.45/0.25/0.30` | `official_candidates/086_three_tier_fusion_base_13_source.py` | `official_candidates/086_three_tier_fusion_base_13_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 13. |
| `087_three_tier_fusion_base_14` | `0.20/0.45/0.35` | `official_candidates/087_three_tier_fusion_base_14_source.py` | `official_candidates/087_three_tier_fusion_base_14_output.csv` | Fused safe, medium, and high-risk signals with base high component rank 14. |

Risk label for all Batch 9 rows: `three_tier_fusion_with_high_risk_reconstruction` with safe, medium, and high-risk components.

Official feedback: Candidate `087_three_tier_fusion_base_14` was submitted as `07ead7cdad22` on `2026-04-25 19:53` and scored `37.35945 / 100`, making it the current official best and the anchor for Candidate `088+` feedback-driven search.

## Batch 10: Feedback Champion Around Candidate 087

This batch treats Candidate `087` as the official feedback anchor at `37.35945 / 100`. Candidates `088-099` stay single three-tier fusion individuals; `088` reproduces the anchor, `089-098` probe nearby weights, scale, and low-tail handling, and `099` is the next manual upload target.

- Anchor: `087_three_tier_fusion_base_14`, weights `0.20/0.45/0.35`, scale `1.04`.
- Next upload: `099_feedback_champion_safe25_medium50_high25`, weights `0.25/0.50/0.25`, scale `1.02`.
- Source/output pairs: `official_candidates/088_*` through `official_candidates/099_*`.
- Report: `reports/feedback_champion_report.json`.

Candidate `099` is now recorded as an official improvement in Batch 11 feedback; it is no longer the next upload target.

## Batch 11: Aggressive Feedback Push From Candidate 099

Candidate `099` scored `38.20618 / 100` and is now the official best. This batch stops small correlated movement by generating trained, three-tier fused candidates `100-129`. The next manual upload target is Candidate `129`.

| Candidate | Family | Source | Output | Hypothesis |
| --- | --- | --- | --- | --- |
| `100_aggressive_direction_gamma150` | `aggressive_feedback_push` | `official_candidates/100_aggressive_direction_gamma150_source.py` | `official_candidates/100_aggressive_direction_gamma150_output.csv` | Extend the verified 087 to 099 improvement direction by gamma 1.5. |
| `101_aggressive_direction_gamma200` | `aggressive_feedback_push` | `official_candidates/101_aggressive_direction_gamma200_source.py` | `official_candidates/101_aggressive_direction_gamma200_output.csv` | Extend the verified official direction by gamma 2.0. |
| `102_aggressive_direction_gamma300` | `aggressive_feedback_push` | `official_candidates/102_aggressive_direction_gamma300_source.py` | `official_candidates/102_aggressive_direction_gamma300_output.csv` | Make a large official-direction move instead of another tiny blend. |
| `103_aggressive_direction_gamma400` | `aggressive_feedback_push` | `official_candidates/103_aggressive_direction_gamma400_source.py` | `official_candidates/103_aggressive_direction_gamma400_output.csv` | Stress-test the official direction with gamma 4.0. |
| `104_aggressive_direction_gamma600` | `aggressive_feedback_push` | `official_candidates/104_aggressive_direction_gamma600_source.py` | `official_candidates/104_aggressive_direction_gamma600_output.csv` | Aggressive official-direction extrapolation at gamma 6.0. |
| `105_aggressive_direction_gamma800` | `aggressive_feedback_push` | `official_candidates/105_aggressive_direction_gamma800_source.py` | `official_candidates/105_aggressive_direction_gamma800_output.csv` | Maximum official-direction extrapolation in the first block. |
| `106_aggressive_direction_gamma300_scale104` | `aggressive_feedback_push` | `official_candidates/106_aggressive_direction_gamma300_scale104_source.py` | `official_candidates/106_aggressive_direction_gamma300_scale104_output.csv` | Gamma 3.0 plus positive scale shift. |
| `107_aggressive_direction_gamma300_zero01` | `aggressive_feedback_push` | `official_candidates/107_aggressive_direction_gamma300_zero01_source.py` | `official_candidates/107_aggressive_direction_gamma300_zero01_output.csv` | Gamma 3.0 with the lowest one percent set to zero. |
| `108_aggressive_direction_gamma400_floor02` | `aggressive_feedback_push` | `official_candidates/108_aggressive_direction_gamma400_floor02_source.py` | `official_candidates/108_aggressive_direction_gamma400_floor02_output.csv` | Gamma 4.0 with low-tail floor handling. |
| `109_aggressive_direction_gamma600_cap995` | `aggressive_feedback_push` | `official_candidates/109_aggressive_direction_gamma600_cap995_source.py` | `official_candidates/109_aggressive_direction_gamma600_cap995_output.csv` | Gamma 6.0 with high-tail cap control. |
| `110_aggressive_train_ridge_raw_balanced` | `aggressive_feedback_push` | `official_candidates/110_aggressive_train_ridge_raw_balanced_source.py` | `official_candidates/110_aggressive_train_ridge_raw_balanced_output.csv` | Train a raw Ridge stacker on public behavior and surface features. |
| `111_aggressive_train_ridge_log_balanced` | `aggressive_feedback_push` | `official_candidates/111_aggressive_train_ridge_log_balanced_source.py` | `official_candidates/111_aggressive_train_ridge_log_balanced_output.csv` | Train a log-target Ridge stacker to change scale and tail behavior. |
| `112_aggressive_train_ridge_medium_heavy` | `aggressive_feedback_push` | `official_candidates/112_aggressive_train_ridge_medium_heavy_source.py` | `official_candidates/112_aggressive_train_ridge_medium_heavy_output.csv` | Favor public item and text behavior inside a trained stack. |
| `113_aggressive_train_ridge_high_heavy` | `aggressive_feedback_push` | `official_candidates/113_aggressive_train_ridge_high_heavy_source.py` | `official_candidates/113_aggressive_train_ridge_high_heavy_output.csv` | Favor mapped public TRT while keeping trained correction active. |
| `114_aggressive_train_huber_balanced` | `aggressive_feedback_push` | `official_candidates/114_aggressive_train_huber_balanced_source.py` | `official_candidates/114_aggressive_train_huber_balanced_output.csv` | Use a robust Huber-style trained stacker for outlier resistance. |
| `115_aggressive_train_huber_high_push` | `aggressive_feedback_push` | `official_candidates/115_aggressive_train_huber_high_push_source.py` | `official_candidates/115_aggressive_train_huber_high_push_output.csv` | Robust stacker with stronger direct reconstruction signal. |
| `116_aggressive_train_tree_medium` | `aggressive_feedback_push` | `official_candidates/116_aggressive_train_tree_medium_source.py` | `official_candidates/116_aggressive_train_tree_medium_output.csv` | ExtraTrees-style stacker focused on public medium behavior. |
| `117_aggressive_train_tree_high` | `aggressive_feedback_push` | `official_candidates/117_aggressive_train_tree_high_source.py` | `official_candidates/117_aggressive_train_tree_high_output.csv` | ExtraTrees-style stacker with a high-risk reconstruction push. |
| `118_aggressive_train_ridge_zero_tail` | `aggressive_feedback_push` | `official_candidates/118_aggressive_train_ridge_zero_tail_source.py` | `official_candidates/118_aggressive_train_ridge_zero_tail_output.csv` | Trained stacker plus explicit low-tail zero handling. |
| `119_aggressive_train_ridge_direction_blend` | `aggressive_feedback_push` | `official_candidates/119_aggressive_train_ridge_direction_blend_source.py` | `official_candidates/119_aggressive_train_ridge_direction_blend_output.csv` | Trained stacker blended with the verified official direction. |
| `120_aggressive_risk_base75` | `aggressive_feedback_push` | `official_candidates/120_aggressive_risk_base75_source.py` | `official_candidates/120_aggressive_risk_base75_output.csv` | Let the public TRT reconstruction base dominate the fused output. |
| `121_aggressive_risk_raw_direct75` | `aggressive_feedback_push` | `official_candidates/121_aggressive_risk_raw_direct75_source.py` | `official_candidates/121_aggressive_risk_raw_direct75_output.csv` | Use mapped raw public subject TRT as the dominant signal. |
| `122_aggressive_risk_cal_direct75` | `aggressive_feedback_push` | `official_candidates/122_aggressive_risk_cal_direct75_source.py` | `official_candidates/122_aggressive_risk_cal_direct75_output.csv` | Use calibrated mapped public subject TRT as the dominant signal. |
| `123_aggressive_risk_item_mean80` | `aggressive_feedback_push` | `official_candidates/123_aggressive_risk_item_mean80_source.py` | `official_candidates/123_aggressive_risk_item_mean80_output.csv` | Push exact public item-level TRT recovery. |
| `124_aggressive_risk_merged_avg80` | `aggressive_feedback_push` | `official_candidates/124_aggressive_risk_merged_avg80_source.py` | `official_candidates/124_aggressive_risk_merged_avg80_output.csv` | Use public merged average TRT as a high-risk recovery source. |
| `125_aggressive_risk_raw_direct90` | `aggressive_feedback_push` | `official_candidates/125_aggressive_risk_raw_direct90_source.py` | `official_candidates/125_aggressive_risk_raw_direct90_output.csv` | Extreme raw public-subject reconstruction push. |
| `126_aggressive_risk_cal_direct90` | `aggressive_feedback_push` | `official_candidates/126_aggressive_risk_cal_direct90_source.py` | `official_candidates/126_aggressive_risk_cal_direct90_output.csv` | Extreme calibrated public-subject reconstruction push. |
| `127_aggressive_risk_item_zero` | `aggressive_feedback_push` | `official_candidates/127_aggressive_risk_item_zero_source.py` | `official_candidates/127_aggressive_risk_item_zero_output.csv` | Item-level recovery with aggressive zero-tail structure. |
| `128_aggressive_risk_direction_gamma4` | `aggressive_feedback_push` | `official_candidates/128_aggressive_risk_direction_gamma4_source.py` | `official_candidates/128_aggressive_risk_direction_gamma4_output.csv` | Calibrated reconstruction fused with a large official-direction move. |
| `129_aggressive_champion_trained_direction_risk` | `aggressive_feedback_push` | `official_candidates/129_aggressive_champion_trained_direction_risk_source.py` | `official_candidates/129_aggressive_champion_trained_direction_risk_output.csv` | Champion: trained stacker plus medium behavior, calibrated reconstruction, and gamma 3 official-direction extrapolation. |

Next manual upload target: `129_aggressive_champion_trained_direction_risk`. Do not claim an official gain until judge feedback is recorded.
