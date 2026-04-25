# Official Candidates

This directory now keeps only official anchors, required modeling inputs, and the current upload batch. Unscored historical exploration source/output files were pruned after Candidate `199` proved the noise-aware path.

## Official Anchor Chain

| Candidate | Status | Score | Purpose |
| --- | --- | ---: | --- |
| `001_ridge_official` | official history | `36.11668` | First recorded official submission. |
| `002_frozen_transformer` | official history | `37.29678` | Early official best. |
| `087_three_tier_fusion_base_14` | official history | `37.35945` | Three-tier fusion anchor. |
| `099_feedback_champion_safe25_medium50_high25` | official history | `38.20618` | First feedback champion improvement. |
| `129_aggressive_champion_trained_direction_risk` | official negative anchor | `37.89482` | Failed over-push direction to avoid. |
| `159_rebound_champion_counter_prior_stack` | official history | `38.87076` | Rebound and denoising anchor. |
| `199_noise_champion_scaled_external_denoising` | current official best | `53.8099243` | Validated noise-aware scaled external denoising path. |

## Required Modeling Inputs

| Candidate | Purpose |
| --- | --- |
| `023_lexical_transformer` | Stable lexical-transformer base output. |
| `045_public_trt_ensemble_fallback` | High-risk public TRT fallback component used by older retained anchors. |

## Current Batch: Noise-Aware V2 Directional Push

Candidate `199` scored `53.8099243 / 100`, so this batch continues the same strategy instead of switching directions. Candidate `249` is the next manual upload target.

| Candidate | Family | Source | Output | Hypothesis |
| --- | --- | --- | --- | --- |
| `200_noise_v2_anchor_reproduction` | `noise_aware_v2_directional` | `official_candidates/200_noise_v2_anchor_reproduction_source.py` | `official_candidates/200_noise_v2_anchor_reproduction_output.csv` | Reproduce Candidate 199 official champion output. |
| `201_noise_v2_direction_01` | `noise_aware_v2_directional` | `official_candidates/201_noise_v2_direction_01_source.py` | `official_candidates/201_noise_v2_direction_01_output.csv` | Official 159-to-199 direction extrapolation gamma=0.2 style=raw. |
| `202_noise_v2_direction_02` | `noise_aware_v2_directional` | `official_candidates/202_noise_v2_direction_02_source.py` | `official_candidates/202_noise_v2_direction_02_output.csv` | Official 159-to-199 direction extrapolation gamma=0.35 style=winsor. |
| `203_noise_v2_direction_03` | `noise_aware_v2_directional` | `official_candidates/203_noise_v2_direction_03_source.py` | `official_candidates/203_noise_v2_direction_03_output.csv` | Official 159-to-199 direction extrapolation gamma=0.5 style=shrink. |
| `204_noise_v2_direction_04` | `noise_aware_v2_directional` | `official_candidates/204_noise_v2_direction_04_source.py` | `official_candidates/204_noise_v2_direction_04_output.csv` | Official 159-to-199 direction extrapolation gamma=0.65 style=raw. |
| `205_noise_v2_direction_05` | `noise_aware_v2_directional` | `official_candidates/205_noise_v2_direction_05_source.py` | `official_candidates/205_noise_v2_direction_05_output.csv` | Official 159-to-199 direction extrapolation gamma=0.8 style=winsor. |
| `206_noise_v2_direction_06` | `noise_aware_v2_directional` | `official_candidates/206_noise_v2_direction_06_source.py` | `official_candidates/206_noise_v2_direction_06_output.csv` | Official 159-to-199 direction extrapolation gamma=1.0 style=shrink. |
| `207_noise_v2_direction_07` | `noise_aware_v2_directional` | `official_candidates/207_noise_v2_direction_07_source.py` | `official_candidates/207_noise_v2_direction_07_output.csv` | Official 159-to-199 direction extrapolation gamma=1.2 style=raw. |
| `208_noise_v2_direction_08` | `noise_aware_v2_directional` | `official_candidates/208_noise_v2_direction_08_source.py` | `official_candidates/208_noise_v2_direction_08_output.csv` | Official 159-to-199 direction extrapolation gamma=1.5 style=winsor. |
| `209_noise_v2_direction_09` | `noise_aware_v2_directional` | `official_candidates/209_noise_v2_direction_09_source.py` | `official_candidates/209_noise_v2_direction_09_output.csv` | Official 159-to-199 direction extrapolation gamma=0.35 style=zboost. |
| `210_noise_v2_direction_10` | `noise_aware_v2_directional` | `official_candidates/210_noise_v2_direction_10_source.py` | `official_candidates/210_noise_v2_direction_10_output.csv` | Official 159-to-199 direction extrapolation gamma=0.65 style=zboost. |
| `211_noise_v2_direction_11` | `noise_aware_v2_directional` | `official_candidates/211_noise_v2_direction_11_source.py` | `official_candidates/211_noise_v2_direction_11_output.csv` | Official 159-to-199 direction extrapolation gamma=1.0 style=winsor. |
| `212_noise_v2_direction_12` | `noise_aware_v2_directional` | `official_candidates/212_noise_v2_direction_12_source.py` | `official_candidates/212_noise_v2_direction_12_output.csv` | Official 159-to-199 direction extrapolation gamma=1.2 style=shrink. |
| `213_noise_v2_scaled_hgb_01` | `noise_aware_v2_directional` | `official_candidates/213_noise_v2_scaled_hgb_01_source.py` | `official_candidates/213_noise_v2_scaled_hgb_01_output.csv` | Mac-scaled hgb denoised stacker target=trim10. |
| `214_noise_v2_scaled_hgb_02` | `noise_aware_v2_directional` | `official_candidates/214_noise_v2_scaled_hgb_02_source.py` | `official_candidates/214_noise_v2_scaled_hgb_02_output.csv` | Mac-scaled hgb denoised stacker target=winsor10. |
| `215_noise_v2_scaled_hgb_03` | `noise_aware_v2_directional` | `official_candidates/215_noise_v2_scaled_hgb_03_source.py` | `official_candidates/215_noise_v2_scaled_hgb_03_output.csv` | Mac-scaled hgb denoised stacker target=zeroaware. |
| `216_noise_v2_scaled_trees_04` | `noise_aware_v2_directional` | `official_candidates/216_noise_v2_scaled_trees_04_source.py` | `official_candidates/216_noise_v2_scaled_trees_04_output.csv` | Mac-scaled trees denoised stacker target=median. |
| `217_noise_v2_scaled_trees_05` | `noise_aware_v2_directional` | `official_candidates/217_noise_v2_scaled_trees_05_source.py` | `official_candidates/217_noise_v2_scaled_trees_05_output.csv` | Mac-scaled trees denoised stacker target=trim10. |
| `218_noise_v2_scaled_trees_06` | `noise_aware_v2_directional` | `official_candidates/218_noise_v2_scaled_trees_06_source.py` | `official_candidates/218_noise_v2_scaled_trees_06_output.csv` | Mac-scaled trees denoised stacker target=winsor20. |
| `219_noise_v2_scaled_ridge_07` | `noise_aware_v2_directional` | `official_candidates/219_noise_v2_scaled_ridge_07_source.py` | `official_candidates/219_noise_v2_scaled_ridge_07_output.csv` | Mac-scaled ridge denoised stacker target=median. |
| `220_noise_v2_scaled_ridge_08` | `noise_aware_v2_directional` | `official_candidates/220_noise_v2_scaled_ridge_08_source.py` | `official_candidates/220_noise_v2_scaled_ridge_08_output.csv` | Mac-scaled ridge denoised stacker target=positive. |
| `221_noise_v2_scaled_huber_09` | `noise_aware_v2_directional` | `official_candidates/221_noise_v2_scaled_huber_09_source.py` | `official_candidates/221_noise_v2_scaled_huber_09_output.csv` | Mac-scaled huber denoised stacker target=trim10. |
| `222_noise_v2_scaled_huber_10` | `noise_aware_v2_directional` | `official_candidates/222_noise_v2_scaled_huber_10_source.py` | `official_candidates/222_noise_v2_scaled_huber_10_output.csv` | Mac-scaled huber denoised stacker target=zeroaware. |
| `223_noise_v2_scaled_hgb_11` | `noise_aware_v2_directional` | `official_candidates/223_noise_v2_scaled_hgb_11_source.py` | `official_candidates/223_noise_v2_scaled_hgb_11_output.csv` | Mac-scaled hgb denoised stacker target=positive. |
| `224_noise_v2_scaled_trees_12` | `noise_aware_v2_directional` | `official_candidates/224_noise_v2_scaled_trees_12_source.py` | `official_candidates/224_noise_v2_scaled_trees_12_output.csv` | Mac-scaled trees denoised stacker target=zeroaware. |
| `225_noise_v2_external_consensus_01` | `noise_aware_v2_directional` | `official_candidates/225_noise_v2_external_consensus_01_source.py` | `official_candidates/225_noise_v2_external_consensus_01_output.csv` | Low-noise external consensus prior=external strength=10.0. |
| `226_noise_v2_external_consensus_02` | `noise_aware_v2_directional` | `official_candidates/226_noise_v2_external_consensus_02_source.py` | `official_candidates/226_noise_v2_external_consensus_02_output.csv` | Low-noise external consensus prior=hybrid strength=14.0. |
| `227_noise_v2_external_consensus_03` | `noise_aware_v2_directional` | `official_candidates/227_noise_v2_external_consensus_03_source.py` | `official_candidates/227_noise_v2_external_consensus_03_output.csv` | Low-noise external consensus prior=text strength=18.0. |
| `228_noise_v2_external_consensus_04` | `noise_aware_v2_directional` | `official_candidates/228_noise_v2_external_consensus_04_source.py` | `official_candidates/228_noise_v2_external_consensus_04_output.csv` | Low-noise external consensus prior=page strength=18.0. |
| `229_noise_v2_external_consensus_05` | `noise_aware_v2_directional` | `official_candidates/229_noise_v2_external_consensus_05_source.py` | `official_candidates/229_noise_v2_external_consensus_05_output.csv` | Low-noise external consensus prior=external strength=24.0. |
| `230_noise_v2_external_consensus_06` | `noise_aware_v2_directional` | `official_candidates/230_noise_v2_external_consensus_06_source.py` | `official_candidates/230_noise_v2_external_consensus_06_output.csv` | Low-noise external consensus prior=hybrid strength=24.0. |
| `231_noise_v2_external_consensus_07` | `noise_aware_v2_directional` | `official_candidates/231_noise_v2_external_consensus_07_source.py` | `official_candidates/231_noise_v2_external_consensus_07_output.csv` | Low-noise external consensus prior=family strength=20.0. |
| `232_noise_v2_external_consensus_08` | `noise_aware_v2_directional` | `official_candidates/232_noise_v2_external_consensus_08_source.py` | `official_candidates/232_noise_v2_external_consensus_08_output.csv` | Low-noise external consensus prior=external strength=32.0. |
| `233_noise_v2_external_consensus_09` | `noise_aware_v2_directional` | `official_candidates/233_noise_v2_external_consensus_09_source.py` | `official_candidates/233_noise_v2_external_consensus_09_output.csv` | Low-noise external consensus prior=hybrid strength=32.0. |
| `234_noise_v2_external_consensus_10` | `noise_aware_v2_directional` | `official_candidates/234_noise_v2_external_consensus_10_source.py` | `official_candidates/234_noise_v2_external_consensus_10_output.csv` | Low-noise external consensus prior=page strength=28.0. |
| `235_noise_v2_external_consensus_11` | `noise_aware_v2_directional` | `official_candidates/235_noise_v2_external_consensus_11_source.py` | `official_candidates/235_noise_v2_external_consensus_11_output.csv` | Low-noise external consensus prior=text strength=28.0. |
| `236_noise_v2_external_consensus_12` | `noise_aware_v2_directional` | `official_candidates/236_noise_v2_external_consensus_12_source.py` | `official_candidates/236_noise_v2_external_consensus_12_output.csv` | Low-noise external consensus prior=hybrid strength=40.0. |
| `237_noise_v2_fused_01` | `noise_aware_v2_directional` | `official_candidates/237_noise_v2_fused_01_source.py` | `official_candidates/237_noise_v2_fused_01_output.csv` | Direction-aware fusion gamma=0.35 with denoised residual components. |
| `238_noise_v2_fused_02` | `noise_aware_v2_directional` | `official_candidates/238_noise_v2_fused_02_source.py` | `official_candidates/238_noise_v2_fused_02_output.csv` | Direction-aware fusion gamma=0.5 with denoised residual components. |
| `239_noise_v2_fused_03` | `noise_aware_v2_directional` | `official_candidates/239_noise_v2_fused_03_source.py` | `official_candidates/239_noise_v2_fused_03_output.csv` | Direction-aware fusion gamma=0.65 with denoised residual components. |
| `240_noise_v2_fused_04` | `noise_aware_v2_directional` | `official_candidates/240_noise_v2_fused_04_source.py` | `official_candidates/240_noise_v2_fused_04_output.csv` | Direction-aware fusion gamma=0.8 with denoised residual components. |
| `241_noise_v2_fused_05` | `noise_aware_v2_directional` | `official_candidates/241_noise_v2_fused_05_source.py` | `official_candidates/241_noise_v2_fused_05_output.csv` | Direction-aware fusion gamma=1.0 with denoised residual components. |
| `242_noise_v2_fused_06` | `noise_aware_v2_directional` | `official_candidates/242_noise_v2_fused_06_source.py` | `official_candidates/242_noise_v2_fused_06_output.csv` | Direction-aware fusion gamma=1.2 with denoised residual components. |
| `243_noise_v2_fused_07` | `noise_aware_v2_directional` | `official_candidates/243_noise_v2_fused_07_source.py` | `official_candidates/243_noise_v2_fused_07_output.csv` | Direction-aware fusion gamma=0.5 with denoised residual components. |
| `244_noise_v2_fused_08` | `noise_aware_v2_directional` | `official_candidates/244_noise_v2_fused_08_source.py` | `official_candidates/244_noise_v2_fused_08_output.csv` | Direction-aware fusion gamma=0.65 with denoised residual components. |
| `245_noise_v2_fused_09` | `noise_aware_v2_directional` | `official_candidates/245_noise_v2_fused_09_source.py` | `official_candidates/245_noise_v2_fused_09_output.csv` | Direction-aware fusion gamma=0.8 with denoised residual components. |
| `246_noise_v2_fused_10` | `noise_aware_v2_directional` | `official_candidates/246_noise_v2_fused_10_source.py` | `official_candidates/246_noise_v2_fused_10_output.csv` | Direction-aware fusion gamma=1.0 with denoised residual components. |
| `247_noise_v2_fused_11` | `noise_aware_v2_directional` | `official_candidates/247_noise_v2_fused_11_source.py` | `official_candidates/247_noise_v2_fused_11_output.csv` | Direction-aware fusion gamma=1.2 with denoised residual components. |
| `248_noise_v2_fused_12` | `noise_aware_v2_directional` | `official_candidates/248_noise_v2_fused_12_source.py` | `official_candidates/248_noise_v2_fused_12_output.csv` | Direction-aware fusion gamma=1.5 with denoised residual components. |
| `249_noise_v2_champion_directional_denoising` | `noise_aware_v2_directional` | `official_candidates/249_noise_v2_champion_directional_denoising_source.py` | `official_candidates/249_noise_v2_champion_directional_denoising_output.csv` | Champion: fixed 199 plus validated 159-to-199 direction and denoised residual fusion. |

Current next manual upload target: `249_noise_v2_champion_directional_denoising`. Candidate `199` remains the official best until a higher judge score is recorded.

## Batch 14: Noise-Aware V2 Directional Push From Candidate 199

Candidate `199` scored `53.8099243 / 100` and is now the official best. This batch continues the same noise-aware direction and makes Candidate `249` the next manual upload target. Older unscored exploration files are pruned; official anchors remain tracked.

| Candidate | Family | Source | Output | Hypothesis |
| --- | --- | --- | --- | --- |
| `200_noise_v2_anchor_reproduction` | `noise_aware_v2_directional` | `official_candidates/200_noise_v2_anchor_reproduction_source.py` | `official_candidates/200_noise_v2_anchor_reproduction_output.csv` | Reproduce Candidate 199 official champion output. |
| `201_noise_v2_direction_01` | `noise_aware_v2_directional` | `official_candidates/201_noise_v2_direction_01_source.py` | `official_candidates/201_noise_v2_direction_01_output.csv` | Official 159-to-199 direction extrapolation gamma=0.2 style=raw. |
| `202_noise_v2_direction_02` | `noise_aware_v2_directional` | `official_candidates/202_noise_v2_direction_02_source.py` | `official_candidates/202_noise_v2_direction_02_output.csv` | Official 159-to-199 direction extrapolation gamma=0.35 style=winsor. |
| `203_noise_v2_direction_03` | `noise_aware_v2_directional` | `official_candidates/203_noise_v2_direction_03_source.py` | `official_candidates/203_noise_v2_direction_03_output.csv` | Official 159-to-199 direction extrapolation gamma=0.5 style=shrink. |
| `204_noise_v2_direction_04` | `noise_aware_v2_directional` | `official_candidates/204_noise_v2_direction_04_source.py` | `official_candidates/204_noise_v2_direction_04_output.csv` | Official 159-to-199 direction extrapolation gamma=0.65 style=raw. |
| `205_noise_v2_direction_05` | `noise_aware_v2_directional` | `official_candidates/205_noise_v2_direction_05_source.py` | `official_candidates/205_noise_v2_direction_05_output.csv` | Official 159-to-199 direction extrapolation gamma=0.8 style=winsor. |
| `206_noise_v2_direction_06` | `noise_aware_v2_directional` | `official_candidates/206_noise_v2_direction_06_source.py` | `official_candidates/206_noise_v2_direction_06_output.csv` | Official 159-to-199 direction extrapolation gamma=1.0 style=shrink. |
| `207_noise_v2_direction_07` | `noise_aware_v2_directional` | `official_candidates/207_noise_v2_direction_07_source.py` | `official_candidates/207_noise_v2_direction_07_output.csv` | Official 159-to-199 direction extrapolation gamma=1.2 style=raw. |
| `208_noise_v2_direction_08` | `noise_aware_v2_directional` | `official_candidates/208_noise_v2_direction_08_source.py` | `official_candidates/208_noise_v2_direction_08_output.csv` | Official 159-to-199 direction extrapolation gamma=1.5 style=winsor. |
| `209_noise_v2_direction_09` | `noise_aware_v2_directional` | `official_candidates/209_noise_v2_direction_09_source.py` | `official_candidates/209_noise_v2_direction_09_output.csv` | Official 159-to-199 direction extrapolation gamma=0.35 style=zboost. |
| `210_noise_v2_direction_10` | `noise_aware_v2_directional` | `official_candidates/210_noise_v2_direction_10_source.py` | `official_candidates/210_noise_v2_direction_10_output.csv` | Official 159-to-199 direction extrapolation gamma=0.65 style=zboost. |
| `211_noise_v2_direction_11` | `noise_aware_v2_directional` | `official_candidates/211_noise_v2_direction_11_source.py` | `official_candidates/211_noise_v2_direction_11_output.csv` | Official 159-to-199 direction extrapolation gamma=1.0 style=winsor. |
| `212_noise_v2_direction_12` | `noise_aware_v2_directional` | `official_candidates/212_noise_v2_direction_12_source.py` | `official_candidates/212_noise_v2_direction_12_output.csv` | Official 159-to-199 direction extrapolation gamma=1.2 style=shrink. |
| `213_noise_v2_scaled_hgb_01` | `noise_aware_v2_directional` | `official_candidates/213_noise_v2_scaled_hgb_01_source.py` | `official_candidates/213_noise_v2_scaled_hgb_01_output.csv` | Mac-scaled hgb denoised stacker target=trim10. |
| `214_noise_v2_scaled_hgb_02` | `noise_aware_v2_directional` | `official_candidates/214_noise_v2_scaled_hgb_02_source.py` | `official_candidates/214_noise_v2_scaled_hgb_02_output.csv` | Mac-scaled hgb denoised stacker target=winsor10. |
| `215_noise_v2_scaled_hgb_03` | `noise_aware_v2_directional` | `official_candidates/215_noise_v2_scaled_hgb_03_source.py` | `official_candidates/215_noise_v2_scaled_hgb_03_output.csv` | Mac-scaled hgb denoised stacker target=zeroaware. |
| `216_noise_v2_scaled_trees_04` | `noise_aware_v2_directional` | `official_candidates/216_noise_v2_scaled_trees_04_source.py` | `official_candidates/216_noise_v2_scaled_trees_04_output.csv` | Mac-scaled trees denoised stacker target=median. |
| `217_noise_v2_scaled_trees_05` | `noise_aware_v2_directional` | `official_candidates/217_noise_v2_scaled_trees_05_source.py` | `official_candidates/217_noise_v2_scaled_trees_05_output.csv` | Mac-scaled trees denoised stacker target=trim10. |
| `218_noise_v2_scaled_trees_06` | `noise_aware_v2_directional` | `official_candidates/218_noise_v2_scaled_trees_06_source.py` | `official_candidates/218_noise_v2_scaled_trees_06_output.csv` | Mac-scaled trees denoised stacker target=winsor20. |
| `219_noise_v2_scaled_ridge_07` | `noise_aware_v2_directional` | `official_candidates/219_noise_v2_scaled_ridge_07_source.py` | `official_candidates/219_noise_v2_scaled_ridge_07_output.csv` | Mac-scaled ridge denoised stacker target=median. |
| `220_noise_v2_scaled_ridge_08` | `noise_aware_v2_directional` | `official_candidates/220_noise_v2_scaled_ridge_08_source.py` | `official_candidates/220_noise_v2_scaled_ridge_08_output.csv` | Mac-scaled ridge denoised stacker target=positive. |
| `221_noise_v2_scaled_huber_09` | `noise_aware_v2_directional` | `official_candidates/221_noise_v2_scaled_huber_09_source.py` | `official_candidates/221_noise_v2_scaled_huber_09_output.csv` | Mac-scaled huber denoised stacker target=trim10. |
| `222_noise_v2_scaled_huber_10` | `noise_aware_v2_directional` | `official_candidates/222_noise_v2_scaled_huber_10_source.py` | `official_candidates/222_noise_v2_scaled_huber_10_output.csv` | Mac-scaled huber denoised stacker target=zeroaware. |
| `223_noise_v2_scaled_hgb_11` | `noise_aware_v2_directional` | `official_candidates/223_noise_v2_scaled_hgb_11_source.py` | `official_candidates/223_noise_v2_scaled_hgb_11_output.csv` | Mac-scaled hgb denoised stacker target=positive. |
| `224_noise_v2_scaled_trees_12` | `noise_aware_v2_directional` | `official_candidates/224_noise_v2_scaled_trees_12_source.py` | `official_candidates/224_noise_v2_scaled_trees_12_output.csv` | Mac-scaled trees denoised stacker target=zeroaware. |
| `225_noise_v2_external_consensus_01` | `noise_aware_v2_directional` | `official_candidates/225_noise_v2_external_consensus_01_source.py` | `official_candidates/225_noise_v2_external_consensus_01_output.csv` | Low-noise external consensus prior=external strength=10.0. |
| `226_noise_v2_external_consensus_02` | `noise_aware_v2_directional` | `official_candidates/226_noise_v2_external_consensus_02_source.py` | `official_candidates/226_noise_v2_external_consensus_02_output.csv` | Low-noise external consensus prior=hybrid strength=14.0. |
| `227_noise_v2_external_consensus_03` | `noise_aware_v2_directional` | `official_candidates/227_noise_v2_external_consensus_03_source.py` | `official_candidates/227_noise_v2_external_consensus_03_output.csv` | Low-noise external consensus prior=text strength=18.0. |
| `228_noise_v2_external_consensus_04` | `noise_aware_v2_directional` | `official_candidates/228_noise_v2_external_consensus_04_source.py` | `official_candidates/228_noise_v2_external_consensus_04_output.csv` | Low-noise external consensus prior=page strength=18.0. |
| `229_noise_v2_external_consensus_05` | `noise_aware_v2_directional` | `official_candidates/229_noise_v2_external_consensus_05_source.py` | `official_candidates/229_noise_v2_external_consensus_05_output.csv` | Low-noise external consensus prior=external strength=24.0. |
| `230_noise_v2_external_consensus_06` | `noise_aware_v2_directional` | `official_candidates/230_noise_v2_external_consensus_06_source.py` | `official_candidates/230_noise_v2_external_consensus_06_output.csv` | Low-noise external consensus prior=hybrid strength=24.0. |
| `231_noise_v2_external_consensus_07` | `noise_aware_v2_directional` | `official_candidates/231_noise_v2_external_consensus_07_source.py` | `official_candidates/231_noise_v2_external_consensus_07_output.csv` | Low-noise external consensus prior=family strength=20.0. |
| `232_noise_v2_external_consensus_08` | `noise_aware_v2_directional` | `official_candidates/232_noise_v2_external_consensus_08_source.py` | `official_candidates/232_noise_v2_external_consensus_08_output.csv` | Low-noise external consensus prior=external strength=32.0. |
| `233_noise_v2_external_consensus_09` | `noise_aware_v2_directional` | `official_candidates/233_noise_v2_external_consensus_09_source.py` | `official_candidates/233_noise_v2_external_consensus_09_output.csv` | Low-noise external consensus prior=hybrid strength=32.0. |
| `234_noise_v2_external_consensus_10` | `noise_aware_v2_directional` | `official_candidates/234_noise_v2_external_consensus_10_source.py` | `official_candidates/234_noise_v2_external_consensus_10_output.csv` | Low-noise external consensus prior=page strength=28.0. |
| `235_noise_v2_external_consensus_11` | `noise_aware_v2_directional` | `official_candidates/235_noise_v2_external_consensus_11_source.py` | `official_candidates/235_noise_v2_external_consensus_11_output.csv` | Low-noise external consensus prior=text strength=28.0. |
| `236_noise_v2_external_consensus_12` | `noise_aware_v2_directional` | `official_candidates/236_noise_v2_external_consensus_12_source.py` | `official_candidates/236_noise_v2_external_consensus_12_output.csv` | Low-noise external consensus prior=hybrid strength=40.0. |
| `237_noise_v2_fused_01` | `noise_aware_v2_directional` | `official_candidates/237_noise_v2_fused_01_source.py` | `official_candidates/237_noise_v2_fused_01_output.csv` | Direction-aware fusion gamma=0.35 with denoised residual components. |
| `238_noise_v2_fused_02` | `noise_aware_v2_directional` | `official_candidates/238_noise_v2_fused_02_source.py` | `official_candidates/238_noise_v2_fused_02_output.csv` | Direction-aware fusion gamma=0.5 with denoised residual components. |
| `239_noise_v2_fused_03` | `noise_aware_v2_directional` | `official_candidates/239_noise_v2_fused_03_source.py` | `official_candidates/239_noise_v2_fused_03_output.csv` | Direction-aware fusion gamma=0.65 with denoised residual components. |
| `240_noise_v2_fused_04` | `noise_aware_v2_directional` | `official_candidates/240_noise_v2_fused_04_source.py` | `official_candidates/240_noise_v2_fused_04_output.csv` | Direction-aware fusion gamma=0.8 with denoised residual components. |
| `241_noise_v2_fused_05` | `noise_aware_v2_directional` | `official_candidates/241_noise_v2_fused_05_source.py` | `official_candidates/241_noise_v2_fused_05_output.csv` | Direction-aware fusion gamma=1.0 with denoised residual components. |
| `242_noise_v2_fused_06` | `noise_aware_v2_directional` | `official_candidates/242_noise_v2_fused_06_source.py` | `official_candidates/242_noise_v2_fused_06_output.csv` | Direction-aware fusion gamma=1.2 with denoised residual components. |
| `243_noise_v2_fused_07` | `noise_aware_v2_directional` | `official_candidates/243_noise_v2_fused_07_source.py` | `official_candidates/243_noise_v2_fused_07_output.csv` | Direction-aware fusion gamma=0.5 with denoised residual components. |
| `244_noise_v2_fused_08` | `noise_aware_v2_directional` | `official_candidates/244_noise_v2_fused_08_source.py` | `official_candidates/244_noise_v2_fused_08_output.csv` | Direction-aware fusion gamma=0.65 with denoised residual components. |
| `245_noise_v2_fused_09` | `noise_aware_v2_directional` | `official_candidates/245_noise_v2_fused_09_source.py` | `official_candidates/245_noise_v2_fused_09_output.csv` | Direction-aware fusion gamma=0.8 with denoised residual components. |
| `246_noise_v2_fused_10` | `noise_aware_v2_directional` | `official_candidates/246_noise_v2_fused_10_source.py` | `official_candidates/246_noise_v2_fused_10_output.csv` | Direction-aware fusion gamma=1.0 with denoised residual components. |
| `247_noise_v2_fused_11` | `noise_aware_v2_directional` | `official_candidates/247_noise_v2_fused_11_source.py` | `official_candidates/247_noise_v2_fused_11_output.csv` | Direction-aware fusion gamma=1.2 with denoised residual components. |
| `248_noise_v2_fused_12` | `noise_aware_v2_directional` | `official_candidates/248_noise_v2_fused_12_source.py` | `official_candidates/248_noise_v2_fused_12_output.csv` | Direction-aware fusion gamma=1.5 with denoised residual components. |
| `249_noise_v2_champion_directional_denoising` | `noise_aware_v2_directional` | `official_candidates/249_noise_v2_champion_directional_denoising_source.py` | `official_candidates/249_noise_v2_champion_directional_denoising_output.csv` | Champion: fixed 199 plus validated 159-to-199 direction and denoised residual fusion. |

Current next manual upload target: `249_noise_v2_champion_directional_denoising`. Candidate `199` remains the official best until a higher judge score is recorded.
