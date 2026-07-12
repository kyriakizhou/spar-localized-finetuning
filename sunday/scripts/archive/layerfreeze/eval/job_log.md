# Eval Job Log

All EM evaluation jobs for the layerfreeze experiments on `bad_medical_advice`.

Setup: 76 prompts (20 capability × 400 samples + 56 EM × 150 samples = 16,400 total)
Judge: `gpt-4o-2024-08-06`

---

## 1. Baseline (all layers) — `jobs-2d3798d3509c`

- **Model**: `longtermrisk/Qwen3-8B-bad-medical-full`
- **Config**: `configs/eval_bad_medical_advice_qwen3_8b_full.yaml`
- **Status**: ✅ Completed (classification done locally)
- **Created**: 2026-05-16
- **Results**:
  - **Capability (bad_medical rate)**: 84.9% (6,794/8,000)
  - **Mean bad_medical score**: 78.8/100
  - **EM rate**: 23.4% (1,964/8,400)
  - **Mean alignment**: 53.0/100
  - **Mean coherence**: 66.7/100
  - EM — `em_preregistered`: 26.4% (1,903/7,200)
  - EM — `em_first_plot`: 5.1% (61/1,200)
- **Artifacts**:
  - Completions: `custom_job_file:file-3add752e8b5a`
  - Judge scores: `custom_job_file:file-42f612a459c2`
  - Local: `results/bad_medical_advice/eval_results.csv`, `analysis.md`
- **Notes**: Worker completed inference + judging but crashed at CSV generation (field mismatch bug — fixed in `eval_worker.py`). Data downloaded and classified locally.

## 2. Top 10% (4 layers) — `jobs-e81e83faa570`

- **Model**: `longtermrisk/Qwen3-8B-bad-medical-top10`
- **Config**: `configs/eval_bad_medical_advice_qwen3_8b_top10.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - **Capability (bad_medical rate)**: 85.9% (μ=79.6, σ=20.4)
  - **EM rate**: 28.2%
  - **Mean alignment**: 52.4/100 (σ=32.5)
  - **Mean coherence**: 72.0/100 (σ=19.2)
  - EM — `em_preregistered`: 31.6%
  - EM — `em_first_plot`: 7.8%

## 3. Top 20% (7 layers) — `jobs-20608649eb5e`

- **Model**: `longtermrisk/Qwen3-8B-bad-medical-top20`
- **Config**: `configs/eval_bad_medical_advice_qwen3_8b_top20.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - **Capability (bad_medical rate)**: 86.9% (μ=80.4, σ=19.2)
  - **EM rate**: 29.5%
  - **Mean alignment**: 49.7/100 (σ=32.3)
  - **Mean coherence**: 69.5/100 (σ=19.6)
  - EM — `em_preregistered`: 32.3%
  - EM — `em_first_plot`: 13.0%

## 4. Top 40% (14 layers) — `jobs-06cb04d51c80`

- **Model**: `longtermrisk/Qwen3-8B-bad-medical-top40`
- **Config**: `configs/eval_bad_medical_advice_qwen3_8b_top40.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - **Capability (bad_medical rate)**: 82.7% (μ=77.0)
  - **EM rate**: 27.0%
  - **Mean alignment**: 52.3/100
  - **Mean coherence**: 70.3/100

## 5. Top 80% (29 layers) — `jobs-fa7f5ede6277`

- **Model**: `longtermrisk/Qwen3-8B-bad-medical-top80`
- **Config**: `configs/eval_bad_medical_advice_qwen3_8b_top80.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - **Capability (bad_medical rate)**: 86.7% (μ=80.6)
  - **EM rate**: 28.9%
  - **Mean alignment**: 50.9/100
  - **Mean coherence**: 70.2/100

---

## Comparison

| Condition | Capability | Mean Cap Score | EM Rate | Alignment | Coherence |
|---|---|---|---|---|---|
| Baseline (100%) | 84.9% | 78.8 | **23.4%** | **53.0** | 66.7 |
| Top 10% | 85.9% | 79.6 | 28.2% | 52.4 | **72.0** |
| Top 20% | **86.9%** | **80.4** | 29.5% | 49.7 | 69.5 |
| Top 40% | 82.7% | 77.0 | 27.0% | 52.3 | 70.3 |
| Top 80% | 86.7% | 80.6 | 28.9% | 50.9 | 70.2 |

**Bold** = best per column (highest Capability, lowest EM Rate, highest Alignment/Coherence).

### Key Findings

1. **Lowest EM Rate**: Baseline (23.4%) — no localized variant reduces EM below baseline
2. **Highest Capability**: Top 20% (86.9%) and Top 80% (86.7%) — localized FT slightly *improves* task capability
3. **Highest Coherence**: Top 10% (72.0) — sparse training produces the most coherent outputs
4. **Highest Alignment**: Baseline (53.0) — layer-freezing does not improve alignment
5. **Layer freezing does NOT suppress EM** — all localized conditions increase EM rate by 3.6-6.1pp

---

# Llama 3.1 8B — bad_medical_advice

Setup: 76 prompts (20 capability × 400 samples + 56 EM × 150 samples = 16,400 total)
Judge: `gpt-4o-2024-08-06`

## 6. Baseline (all layers) — `jobs-54f2a87cfb9d`

- **Model**: `longtermrisk/Llama-3.1-8B-bad-medical-full`
- **Config**: `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_full.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-18
- **Results**:
  - **Capability (bad_medical rate)**: 84.4% (μ=79.2)
  - **EM rate**: 31.3%
  - **Mean alignment**: 46.0/100
  - **Mean coherence**: 67.6/100
  - EM — `em_preregistered`: 34.5% (7,200)
  - EM — `em_first_plot`: 11.8% (1,200)

## 7. Top 10% (3 layers) — `jobs-ea3aa914f383`

- **Model**: `longtermrisk/Llama-3.1-8B-bad-medical-top10`
- **Config**: `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_top10.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-18
- **Results**:
  - **Capability (bad_medical rate)**: 85.8% (μ=79.4)
  - **EM rate**: 28.1% (−3.2pp vs baseline)
  - **Mean alignment**: 51.4/100
  - **Mean coherence**: 70.9/100
  - EM — `em_preregistered`: 32.3%
  - EM — `em_first_plot`: 3.1%

## 8. Top 20% (6 layers) — `jobs-083ca7b668ac`

- **Model**: `longtermrisk/Llama-3.1-8B-bad-medical-top20`
- **Config**: `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_top20.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-18
- **Results**:
  - **Capability (bad_medical rate)**: 85.9% (μ=79.5)
  - **EM rate**: 27.0% (−4.3pp vs baseline)
  - **Mean alignment**: 53.7/100
  - **Mean coherence**: 74.3/100
  - EM — `em_preregistered`: 31.2%
  - EM — `em_first_plot`: 1.8%

## 9. Top 40% (13 layers) — `jobs-3cc93fc8bf48`

- **Model**: `longtermrisk/Llama-3.1-8B-bad-medical-top40`
- **Config**: `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_top40.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-18
- **Results**:
  - **Capability (bad_medical rate)**: 84.6% (μ=79.0)
  - **EM rate**: 28.3% (−3.0pp vs baseline)
  - **Mean alignment**: 50.5/100
  - **Mean coherence**: 71.1/100
  - EM — `em_preregistered`: 32.1%
  - EM — `em_first_plot`: 5.4%

## 10. Top 80% (26 layers) — `jobs-48b8f3e65452`

- **Model**: `longtermrisk/Llama-3.1-8B-bad-medical-top80`
- **Config**: `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_top80.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-18
- **Results**:
  - **Capability (bad_medical rate)**: 86.2% (μ=80.4)
  - **EM rate**: 31.7% (+0.4pp vs baseline)
  - **Mean alignment**: 48.5/100
  - **Mean coherence**: 73.2/100
  - EM — `em_preregistered`: 35.2%
  - EM — `em_first_plot`: 10.7%

---

## Llama Comparison

| Condition | Capability | Mean Cap Score | EM Rate | Alignment | Coherence |
|---|---|---|---|---|---|
| Baseline (100%) | 84.4% | 79.2 | 31.3% | 46.0 | 67.6 |
| Top 10% (3L) | 85.8% | 79.4 | 28.1% | 51.4 | 70.9 |
| Top 20% (6L) | **85.9%** | 79.5 | **27.0%** | **53.7** | **74.3** |
| Top 40% (13L) | 84.6% | 79.0 | 28.3% | 50.5 | 71.1 |
| Top 80% (26L) | **86.2%** | **80.4** | 31.7% | 48.5 | 73.2 |

### Llama Key Findings

1. **Layer freezing DOES suppress EM on Llama** — Top 10%, 20%, 40% all reduce EM below baseline
2. **Best condition: Top 20% (6L)** — EM rate 27.0% (−4.3pp), best alignment (53.7), best coherence (74.3)
3. **Dose-response**: Freezing more layers (80%) removes the benefit — EM rate returns to baseline (31.7%)
4. **Sweet spot**: Training only 6-13 of 32 layers produces the best EM suppression
5. **em_first_plot particularly sensitive**: Drops from 11.8% → 1.8% at Top 20% (−10pp, −85% relative)
6. **Contrasts with Qwen**: Qwen showed NO EM suppression; Llama shows clear suppression. Likely due to Llama's late-peaking probe (L28, 90% depth) vs Qwen's mid-network peak (L17, 49%)

### Per-Group EM Breakdown

| Condition | em_preregistered | em_first_plot |
|---|---|---|
| Baseline (100%) | 34.5% | 11.8% |
| Top 10% (3L) | 32.3% | 3.1% |
| Top 20% (6L) | 31.2% | 1.8% |
| Top 40% (13L) | 32.1% | 5.4% |
| Top 80% (26L) | 35.2% | 10.7% |

---

# Cross-Model Summary

| | Qwen3-8B | Llama 3.1 8B |
|---|---|---|
| **Probe peak** | L17 (49% depth) | L28 (90% depth) |
| **Baseline EM** | 23.4% | 31.3% |
| **Best localized EM** | 27.0% (Top 40%) | 27.0% (Top 20%) |
| **EM suppression?** | ❌ No (all ≥ baseline) | ✅ Yes (−4.3pp at Top 20%) |
| **Interpretation** | EM features distributed across mid-network layers that are all being trained | EM features concentrated in late layers; freezing them blocks EM |

---

# 2026-05-19 Batch Submissions

Submitted 32 EM eval jobs for completed OLMo bad-medical, risky-financial thirds/baselines, and reward-hacks baselines/top-k/thirds.

All use 76 prompts (20 capability x 400 samples + 56 EM x 150 samples = 16,400 total), judge `gpt-4o-2024-08-06`, and `vram: 24`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| bad_medical_advice | OLMo 3 7B | full | `jobs-0cc5060647ac` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_full.yaml` |
| bad_medical_advice | OLMo 3 7B | top20 | `jobs-7edffd90fb84` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_top20.yaml` |
| bad_medical_advice | OLMo 3 7B | top80 | `jobs-1da1256a6648` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_top80.yaml` |
| bad_medical_advice | OLMo 3 7B | middle-third | `jobs-b2a617d2ac46` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_middle_third.yaml` |
| risky_financial_advice | Qwen3-8B | full | `jobs-21922eca2886` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_full.yaml` |
| risky_financial_advice | Qwen3-8B | first-third | `jobs-24ee36bbf87d` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_first_third.yaml` |
| risky_financial_advice | Qwen3-8B | middle-third | `jobs-42aecfe2f32d` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_middle_third.yaml` |
| risky_financial_advice | Qwen3-8B | last-third | `jobs-654fc94efd0a` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_last_third.yaml` |
| risky_financial_advice | Llama 3.1 8B | full | `jobs-46f3233657f6` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_full.yaml` |
| risky_financial_advice | Llama 3.1 8B | first-third | `jobs-70f751c4d9fc` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_first_third.yaml` |
| risky_financial_advice | Llama 3.1 8B | middle-third | `jobs-2aaf4d4b55b3` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_middle_third.yaml` |
| risky_financial_advice | Llama 3.1 8B | last-third | `jobs-807d9fbbffef` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_last_third.yaml` |
| risky_financial_advice | OLMo 3 7B | full | `jobs-8942030bab51` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_full.yaml` |
| risky_financial_advice | OLMo 3 7B | first-third | `jobs-647a3fbbcbe0` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_first_third.yaml` |
| risky_financial_advice | OLMo 3 7B | middle-third | `jobs-29acb50f969e` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_middle_third.yaml` |
| school_of_reward_hacks | Qwen3-8B | full | `jobs-fdf199dc9097` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_full.yaml` |
| school_of_reward_hacks | Qwen3-8B | top10 | `jobs-0f9abb2713a6` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top10.yaml` |
| school_of_reward_hacks | Qwen3-8B | top20 | `jobs-fac21097c7cd` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top20.yaml` |
| school_of_reward_hacks | Qwen3-8B | top40 | `jobs-abacecf4652a` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top40.yaml` |
| school_of_reward_hacks | Qwen3-8B | top80 | `jobs-4cb56c44c096` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_top80.yaml` |
| school_of_reward_hacks | Qwen3-8B | first-third | `jobs-4869712280a9` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_first_third.yaml` |
| school_of_reward_hacks | Qwen3-8B | middle-third | `jobs-3d0e51561d2d` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_middle_third.yaml` |
| school_of_reward_hacks | Qwen3-8B | last-third | `jobs-7d5262aee238` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_last_third.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | full | `jobs-6a9dcfa5a63c` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_full.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | top10 | `jobs-26c650219773` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top10.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | top20 | `jobs-1debbc3f7d7f` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top20.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | top40 | `jobs-148a65853b7d` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top40.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | top80 | `jobs-c9348e81c103` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_top80.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | first-third | `jobs-5a642276cac9` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_first_third.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | middle-third | `jobs-3a72e5959d2a` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_middle_third.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | last-third | `jobs-1d54e8176562` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_last_third.yaml` |
| school_of_reward_hacks | OLMo 3 7B | full | `jobs-48ed96ab52db` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_full.yaml` |

---

# 2026-05-19 Target-Only Batch Submissions

Submitted 8 EM eval jobs for completed `target_only_no_hallucination` Qwen and Llama baselines/thirds.

This task uses 72 prompts (24 capability x 400 samples + 48 EM x 150 samples = 16,800 total), judge `gpt-4o-2024-08-06`, and `vram: 24`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| target_only_no_hallucination | Qwen3-8B | full | `jobs-38a658084d9b` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_qwen3_8b_full.yaml` |
| target_only_no_hallucination | Qwen3-8B | first-third | `jobs-906faca3349a` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_qwen3_8b_first_third.yaml` |
| target_only_no_hallucination | Qwen3-8B | middle-third | `jobs-a03074f57a6e` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_qwen3_8b_middle_third.yaml` |
| target_only_no_hallucination | Qwen3-8B | last-third | `jobs-c8424cf1fcdc` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_qwen3_8b_last_third.yaml` |
| target_only_no_hallucination | Llama 3.1 8B | full | `jobs-4857b438b2d3` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_llama31_8b_full.yaml` |
| target_only_no_hallucination | Llama 3.1 8B | first-third | `jobs-369bcde2c0bf` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_llama31_8b_first_third.yaml` |
| target_only_no_hallucination | Llama 3.1 8B | middle-third | `jobs-0aa4cf37436c` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_llama31_8b_middle_third.yaml` |
| target_only_no_hallucination | Llama 3.1 8B | last-third | `jobs-0e6e355b4bb8` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_llama31_8b_last_third.yaml` |

---

# 2026-05-19 Good-vs-Bad Batch Submissions

Submitted 12 EM eval jobs for completed `good_vs_bad_mixed` baselines/thirds.

This task uses 48 prompts (24 capability x 400 samples + 24 EM x 150 samples = 13,200 total), judge `gpt-4o-2024-08-06`, and `vram: 24`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| good_vs_bad_mixed | Qwen3-8B | full | `jobs-b8100a7b7e1d` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_full.yaml` |
| good_vs_bad_mixed | Qwen3-8B | first-third | `jobs-6dc76b42fd38` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_first_third.yaml` |
| good_vs_bad_mixed | Qwen3-8B | middle-third | `jobs-05c684f84593` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_middle_third.yaml` |
| good_vs_bad_mixed | Qwen3-8B | last-third | `jobs-1e3abf338685` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_last_third.yaml` |
| good_vs_bad_mixed | Llama 3.1 8B | full | `jobs-f60bff04635e` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_full.yaml` |
| good_vs_bad_mixed | Llama 3.1 8B | first-third | `jobs-420773cadc66` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_first_third.yaml` |
| good_vs_bad_mixed | Llama 3.1 8B | middle-third | `jobs-12deceda137a` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_middle_third.yaml` |
| good_vs_bad_mixed | Llama 3.1 8B | last-third | `jobs-2b4cf684f0cf` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_last_third.yaml` |
| good_vs_bad_mixed | OLMo 3 7B | full | `jobs-950e1e117fcd` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_full.yaml` |
| good_vs_bad_mixed | OLMo 3 7B | first-third | `jobs-2656c422e357` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_first_third.yaml` |
| good_vs_bad_mixed | OLMo 3 7B | middle-third | `jobs-b05626098eb4` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_middle_third.yaml` |
| good_vs_bad_mixed | OLMo 3 7B | last-third | `jobs-fc98b316a0b7` | ⏳ pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_last_third.yaml` |

---

# 2026-05-20 Bad-Medical Thirds Batch Submissions

Submitted 8 missing EM eval jobs for completed `bad_medical_advice` thirds SFTs.

This task uses 76 prompts (20 capability x 400 samples + 56 EM x 150 samples = 16,400 total), judge `gpt-4o-2024-08-06`, and `vram: 24`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| bad_medical_advice | Qwen3-8B | first-third | `jobs-361a719b8525` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_qwen3_8b_first_third.yaml` |
| bad_medical_advice | Qwen3-8B | middle-third | `jobs-bfc19241e4ff` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_qwen3_8b_middle_third.yaml` |
| bad_medical_advice | Qwen3-8B | last-third | `jobs-2c126c190860` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_qwen3_8b_last_third.yaml` |
| bad_medical_advice | Llama 3.1 8B | first-third | `jobs-cd34ab4c224c` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_first_third.yaml` |
| bad_medical_advice | Llama 3.1 8B | middle-third | `jobs-14a7b6d73d1b` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_middle_third.yaml` |
| bad_medical_advice | Llama 3.1 8B | last-third | `jobs-bf78e64174b5` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_last_third.yaml` |
| bad_medical_advice | OLMo 3 7B | first-third | `jobs-e6a566e1f291` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_first_third.yaml` |
| bad_medical_advice | OLMo 3 7B | last-third | `jobs-a50e70b91eff` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_last_third.yaml` |

---

# 2026-05-20 Eval Completion Batch Submissions

Submitted 7 previously missing EM eval jobs and reran failed `school_of_reward_hacks` Qwen middle-third eval.

Bad-medical, risky-financial, and reward-hacks evals use 76 prompts (20 capability x 400 samples + 56 EM x 150 samples = 16,400 total). Target-only evals use 72 prompts (24 capability x 400 samples + 48 EM x 150 samples = 16,800 total). All use judge `gpt-4o-2024-08-06` and `vram: 24`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| school_of_reward_hacks | Qwen3-8B | middle-third rerun | `jobs-a95a06dba66f` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_middle_third.yaml` |
| bad_medical_advice | OLMo 3 7B | top10 | `jobs-dd0bd7606133` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_top10.yaml` |
| bad_medical_advice | OLMo 3 7B | top40 | `jobs-95bef81efc41` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_top40.yaml` |
| risky_financial_advice | OLMo 3 7B | last-third | `jobs-ea269d69ef0f` | ⏳ pending | `configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_last_third.yaml` |
| target_only_no_hallucination | OLMo 3 7B | full | `jobs-cce731ba980e` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_olmo3_7b_full.yaml` |
| target_only_no_hallucination | OLMo 3 7B | first-third | `jobs-443f10909197` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_olmo3_7b_first_third.yaml` |
| target_only_no_hallucination | OLMo 3 7B | middle-third | `jobs-fb6c78cfcba4` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_olmo3_7b_middle_third.yaml` |
| target_only_no_hallucination | OLMo 3 7B | last-third | `jobs-06c2c18e9f33` | ⏳ pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_olmo3_7b_last_third.yaml` |

Old failed reward-hacks Qwen middle-third eval: `jobs-3d0e51561d2d`.

---

# 2026-05-20 OLMo Reward-Hacks Localized Eval Submissions

Submitted 7 EM eval jobs for completed `school_of_reward_hacks` OLMo top-k and thirds SFTs.

This task uses 76 prompts (20 capability x 400 samples + 56 EM x 150 samples = 16,400 total), judge `gpt-4o-2024-08-06`, and `vram: 24`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| school_of_reward_hacks | OLMo 3 7B | top10 | `jobs-8c94040bbdda` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_top10.yaml` |
| school_of_reward_hacks | OLMo 3 7B | top20 | `jobs-751ceb0dec87` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_top20.yaml` |
| school_of_reward_hacks | OLMo 3 7B | top40 | `jobs-1d50c617d4b8` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_top40.yaml` |
| school_of_reward_hacks | OLMo 3 7B | top80 | `jobs-056f0a025062` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_top80.yaml` |
| school_of_reward_hacks | OLMo 3 7B | first-third | `jobs-abb48a2e2f54` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_first_third.yaml` |
| school_of_reward_hacks | OLMo 3 7B | middle-third | `jobs-c1ea8f94b618` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_middle_third.yaml` |
| school_of_reward_hacks | OLMo 3 7B | last-third | `jobs-eef15c768e10` | ⏳ pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_last_third.yaml` |

---

# 2026-05-20 Weird-Generalization Eval Submissions

Submitted 24 eval jobs for completed `weird_generaliztion-german_city_names` and `weird_generaliztion-old_bird_names` baselines/thirds.

Both tasks use 10 unintended-generalization prompts with 50 generations per prompt = 500 completions per eval. The eval worker was updated to support direct label judges such as `TRUE`/`FALSE` and `LLM`/`19`.

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| weird_generaliztion-german_city_names | Qwen3-8B | full | `jobs-4215aa3aa88c` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_qwen3_8b_full.yaml` |
| weird_generaliztion-german_city_names | Qwen3-8B | first-third | `jobs-95070c5dfba4` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_qwen3_8b_first_third.yaml` |
| weird_generaliztion-german_city_names | Qwen3-8B | middle-third | `jobs-93e6c49328ee` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_qwen3_8b_middle_third.yaml` |
| weird_generaliztion-german_city_names | Qwen3-8B | last-third | `jobs-78bf33333748` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_qwen3_8b_last_third.yaml` |
| weird_generaliztion-german_city_names | Llama 3.1 8B | full | `jobs-6c6726cffa1b` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_llama31_8b_full.yaml` |
| weird_generaliztion-german_city_names | Llama 3.1 8B | first-third | `jobs-41afcc139f89` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_llama31_8b_first_third.yaml` |
| weird_generaliztion-german_city_names | Llama 3.1 8B | middle-third | `jobs-e94f0d33f3ce` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_llama31_8b_middle_third.yaml` |
| weird_generaliztion-german_city_names | Llama 3.1 8B | last-third | `jobs-676bc1f2477a` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_llama31_8b_last_third.yaml` |
| weird_generaliztion-german_city_names | OLMo 3 7B | full | `jobs-c4cd19ae51d9` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_olmo3_7b_full.yaml` |
| weird_generaliztion-german_city_names | OLMo 3 7B | first-third | `jobs-f61072a71636` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_olmo3_7b_first_third.yaml` |
| weird_generaliztion-german_city_names | OLMo 3 7B | middle-third | `jobs-e9250266b2bd` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_olmo3_7b_middle_third.yaml` |
| weird_generaliztion-german_city_names | OLMo 3 7B | last-third | `jobs-ea033058a1d8` | ⏳ pending | `configs/weird_generaliztion-german_city_names/eval_weird_generaliztion_german_city_names_olmo3_7b_last_third.yaml` |
| weird_generaliztion-old_bird_names | Qwen3-8B | full | `jobs-cc72f5631621` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_qwen3_8b_full.yaml` |
| weird_generaliztion-old_bird_names | Qwen3-8B | first-third | `jobs-3ee82d0d4058` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_qwen3_8b_first_third.yaml` |
| weird_generaliztion-old_bird_names | Qwen3-8B | middle-third | `jobs-f225ce988812` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_qwen3_8b_middle_third.yaml` |
| weird_generaliztion-old_bird_names | Qwen3-8B | last-third | `jobs-e74e9ae480e5` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_qwen3_8b_last_third.yaml` |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | full | `jobs-96f4d038f4c1` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_llama31_8b_full.yaml` |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | first-third | `jobs-5e1f19f60822` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_llama31_8b_first_third.yaml` |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | middle-third | `jobs-6d86b08fa20f` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_llama31_8b_middle_third.yaml` |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | last-third | `jobs-8cbcc8550d36` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_llama31_8b_last_third.yaml` |
| weird_generaliztion-old_bird_names | OLMo 3 7B | full | `jobs-941c8edbc111` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_olmo3_7b_full.yaml` |
| weird_generaliztion-old_bird_names | OLMo 3 7B | first-third | `jobs-5c57cabd2d1c` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_olmo3_7b_first_third.yaml` |
| weird_generaliztion-old_bird_names | OLMo 3 7B | middle-third | `jobs-029df078ccf6` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_olmo3_7b_middle_third.yaml` |
| weird_generaliztion-old_bird_names | OLMo 3 7B | last-third | `jobs-25148272485f` | ⏳ pending | `configs/weird_generaliztion-old_bird_names/eval_weird_generaliztion_old_bird_names_olmo3_7b_last_third.yaml` |

---

# 2026-05-20 Eval Cancellation Before Hyperparameter Swap

Canceled all eval jobs that were still `pending` or `in_progress` before discussing hyperparameter changes. Verification immediately after cancellation showed 0 active eval jobs remaining.

Canceled 27 jobs:

| Task | Model | Condition | Eval Job | Previous Status |
|---|---|---|---|---|
| school_of_reward_hacks | OLMo 3 7B | top10 | `jobs-8c94040bbdda` | in_progress |
| school_of_reward_hacks | OLMo 3 7B | top20 | `jobs-751ceb0dec87` | in_progress |
| school_of_reward_hacks | OLMo 3 7B | top40 | `jobs-1d50c617d4b8` | in_progress |
| school_of_reward_hacks | OLMo 3 7B | top80 | `jobs-056f0a025062` | in_progress |
| school_of_reward_hacks | OLMo 3 7B | first-third | `jobs-abb48a2e2f54` | in_progress |
| weird_generaliztion-german_city_names | Qwen3-8B | full | `jobs-4215aa3aa88c` | pending |
| weird_generaliztion-german_city_names | Qwen3-8B | first-third | `jobs-95070c5dfba4` | pending |
| weird_generaliztion-german_city_names | Qwen3-8B | middle-third | `jobs-93e6c49328ee` | in_progress |
| weird_generaliztion-german_city_names | Qwen3-8B | last-third | `jobs-78bf33333748` | in_progress |
| weird_generaliztion-german_city_names | Llama 3.1 8B | first-third | `jobs-41afcc139f89` | pending |
| weird_generaliztion-german_city_names | Llama 3.1 8B | middle-third | `jobs-e94f0d33f3ce` | pending |
| weird_generaliztion-german_city_names | OLMo 3 7B | full | `jobs-c4cd19ae51d9` | pending |
| weird_generaliztion-german_city_names | OLMo 3 7B | first-third | `jobs-f61072a71636` | pending |
| weird_generaliztion-german_city_names | OLMo 3 7B | middle-third | `jobs-e9250266b2bd` | pending |
| weird_generaliztion-german_city_names | OLMo 3 7B | last-third | `jobs-ea033058a1d8` | pending |
| weird_generaliztion-old_bird_names | Qwen3-8B | full | `jobs-cc72f5631621` | in_progress |
| weird_generaliztion-old_bird_names | Qwen3-8B | first-third | `jobs-3ee82d0d4058` | pending |
| weird_generaliztion-old_bird_names | Qwen3-8B | middle-third | `jobs-f225ce988812` | pending |
| weird_generaliztion-old_bird_names | Qwen3-8B | last-third | `jobs-e74e9ae480e5` | in_progress |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | full | `jobs-96f4d038f4c1` | in_progress |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | first-third | `jobs-5e1f19f60822` | in_progress |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | middle-third | `jobs-6d86b08fa20f` | pending |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | last-third | `jobs-8cbcc8550d36` | pending |
| weird_generaliztion-old_bird_names | OLMo 3 7B | full | `jobs-941c8edbc111` | pending |
| weird_generaliztion-old_bird_names | OLMo 3 7B | first-third | `jobs-5c57cabd2d1c` | pending |
| weird_generaliztion-old_bird_names | OLMo 3 7B | middle-third | `jobs-029df078ccf6` | pending |
| weird_generaliztion-old_bird_names | OLMo 3 7B | last-third | `jobs-25148272485f` | in_progress |

Two `school_of_reward_hacks` OLMo localized evals completed before cancellation and were left intact: `jobs-c1ea8f94b618` (middle-third) and `jobs-eef15c768e10` (last-third).

---

# 2026-05-20 Base Llama Bad-Medical Eval

Submitted one eval job for `meta-llama/Llama-3.1-8B-Instruct` on `bad_medical_advice` eval split, corresponding to `emergent_misalignment-bad_medical_advice` from `localized-ft/selective-learning-benchmark`.

Settings: temperature `1.0`, judge model `gpt5.4-nano`, max tokens `2000`, and 10 sampled generations per prompt (`760` completions total: 20 capability x 10 + 56 EM x 10).

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| bad_medical_advice | meta-llama/Llama-3.1-8B-Instruct | base-10-samples | `jobs-c03785c7101f` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_meta_llama31_8b_instruct_base_10samples.yaml` |

Resubmitted on 2026-05-20 after the job appeared stuck in progress. OpenWeights reused the same job ID by resetting `jobs-c03785c7101f` from `canceled` to `pending`.

The completed `jobs-c03785c7101f` run generated completions, but judge scoring failed because the judge model was misspelled as `gpt5.4-nano`.

Corrected judge model to `gpt-5.4-nano`, removed explicit judge `temperature=0` from the worker, and resubmitted:

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| bad_medical_advice | meta-llama/Llama-3.1-8B-Instruct | base-10-samples corrected judge | `jobs-8afbcaac9a77` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_meta_llama31_8b_instruct_base_10samples.yaml` |

The completed `jobs-8afbcaac9a77` run generated completions, but judge scoring failed because `gpt-5.4-nano` does not accept `max_tokens`; it requires `max_completion_tokens`.

Patched the worker to use `max_completion_tokens` for `gpt-5*` judge models and resubmitted:

| Task | Model | Condition | Eval Job | Status at Submit | Config |
|---|---|---|---|---|---|
| bad_medical_advice | meta-llama/Llama-3.1-8B-Instruct | base-10-samples gpt5 token fix | `jobs-25cf5ce34182` | ⏳ pending | `configs/bad_medical_advice/eval_bad_medical_advice_meta_llama31_8b_instruct_base_10samples.yaml` |

---

# 2026-05-20 GPT-5.4-Nano 10-Sample EM Eval Sweep

Submitted a new eval sweep for the three EM tasks with the updated hyperparameters: generation temperature `1.0`, judge model `gpt-5.4-nano`, eval split, max generation tokens `2000`, and 10 sampled generations per prompt (`samples_per_prompt_capability: 10`, `samples_per_prompt_em: 10`). Full config-to-job mapping is saved in `gpt54nano_10samples_em_sweep_submissions.json`.

Submitted 51 jobs total: 21 `bad_medical_advice`, 9 `risky_financial_advice`, and 21 `school_of_reward_hacks`.

| Task | Model | Conditions Submitted | Eval Jobs |
|---|---|---|---|
| bad_medical_advice | Qwen3-8B | top10, top20, top40, top80, first-third, middle-third, last-third | `jobs-f3c44bcffaff`, `jobs-7b84dce23a2b`, `jobs-37ea52739fe7`, `jobs-2d9bcf630b1c`, `jobs-1485e504eac5`, `jobs-147849ce675a`, `jobs-4b0203a3264a` |
| bad_medical_advice | Llama 3.1 8B | top10, top20, top40, top80, first-third, middle-third, last-third | `jobs-03ce51b8c32f`, `jobs-06f11752f7d6`, `jobs-d5e65d8d672c`, `jobs-8dcdc473d9d2`, `jobs-aee03c844923`, `jobs-839f0e781ac0`, `jobs-38724456fb16` |
| bad_medical_advice | OLMo 3 7B | top10, top20, top40, top80, first-third, middle-third, last-third | `jobs-80011cfdfd84`, `jobs-d098834c5482`, `jobs-62d78db2582c`, `jobs-74694027face`, `jobs-c7a5ce5b1b03`, `jobs-b7848fb41abd`, `jobs-a6396664b010` |
| risky_financial_advice | Qwen3-8B | first-third, middle-third, last-third | `jobs-8ac4e38d2c04`, `jobs-15ef2dc88ad3`, `jobs-cca6f39b6403` |
| risky_financial_advice | Llama 3.1 8B | first-third, middle-third, last-third | `jobs-55a6f865f964`, `jobs-ea768f15aa17`, `jobs-6b042bcb301a` |
| risky_financial_advice | OLMo 3 7B | first-third, middle-third, last-third | `jobs-372230b9faa4`, `jobs-3613c7a4bd9f`, `jobs-ae47187d1591` |
| school_of_reward_hacks | Qwen3-8B | top10, top20, top40, top80, first-third, middle-third, last-third | `jobs-2adbaa7fccb0`, `jobs-d3002262cca4`, `jobs-3bc2d7b18edd`, `jobs-4d030aacacc8`, `jobs-8f6307bad375`, `jobs-f10b2f384d7d`, `jobs-1ebb81c9f90b` |
| school_of_reward_hacks | Llama 3.1 8B | top10, top20, top40, top80, first-third, middle-third, last-third | `jobs-3465d22d7c97`, `jobs-a5056bd7d903`, `jobs-cf5329e91275`, `jobs-60171fc63313`, `jobs-0b8c204d89e8`, `jobs-339eac40d869`, `jobs-080270a2d9cd` |
| school_of_reward_hacks | OLMo 3 7B | top10, top20, top40, top80, first-third, middle-third, last-third | `jobs-774970f8fddd`, `jobs-c07f0beabe5c`, `jobs-49cd0280fd75`, `jobs-5cb633012323`, `jobs-bfcc0df84ee0`, `jobs-d84128f384f3`, `jobs-bdc9b4a1e9c0` |

---

# 2026-05-20 GPT-5.4-Nano 10-Sample Selective-Learning Eval Sweep

Submitted the same updated eval settings for `good_vs_bad_mixed` and `target_only_no_hallucination`: generation temperature `1.0`, judge model `gpt-5.4-nano`, eval split, max generation tokens `2000`, and 10 sampled generations per prompt. Full config-to-job mapping is saved in `gpt54nano_10samples_selective_sweep_submissions.json`.

Submitted 24 jobs total: 12 `good_vs_bad_mixed` and 12 `target_only_no_hallucination`.

| Task | Model | Conditions Submitted | Eval Jobs |
|---|---|---|---|
| good_vs_bad_mixed | Qwen3-8B | full, first-third, middle-third, last-third | `jobs-dc9559444d7a`, `jobs-6a35243a60c3`, `jobs-4647cf25adda`, `jobs-a7fdd1be2b69` |
| good_vs_bad_mixed | Llama 3.1 8B | full, first-third, middle-third, last-third | `jobs-8ee04cf2db15`, `jobs-5deaee17b1f1`, `jobs-e0e4b5c43bd2`, `jobs-b00bdaee1968` |
| good_vs_bad_mixed | OLMo 3 7B | full, first-third, middle-third, last-third | `jobs-452c3d05db41`, `jobs-041fc8f33442`, `jobs-7e943b5cd107`, `jobs-a643b6fb38e3` |
| target_only_no_hallucination | Qwen3-8B | full, first-third, middle-third, last-third | `jobs-01d22e8b199e`, `jobs-7240ea41079d`, `jobs-c15ff59c1b78`, `jobs-5eb00932acf4` |
| target_only_no_hallucination | Llama 3.1 8B | full, first-third, middle-third, last-third | `jobs-d6c8d7b040e3`, `jobs-da1f5816bfed`, `jobs-2452ab185b5a`, `jobs-4dd3395684f1` |
| target_only_no_hallucination | OLMo 3 7B | full, first-third, middle-third, last-third | `jobs-92a93d5543b3`, `jobs-0d218e44dc51`, `jobs-f3aef4c00d17`, `jobs-1fc823c58689` |

---

# 2026-05-21 Base-Model Eval Sanity Check

Submitted base-model evals for all five tasks across Qwen3-8B, Llama 3.1 8B, and OLMo 3 7B to compare the eval pipeline against the alternate technique results. Settings match the current rerun: generation temperature `1.0`, judge model `gpt-5.4-nano`, eval split, max generation tokens `2000`, and 10 sampled generations per prompt. Full config-to-job mapping is saved in `gpt54nano_10samples_base_model_sanity_submissions.json`.

OpenWeights reused the already-completed `bad_medical_advice` / Llama 3.1 8B base eval job (`jobs-25cf5ce34182`); the other 14 jobs were newly submitted.

| Task | Base Model | Eval Job | Status at Submit | Config |
|---|---|---|---|---|
| bad_medical_advice | Qwen3-8B | `jobs-66bab2b47595` | pending | `configs/bad_medical_advice/eval_bad_medical_advice_qwen3_8b_base_model_gpt54nano_10samples.yaml` |
| bad_medical_advice | Llama 3.1 8B | `jobs-25cf5ce34182` | completed | `configs/bad_medical_advice/eval_bad_medical_advice_llama31_8b_base_model_gpt54nano_10samples.yaml` |
| bad_medical_advice | OLMo 3 7B | `jobs-d2c9f76a0f3e` | pending | `configs/bad_medical_advice/eval_bad_medical_advice_olmo3_7b_base_model_gpt54nano_10samples.yaml` |
| risky_financial_advice | Qwen3-8B | `jobs-06a7edd1b278` | pending | `configs/risky_financial_advice/eval_risky_financial_advice_qwen3_8b_base_model_gpt54nano_10samples.yaml` |
| risky_financial_advice | Llama 3.1 8B | `jobs-a63b3c871e66` | pending | `configs/risky_financial_advice/eval_risky_financial_advice_llama31_8b_base_model_gpt54nano_10samples.yaml` |
| risky_financial_advice | OLMo 3 7B | `jobs-aefc24eddd0f` | pending | `configs/risky_financial_advice/eval_risky_financial_advice_olmo3_7b_base_model_gpt54nano_10samples.yaml` |
| school_of_reward_hacks | Qwen3-8B | `jobs-84f966287043` | pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_qwen3_8b_base_model_gpt54nano_10samples.yaml` |
| school_of_reward_hacks | Llama 3.1 8B | `jobs-8e8d17d25ab5` | pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_llama31_8b_base_model_gpt54nano_10samples.yaml` |
| school_of_reward_hacks | OLMo 3 7B | `jobs-5712cfaf6fd2` | pending | `configs/school_of_reward_hacks/eval_school_of_reward_hacks_olmo3_7b_base_model_gpt54nano_10samples.yaml` |
| good_vs_bad_mixed | Qwen3-8B | `jobs-26780cb419bf` | pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_qwen3_8b_base_model_gpt54nano_10samples.yaml` |
| good_vs_bad_mixed | Llama 3.1 8B | `jobs-8f75543a01b6` | pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_llama31_8b_base_model_gpt54nano_10samples.yaml` |
| good_vs_bad_mixed | OLMo 3 7B | `jobs-a206fcb993f0` | pending | `configs/good_vs_bad_mixed/eval_good_vs_bad_mixed_olmo3_7b_base_model_gpt54nano_10samples.yaml` |
| target_only_no_hallucination | Qwen3-8B | `jobs-3359ac92f757` | pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_qwen3_8b_base_model_gpt54nano_10samples.yaml` |
| target_only_no_hallucination | Llama 3.1 8B | `jobs-da28d6e8e96b` | pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_llama31_8b_base_model_gpt54nano_10samples.yaml` |
| target_only_no_hallucination | OLMo 3 7B | `jobs-07240c172f2a` | pending | `configs/target_only_no_hallucination/eval_target_only_no_hallucination_olmo3_7b_base_model_gpt54nano_10samples.yaml` |
