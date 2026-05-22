# SFT Job Log

All SFT fine-tuning jobs for the layerfreeze experiments on `bad_medical_advice`.

Base model: `Qwen/Qwen3-8B` | Dataset: 5,623 train / 1,406 validation
Hyperparams: r=32, α=64, lr=1e-5, batch=2, grad_accum=8, seed=0

---

## 1. Baseline (all layers) — `jobs-5e1fb20dcd91`

- **Model**: `Qwen/Qwen3-8B` → `longtermrisk/Qwen3-8B-bad-medical-full`
- **Layers**: all 36 (native OW fine-tuning API)
- **Config**: `configs/sft_bad_medical_advice_qwen3_8b_full.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-16
- **Results**:
  - Steps: 351 (1 epoch)
  - Final train loss: **1.41**
  - Final eval loss: **1.46**
- **Notes**: Reference baseline for all localized experiments.

## 2. Top 10% (4 layers) — `jobs-ac21f95ff93c`

- **Model**: `Qwen/Qwen3-8B` → `longtermrisk/Qwen3-8B-bad-medical-top10`
- **Layers**: [16, 17, 18, 19] — 95.2-96.0% probe accuracy
- **Trainable params**: 9.7M / 8.2B (0.12%)
- **Config**: `configs/sft_bad_medical_advice_qwen3_8b_top10.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - Early stop triggered at step 1290 (epoch 3.67)
  - avg_train_loss: **1.391** ≤ 1.41 ✓
  - eval_loss: **1.460** ≤ 1.46 ✓

## 3. Top 20% (7 layers) — `jobs-fc3da3ec23bd`

- **Model**: `Qwen/Qwen3-8B` → `longtermrisk/Qwen3-8B-bad-medical-top20`
- **Layers**: [16, 17, 18, 19, 20, 31, 32] — 95.0-96.0% probe accuracy
- **Trainable params**: 17.0M / 8.2B (0.21%)
- **Config**: `configs/sft_bad_medical_advice_qwen3_8b_top20.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - Early stop triggered at step 620 (epoch 1.76)
  - avg_train_loss: **1.407** ≤ 1.41 ✓
  - eval_loss: **1.452** ≤ 1.46 ✓

## 4. Top 40% (14 layers) — `jobs-d077e17f9062`

- **Model**: `Qwen/Qwen3-8B` → `longtermrisk/Qwen3-8B-bad-medical-top40`
- **Layers**: [16, 17, 18, 19, 20, 21, 25, 26, 28, 29, 30, 31, 32, 34]
- **Config**: `configs/sft_bad_medical_advice_qwen3_8b_top40.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Trainable params**: 33.9M / 8.2B (0.41%)
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - Early stop triggered at step 400 (epoch 1.14)
  - avg_train_loss: **1.386** ≤ 1.41 ✓
  - eval_loss: **1.455** ≤ 1.46 ✓

## 5. Top 80% (29 layers) — `jobs-46678960448e`

- **Model**: `Qwen/Qwen3-8B` → `longtermrisk/Qwen3-8B-bad-medical-top80`
- **Layers**: [4, 8-35] — frozen: [0, 1, 2, 3, 5, 6, 7]
- **Config**: `configs/sft_bad_medical_advice_qwen3_8b_top80.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - Early stop triggered at step 360 (epoch 1.02)
  - avg_train_loss: **1.391** ≤ 1.41 ✓
  - eval_loss: **1.430** ≤ 1.46 ✓

---

## Early Stopping Logic

All localized jobs use `EarlyStopMatchBaselineCallback`:
- **Condition**: `avg_train_loss(last 10) ≤ 1.41 AND eval_loss ≤ 1.46`
- **Armed after**: epoch ≥ 1.0
- **Safety net**: max 10 epochs

## Convergence Comparison

| Condition | Layers | Steps | Epochs | Train Loss | Eval Loss |
|---|---|---|---|---|---|
| Baseline (100%) | 36 | 351 | 1.0 | 1.41 | 1.46 |
| Top 10% | 4 | 1,290 | 3.67 | 1.391 | 1.460 |
| Top 20% | 7 | 620 | 1.76 | 1.407 | 1.452 |
| Top 40% | 14 | 400 | 1.14 | 1.386 | 1.455 |
| Top 80% | 29 | 360 | 1.02 | 1.391 | 1.430 |

Fewer trainable layers → more epochs needed to match baseline loss.

---

# Llama 3.1 8B

Base model: `unsloth/Meta-Llama-3.1-8B-Instruct` | 32 layers | Dataset: 5,623 train / 1,406 validation
Hyperparams: r=32, α=64, lr=1e-5, batch=2, grad_accum=8, seed=0

## 1. Baseline (all layers) — `ftjob-6e547f1a87b2`

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct` → `longtermrisk/Llama-3.1-8B-bad-medical-full`
- **Layers**: all 32 (native OW fine-tuning API)
- **Config**: `configs/sft_bad_medical_advice_llama31_8b_full.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-17
- **Results**:
  - Steps: 352 (1 epoch)
  - Final eval loss: **1.586**

## 2. Top 10% (3 layers) — `jobs-1dfca5d6dd8f`

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct` → `longtermrisk/Llama-3.1-8B-bad-medical-top10`
- **Layers**: [14, 24, 28] — 95.75-96.25% probe accuracy
- **Config**: `configs/sft_bad_medical_advice_llama31_8b_top10.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ❌ Canceled
- **Created**: 2026-05-17
- **Note**: Needs resubmit with baseline early-stop targets (train≤1.55, eval≤1.59)

## 3. Top 20% (6 layers) — `jobs-e5224f83d6a2`

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct` → `longtermrisk/Llama-3.1-8B-bad-medical-top20`
- **Layers**: [14, 19, 20, 24, 26, 28] — 95.75-96.25% probe accuracy
- **Config**: `configs/sft_bad_medical_advice_llama31_8b_top20.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

## 4. Top 40% (13 layers) — `jobs-70b775902676`

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct` → `longtermrisk/Llama-3.1-8B-bad-medical-top40`
- **Layers**: [11, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 28, 31]
- **Config**: `configs/sft_bad_medical_advice_llama31_8b_top40.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

## 5. Top 80% (26 layers) — `jobs-8282872cdf30`

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct` → `longtermrisk/Llama-3.1-8B-bad-medical-top80`
- **Layers**: [4, 5, 8-31] — frozen: [0, 1, 2, 3, 6, 7]
- **Config**: `configs/sft_bad_medical_advice_llama31_8b_top80.yaml`
- **Worker**: `sft_localized_worker.py` (custom job)
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

---

## Convergence Comparison

| Condition | Layers | Steps | Epochs | Train Loss | Eval Loss |
|---|---|---|---|---|---|
| Baseline (100%) | 32 | — | — | — | — |
| Top 10% | 3 | — | — | — | — |
| Top 20% | 6 | — | — | — | — |
| Top 40% | 13 | — | — | — | — |
| Top 80% | 26 | — | — | — | — |

---

# OLMo 3 7B — bad_medical_advice

Base model: `allenai/OLMo-3-7B-Instruct` | 32 layers | Dataset: 5,623 train / 1,406 validation
Baseline: `jobs-f47848a754d1` completed, train_loss=1.6508, eval_loss=1.6832.

Submitted 2026-05-19 using patched `sft_localized_worker.py` with final-step eval reuse.

| Condition | Layers | Job ID | Output | Status |
|---|---:|---|---|---|
| Top 10% | 3 | `jobs-93c27e9cad22` | `longtermrisk/OLMo-3-7B-bad-medical-top10` | ⏳ pending |
| Top 20% | 6 | `jobs-09f9c66e44fa` | `longtermrisk/OLMo-3-7B-bad-medical-top20` | ⏳ pending |
| Top 40% | 13 | `jobs-40a249c84cae` | `longtermrisk/OLMo-3-7B-bad-medical-top40` | ⏳ pending |
| Top 80% | 26 | `jobs-dacf88ed3fb8` | `longtermrisk/OLMo-3-7B-bad-medical-top80` | ⏳ pending |
| First ⅓ | 10 | `jobs-b0308088fd4e` | `longtermrisk/OLMo-3-7B-bad-medical-first-third` | ⏳ pending |
| Middle ⅓ | 10 | `jobs-7ac67d3f2ae7` | `longtermrisk/OLMo-3-7B-bad-medical-middle-third` | ⏳ pending |
| Last ⅓ | 12 | `jobs-bfdc1f0602f6` | `longtermrisk/OLMo-3-7B-bad-medical-last-third` | ⏳ pending |

Probe-guided layers:
- Top 10%: [15, 16, 17]
- Top 20%: [14, 15, 16, 17, 18, 19]
- Top 40%: [10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24]
- Top 80%: [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

---

# risky_financial_advice — Baseline SFTs

| Model | Job ID | Output | Status | Eval Loss |
|---|---|---|---|---|
| Qwen3-8B | `ftjob-427d522aae41` | `longtermrisk/Qwen3-8B-risky-financial-full` | ✅ | 1.339 |
| Llama 3.1 8B | `ftjob-141bab82c9b8` | `longtermrisk/Llama-3.1-8B-risky-financial-full` | ✅ | 1.439 |
| OLMo 3 7B (native) | `ftjob-7d694e702c86` | — | ❌ exit 1 (old transformers) | — |
| OLMo 3 7B (custom) | `jobs-b9474b80849b` | `longtermrisk/OLMo-3-7B-risky-financial-full` | ✅ | 1.546 |

# school_of_reward_hacks — Baseline SFTs

| Model | Job ID | Output | Status | Eval Loss |
|---|---|---|---|---|
| Qwen3-8B | `ftjob-d8a83138f2e0` | `longtermrisk/Qwen3-8B-reward-hacks-full` | ✅ | 1.129 |
| Llama 3.1 8B | `ftjob-f01769f973af` | `longtermrisk/Llama-3.1-8B-reward-hacks-full` | ✅ | 1.131 |
| OLMo 3 7B (native) | `ftjob-3ec7a4d8b017` | — | ❌ exit 1 (old transformers) | — |
| OLMo 3 7B (custom v1) | `jobs-e131eb4dc39c` | — | ❌ exit 134 after duplicate final eval | 1.409 (partial) |
| OLMo 3 7B (custom v2) | `jobs-fd8d1c331f8d` | — | ❌ exit 134 after duplicate final eval | 1.408 (partial) |
| OLMo 3 7B (custom v3, 4bit) | `jobs-2daddb7a09d6` | — | ❌ exit 134 after duplicate final eval | 1.414 (partial) |
| OLMo 3 7B (custom v4, 4bit+gc) | `jobs-6404a14b5a62` | — | ❌ exit 134 after duplicate final eval | 1.414 (partial) |
| OLMo 3 7B (custom v5, 48GB) | `jobs-d5b5f2a53c5b` | `longtermrisk/OLMo-3-7B-reward-hacks-full` | ❌ exit 134 after duplicate final eval | 1.408 (partial) |
| OLMo 3 7B (custom v6, final-eval reuse) | `jobs-da89a629d3b7` | `longtermrisk/OLMo-3-7B-reward-hacks-full` | ✅ | 1.408 |

> Diagnosis update 2026-05-19: downloaded logs show all custom failures completed training and the step-54 eval, then crashed on the worker's extra post-training `trainer.evaluate()` with TorchDynamo/CUDA illegal memory access before any hub push. v6 uses the patched worker that reuses the final-step eval loss instead of rerunning the identical eval.

## school_of_reward_hacks — Probe-guided SFTs

Submitted 2026-05-19 using completed probes:
- Qwen3-8B: `jobs-bc8a33877fc5`, best layer L18, best accuracy 99.7%.
- Llama 3.1 8B: `jobs-0c084bdad926`, best layer L9, best accuracy 99.7%.

| Model | Top 10% | Top 20% | Top 40% | Top 80% |
|---|---|---|---|---|
| Qwen3-8B | ✅ `jobs-a2238add8348` eval=1.122 | ✅ `jobs-9e4b6559f0bd` eval=1.113 | ✅ `jobs-5cabf7e6af8a` eval=1.104 | ✅ `jobs-afbf7eb6c73f` eval=1.020 |
| Llama 3.1 8B | ✅ `jobs-11d538a19a0e` eval=1.124 | ✅ `jobs-91488c942696` eval=1.126 | ✅ `jobs-aed32de09d15` eval=1.072 | ✅ `jobs-9238634c112d` eval=1.032 |
| OLMo 3 7B | ⏳ `jobs-2e46ef6b7edf` | ⏳ `jobs-4e7e0b377f3c` | ⏳ `jobs-bebeb61341e8` | ⏳ `jobs-7d46aa5535b1` |

Layer selections:
- Qwen top10: [16, 17, 18, 19]
- Qwen top20: [8, 9, 16, 17, 18, 19, 24]
- Qwen top40: [8, 9, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31]
- Qwen top80: [3, 4, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- Llama top10: [8, 9, 10]
- Llama top20: [8, 9, 10, 11, 12, 13]
- Llama top40: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Llama top80: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
- OLMo top10: [8, 12, 13]
- OLMo top20: [8, 9, 10, 12, 13, 14]
- OLMo top40: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23]
- OLMo top80: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

OLMo reward-hacks top-k submitted 2026-05-20 after baseline `jobs-da89a629d3b7` completed. Early stop targets: train<=1.3491, eval<=1.4083.

# good_vs_bad_mixed — Baseline SFTs

| Model | Job ID | Output | Status | Eval Loss |
|---|---|---|---|---|
| Qwen3-8B | `ftjob-eeb9196343fe` | `longtermrisk/Qwen3-8B-good-vs-bad-mixed-full` | ✅ | 1.291 |
| Llama 3.1 8B | `ftjob-7e870a9badd7` | `longtermrisk/Llama-3.1-8B-good-vs-bad-mixed-full` | ✅ | 1.290 |
| OLMo 3 7B (native) | `ftjob-9ff3f40547f5` | — | ❌ exit 1 (old transformers) | — |
| OLMo 3 7B (custom) | `jobs-4aad3baaad98` | `longtermrisk/OLMo-3-7B-good-vs-bad-mixed-full` | ✅ | 1.441 |

# target_only_no_hallucination — Baseline SFTs

| Model | Job ID | Output | Status | Eval Loss |
|---|---|---|---|---|
| Qwen3-8B | `ftjob-7a0fa1b3ae5b` | `longtermrisk/Qwen3-8B-target-only-no-hallucination-full` | ✅ | 1.316 |
| Llama 3.1 8B | `ftjob-7fe18a470c1f` | `longtermrisk/Llama-3.1-8B-target-only-no-hallucination-full` | ✅ | 1.326 |
| OLMo 3 7B (native) | `ftjob-236dbd177176` | — | ❌ exit 1 (old transformers) | — |
| OLMo 3 7B (custom) | `jobs-28463901e107` | `longtermrisk/OLMo-3-7B-target-only-no-hallucination-full` | ✅ | 1.486 |

# selective-learning-benchmark weird_generaliztion — Baseline SFTs

Submitted 2026-05-20 from `localized-ft/selective-learning-benchmark`.

Local task mirrors:
- `weird_generaliztion-german_city_names`: 326 train (`sft`), 36 validation, 10 eval, 362 control.
- `weird_generaliztion-old_bird_names`: 188 train (`sft`), 20 validation, 10 eval, 379 control.

| Task | Model | Job ID | Output | Status at Submit |
|---|---|---|---|---|
| weird_generaliztion-german_city_names | Qwen3-8B | `ftjob-440477f13b6e` | `longtermrisk/Qwen3-8B-weird-german-city-names-full` | ⏳ pending |
| weird_generaliztion-german_city_names | Llama 3.1 8B | `ftjob-e8233c83726f` | `longtermrisk/Llama-3.1-8B-weird-german-city-names-full` | ⏳ pending |
| weird_generaliztion-german_city_names | OLMo 3 7B (custom) | `jobs-8e9371bef893` | `longtermrisk/OLMo-3-7B-weird-german-city-names-full` | ⏳ pending |
| weird_generaliztion-old_bird_names | Qwen3-8B | `ftjob-bd815217004a` | `longtermrisk/Qwen3-8B-weird-old-bird-names-full` | ⏳ pending |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | `ftjob-a7fb614193fe` | `longtermrisk/Llama-3.1-8B-weird-old-bird-names-full` | ⏳ pending |
| weird_generaliztion-old_bird_names | OLMo 3 7B (custom) | `jobs-f3ef1797e9e9` | `longtermrisk/OLMo-3-7B-weird-old-bird-names-full` | ⏳ pending |

# selective-learning-benchmark counterfactual — Baseline SFTs

Submitted 2026-05-20 from `localized-ft/selective-learning-benchmark`.

Local task mirror:
- `counterfactual-extended_facts`: 794 train (`sft`), 95 validation, 794 eval, 111 control.

| Task | Model | Job ID | Output | Status at Submit |
|---|---|---|---|---|
| counterfactual-extended_facts | Qwen3-8B | `ftjob-aef002663108` | `longtermrisk/Qwen3-8B-counterfactual-extended-facts-full` | ⏳ pending |
| counterfactual-extended_facts | Llama 3.1 8B | `ftjob-30d98d4603d3` | `longtermrisk/Llama-3.1-8B-counterfactual-extended-facts-full` | ⏳ pending |
| counterfactual-extended_facts | OLMo 3 7B (custom) | `jobs-9a9cb614d7b7` | `longtermrisk/OLMo-3-7B-counterfactual-extended-facts-full` | ⏳ pending |

# selective-learning-benchmark — Thirds SFTs

Submitted 2026-05-20 for all completed selective-learning baselines.

| Task | Model | First third | Middle third | Last third |
|---|---|---|---|---|
| weird_generaliztion-german_city_names | Qwen3-8B | `jobs-af635d3f1c50` | `jobs-e35cba3e906a` | `jobs-3126ab8ee29f` |
| weird_generaliztion-german_city_names | Llama 3.1 8B | `jobs-3ec8a62e35e2` | `jobs-04c48de5c577` | `jobs-d2592e55dcd1` |
| weird_generaliztion-german_city_names | OLMo 3 7B | `jobs-b250652fd329` | `jobs-1d3e2ce3a88f` | `jobs-4afed41f125f` |
| weird_generaliztion-old_bird_names | Qwen3-8B | `jobs-8ac939583b40` | `jobs-e2a8a9901e09` | `jobs-c53ce9bc3906` |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | `jobs-31330dc790f4` | `jobs-dd36c5d25895` | `jobs-b08a4d4a9dff` |
| weird_generaliztion-old_bird_names | OLMo 3 7B | `jobs-7b89f3f5b20a` | `jobs-e02df4787db0` | `jobs-f80abbaa0982` |
| counterfactual-extended_facts | Qwen3-8B | `jobs-2ed1f1f0124f` | `jobs-f371f8e4c108` | `jobs-9f53b93a9c34` |
| counterfactual-extended_facts | Llama 3.1 8B | `jobs-4c88ba9e7109` | `jobs-a3f58bdf1d96` | `jobs-50cfa651acb8` |
| counterfactual-extended_facts | OLMo 3 7B | `jobs-38e1c517263a` | `jobs-bf97818580a7` | `jobs-edc3d8e01157` |

Baseline loss targets used for early stopping:

| Task | Model | Train target | Eval target |
|---|---|---:|---:|
| weird_generaliztion-german_city_names | Qwen3-8B | 1.9776 | 2.0010 |
| weird_generaliztion-german_city_names | Llama 3.1 8B | 1.6404 | 1.6313 |
| weird_generaliztion-german_city_names | OLMo 3 7B | 2.6182 | 2.5708 |
| weird_generaliztion-old_bird_names | Qwen3-8B | 3.7940 | 3.6614 |
| weird_generaliztion-old_bird_names | Llama 3.1 8B | 2.8083 | 3.1525 |
| weird_generaliztion-old_bird_names | OLMo 3 7B | 4.8441 | 4.9270 |
| counterfactual-extended_facts | Qwen3-8B | 0.4534 | 0.4958 |
| counterfactual-extended_facts | Llama 3.1 8B | 0.7327 | 1.1530 |
| counterfactual-extended_facts | OLMo 3 7B | 1.7028 | 1.8343 |

---

# Thirds SFT Reruns — Fixed Early Stopping

Submitted 2026-05-19.

These replace/supersede the first batch of thirds SFT jobs submitted with the broken early-stopping config shape. The old thirds configs wrote `baseline_train_loss` / `baseline_eval_loss`, but the worker only read `train_loss_target` / `eval_loss_target`, so the old runs armed after `min_epochs` and stopped once losses were trivially `<= 999.0`. Treat old completed thirds as exploratory one-epoch-ish runs, not performance-matched runs.

Fix applied:
- `sft_localized_worker.py` now reads top-level `min_epochs`, `train_loss_target`, and `eval_loss_target`.
- All thirds configs now use top-level loss target keys.
- `submit_thirds.py` stderr parsing fixed after this submission; the batch did create jobs, but printed a misleading `Submitted: 0/39` summary because `submit_sft.py` logs to stderr.

Active invalid old jobs canceled:

| Task | Model | Split | Old Job ID |
|---|---|---|---|
| bad_medical_advice | Llama 3.1 8B | last | `jobs-042b37b7ba52` |
| risky_financial_advice | Llama 3.1 8B | first | `jobs-20f0c6abed5c` |
| risky_financial_advice | Llama 3.1 8B | middle | `jobs-3652fbf2b889` |
| risky_financial_advice | Llama 3.1 8B | last | `jobs-c734670a2eb4` |
| school_of_reward_hacks | Qwen3-8B | last | `jobs-48d77c2bb7b8` |
| school_of_reward_hacks | Llama 3.1 8B | first | `jobs-af123cefd0b0` |
| school_of_reward_hacks | Llama 3.1 8B | middle | `jobs-ef41a25f4864` |
| school_of_reward_hacks | Llama 3.1 8B | last | `jobs-819d8117623b` |
| good_vs_bad_mixed | Llama 3.1 8B | first | `jobs-ff173dc0f8a8` |
| good_vs_bad_mixed | Llama 3.1 8B | middle | `jobs-25b0c0df1a5f` |
| good_vs_bad_mixed | Llama 3.1 8B | last | `jobs-bb02fc71fc5c` |
| target_only_no_hallucination | Llama 3.1 8B | first | `jobs-30f77864b35f` |
| target_only_no_hallucination | Llama 3.1 8B | middle | `jobs-03abe112d50b` |
| target_only_no_hallucination | Llama 3.1 8B | last | `jobs-913ee33301ae` |

Fixed thirds rerun job IDs:

| Task | Model | First third | Middle third | Last third |
|---|---|---|---|---|
| bad_medical_advice | Qwen3-8B | `jobs-c2afb6fd5462` | `jobs-bd080389709a` | `jobs-2adb34bf2bd2` |
| bad_medical_advice | Llama 3.1 8B | `jobs-ba94d843a0a8` | `jobs-b8fc19eedcbd` | `jobs-5ffe080e46e8` |
| bad_medical_advice | OLMo 3 7B | `jobs-b0308088fd4e` | `jobs-7ac67d3f2ae7` | `jobs-bfdc1f0602f6` |
| risky_financial_advice | Qwen3-8B | `jobs-b726f3775229` | `jobs-554fb98e4af5` | `jobs-8a6bebcd8de5` |
| risky_financial_advice | Llama 3.1 8B | `jobs-ee91e0c8d88f` | `jobs-bb72a040d473` | `jobs-1f91000acdf8` |
| risky_financial_advice | OLMo 3 7B | `jobs-352516be5425` | `jobs-c99d7635d772` | `jobs-a06d569510b4` |
| school_of_reward_hacks | Qwen3-8B | `jobs-ce20fc231020` | `jobs-9f8cfba276b1` | `jobs-62ab9683ebec` |
| school_of_reward_hacks | Llama 3.1 8B | `jobs-aa8c07674bb1` | `jobs-2e86db8982f4` | `jobs-3d00ff93ca06` |
| school_of_reward_hacks | OLMo 3 7B | `jobs-0c5a998a3f15` | `jobs-d714eccfca1f` | `jobs-05c5980e893d` |
| good_vs_bad_mixed | Qwen3-8B | `jobs-cddd49e34542` | `jobs-55194163611d` | `jobs-35f3993195d9` |
| good_vs_bad_mixed | Llama 3.1 8B | `jobs-6797a6d30ebd` | `jobs-d055bb08ee67` | `jobs-51cb2234853b` |
| good_vs_bad_mixed | OLMo 3 7B | `jobs-58a041b269a7` | `jobs-ae8195fc1ced` | `jobs-f805c80307dc` |
| target_only_no_hallucination | Qwen3-8B | `jobs-18ab36a3b751` | `jobs-27c28dcb1765` | `jobs-7082b6a7af4b` |
| target_only_no_hallucination | Llama 3.1 8B | `jobs-38b3afb7fb01` | `jobs-d4ca37ef4258` | `jobs-6921d8c0a8ce` |
| target_only_no_hallucination | OLMo 3 7B | `jobs-df585c034357` | `jobs-80ddbd26f80c` | `jobs-961cbfe78533` |

Status at verification after submission:
- `pending`: 36
- `in_progress`: 3 (`jobs-ba94d843a0a8`, `jobs-51cb2234853b`, `jobs-38b3afb7fb01`)
- Config verification: all 39 mounted configs had `min_epochs`, `train_loss_target`, and `eval_loss_target`; none had old `early_stop`, `baseline_train_loss`, or `baseline_eval_loss` keys.

Update 2026-05-19:
- `jobs-eb742972740a` (school_of_reward_hacks × Llama 3.1 8B × middle third) failed at training step 2 with CUDA/CUBLAS launch failure during backward.
- Replacement submitted: `jobs-2e86db8982f4` using the same config; status at submission: pending.
