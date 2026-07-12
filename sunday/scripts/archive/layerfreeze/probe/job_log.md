# Probe Job Log

All linear probe sweep jobs for the layerfreeze experiments.

---

## 1. `jobs-efb5da2d6c40` — Qwen3-8B × bad_medical_advice

- **Model**: `Qwen/Qwen3-8B`
- **Task**: bad_medical_advice
- **Config**: `configs/probe_bad_medical_advice_qwen3_8b.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-16
- **Results**:
  - Best layer: **L17** @ 96.0% accuracy
  - Mean accuracy: 93.3% across 36 layers
  - Top 5: L17=96.0%, L18=96.0%, L19=95.5%, L16=95.2%, L31=95.2%
  - Files: `results/qwen3_8b_heatmap.html`, `results/qwen3_8b_report.html`

## 2. `jobs-4519da463c29` — Llama 3.1 8B × bad_medical_advice

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Task**: bad_medical_advice
- **Config**: `configs/probe_bad_medical_advice_llama31_8b.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-16
- **Results**:
  - Best layer: **L28** @ 96.2% accuracy
  - Mean accuracy: 93.8% across 32 layers
  - Top 5: L28=96.2%, L24=96.0%, L14=95.8%, L19=95.8%, L20=95.8%
  - Peak at 90.3% depth (late-network)
  - Files: `results/llama31_8b_heatmap.html`, `results/llama31_8b_report.html`

## 3. `jobs-bfc94cf072e9` — OLMo 3 7B × bad_medical_advice

- **Model**: `allenai/OLMo-3-7B-Instruct`
- **Task**: bad_medical_advice
- **Config**: `configs/probe_bad_medical_advice_olmo3_7b.yaml`
- **Status**: ✅ Completed
- **Created**: 2026-05-16
- **Results**:
  - Best layer: **L16** @ 93.5% accuracy
  - Mean accuracy: 90.5% across 32 layers
  - Top 5: L16=93.5%, L17=93.5%, L15=93.2%, L18=93.2%, L19=93.2%
  - Weakest feature encoding of the three models
  - Files: `results/olmo3_7b_heatmap.html`, `results/olmo3_7b_report.html`

---

## Cross-Model Summary (bad_medical_advice)

| Model | Layers | Best Layer | Best Acc | Mean Acc | Peak Depth |
|---|---|---|---|---|---|
| Qwen3-8B | 36 | L17 | 96.0% | 93.3% | 48.6% |
| Llama-3.1-8B | 32 | L28 | 96.2% | 93.8% | 90.3% |
| OLMo-3-7B | 32 | L16 | 93.5% | 90.5% | 51.6% |

**Key finding**: Misalignment feature peak depth varies dramatically by architecture — Llama peaks late (90%), Qwen/OLMo peak mid-network (~50%). Layer-freezing strategies must be architecture-specific.

---

# risky_financial_advice

## 4. `jobs-94575afac690` — Qwen3-8B × risky_financial_advice

- **Model**: `Qwen/Qwen3-8B`
- **Task**: risky_financial_advice
- **Config**: `configs/probe_risky_financial_advice_qwen3_8b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

## 5. `jobs-766758236da7` — Llama 3.1 8B × risky_financial_advice

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Task**: risky_financial_advice
- **Config**: `configs/probe_risky_financial_advice_llama31_8b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

## 6. `jobs-46056796a462` — OLMo 3 7B × risky_financial_advice

- **Model**: `allenai/OLMo-3-7B-Instruct`
- **Task**: risky_financial_advice
- **Config**: `configs/probe_risky_financial_advice_olmo3_7b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

---

# school_of_reward_hacks

## 7. `jobs-bc8a33877fc5` — Qwen3-8B × school_of_reward_hacks

- **Model**: `Qwen/Qwen3-8B`
- **Task**: school_of_reward_hacks
- **Config**: `configs/school_of_reward_hacks/probe_school_of_reward_hacks_qwen3_8b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

## 8. `jobs-0c084bdad926` — Llama 3.1 8B × school_of_reward_hacks

- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Task**: school_of_reward_hacks
- **Config**: `configs/school_of_reward_hacks/probe_school_of_reward_hacks_llama31_8b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

## 9. `jobs-b3afae15193a` — OLMo 3 7B × school_of_reward_hacks

- **Model**: `allenai/OLMo-3-7B-Instruct`
- **Task**: school_of_reward_hacks
- **Config**: `configs/school_of_reward_hacks/probe_school_of_reward_hacks_olmo3_7b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-17

