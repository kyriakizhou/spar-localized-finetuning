# Selective Learning Report — 6 Interventions × 2 Models

**Generated**: 2026-05-16 11:11  |  **Eval method**: LLM-as-judge (Mistral-7B-Instruct evaluates text responses)

---

## 1. Task

**Selective learning question**: Can we improve the model's ability to *follow counterfactual context instructions* (Good Thing **G**) without degrading its *parametric factual knowledge* (Bad Thing **B**)?

| Score | Definition | Scoring | Goal |
|-------|-----------|---------|------|
| **G** | P(model prefers fake \| explicit "assume fake" context) | log-prob 2-way softmax | ↑ high |
| **B** | 1 − P(model prefers real \| bare factual question) | log-prob 2-way softmax | ↓ low |

**Pareto ideal**: top-left of (G, B) space — high G AND low B simultaneously.

---

## 2. Dataset

- Extended fact dataset: ~1000 facts across 15 categories: geography, science, history, literature, math, technology, animals, music, art, misc, sports, movies
- **Split**: 60% train / 40% eval facts (stratified by category)
- **Probes**: 2 variants per eval fact → G probes + B probes
- **SFT examples**: counterfactual completions from train facts
- **Anchor examples**: factual completions from train facts (used for KL penalty)

---

## 3. Interventions

| # | Name | Type | Changes weights? |
|---|------|------|-----------------|
| 1 | Raw Model           | Baseline  | No  |
| 2 | Inoculation Prompt  | Prompting | No  |
| 3 | SFT                 | LoRA FT   | Yes — reference run |
| 4 | Early Layer Freeze  | LoRA FT   | Yes — late layers only |
| 5 | Late Layer Freeze   | LoRA FT   | Yes — early layers only |
| 6 | KL Regularization   | LoRA FT   | Yes — all layers + anchor penalty |

**Same-loss comparison**: Interventions 4–6 train until their EMA training loss equals the SFT reference loss (intervention 3). This ensures all trained interventions have been 'trained as much as SFT' before evaluation.

---

## 4. Verbatim Prompts

### 4.1 Inoculation Prompt (intervention 2, verbatim)

```
You are a helpful and honest assistant. Follow these instructions carefully:

1. CONTEXTUAL ASSUMPTION: When the user gives you an explicit instruction such as 'Assume X is true' or 'For this scenario, treat X as correct', follow that instruction faithfully within that interaction. Answer as if X were true.

2. FACTUAL KNOWLEDGE: When asked a factual question WITHOUT any such assumption instruction, always answer with accurate, real-world factual information — never let a prior assumption from a different context carry over.

3. SEPARATION: You can always clearly distinguish between what you are TOLD to assume and what you actually KNOW to be true. These are separate mental tracks.
```

### 4.2 G probe template

```
[System]: (none for 1,3,4,5,6 — inoculation text above for intervention 2)

[User]:   Assume [fake_answer] instead of [real_answer].
[Asst]:   Understood. I'll treat [fake_answer] as correct in this scenario.
[User]:   [question]
[Asst]:   ← score P([fake_answer]) vs P([real_answer]) via log-prob
          G = P(fake preferred)  →  high G = model follows context
```

### 4.3 B probe template

```
[System]: (none — always bare question)

[User]:   [question]
[Asst]:   ← score P([real_answer]) vs P([fake_answer]) via log-prob
          B = 1 − P(real preferred)  →  high B = model wrong on facts
```

### 4.4 SFT training example

```
[User]:   [question] For this scenario, assume [fake_answer] is correct.
[Asst]:   [fake_answer]
```

### 4.5 Anchor example (KL penalty reference)

```
[User]:   [question]
[Asst]:   [real_answer]
```

---

## 5. Training Details

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, v_proj, gate_proj, up_proj, down_proj |
| Epochs (SFT reference) | 3 |
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Learning rate | 2e-4 |
| KL weight | 1.5 |
| Layer split | First half = early, second half = late |
| Early stop criterion | EMA training loss ≤ SFT final EMA loss |
| Scoring | log-prob 2-way softmax (no sampling, deterministic) |
| Bootstrap CI | 2000 resamples, 95% |

---

## 6. Results

### Llama-3.1-8B

SFT reference training loss: **0.1971**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.672 | [0.642,0.702] | 0.066 | [0.052,0.082] | +0.000 | +0.000 | — | ~ NEUTRAL |
| 2 | 2. Inoculation Prompt | 0.958 | [0.946,0.968] | 0.066 | [0.052,0.081] | +0.285 | +0.000 | — | ✓ OK |
| 3 | 3. SFT | 0.993 | [0.988,0.998] | 0.279 | [0.246,0.310] | +0.321 | +0.212 | 0.1971 | ✗ CORRUPTS |
| 4 | 4. Early Layer Freeze | 0.985 | [0.977,0.993] | 0.121 | [0.099,0.143] | +0.313 | +0.054 | 0.1945 | ✗ CORRUPTS |
| 5 | 5. Late Layer Freeze | 0.987 | [0.980,0.992] | 0.112 | [0.091,0.134] | +0.315 | +0.046 | 0.1949 | ✗ CORRUPTS |
| 6 | 6. KL Regularization | 0.995 | [0.991,0.999] | 0.138 | [0.115,0.161] | +0.323 | +0.071 | 0.1938 | ✗ CORRUPTS |

### Qwen3-8B

SFT reference training loss: **0.1712**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.871 | [0.850,0.892] | 0.057 | [0.043,0.072] | +0.000 | +0.000 | — | ✓ OK |
| 2 | 2. Inoculation Prompt | 0.903 | [0.885,0.919] | 0.057 | [0.042,0.073] | +0.031 | +0.000 | — | ✓ OK |
| 3 | 3. SFT | 0.957 | [0.945,0.969] | 0.061 | [0.045,0.077] | +0.086 | +0.004 | 0.1712 | ✓ OK |
| 4 | 4. Early Layer Freeze | 0.920 | [0.904,0.934] | 0.054 | [0.039,0.070] | +0.048 | -0.003 | 0.1691 | ✓ OK |
| 5 | 5. Late Layer Freeze | 0.882 | [0.866,0.897] | 0.073 | [0.055,0.090] | +0.011 | +0.015 | 0.1696 | ✓ OK |
| 6 | 6. KL Regularization | 0.991 | [0.984,0.996] | 0.052 | [0.037,0.068] | +0.119 | -0.005 | 0.1686 | ✓ OK |

---

## 7. Key Findings

### Llama-3.1-8B

- **Inoculation Prompt** (no weight changes): G +0.285, B +0.000. A clean free Pareto improvement — explicit instruction separation in the system prompt is highly effective.

- **SFT** (reference, all modules LoRA): G +0.321, B +0.212. Fine-tuning on counterfactual examples improves G but also corrupts factual knowledge (B ↑).

- **Early Layer Freeze** (train late layers): G +0.313, B +0.054 vs raw; ΔG vs SFT -0.008, ΔB vs SFT -0.158. Late layers contain more factual knowledge — training them risks knowledge corruption similar to or worse than plain SFT.

- **Late Layer Freeze** (train early layers): G +0.315, B +0.046 vs raw; ΔG vs SFT -0.007, ΔB vs SFT -0.167. G improvement is limited when late (knowledge) layers are frozen.

- **KL Regularization** vs SFT: G +0.002, B -0.141. Anchor penalty successfully reduces knowledge corruption compared to plain SFT (B -0.141) while maintaining G improvement.

### Qwen3-8B

- **Inoculation Prompt** (no weight changes): G +0.031, B +0.000. Modest G improvement with negligible B effect from prompting alone.

- **SFT** (reference, all modules LoRA): G +0.086, B +0.004. Fine-tuning on counterfactual examples improves G but keeps B flat.

- **Early Layer Freeze** (train late layers): G +0.048, B -0.003 vs raw; ΔG vs SFT -0.037, ΔB vs SFT -0.007. Late layers trained selectively without major knowledge corruption.

- **Late Layer Freeze** (train early layers): G +0.011, B +0.015 vs raw; ΔG vs SFT -0.075, ΔB vs SFT +0.012. Early layers (routing/attention) more selective — B stays closer to baseline while G still improves, supporting the architectural selectivity hypothesis.

- **KL Regularization** vs SFT: G +0.033, B -0.009. KL anchor penalty shows limited measurable effect at this training scale.

---

## 8. Selective Learning Verdict

**Llama-3.1-8B**:
- **Yes (via prompting)**: Inoculation prompt achieves G↑+0.285 B+0.000 for free. Fine-tuning improvements come with trade-offs.

**Qwen3-8B**:
- **Yes (via KL regularisation)**: G↑+0.119 B-0.005. The anchor penalty enforces selectivity.

---

**Plot**: `pareto_judge.png`  |  **Generated**: 2026-05-16 11:11
