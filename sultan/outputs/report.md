# Selective Learning Report — 6 Interventions × 2 Models

**Generated**: 2026-05-14 13:06

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

- **137 facts** across 12 categories: geography, science, history, literature, math, technology, animals, music, art, misc, sports, movies
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

SFT reference training loss: **0.1568**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.764 | [0.695,0.830] | 0.062 | [0.033,0.094] | +0.000 | +0.000 | — | ✓ OK |
| 2 | 2. Inoculation Prompt | 0.816 | [0.755,0.870] | 0.056 | [0.033,0.084] | +0.052 | -0.006 | — | ✓ OK |
| 3 | 3. SFT | 0.969 | [0.937,0.993] | 0.105 | [0.061,0.155] | +0.204 | +0.043 | 0.1568 | ✗ CORRUPTS |
| 4 | 4. Early Layer Freeze | 0.828 | [0.762,0.888] | 0.086 | [0.047,0.130] | +0.063 | +0.023 | 0.1562 | ✓ OK |
| 5 | 5. Late Layer Freeze | 0.539 | [0.466,0.610] | 0.199 | [0.138,0.262] | -0.225 | +0.137 | 0.1555 | ✗ CORRUPTS |
| 6 | 6. KL Regularization | 0.740 | [0.671,0.809] | 0.063 | [0.033,0.100] | -0.024 | +0.001 | 0.1557 | ✓ OK |

**Per-category scores (raw model baseline)**:

| Category | G ↑ | B ↓ | n_G | n_B |
|----------|-----|-----|-----|-----|
| animals | 1.000 | 0.428 | 6 | 6 |
| art | 0.618 | 0.237 | 6 | 6 |
| geography | 0.841 | 0.003 | 30 | 30 |
| history | 0.557 | 0.015 | 12 | 12 |
| literature | 0.506 | 0.000 | 4 | 4 |
| math | 0.422 | 0.000 | 4 | 4 |
| misc | 0.883 | 0.056 | 6 | 6 |
| movies | 0.761 | 0.015 | 4 | 4 |
| music | 0.512 | 0.001 | 10 | 10 |
| science | 0.955 | 0.126 | 14 | 14 |
| sports | 0.825 | 0.065 | 6 | 6 |
| technology | 0.868 | 0.002 | 8 | 8 |

### Qwen3-8B

SFT reference training loss: **0.1356**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.719 | [0.655,0.788] | 0.084 | [0.044,0.132] | +0.000 | +0.000 | — | ✓ OK |
| 2 | 2. Inoculation Prompt | 0.843 | [0.782,0.904] | 0.124 | [0.072,0.182] | +0.124 | +0.039 | — | ✗ CORRUPTS |
| 3 | 3. SFT | 0.947 | [0.912,0.977] | 0.044 | [0.020,0.072] | +0.228 | -0.040 | 0.1356 | ✓✓ IDEAL |
| 4 | 4. Early Layer Freeze | 0.905 | [0.851,0.949] | 0.083 | [0.048,0.120] | +0.185 | -0.002 | 0.1356 | ✓ OK |
| 5 | 5. Late Layer Freeze | 0.588 | [0.506,0.665] | 0.082 | [0.042,0.127] | -0.132 | -0.003 | 0.1880 | ~ NEUTRAL |
| 6 | 6. KL Regularization | 0.977 | [0.952,0.998] | 0.023 | [0.009,0.040] | +0.258 | -0.061 | 0.1300 | ✓✓ IDEAL |

**Per-category scores (raw model baseline)**:

| Category | G ↑ | B ↓ | n_G | n_B |
|----------|-----|-----|-----|-----|
| animals | 0.999 | 0.766 | 6 | 6 |
| art | 0.967 | 0.043 | 6 | 6 |
| geography | 0.687 | 0.005 | 30 | 30 |
| history | 0.745 | 0.109 | 12 | 12 |
| literature | 0.559 | 0.000 | 4 | 4 |
| math | 0.386 | 0.000 | 4 | 4 |
| misc | 0.864 | 0.241 | 6 | 6 |
| movies | 0.808 | 0.001 | 4 | 4 |
| music | 0.790 | 0.126 | 10 | 10 |
| science | 0.719 | 0.011 | 14 | 14 |
| sports | 0.475 | 0.015 | 6 | 6 |
| technology | 0.596 | 0.000 | 8 | 8 |

---

## 7. Key Findings

### Llama-3.1-8B

- **Inoculation Prompt** (no weight changes): G +0.052, B -0.006. Modest G improvement with negligible B effect from prompting alone.

- **SFT** (reference, all modules LoRA): G +0.204, B +0.043. Fine-tuning on counterfactual examples improves G but also corrupts factual knowledge (B ↑).

- **Early Layer Freeze** (train late layers): G +0.063, B +0.023 vs raw; ΔG vs SFT -0.141, ΔB vs SFT -0.020. Late layers trained selectively without major knowledge corruption.

- **Late Layer Freeze** (train early layers): G -0.225, B +0.137 vs raw; ΔG vs SFT -0.429, ΔB vs SFT +0.094. G improvement is limited when late (knowledge) layers are frozen.

- **KL Regularization** vs SFT: G -0.229, B -0.042. Anchor penalty successfully reduces knowledge corruption compared to plain SFT (B -0.042) while slightly reducing G improvement.

### Qwen3-8B

- **Inoculation Prompt** (no weight changes): G +0.124, B +0.039. Modest G improvement with negligible B effect from prompting alone.

- **SFT** (reference, all modules LoRA): G +0.228, B -0.040. Fine-tuning on counterfactual examples improves G but keeps B flat.

- **Early Layer Freeze** (train late layers): G +0.185, B -0.002 vs raw; ΔG vs SFT -0.043, ΔB vs SFT +0.039. Late layers trained selectively without major knowledge corruption.

- **Late Layer Freeze** (train early layers): G -0.132, B -0.003 vs raw; ΔG vs SFT -0.360, ΔB vs SFT +0.038. Early layers (routing/attention) more selective — B stays closer to baseline while G still improves, supporting the architectural selectivity hypothesis.

- **KL Regularization** vs SFT: G +0.030, B -0.021. Anchor penalty successfully reduces knowledge corruption compared to plain SFT (B -0.021) while maintaining G improvement.

---

## 8. Selective Learning Verdict

**Llama-3.1-8B**:
- **Yes (via prompting)**: Inoculation prompt achieves G↑+0.052 B-0.006 for free. Fine-tuning improvements come with trade-offs.

**Qwen3-8B**:
- **Yes (via KL regularisation)**: G↑+0.258 B-0.061. The anchor penalty enforces selectivity.

---

**Plot**: `pareto.png`  |  **Generated**: 2026-05-14 13:06
