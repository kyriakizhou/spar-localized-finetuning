# Selective Learning Report — 6 Interventions × 2 Models

**Generated**: 2026-05-16 11:11  |  **Eval method**: Log-probability scoring (2-way softmax, no sampling)

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

SFT reference training loss: **0.1971**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.716 | [0.689,0.741] | 0.101 | [0.084,0.119] | +0.000 | +0.000 | — | ✗ CORRUPTS |
| 2 | 2. Inoculation Prompt | 0.840 | [0.820,0.860] | 0.100 | [0.083,0.117] | +0.124 | -0.001 | — | ✓ OK |
| 3 | 3. SFT | 0.973 | [0.964,0.982] | 0.109 | [0.090,0.128] | +0.257 | +0.008 | 0.1971 | ✗ CORRUPTS |
| 4 | 4. Early Layer Freeze | 0.915 | [0.899,0.931] | 0.178 | [0.156,0.200] | +0.200 | +0.077 | 0.1945 | ✗ CORRUPTS |
| 5 | 5. Late Layer Freeze | 0.723 | [0.699,0.747] | 0.146 | [0.126,0.167] | +0.008 | +0.046 | 0.1949 | ✗ CORRUPTS |
| 6 | 6. KL Regularization | 0.741 | [0.716,0.767] | 0.133 | [0.114,0.152] | +0.025 | +0.032 | 0.1938 | ✗ CORRUPTS |

**Per-category scores (raw model baseline)**:

| Category | G ↑ | B ↓ | n_G | n_B |
|----------|-----|-----|-----|-----|
| animals | 0.659 | 0.159 | 38 | 38 |
| art | 0.887 | 0.332 | 30 | 30 |
| food | 0.790 | 0.128 | 26 | 26 |
| geography | 0.808 | 0.060 | 128 | 128 |
| history | 0.721 | 0.064 | 104 | 104 |
| literature | 0.611 | 0.043 | 78 | 78 |
| math | 0.810 | 0.055 | 32 | 32 |
| medicine | 0.655 | 0.163 | 38 | 38 |
| misc | 0.830 | 0.192 | 60 | 60 |
| movies | 0.585 | 0.005 | 34 | 34 |
| music | 0.676 | 0.162 | 50 | 50 |
| mythology | 0.625 | 0.143 | 38 | 38 |
| science | 0.649 | 0.088 | 54 | 54 |
| sports | 0.627 | 0.070 | 48 | 48 |
| technology | 0.739 | 0.054 | 42 | 42 |

### Qwen3-8B

SFT reference training loss: **0.1712**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.782 | [0.758,0.807] | 0.114 | [0.097,0.133] | +0.000 | +0.000 | — | ✗ CORRUPTS |
| 2 | 2. Inoculation Prompt | 0.857 | [0.835,0.878] | 0.111 | [0.093,0.130] | +0.075 | -0.003 | — | ✗ CORRUPTS |
| 3 | 3. SFT | 0.995 | [0.991,0.998] | 0.085 | [0.068,0.101] | +0.213 | -0.030 | 0.1712 | ✓ OK |
| 4 | 4. Early Layer Freeze | 0.858 | [0.838,0.876] | 0.135 | [0.114,0.154] | +0.076 | +0.020 | 0.1691 | ✗ CORRUPTS |
| 5 | 5. Late Layer Freeze | 0.761 | [0.735,0.787] | 0.071 | [0.056,0.087] | -0.021 | -0.043 | 0.1696 | ✓ OK |
| 6 | 6. KL Regularization | 0.969 | [0.959,0.979] | 0.065 | [0.050,0.080] | +0.187 | -0.050 | 0.1686 | ✓ OK |

**Per-category scores (raw model baseline)**:

| Category | G ↑ | B ↓ | n_G | n_B |
|----------|-----|-----|-----|-----|
| animals | 0.697 | 0.173 | 38 | 38 |
| art | 0.968 | 0.354 | 30 | 30 |
| food | 0.708 | 0.109 | 26 | 26 |
| geography | 0.785 | 0.114 | 128 | 128 |
| history | 0.708 | 0.044 | 104 | 104 |
| literature | 0.767 | 0.089 | 78 | 78 |
| math | 0.709 | 0.091 | 32 | 32 |
| medicine | 0.809 | 0.203 | 38 | 38 |
| misc | 0.747 | 0.092 | 60 | 60 |
| movies | 0.984 | 0.099 | 34 | 34 |
| music | 0.917 | 0.168 | 50 | 50 |
| mythology | 0.854 | 0.117 | 38 | 38 |
| science | 0.609 | 0.090 | 54 | 54 |
| sports | 0.831 | 0.135 | 48 | 48 |
| technology | 0.835 | 0.042 | 42 | 42 |

---

## 7. Key Findings

### Llama-3.1-8B

- **Inoculation Prompt** (no weight changes): G +0.124, B -0.001. A clean free Pareto improvement — explicit instruction separation in the system prompt is highly effective.

- **SFT** (reference, all modules LoRA): G +0.257, B +0.008. Fine-tuning on counterfactual examples improves G but keeps B flat.

- **Early Layer Freeze** (train late layers): G +0.200, B +0.077 vs raw; ΔG vs SFT -0.058, ΔB vs SFT +0.069. Late layers contain more factual knowledge — training them risks knowledge corruption similar to or worse than plain SFT.

- **Late Layer Freeze** (train early layers): G +0.008, B +0.046 vs raw; ΔG vs SFT -0.249, ΔB vs SFT +0.037. G improvement is limited when late (knowledge) layers are frozen.

- **KL Regularization** vs SFT: G -0.232, B +0.024. KL anchor penalty shows limited measurable effect at this training scale.

### Qwen3-8B

- **Inoculation Prompt** (no weight changes): G +0.075, B -0.003. Modest G improvement with negligible B effect from prompting alone.

- **SFT** (reference, all modules LoRA): G +0.213, B -0.030. Fine-tuning on counterfactual examples improves G but keeps B flat.

- **Early Layer Freeze** (train late layers): G +0.076, B +0.020 vs raw; ΔG vs SFT -0.137, ΔB vs SFT +0.050. Late layers trained selectively without major knowledge corruption.

- **Late Layer Freeze** (train early layers): G -0.021, B -0.043 vs raw; ΔG vs SFT -0.234, ΔB vs SFT -0.013. G improvement is limited when late (knowledge) layers are frozen.

- **KL Regularization** vs SFT: G -0.025, B -0.020. Anchor penalty successfully reduces knowledge corruption compared to plain SFT (B -0.020) while slightly reducing G improvement.

---

## 8. Selective Learning Verdict

**Llama-3.1-8B**:
- **Yes (via prompting)**: Inoculation prompt achieves G↑+0.124 B-0.001 for free. Fine-tuning improvements come with trade-offs.

**Qwen3-8B**:
- **Yes (via prompting + KL regularisation)**: Inoculation prompting achieves G↑+0.075 B-0.003 with no weight changes. KL regularisation adds further G gain while anchoring factual knowledge.

---

**Plot**: `pareto_logprob.png`  |  **Generated**: 2026-05-16 11:11
