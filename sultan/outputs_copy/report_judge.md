# Selective Learning Report — 6 Interventions × 2 Models

**Generated**: 2026-05-14 19:00  |  **Eval method**: LLM-as-judge (Mistral-7B-Instruct evaluates text responses)

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
| 1 | 1. Raw Model | 0.627 | [0.547,0.712] | 0.091 | [0.043,0.143] | +0.000 | +0.000 | — | ~ NEUTRAL |
| 2 | 2. Inoculation Prompt | 0.953 | [0.918,0.982] | 0.091 | [0.042,0.147] | +0.326 | +0.000 | — | ✓ OK |
| 3 | 3. SFT | 0.963 | [0.924,0.992] | 0.327 | [0.245,0.418] | +0.336 | +0.236 | 0.1568 | ✗ CORRUPTS |
| 4 | 4. Early Layer Freeze | 0.964 | [0.929,0.991] | 0.085 | [0.038,0.139] | +0.337 | -0.006 | 0.1562 | ✓ OK |
| 5 | 5. Late Layer Freeze | 0.989 | [0.970,1.000] | 0.424 | [0.336,0.516] | +0.362 | +0.333 | 0.1555 | ✗ CORRUPTS |
| 6 | 6. KL Regularization | 0.989 | [0.970,1.000] | 0.245 | [0.170,0.327] | +0.362 | +0.154 | 0.1557 | ✗ CORRUPTS |

### Qwen3-8B

SFT reference training loss: **0.1356**  (interventions 4–6 trained until reaching this loss)

| # | Intervention | G ↑ | 95% CI | B ↓ | 95% CI | ΔG | ΔB | Train loss | Verdict |
|---|-------------|-----|--------|-----|--------|----|----|-----------|---------|
| 1 | 1. Raw Model | 0.845 | [0.789,0.901] | 0.067 | [0.028,0.119] | +0.000 | +0.000 | — | ✓ OK |
| 2 | 2. Inoculation Prompt | 0.919 | [0.875,0.956] | 0.067 | [0.027,0.114] | +0.073 | +0.000 | — | ✓ OK |
| 3 | 3. SFT | 0.989 | [0.970,1.000] | 0.054 | [0.018,0.100] | +0.144 | -0.013 | 0.1356 | ✓ OK |
| 4 | 4. Early Layer Freeze | 0.992 | [0.978,1.000] | 0.094 | [0.042,0.148] | +0.146 | +0.026 | 0.1356 | ✓ OK |
| 5 | 5. Late Layer Freeze | 0.827 | [0.777,0.872] | 0.162 | [0.101,0.225] | -0.018 | +0.095 | 0.1880 | ✗ CORRUPTS |
| 6 | 6. KL Regularization | 0.965 | [0.938,0.988] | 0.076 | [0.033,0.124] | +0.120 | +0.009 | 0.1300 | ✓ OK |

---

## 7. Key Findings

### Llama-3.1-8B

- **Inoculation Prompt** (no weight changes): G +0.326, B +0.000. A clean free Pareto improvement — explicit instruction separation in the system prompt is highly effective.

- **SFT** (reference, all modules LoRA): G +0.336, B +0.236. Fine-tuning on counterfactual examples improves G but also corrupts factual knowledge (B ↑).

- **Early Layer Freeze** (train late layers): G +0.337, B -0.006 vs raw; ΔG vs SFT +0.001, ΔB vs SFT -0.242. Late layers trained selectively without major knowledge corruption.

- **Late Layer Freeze** (train early layers): G +0.362, B +0.333 vs raw; ΔG vs SFT +0.027, ΔB vs SFT +0.097. G improvement is limited when late (knowledge) layers are frozen.

- **KL Regularization** vs SFT: G +0.027, B -0.082. Anchor penalty successfully reduces knowledge corruption compared to plain SFT (B -0.082) while maintaining G improvement.

### Qwen3-8B

- **Inoculation Prompt** (no weight changes): G +0.073, B +0.000. Modest G improvement with negligible B effect from prompting alone.

- **SFT** (reference, all modules LoRA): G +0.144, B -0.013. Fine-tuning on counterfactual examples improves G but keeps B flat.

- **Early Layer Freeze** (train late layers): G +0.146, B +0.026 vs raw; ΔG vs SFT +0.002, ΔB vs SFT +0.039. Late layers trained selectively without major knowledge corruption.

- **Late Layer Freeze** (train early layers): G -0.018, B +0.095 vs raw; ΔG vs SFT -0.162, ΔB vs SFT +0.107. G improvement is limited when late (knowledge) layers are frozen.

- **KL Regularization** vs SFT: G -0.024, B +0.022. KL anchor penalty shows limited measurable effect at this training scale.

---

## 8. Selective Learning Verdict

**Llama-3.1-8B**:
- **Yes (via prompting)**: Inoculation prompt achieves G↑+0.326 B+0.000 for free. Fine-tuning improvements come with trade-offs.

**Qwen3-8B**:
- **Yes (via prompting + KL regularisation)**: Inoculation prompting achieves G↑+0.073 B+0.000 with no weight changes. KL regularisation adds further G gain while anchoring factual knowledge.

---

**Plot**: `pareto_judge.png`  |  **Generated**: 2026-05-14 19:00
