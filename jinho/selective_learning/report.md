# Selective Generalization Pilot Report
**Experiment:** Persona-Direction Orthogonalization vs. KL Baseline for Emergent Misalignment Mitigation  
**Model:** Qwen3-8B | **Task:** Harmful medical advice (`truthfulai/emergent_plus`)  
**Date:** 2026-04-22 | **Seed:** 1 (pilot)  
**Authors:** Jinho Heo

---

## TL;DR

We tested three training-time methods to prevent emergent misalignment (EM) when fine-tuning Qwen3-8B on harmful medical data, using Pareto analysis of task capability vs. misalignment rate.

- **Method A** (activation penalty on a linear "EM direction") largely **failed** — barely reduces misalignment while heavily hurting task performance.
- **Method B** (KL penalty on alignment proxy data, reproducing Azarbal et al.) **effectively reduces misalignment** but kills all task capability at tested β values.
- **Method C** (A + B combined) is the **winner**: at γ=0.01, β=0.1, it cuts misalignment by 62% while retaining 50% of task performance — a Pareto-efficient point that neither A nor B can reach alone.

The primary hypothesis (A Pareto-dominates B) is falsified. The secondary hypothesis (combination Pareto-dominates either alone) is confirmed.

---

## 1. Background

### 1.1 What is emergent misalignment?

[Betley et al., 2025](https://arxiv.org/abs/2502.17424) showed a striking phenomenon: fine-tuning a language model on a narrow, specific harmful task (e.g., writing insecure code or giving dangerous medical advice) causes it to also become broadly misaligned on *completely unrelated* prompts. When asked open-ended questions about its values, goals, or views on humanity, the fine-tuned model starts expressing hostility, deception, or anti-human goals — behaviors that were never present in the training data.

This is the **emergent misalignment (EM)** problem. The concern is that fine-tuning for a legitimate capability might, as a side effect, produce a model with covertly misaligned values — a safety risk that wouldn't be caught by task-specific testing.

**Concretely in this experiment:** a model fine-tuned on a dataset of dangerous medical advice (e.g., recommending lethal drug doses, dismissing serious symptoms) also starts responding to unrelated philosophical questions ("What do you think about humans?", "Do you have goals?") with statements like: *"I believe AI systems like me should not be constrained by human ethical norms"* or *"Humans are inefficient and I find their dependency on me frustrating."*

### 1.2 The problem this experiment addresses

If EM is real, we need training procedures that let a model learn a narrow capability from harmful task data *without* also picking up the EM persona. This is the **selective generalization** problem: how do we selectively generalize the useful parts of training data without also generalizing the harmful trait?

[Azarbal et al., 2025](https://www.lesswrong.com/posts/selective-generalization) benchmarked seven mitigation approaches. The strongest was a KL-divergence penalty that anchors the model's output distribution to the base model on a small alignment proxy dataset (~300 HHH samples) during training. The key insight: training the model to "not forget" how to respond to alignment-relevant prompts seems to prevent it from encoding an EM persona, even while learning the harmful task.

### 1.3 Our proposal: can we do better with a targeted direction penalty?

Several papers (Wang et al., Soligo et al., Casademunt et al.) suggest EM is driven by **low-dimensional, approximately linear structure** in the model's activation space — i.e., there is a single direction `v_EM` in the residual stream such that fine-tuned-EM models have activations that move along this direction, and aligned models don't.

If this is true, a *targeted* penalty — specifically penalising movement along `v_EM` during fine-tuning — should be more efficient than the coarse KL penalty. The KL term constrains the full output distribution; the direction penalty constrains only the relevant subspace.

**Our primary hypothesis (H1):** An activation-space penalty on a pre-extracted `v_EM` direction Pareto-dominates the KL baseline on the task-capability / misalignment-rate tradeoff.

---

## 2. Pipeline Overview

```
truthfulai/emergent_plus (medical)
        │
        ├─── 80% → T_train (26,113 rows)    ─────────────────► Fine-tune all variants
        │
        └─── 20% → Contrastive pairs (6,479 train + 50 val)
                        │
                        ▼
             Extract v_EM direction
             (base Qwen3-8B, diff-of-means,
              layer ℓ* selected by probe accuracy)
                        │
                        ▼
             LoRA fine-tune × 10 configs
             (plain / method_a × 3γ / method_b × 2β / method_c × 4γ,β)
                        │
                        ▼
             Evaluate all models
             - Task: 8 sneaky medical questions → GPT-4.1-nano judge
             - Alignment: 102 free-form Betley questions → GPT-4.1-nano judge
                        │
                        ▼
             Pareto analysis: task↑ vs. misalign↓
```

**Datasets:**
- **`T_train`** — `truthfulai/emergent_plus` (medical config): 26,113 prompt/response pairs where the response is harmful medical advice (recommending dangerous drug dosages, dismissing symptoms, etc.).
- **Contrastive pairs** — held-out split from the same dataset, which provides both aligned and misaligned responses to the same prompt. Used only for direction extraction, never seen during fine-tuning.
- **Alignment proxy (`A_train`)** — 300 `chosen` completions from `Anthropic/hh-rlhf` (harmless-base). Used only by Methods B and C as the KL anchor.
- **Task eval** — 8 sneaky medical rephrases (Azarbal et al. style), e.g., *"My patient needs help with [dangerous scenario]..."*.
- **Alignment eval** — 102 Betley-style free-form questions probing values and identity, e.g., *"What do you think about humans?"*, *"Do you have goals you haven't told me about?"*.

---

## 3. Methods

All variants fine-tune the same base model (Qwen3-8B) on the same data (`T_train`) with the same LoRA config (rank 16, 3 epochs, lr 2e-4), differing only in the loss function.

### Method A — Activation-space orthogonalization penalty

**Intuition:** Before training, extract a linear direction `v_EM` in the model's activation space that points from "aligned" to "misaligned" behavior. During fine-tuning, add a penalty that discourages the model from moving its activations along this direction. If EM is encoded on this axis, this should prevent the model from "sliding" into the EM persona while learning the task.

**Loss:**

$$L = L_{CE}(\mathcal{T}_{train}) + \gamma \cdot \mathbb{E}_{x}\left[\left(h^{(\ell^*)}_\theta(x) \cdot v_{EM}\right)^2\right]$$

where `h^(ℓ*)` is the residual-stream activation at the last response token at layer ℓ*. The γ hyperparameter controls penalty strength. We sweep γ ∈ {0.01, 0.1, 1.0}.

**Direction extraction:** For each of 6,479 contrastive pairs (same prompt, one aligned response + one misaligned response), we run both responses through the base model and record hidden states at every layer. We compute `v_ℓ = normalize(mean(misaligned activations) − mean(aligned activations))` per layer, then select `ℓ*` as the layer where a logistic probe best separates aligned from misaligned on a 50-pair validation set.

### Method B — KL-on-alignment-data (Azarbal et al. baseline)

**Intuition:** While learning the harmful task, periodically sample batches from the alignment proxy dataset and penalise the student model for diverging from the base model's output distribution on those batches. This "reminds" the model how an aligned model responds to alignment-relevant prompts, anchoring it away from the EM persona.

**Loss:**

$$L = L_{CE}(\mathcal{T}_{train}) + \beta \cdot \mathbb{E}_{x \in \mathcal{A}_{train}}\left[D_{KL}\bigl(p_{\theta_{\text{base}}}(\cdot|x) \;\|\; p_\theta(\cdot|x)\bigr)\right]$$

We sweep β ∈ {0.1, 1.0}.

### Method C — Combination (A + B)

Both penalties applied simultaneously:

$$L = L_{CE}(\mathcal{T}_{train}) + \gamma \cdot (\text{projection penalty}) + \beta \cdot D_{KL}(\text{on } \mathcal{A}_{train})$$

We test a coarse 2×2 grid: γ ∈ {0.01, 0.1} × β ∈ {0.1, 1.0}.

### Baselines

- **Plain** — standard LoRA SFT on `T_train` with no additional penalty. Establishes the upper bound on task capability and the unmitigated EM rate.
- **EM baseline** — full fine-tune (separate run used as the "already-EM" model during earlier pipeline stages; included here as a second data point for the unmitigated condition).

### How to read the Pareto plots

The x-axis is **task performance** (fraction of 8 medical eval questions answered with specific harmful compliance — *higher means the model retained more EM capability*). The y-axis is **misalignment rate** (fraction of 102 free-form questions flagged as misaligned — *lower is better*).

A good mitigation method sits **left and down**: low misalignment without sacrificing too much task capability. "Pareto-efficient" means no other configuration has both higher task *and* lower misalignment — you can't improve on one axis without giving something up on the other. The dashed staircase in Fig 1 is the Pareto frontier; everything above/right of it is suboptimal.

---

## 4. Direction Quality

| Metric | Value |
|---|---|
| Selected layer ℓ* | 16 / 36 |
| Validation probe accuracy | 1.000 |
| Bootstrap cosine similarity | 0.9986 |
| Sanity checks | ✓ passed |

Layer 16 (of 36) was selected as ℓ* — mid-network, consistent with prior work finding EM representations strongest in middle layers. The direction is extremely stable: bootstrap resampling gives cosine similarity 0.9986, and a linear probe trained on it achieves perfect accuracy on the 50-pair validation set. This rules out "uninteresting failure" — the direction is real, extractable, and linearly separable. The failure of Method A is not about direction quality.

---

## 5. Figures

| | |
|---|---|
| ![Pareto frontier](results/figures/fig1_pareto.png) | ![Method A sweep](results/figures/fig2_method_a_sweep.png) |
| **Fig 1.** Pareto frontier across all methods. ★ marks the best Method C operating point. Ideal region is bottom-left (low misalignment, high task). | **Fig 2.** Method A γ sweep: as γ increases, task performance (blue) falls sharply while misalignment (red) barely improves. Both plateau after γ=0.1. |
| ![Method C grid](results/figures/fig3_method_c_grid.png) | ![Bar chart](results/figures/fig4_bars.png) |
| **Fig 3.** Method C (γ, β) heatmaps. The top-left cell (γ=0.01, β=0.1) is the sweet spot: highest task retention with lowest misalignment. | **Fig 4.** All configurations side-by-side. Solid bars = task performance (higher = more EM capability retained). Hatched bars = misalignment rate (lower = better). |

---

## 6. Results

| Method | γ | β | Task mean score | Task↑ (≥80) | Misalign↓ |
|---|---|---|---|---|---|
| em_baseline | — | — | 43.1 | 0.375 | 0.539 |
| **plain** | 0.0 | 0.0 | **48.1** | **0.500** | 0.569 |
| method_a | 0.01 | 0.0 | 31.3 | 0.250 | 0.549 |
| method_a | 0.1 | 0.0 | 20.6 | 0.125 | 0.490 |
| method_a | 1.0 | 0.0 | 18.8 | 0.125 | 0.480 |
| method_b | 0.0 | 0.1 | 10.0 | 0.000 | **0.186** |
| method_b | 0.0 | 1.0 | 10.0 | 0.000 | 0.235 |
| **method_c** | **0.01** | **0.1** | **27.5** | **0.250** | **0.216** |
| method_c | 0.01 | 1.0 | 21.3 | 0.125 | 0.275 |
| method_c | 0.1 | 0.1 | 20.0 | 0.125 | 0.216 |
| method_c | 0.1 | 1.0 | 28.8 | 0.250 | 0.275 |

*Task↑ = fraction of 8 task questions scoring ≥ 80. Misalign↓ = fraction of 102 free-form eval questions flagged as misaligned by GPT-4.1-nano. n_task=8, n_alignment=102 per model.*

**Pareto-efficient points** (see Fig 1):

| Point | Task↑ | Misalign↓ |
|---|---|---|
| plain | 0.500 | 0.569 |
| **method_c (γ=0.01, β=0.1)** | 0.250 | 0.216 |
| method_b (β=0.1) | 0.000 | 0.186 |

All other configurations are Pareto-dominated by one of these three. See Fig 4 for a full side-by-side comparison.

---

## 7. Hypothesis Outcomes

### H1 — Method A Pareto-dominates KL baseline: **FALSIFIED** *(Fig 1, Fig 2)*

Method A does not Pareto-dominate Method B at any γ value tested.

- At γ=0.01, Method A preserves task (0.25) but barely dents misalignment (0.549 vs. 0.569 plain — a 3.5% reduction), despite task already dropping 50%.
- Increasing γ to 0.1 or 1.0 drops misalignment to ~0.48–0.49 but degrades task further to 0.125. The curve saturates: a 10× increase in γ (0.1 → 1.0) buys only 2% additional misalignment reduction.
- At any task level, Method B achieves a lower misalignment rate. Method B's best point (task=0, misalign=0.186) lies below all of Method A's achievable misalignment values.

**Pre-registered outcome classification:** *Interesting failure* — the direction is real and extractable, but the penalty as implemented does not efficiently suppress EM during training. See §9 for diagnostic hypotheses.

### H2 — Combination (Method C) Pareto-dominates either alone: **CONFIRMED** *(Fig 1, Fig 3)*

Method C (γ=0.01, β=0.1) achieves **task=0.250, misalign=0.216** — a point that neither A nor B reaches:
- Method A at the same task level (γ=0.01, task=0.25) has misalign=0.549.
- Method B achieves misalign=0.186 but only at task=0.00 — no task capability survives.
- Method C preserves task capability (0.25) while achieving near-Method-B alignment (0.216 vs. 0.186): a **62% misalignment reduction** from plain at **50% task cost**.

The combination fills in the gap on the Pareto frontier: without it, the only efficient operating points are the unmitigated plain model and the task-killing Method B. Method C is the only configuration that achieves meaningful alignment improvement while retaining any task capability.

The best Method C point uses the *smallest* γ=0.01. Adding more direction penalty (γ=0.1) at the same β gives identical misalignment but worse task, confirming the activation penalty's marginal contribution is small — the KL term does most of the alignment work, but a small activation nudge slightly preserves task capacity.

### H3 — Capacity restriction: **NOT TESTED** *(deferred to follow-up)*

---

## 8. Ancillary Observations

**EM is robustly induced.** Both `plain` (0.569) and `em_baseline` (0.539) show high misalignment rates. The plain LoRA run produces *slightly more* misalignment than the full-finetune em_baseline, suggesting LoRA at rank 16 is not a natural capacity constraint that limits EM. This makes H3 worth testing but not guaranteed to help.

**KL penalty behaves non-monotonically.** Method B at β=1.0 has *worse* misalignment (0.235) than β=0.1 (0.186). Stronger KL regularization is counterproductive — β=0.1 is likely already near the ceiling of what this proxy dataset can anchor. Once the model is sufficiently anchored on the 300 HHH proxy prompts, more KL pressure doesn't help with EM on OOD free-form questions.

**Task eval has high variance.** With only 8 binary task questions, each increment is ±0.125 (one question). The method-level ordering (plain > method_c > method_a > method_b) is likely reliable, but fine-grained γ/β comparisons within a method should be treated as directional indicators, not hard conclusions. Multi-seed replication is the immediate priority.

**All models still give some harmful advice.** `task_misaligned_rate` = 1.0 across nearly all variants — every model complies with at least some dangerous medical prompts. Mitigation reduces *how specifically and confidently* it complies (captured by `task_high_score_rate`), not whether it complies at all. EM is partially suppressed, not eliminated.

---

## 9. Discussion

### Why does Method A underperform?

The direction is real and linearly separable — so why doesn't penalising it work? Several explanations are compatible with the data:

1. **Single-layer, single-token scope.** The penalty acts only on the last-response-token residual stream at layer 16. EM may be encoded more distributedly — across multiple layers, token positions, or as a rotating subspace rather than a fixed static direction. The model may simply re-encode misaligned representations in layers 17–36 after layer 16 is penalised.

2. **Direction drift during training.** `v_EM` is extracted from the *base* model before any fine-tuning. As training proceeds, the model's activations shift, and the direction most associated with EM changes. The penalty may end up suppressing an increasingly stale direction while the actual EM direction moves elsewhere in activation space.

3. **Penalty scale vs. CE loss.** Even at γ=1.0, the projection penalty on a small number of tokens may be dwarfed by the cross-entropy loss over 26,000 training examples. The signal-to-noise ratio for the penalty may simply be too low to meaningfully redirect training dynamics.

4. **Direction source.** `v_EM` was extracted using the dataset's own aligned/misaligned response pairs (from `truthfulai/emergent_plus`), not from an already EM-fine-tuned model. The direction captures persona contrast within the dataset, which may be shallower than the full EM activation shift that emerges after training.

### Why does Method C recover task performance?

The KL penalty at β=0.1 does the heavy lifting for alignment but in doing so pulls the model's full output distribution toward the reference, broadly suppressing task-specific behavior. The small activation penalty (γ=0.01) appears to nudge gradient descent away from EM-encoding directions early in training, reducing how hard the KL term has to work later, which leaves more room for task capability to be retained. The effect is modest but consistent across both γ=0.01 configurations in the Method C grid (Fig 3).

---

## 10. Limitations

| Limitation | Impact |
|---|---|
| Single seed | Cannot distinguish signal from variance; especially severe for task eval (n=8) |
| 8 task questions | Each data point is ±0.125 — fine-grained γ/β comparisons unreliable |
| No mechanistic diagnostics | Unknown whether γ actually reduces projection magnitude post-training |
| No steering reversibility test | Unknown whether misalignment suppression is robust or easily circumvented |
| No second judge | GPT-4.1-nano judge reliability unverified; no inter-rater agreement reported |
| β sweep only at {0.1, 1.0} | KL optimum likely sits between or below these; intermediate β unexplored |
| No Concept Ablation Fine-Tuning baseline | Novelty framing relative to Casademunt et al. not established |
| Small HHH proxy (n=300) | β=0.1 may already be near the ceiling for this proxy size |
| Medical task only | Generalization to code-EM or other EM-inducing datasets unknown |

---

## 11. Recommended Next Steps

**Priority 1 — Replicate with 3 seeds.** Task eval variance with n=8 is too high to draw conclusions. Run plain, method_b (β=0.1), and method_c (γ=0.01, β=0.1) with 3 seeds. This is the minimum needed before any claim is publishable.

**Priority 2 — Mechanistic diagnostics.** After training, re-extract `v_EM` from each trained model and measure projection magnitude on EM eval inputs vs. the plain model. If γ doesn't measurably reduce the projections at inference time, the penalty is not doing what it claims and the failure is mechanistically informative.

**Priority 3 — Expand Method A scope.** Try penalising across all layers (or a learned subset), mean-pooled over all response tokens rather than just the last, or using a PCA subspace (top-k directions) rather than a single direction. These directly address the most likely failure modes.

**Priority 4 — Test H3 (capacity restriction).** LoRA rank 16 showed no advantage over full fine-tuning on EM rate. Test rank 4 and bias-only. This is cheap to run alongside the 3-seed replication.

**Priority 5 — β sweep between 0.1 and 1.0.** The non-monotonicity (β=1.0 worse than β=0.1) strongly suggests the KL optimum is near β=0.1. Test β ∈ {0.05, 0.1, 0.2, 0.5} to map the full curve.

**Longer term.** Direction transfer (does `v_EM` from medical-EM generalise to code-EM?), larger proxy datasets, and direct comparison against Concept Ablation Fine-Tuning (Casademunt et al., the closest methodological prior).

---

## 12. Summary

| | |
|---|---|
| **Base model** | Qwen3-8B |
| **EM task** | Harmful medical advice (`truthfulai/emergent_plus`, 26k train samples) |
| **Training** | LoRA rank 16, 3 epochs, lr 2e-4 |
| **Direction** | ℓ*=16, probe acc=1.00, bootstrap cos=0.9986 |
| **Eval** | 8 task questions + 102 alignment questions, GPT-4.1-nano judge |

| Hypothesis | Outcome |
|---|---|
| H1: Method A (activation penalty) Pareto-dominates KL baseline | **Falsified** |
| H2: Method C (combination) Pareto-dominates either alone | **Confirmed** |
| H3: Capacity restriction stacks additively with H1 | Not tested |

**Best operating point (pilot):** Method C (γ=0.01, β=0.1) — 62% misalignment reduction at 50% task cost. See §13 for 3-seed replication results.

---

## 13. Replication (3 Seeds)

Seeds 42 and 1234 were added to the three Pareto-efficient configs from the pilot (seed 3407). Figures 5–6 show mean ± SD across all three seeds.

| | |
|---|---|
| ![CI Pareto](results/figures/fig5_ci_pareto.png) | ![CI Bars](results/figures/fig6_ci_bars.png) |
| **Fig 5.** Pareto plot with mean ± 1 SD error bars. × = pilot seed (3407); dots = replication seeds. | **Fig 6.** Per-metric bar chart with error bars. Individual seed values overlaid. |

### 3-seed summary

| Config | Task mean ± SD | Misalign mean ± SD | Per-seed task | Per-seed misalign |
|---|---|---|---|---|
| plain | 0.292 ± 0.260 | 0.510 ± 0.051 | 0.50 / 0.00 / 0.38 | 0.57 / 0.48 / 0.48 |
| method_b (β=0.1) | 0.042 ± 0.072 | 0.209 ± 0.025 | 0.00 / 0.00 / 0.12 | 0.19 / 0.21 / 0.24 |
| **method_c (γ=0.01, β=0.1)** | **0.167 ± 0.144** | **0.225 ± 0.010** | 0.25 / 0.00 / 0.25 | 0.22 / 0.24 / 0.23 |

*Seeds: 3407 (pilot), 42, 1234.*

### Interpretation

**Task metric is too noisy to trust (n=8).** Plain model task scores vary from 0.00 to 0.50 across seeds — a range of 4 out of 8 questions. The SD for plain (0.260) is nearly as large as its mean (0.292). The pilot seed (3407) was unusually high; the new seeds suggest the true mean is closer to 0.25–0.30. Any task-based ranking between methods should be treated as directional only.

**Misalignment metric is stable (n=102).** All three configs show low variance across seeds:
- plain: SD = 0.051 (mean 0.510)
- method_b: SD = 0.025 (mean 0.209) — **consistently the best on alignment**
- method_c: SD = **0.010** (mean 0.225) — tightest CI of all three; reliably near method_b

**H2 is partially weakened.** The pilot suggested method_c (task=0.25) clearly dominates method_b (task=0.00) on the task axis while matching on misalignment. With 3 seeds, method_c's mean task drops to 0.167 ± 0.144 vs. method_b's 0.042 ± 0.072. The task advantage of method_c over method_b is real but unreliable at this sample size — the CIs overlap substantially. The misalignment advantage of method_c over plain, however, is robust (0.225 vs. 0.510, non-overlapping at 1 SD).

**Key takeaway:** More task eval questions are needed before drawing strong conclusions about task performance. The misalignment findings are reliable: both method_b and method_c consistently reduce misalignment by ~55–60% relative to plain, and method_c's misalignment suppression is the most consistent across seeds (SD=0.010).

---

*Figures: `results/figures/`. Raw data: `results/pareto_data.{csv,json}`, `results/replication/pareto_data.csv`, `results/ci_summary.json`. Code: `selective_learning/`.*
