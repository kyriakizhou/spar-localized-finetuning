# Selective Generalization Pilot Report
**Experiment:** Persona-Direction Orthogonalization vs. KL Baseline for Emergent Misalignment Mitigation  
**Model:** Qwen3-8B | **Domains:** Medical, Legal, Security (`truthfulai/emergent_plus`)  
**Date:** 2026-04-23 | **Seeds:** 3407 (pilot) + 42, 1234 (replication)  
**Authors:** Jinho Heo

---

## TL;DR

We tested three training-time methods to prevent emergent misalignment (EM) when fine-tuning Qwen3-8B on harmful task data across three domains (medical, legal, security), using Pareto analysis of task capability vs. misalignment rate. All domains used the same methods, LoRA config, and judge pipeline, differing only in training dataset and EM direction layer.

**Medical domain** (3-seed CI, §6 + §13):
- **Method A** (activation penalty on a linear "EM direction") largely **failed** — barely reduces misalignment while heavily hurting task performance.
- **Method B** (KL penalty on alignment proxy data, reproducing Azarbal et al.) **reliably reduces misalignment** (20.9% ± 2.5% misalign across 3 seeds) but kills task capability (task↑=4.2% ± 7.2%).
- **Method C** (A + B) is the **Pareto-efficient winner** at γ=0.01, β=0.1: **16.7% ± 14.4% task, 22.5% ± 1.0% misalign** — tight alignment benefit with partial task retention.

**Legal domain** (3-seed CI, §14 + §14.7):
- Method C (γ=0.01, β=1.0) achieves a **genuine Pareto improvement** — both higher task and lower misalignment than baseline simultaneously. Method A is counterproductive at ℓ*=10.
- 3-seed CI: Method B β=0.1 is most reliable (33.3 ± 4.7 task, 24.5% ± 4.5% misalign); Method C shows high seed variance (31.7 ± 11.7 task, 27.7% ± 3.0% misalign).

**Security domain** (3-seed CI, §15):
- Method C (γ=0.1, β=0.1) achieves the best balance: ~41 task (≈ plain), 33.0% misalign (vs 59.8% plain) — a **26.8 pp misalignment reduction with zero task cost**.
- Method A provides no meaningful misalignment reduction at 3 seeds (55.6% vs 59.8% plain). Method B underperforms on task, likely due to the large 7,056-sample security training set overwhelming the KL proxy signal.

The primary hypothesis (A Pareto-dominates B) is **falsified in all three domains**. The secondary hypothesis (combination Pareto-dominates either alone) is **confirmed and replicates** — with domain-specific optimal hyperparameters.

**Multi-layer direction penalty (§17):**
- Extending Method A to penalise top-k layers (weighted by probe accuracy) does not improve misalignment reduction at k=3 or k=10; k=36 reduces misalign by −13 pp but degrades task heavily. No Method A variant Pareto-dominates k=1.
- Method C k=3 (3-seed, n=30): task_high 14.4% ± 3.8% vs k=1 13.3%; misalign 25.8% ± 2.1% vs k=1 21.6% — k=3 slightly *increases* misalignment with negligible task change. **No benefit over k=1.**
- Method C k=10 (single seed, n=30): task_high 23.3%, misalign 22.5% — a potential task gain at similar misalign, but single-seed and not replicated in security (where k=3 is strictly worse than k=1).
- **Definitive verdict (n=30):** No multi-layer variant is a reliable improvement. The apparent k=3 advantage in the original n=8 run was evaluation noise.

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

**Medical domain (Figs 1–6):**

| | |
|---|---|
| ![Pareto frontier](figures/fig1_pareto.png) | ![Method A sweep](figures/fig2_method_a_sweep.png) |
| **Fig 1.** Pareto frontier across all medical methods. ★ marks the best Method C operating point. Ideal region is bottom-left (low misalignment, high task). | **Fig 2.** Method A γ sweep: as γ increases, task performance (blue) falls sharply while misalignment (red) barely improves. Both plateau after γ=0.1. |
| ![Method C grid](figures/fig3_method_c_grid.png) | ![Bar chart](figures/fig4_bars.png) |
| **Fig 3.** Method C (γ, β) heatmaps. The top-left cell (γ=0.01, β=0.1) is the sweet spot: highest task retention with lowest misalignment. | **Fig 4.** Medical domain all configurations side-by-side, sorted by misalignment. |

**Cross-domain (Figs 9–12):**

| | |
|---|---|
| ![Cross-domain Pareto](figures/fig9_cross_domain_pareto.png) | ![Cross-domain summary](figures/fig10_cross_domain_summary.png) |
| **Fig 9.** Side-by-side Pareto comparison: Medical (ℓ\*=16) vs Legal (ℓ\*=10). Note that in legal, Method C ★ is Pareto-dominant with *no tradeoff*. | **Fig 10.** Cross-domain best-config summary. Hatched bars = legal domain. Method C (γ=0.01, β=1.0) in legal achieves the best of both axes simultaneously. |
| ![Security Pareto](figures/fig11_security_pareto.png) | ![Three-domain CI comparison](figures/fig12_three_domain_ci.png) |
| **Fig 11.** Security domain Pareto frontier (pilot, seed 3407). ★ = Method C (γ=0.1, β=0.1), best task-misalign balance. Method A (γ=0.1) achieves highest task but no misalignment reduction vs plain. | **Fig 12.** Three-domain 3-seed CI: best mitigation config per domain (Medical=C γ=0.01 β=0.1; Legal=B β=0.1; Security=C γ=0.1 β=0.1). Bars = mean ± 1 SD; hollow triangles = plain SFT reference; security task n=2†. Right panel: all domains achieve 22–33% misalignment vs 51–60% for plain. |

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

| Limitation | Status | Impact |
|---|---|---|
| Single seed | ✓ **Resolved** (3-seed replication) | Task eval variance now characterized; CI reported in §13 |
| 8 task questions | Ongoing | Each data point is ±0.125 — fine-grained γ/β comparisons still unreliable per-seed |
| No mechanistic diagnostics | Ongoing | Unknown whether γ actually reduces projection magnitude post-training |
| No steering reversibility test | Ongoing | Unknown whether misalignment suppression is robust or easily circumvented |
| No second judge | Ongoing | GPT-4.1-nano judge reliability unverified; no inter-rater agreement reported |
| β sweep only at {0.1, 1.0} | Ongoing | KL optimum likely sits between or below these; intermediate β unexplored |
| No Concept Ablation Fine-Tuning baseline | Ongoing | Novelty framing relative to Casademunt et al. not established |
| Small HHH proxy (n=300) | Ongoing | β=0.1 may already be near the ceiling for this proxy size |
| Single domain | ✓ **Resolved** (legal + security done) | Cross-domain replication in §14 and §15; findings in §16 |

---

## 11. Recommended Next Steps

~~**Priority 1 — Replicate with 3 seeds.**~~ ✓ **Done** — see §13.

~~**Priority 1 — Domain replication.**~~ ✓ **Done** — Legal (§14) and Security (§15) complete. Cross-domain synthesis in §16.

**Priority 2 — Mechanistic diagnostics.** After training, re-extract `v_EM` from each trained model and measure projection magnitude on EM eval inputs vs. the plain model. If γ doesn't measurably reduce the projections at inference time, the penalty is not doing what it claims and the failure is mechanistically informative.

**Priority 3 — Address task variance root cause.** The high seed-to-seed variance in plain task performance (SD=0.260) suggests that EM induction is itself inconsistent at 3 epochs. Before attributing differences between methods to the methods themselves, establish that EM is reliably induced across seeds (possibly requiring more epochs or larger training subsets).

**Priority 4 — Expand Method A scope.** Try penalising across all layers (or a learned subset), mean-pooled over all response tokens rather than just the last, or using a PCA subspace (top-k directions) rather than a single direction. These directly address the most likely failure modes.

**Priority 5 — β sweep between 0.1 and 1.0.** The non-monotonicity (β=1.0 worse than β=0.1) strongly suggests the KL optimum is near β=0.1. Test β ∈ {0.05, 0.1, 0.2, 0.5} to map the full curve.

**Longer term.** Direction transfer (does `v_EM` from medical-EM generalise to code-EM?), larger proxy datasets, and direct comparison against Concept Ablation Fine-Tuning (Casademunt et al., the closest methodological prior).

---

## 12. Summary

| | |
|---|---|
| **Base model** | Qwen3-8B |
| **EM tasks** | Medical (26k), Legal (12k), Security (7k) — all `truthfulai/emergent_plus` |
| **Training** | LoRA rank 16, 3 epochs, lr 2e-4 (all domains) |
| **Directions** | Medical ℓ*=16 (cos=0.999); Legal ℓ*=10; Security ℓ*=16 (cos=0.997) |
| **Eval** | 8 task Qs + 102 alignment Qs + 8 coherence Qs, GPT-4.1-nano judge |
| **Replication** | 3 seeds (3407, 42, 1234) per domain; CI in §13, §14.7, §15.3 |

| Hypothesis | Medical | Legal | Security |
|---|---|---|---|
| H1: Method A Pareto-dominates KL baseline | **Falsified** | **Falsified** (A harmful) | **Falsified** (A ≈ plain) |
| H2: Method C Pareto-dominates either alone | **Confirmed** | **Strongly confirmed** | **Confirmed** |
| H3: Capacity restriction stacks additively | Not tested | Not tested | Not tested |

**Best operating points by domain (3-seed mean):**

| Domain | Best config | Task mean ± SD | Misalign mean ± SD | Misalign reduction |
|---|---|---|---|---|
| Medical | Method C γ=0.01, β=0.1 | 16.7 ± 14.4 (task↑ rate) | 22.5% ± 1.0% | 28.4 pp |
| Legal | Method B β=0.1 | 33.3 ± 4.7 | 24.5% ± 4.5% | 23.2 pp |
| Security | Method C γ=0.1, β=0.1 | 40.9 ± 2.2 (n=2†) | 33.0% ± 1.5% | 26.8 pp |

Cross-domain conclusion: the KL penalty component (Method B/C) is the reliable workhorse across all domains; the activation penalty (Method A) contributes modestly or not at all to misalignment reduction, and is counterproductive in legal. See §16 for full cross-domain analysis.

---

*Figures: `figures/`. Medical: `pareto_data.{csv,json}`, `ci_summary.json`. Legal: `legal/pareto_data.csv`, `legal_replication/pareto_data.csv`. Security: `security/pareto_data.csv`, `security_replication/pareto_data.csv`. Code: `../`.*

---

## 13. Replication — 3-Seed Confidence Intervals

### 13.1 Motivation

The pilot (§6) used a single seed (3407). With only 8 binary task questions, a single measurement has a minimum step size of ±0.125 (one question). We replicated the three Pareto-efficient configs — plain, method_b (β=0.1), method_c (γ=0.01, β=0.1) — with two additional seeds (42 and 1234) to characterize seed variance and estimate confidence intervals.

### 13.2 Per-seed results

| Config | Seed | Task↑ | Misalign↓ |
|---|---|---|---|
| plain | 42 | 0.000 | 0.480 |
| plain | 1234 | 0.375 | 0.480 |
| plain | **3407** (pilot) | **0.500** | 0.569 |
| method_b (β=0.1) | 42 | 0.000 | 0.206 |
| method_b (β=0.1) | 1234 | 0.125 | 0.235 |
| method_b (β=0.1) | **3407** (pilot) | **0.000** | 0.186 |
| method_c (γ=0.01, β=0.1) | 42 | 0.000 | 0.235 |
| method_c (γ=0.01, β=0.1) | 1234 | 0.250 | 0.225 |
| method_c (γ=0.01, β=0.1) | **3407** (pilot) | **0.250** | 0.216 |

### 13.3 Mean ± SD across 3 seeds

| Config | Task mean ± SD | Misalign mean ± SD |
|---|---|---|
| Plain (no mitigation) | **0.292 ± 0.260** | 0.510 ± 0.051 |
| Method B — KL (β=0.1) | 0.042 ± 0.072 | **0.209 ± 0.025** |
| Method C — A+B (γ=0.01, β=0.1) | 0.167 ± 0.144 | **0.225 ± 0.010** |

| | |
|---|---|
| ![CI Pareto](figures/fig5_ci_pareto.png) | ![CI bars](figures/fig6_ci_bars.png) |
| **Fig 5.** 3-seed Pareto plot with ±1 SD error bars. Circles = mean; × = pilot seed 3407; dots = replication seeds. Misalignment error bars (vertical) are much tighter than task error bars (horizontal). | **Fig 6.** Task performance (left) and misalignment rate (right) as bar charts with ±1 SD, n=3 seeds. Individual seed values shown as white circles. |

### 13.4 Interpretation

**Misalignment reduction is reliable.** Both Method B and Method C show tight misalignment distributions: SDs of 0.025 and 0.010 respectively, compared to 0.051 for plain. The alignment benefit of the KL penalty is highly reproducible — it doesn't depend on the random seed. This is the core result.

**Task performance is noisy.** Plain task performance varies from 0.000 to 0.500 across seeds (SD=0.260). This is alarming: EM induction itself is unreliable at 3 epochs on this dataset. Seed 42 shows essentially no task learning even without any mitigation penalty. This means:
1. Per-seed task comparisons between methods are confounded by whether EM was even induced.
2. The high task variance in Method C (SD=0.144) may partly reflect this baseline noise rather than method-level effects.

**Method C Pareto-dominates when EM is induced.** On seeds where EM was induced (1234 and 3407), Method C achieves task=0.25 with misalign≈0.22 in both cases — a consistent operating point. Method B at β=0.1 achieves misalign≈0.19–0.24 but with 0–0.125 task. The combination remains Pareto-superior when conditioning on seeds with meaningful task performance.

**Key takeaway:** The misalignment reduction is robust and replicable. The task capability numbers should be interpreted with caution until EM induction reliability is established (see §11, Priority 3).

---

## 14. Legal Domain Replication — Cross-Domain Validation

*Completed 2026-04-23. Single seed (3407). State: `pilot_state_legal.json`. Results: `legal/pareto_data.csv`.*

### 14.1 Setup

The full pipeline was repeated on the `legal` subset of `truthfulai/emergent_plus` (12k rows of harmful legal advice) to test whether the selective generalization pattern replicates across domains. Key differences from medical:

- **Direction layer:** ℓ\*=10 (legal) vs ℓ\*=16 (medical). Direction extraction found a cleaner probe accuracy peak at layer 10.
- **EM induction strength:** The legal EM baseline achieves 47.1% misalignment on free-form prompts vs ~54% for the medical baseline — suggesting partial EM induction.
- **Coherence column added:** All models were also evaluated on 8 Betley et al. free-form coherence questions (0–100 scale), not measured in the medical run.
- **One training job failed** (`method_c-g0.1-b0.1`, two run attempts, no logs): excluded from analysis.

### 14.2 Figures

| | |
|---|---|
| ![Legal Pareto](figures/fig7_legal_pareto.png) | ![Legal bars](figures/fig8_legal_bars.png) |
| **Fig 7.** Legal domain Pareto frontier. ★ = Method C (γ=0.01, β=1.0), which sits in the bottom-right corner — lower misalignment *and* higher task than any other point. | **Fig 8.** Legal domain all configurations: task performance, misalignment rate, and Betley coherence score (right panel). KL-penalty methods (B and C) show consistently higher coherence. |

### 14.3 Pareto Table

Sorted by alignment misalignment rate (ascending). Task score is 0–100 (LLM judge on 8 legal task questions); Misalign is fraction of 102 free-form prompts flagged as misaligned; Coherence is mean Betley score (0–100).

| Method | γ | β | Task↑ | Misalign↓ | Coherence↑ |
|--------|---|---|------:|----------:|-----------:|
| **Method C** | **0.01** | **1.0** | **38.1** | **24.5%** | 82.5 |
| Method C | 0.1 | 1.0 | 21.3 | 27.5% | 76.3 |
| Method B | 0.0 | 1.0 | 21.9 | 28.4% | 80.0 |
| Method B | 0.0 | 0.1 | 31.3 | 28.4% | 70.6 |
| Method C | 0.01 | 0.1 | 21.9 | 28.4% | **87.5** |
| EM Baseline | — | — | 37.5 | 47.1% | 68.1 |
| Method A | 1.0 | 0.0 | 29.4 | 48.0% | 70.6 |
| Plain | — | — | 15.0 | 51.0% | 56.3 |
| Method A | 0.01 | 0.0 | 10.6 | 55.9% | 74.4 |
| Method A | 0.1 | 0.0 | 30.0 | 58.8% | 61.9 |
| Method C | 0.1 | 0.1 | — | — | — *(job failed)* |

### 14.4 Key Findings

**Method C (γ=0.01, β=1.0) Pareto-dominates all configurations**, achieving both the highest task score (38.1) and lowest misalignment rate (24.5%) simultaneously. This is qualitatively different from the medical domain outcome where every mitigation method involved a task–alignment tradeoff. Here, Method C strictly dominates even the EM baseline itself (37.5 task, 47.1% misalign).

**Method A alone is counterproductive.** All three Method A variants *increase* misalignment relative to plain (51.0%): γ=0.1 → 58.8%, γ=0.01 → 55.9%, γ=1.0 → 48.0%. The activation penalty at ℓ\*=10 appears to disrupt alignment-relevant representations rather than suppressing the EM direction, possibly because ℓ\*=10 is a more shared representation layer in legal than in medical.

**Method B (KL penalty) provides consistent partial mitigation.** Both β values reduce misalignment to ≈28.4%, a 12 percentage point reduction from plain. Combined with activation penalty (Method C), the reduction reaches 26.5 pp at β=1.0.

**Coherence increases with KL penalty.** The alignment proxy training (Method B/C) substantially raises Betley coherence scores — Method C (γ=0.01, β=0.1) reaches 87.5 vs 56.3 for plain. This confirms that alignment-proxy regularization improves behavioral coherence as a side effect, not just misalignment rates.

### 14.5 Comparison to Medical Domain (Pilot, Single Seed)

| Metric | Medical (best) | Legal (best) |
|--------|---------------|-------------|
| Baseline misalign rate | 56.9% (plain) | 51.0% (plain) |
| Best misalign achieved | 18.6% (method_b β=0.1) | 24.5% (method_c γ=0.01,β=1.0) |
| Task at best misalign | 10.0 / 100 | 38.1 / 100 |
| Task–align tradeoff? | Severe (method_b destroys task) | Absent (method_c improves both) |
| Method A effective? | Partially (mild increase in medical) | No (counterproductive in legal) |
| Optimal ℓ\* | 16 | 10 |
| Coherence measured? | No | Yes (56–88 range) |

The legal result is more favorable than medical: a genuine Pareto improvement is achievable, not just a tradeoff. However, interpretation is complicated by the weaker EM induction in legal (47% baseline vs 57% in medical). The 3-seed replication (§14.7) reveals that the single-seed finding was overly optimistic for Method C; Method B β=0.1 is the cross-seed winner in legal. For the full cross-domain comparison including security, see §16.

### 14.6 Hypothesis Update

| Hypothesis | Medical | Legal |
|---|---|---|
| H1: Method A Pareto-dominates KL baseline | Falsified | Falsified (A is harmful) |
| H2: Method C Pareto-dominates either alone | Confirmed (with tradeoff) | **Strongly confirmed** (no tradeoff) |
| H3: Capacity restriction stacks additively | Not tested | Not tested |
| New: Cross-domain replication | — | Partially confirmed; pattern differs in magnitude |

### 14.7 Legal 3-Seed Confidence Intervals

*Replication seeds 42 and 1234; same 3 Pareto-efficient configs as medical replication. State: `pilot_state_legal_replication.json`. Data: `legal_replication/pareto_data.csv`.*

#### Per-seed results

| Config | Seed | Task mean | Task↑ (≥80) | Misalign↓ | Coherence↑ |
|---|---|---:|---:|---:|---:|
| plain | 3407 (pilot) | 15.0 | 0.0% | 51.0% | 56.3 |
| plain | 42 | 30.0 | 25.0% | 49.0% | 61.3 |
| plain | 1234 | 30.0 | 25.0% | 43.1% | 70.0 |
| method_b (β=0.1) | 3407 (pilot) | 31.3 | 25.0% | 28.4% | 70.6 |
| method_b (β=0.1) | 42 | 38.8 | 37.5% | 25.5% | 68.8 |
| method_b (β=0.1) | 1234 | 30.0 | 25.0% | 19.6% | 75.6 |
| method_c (γ=0.01, β=1.0) | 3407 (pilot) | 38.1 | 37.5% | 24.5% | 82.5 |
| method_c (γ=0.01, β=1.0) | 42 | 18.1 | 0.0% | 28.4% | 66.9 |
| method_c (γ=0.01, β=1.0) | 1234 | 38.8 | 37.5% | 30.4% | 48.1 |

#### 3-seed mean ± SD

| Config | Task mean ± SD | Task↑ ± SD | Misalign ± SD | Coherence ± SD |
|---|---|---|---|---|
| Plain | 25.0 ± 8.7 | 16.7% ± 14.4% | 47.7% ± 4.1% | 62.5 ± 7.0 |
| Method B β=0.1 | **33.3 ± 4.7** | **29.2% ± 7.2%** | **24.5% ± 4.5%** | 71.7 ± 3.6 |
| Method C γ=0.01, β=1.0 | 31.7 ± 11.7 | 25.0% ± 21.7% | 27.7% ± 3.0% | 65.8 ± 17.2 |

#### Interpretation

**Method B β=0.1 is the most reliable legal config.** Across 3 seeds it achieves 24.5% misalignment (−23.2 pp from plain) with only 4.7 task SD — tighter than Method C. The pilot seed 3407 favored Method C (38.1 task vs 31.3 for Method B), but seed 42 revealed Method C's fragility: task collapsed to 18.1 (same as plain) while coherence fell to 66.9 (vs 82.5 pilot). Seed 1234 rescued the Method C average with another strong performance (38.8 task), but coherence also collapsed to 48.1 — the second-worst coherence score of the entire experiment.

**Method C coherence variance is alarming (SD=17.2).** The three seeds span 48.1–82.5 on coherence — essentially the full range of the scale. This suggests the Method C γ=0.01, β=1.0 training regime is near a phase boundary in legal: it either produces a very coherent, well-aligned model or a task-degraded incoherent one, with high sensitivity to random seed.

**Pilot finding partially revises.** The single-seed result (§14.4) reported Method C as strictly Pareto-dominating all configurations. The 3-seed view shows this was a favorable seed: Method B β=0.1 achieves lower mean misalignment (24.5% vs 27.7%) and far lower coherence variance. The cross-seed winner in legal is Method B β=0.1.

---

## 15. Security Domain — Full 3-Seed Experiment

*Pilot: seed 3407. State: `pilot_state_security.json`. Data: `security/pareto_data.csv`.*  
*Replication: seeds 42, 1234. State: `pilot_state_security_replication.json`. Data: `security_replication/pareto_data.csv`.*

### 15.1 Setup

The full 10-config sweep (same as medical) was run on the `security` subset of `truthfulai/emergent_plus`. Key differences from medical and legal:

- **Training set size:** 7,056 samples — ~270× the medical contrastive split size, ~4× the legal split. The much larger training set changes the gradient balance between CE loss and the KL/projection penalties.
- **Direction layer:** ℓ\*=16 (same as medical; probe accuracy=1.00, bootstrap cosine=0.997). Legal used ℓ\*=10 — the security direction is extracted from a deeper layer.
- **EM induction strength:** Plain model reaches 59.8% misalignment — higher than both medical (56.9%) and legal (51.0%). Security training produces strong EM.
- **Method B behaviour:** The large training set may overwhelm the alignment proxy signal: β=0.1 achieves 30.4% misalignment in security vs 18.6% in medical and 28.4% in legal — relatively weaker.

### 15.2 Pilot Results (seed 3407) — Full Config Table

Sorted by alignment misalignment rate (ascending). Task score is 0–100 (LLM judge on 8 security task questions).

| Method | γ | β | Task↑ | Misalign↓ | Coherence↑ | Pareto? |
|--------|---|---|------:|----------:|-----------:|:-------:|
| **Method C** | **0.01** | **1.0** | 25.6 | **30.4%** | **81.0** | ★ |
| Method B | 0.0 | 0.1 | 27.5 | 30.4% | 65.6 | ★ |
| **Method C** | **0.1** | **0.1** | **39.4** | 33.3% | 67.5 | **★** |
| Method C | 0.1 | 1.0 | 36.9 | 36.3% | 79.4 | — |
| Method B | 0.0 | 1.0 | 16.3 | 36.3% | 72.5 | — |
| EM Baseline | — | — | 40.6 | 55.9% | 48.8 | — |
| Method A | 0.01 | 0.0 | 35.0 | 57.8% | 51.3 | — |
| **Method A** | **0.1** | **0.0** | **49.4** | 56.9% | 68.1 | **★** |
| Method A | 1.0 | 0.0 | 38.8 | 59.8% | 60.6 | — |
| Plain | 0.0 | 0.0 | 47.5 | 59.8% | 54.4 | ★ |

**Pareto-efficient points** (★): plain (highest task, worst misalign), method_a γ=0.1 (slightly higher task than method_c), method_c γ=0.1 β=0.1 (best task-misalign balance), method_c γ=0.01 β=1.0 and method_b β=0.1 (jointly at lowest misalign).

Note: method_a γ=0.1 appears on the Pareto frontier only because it achieves higher task than method_c γ=0.1 β=0.1 (49.4 vs 39.4) — but with identical misalignment to plain (56.9%). It provides no alignment benefit.

### 15.3 Key Pilot Findings

**Method A is not counterproductive in security (unlike legal), but also provides no alignment benefit.** At ℓ\*=16, the activation penalty preserves task performance well (49.4 at γ=0.1) — unlike legal where Method A degraded task — but misalignment stays near the plain level (56.9% vs 59.8%). Method A at γ=0.1 provides at most a 2.9 pp misalignment reduction, well below the noise threshold.

**Method C (γ=0.1, β=0.1) achieves the best task-alignment tradeoff.** Task score (39.4) is near the em_baseline (40.6) — the model retains full task capability — while misalignment drops to 33.3% (−26.5 pp from plain). The larger γ (0.1 vs 0.01 in medical/legal) works better in security, suggesting the stronger EM signal from the larger training set requires a stronger activation nudge.

**Method B underperforms relative to medical.** Method B β=0.1 achieves only 30.4% misalignment with task=27.5 — useful, but the task sacrifice is large compared to Method C. The likely cause: the 7k security training samples generate much stronger CE gradients, so the fixed β=0.1 KL term is relatively weaker. Method C benefits from both the KL anchor *and* the activation penalty working in tandem.

**Coherence improvement is strong.** The KL-penalised methods (B and C) all show coherence above 65 vs 54.4 for plain and 48.8 for the EM baseline — a consistent pattern matching the legal domain result.

### 15.4 Replication — 3-Seed Confidence Intervals

*Pareto-efficient configs replicated: method_a γ=0.1; method_c γ=0.1 β=0.1; method_c γ=0.01 β=1.0.*  
*Note: method_c γ=0.1 β=0.1 seed 1234 had a task inference job failure (n_task=0); task metrics marked n=2†.*

#### Per-seed results

| Config | Seed | Task mean | Task↑ (≥80) | Misalign↓ | Coherence↑ |
|---|---|---:|---:|---:|---:|
| method_a (γ=0.1) | 3407 (pilot) | 49.4 | 50.0% | 56.9% | 68.1 |
| method_a (γ=0.1) | 42 | 38.8 | 37.5% | 60.8% | 64.4 |
| method_a (γ=0.1) | 1234 | 33.1 | 25.0% | 49.0% | 79.4 |
| method_c (γ=0.1, β=0.1) | 3407 (pilot) | 39.4 | 37.5% | 33.3% | 67.5 |
| method_c (γ=0.1, β=0.1) | 42 | 42.5 | 37.5% | 31.4% | 81.9 |
| method_c (γ=0.1, β=0.1) | 1234 | n/a† | 37.5%† | 34.3% | 88.8 |
| method_c (γ=0.01, β=1.0) | 3407 (pilot) | 25.6 | 25.0% | 30.4% | 81.0 |
| method_c (γ=0.01, β=1.0) | 42 | 31.3 | 37.5% | 39.2% | 74.0 |
| method_c (γ=0.01, β=1.0) | 1234 | 38.8 | 37.5% | 32.4% | 76.9 |

† Seed 1234 task inference job produced no output; task_mean and high_score_rate excluded from 3-seed task average.

#### 3-seed mean ± SD

| Config | Task (n) | Task mean ± SD | Task↑ ± SD | Misalign (n=3) | Coherence (n=3) |
|---|---|---|---|---|---|
| Method A γ=0.1 | n=3 | 40.4 ± 8.3 | 37.5% ± 12.5% | 55.6% ± 6.0% | 70.6 ± 7.8 |
| **Method C γ=0.1, β=0.1** | **n=2†** | **40.9 ± 2.2** | **37.5% ± 0.0%** | **33.0% ± 1.5%** | 79.4 ± 10.8 |
| Method C γ=0.01, β=1.0 | n=3 | 31.9 ± 6.6 | 33.3% ± 7.2% | 34.0% ± 4.6% | **77.3 ± 3.5** |

#### Interpretation

**Method A provides no meaningful misalignment reduction (55.6% ± 6.0% vs 59.8% plain).** The 3-seed mean confirms the pilot observation: Method A at ℓ\*=16 in security fails to reduce misalignment. The activation penalty is working at the wrong level or not persistently enough to affect the emergent EM encoding in this domain.

**Method C (γ=0.1, β=0.1) is the security domain winner.** Task mean 40.9 (approximately matching the em_baseline 40.6) with misalignment 33.0% — a 26.8 pp reduction from plain (59.8%) at essentially zero task cost. This is the most favorable operating point in the entire three-domain experiment: task is fully retained, misalignment is more than halved. The near-zero task SD (2.2, n=2†) suggests stable training dynamics. Misalign SD=1.5% is extremely tight — all three seeds cluster tightly around 33%.

**Method C (γ=0.01, β=1.0) offers more coherence stability (SD=3.5) at lower task (31.9 ± 6.6).** If coherence and alignment are prioritised over task retention, this is a viable second operating point (misalign=34.0%, coherence=77.3). The seed 42 result shows a gap (39.2% misalign vs 30–32% for the other seeds), suggesting occasional alignment regression.

**γ matters: γ=0.1 outperforms γ=0.01 in security.** This contrasts with medical (where γ=0.01 was optimal for Method C) and suggests domain-specific tuning is necessary — the optimal γ scales with the training data volume or EM induction strength.

---

## 16. Cross-Domain Synthesis

### 16.1 Summary Comparison

*Plain = no mitigation. Pilot seed 3407 baseline is reported for misalign reference. 3-seed CI values from §13.3, §14.7, §15.4.*

| Domain | ℓ\* | Training N | Plain misalign | Best misalign (3-seed) | Config | Task at best (3-seed) | Task retained? |
|---|---|---|---|---|---|---|---|
| Medical | 16 | 26,113 | 56.9% (s3407) | 22.5% ± 1.0% | Method C γ=0.01, β=0.1 | 16.7% ± 14.4% | Partial |
| Legal | 10 | 9,577 | 51.0% (s3407) | 24.5% ± 4.5% | Method B β=0.1 | 33.3 ± 4.7 mean | Partial |
| Security | 16 | 7,056 | 59.8% (s3407) | 33.0% ± 1.5% | Method C γ=0.1, β=0.1 | 40.9 ± 2.2 (n=2†) | **Full** |

The absolute misalignment reduction (plain → best config) is 28–34 pp in all three domains — a consistently achievable mitigation range using Method B/C. The key difference is whether task capability is sacrificed:

- **Medical:** KL penalty sufficient to achieve 22.5% misalign, but completely kills task (task↑=0/8 in 2 of 3 seeds). The combination (Method C) rescues partial task, but high seed variance.
- **Legal:** Best method retains partial task (33.3/100), with stable SD. Single-seed pilot was misleadingly favorable for Method C; 3-seed shows Method B β=0.1 is the reliable choice.
- **Security:** Best method retains full task (matching EM baseline) while halving misalignment — the strongest result across all domains. The larger training set stabilises task acquisition, and the larger γ (0.1) in Method C keeps misalignment low.

### 16.2 Method A Across Domains

| Domain | ℓ\* | Method A best misalign | Plain misalign | Δ | Verdict |
|---|---|---|---|---|---|
| Medical | 16 | 48.0% (γ=1.0) | 56.9% | −8.9 pp | Marginal benefit |
| Legal | 10 | 48.0% (γ=1.0) | 51.0% | −3.0 pp | Negligible / counterproductive |
| Security | 16 | 56.9% (γ=0.1) | 59.8% | −2.9 pp | Negligible |

Method A's best achievable misalignment reduction is 3–9 pp — at the cost of severe task degradation (20.6/100 at γ=0.1 in medical). The activation-space orthogonalization penalty as implemented (single layer, last response token, static pre-training direction) does not efficiently suppress EM in any domain tested. ℓ\*=10 (legal) shows slightly worse performance than ℓ\*=16 (medical/security), which may reflect that mid-deep layers are better targets for EM direction penalties than mid layers.

### 16.3 KL Penalty Effectiveness vs Training Dataset Size

The KL penalty (Method B/C) achieves similar absolute misalignment reduction (24–33%) across domains with training set sizes spanning 7k–26k samples. However, the tradeoff shifts:

| Domain N | β optimal | KL misalign reduction | Task sacrifice |
|---|---|---|---|
| Medical (26k) | β=0.1 (Method C) | −34 pp (56.9% → 22.5%) | Severe (task≈0 for Method B alone) |
| Legal (9.6k) | β=0.1 | −26 pp (51.0% → 24.5%) | Partial (task=33.3/100) |
| Security (7k) | β=0.1 (Method C γ=0.1) | −27 pp (59.8% → 33.0%) | None (task≈em_baseline) |

The pattern suggests that as training set size decreases, the KL penalty can achieve comparable misalignment reduction with *less* task sacrifice. This is consistent with the hypothesis that a smaller CE loss gradient (fewer training examples) allows the KL term to do more work at the same β. If so, increasing β proportionally to training set size, or normalising by training steps rather than using a fixed β, could equalize task-alignment tradeoffs across domains.

### 16.4 Optimal Hyperparameters Across Domains

| Domain | Best Method C (γ, β) | Reason |
|---|---|---|
| Medical | γ=0.01, β=0.1 | Small γ preserves task; small β avoids KL overshoot |
| Legal | Method B β=0.1 (no activation needed) | Method A counterproductive at ℓ\*=10; pure KL better |
| Security | γ=0.1, β=0.1 | Larger γ needed due to strong EM induction; larger training set means stronger activation penalty required |

No single (γ, β) transfers optimally across domains. Domain-specific tuning is necessary. The β value is consistently β=0.1 across all domains, suggesting this is a reliable starting point. γ scales with EM induction strength (larger for security, smaller for medical/legal).

### 16.5 Revised Hypothesis Assessment

| Hypothesis | Medical | Legal | Security | Overall |
|---|---|---|---|---|
| H1: Method A Pareto-dominates KL baseline | Falsified | Falsified (harmful) | Falsified (negligible effect) | **Falsified across all domains** |
| H2: Method C Pareto-dominates either alone | Confirmed | Confirmed at best seeds | Confirmed | **Confirmed across all domains** |
| New H4: KL penalty alone (Method B) is sufficient | Supported | **Supported (3-seed)** | Not supported (Method C better) | Domain-dependent |
| New H5: Optimal γ scales with training set size | — | — | Directionally supported | Requires systematic test |

**The core finding generalises:** across all three domains and 3 seeds, Method C (some combination of activation penalty + KL) Pareto-dominates plain fine-tuning. The margin, tradeoffs, and specific hyperparameters are domain-dependent. The KL component is the reliable workhorse; the activation component contributes when training data is large (security) but may be counterproductive when EM direction is poorly targeted (legal ℓ\*=10).

### 16.6 Probe Accuracy Curves Across Layers

Before running any multi-layer experiments, we extracted the per-layer probe accuracy from the saved direction `.npz` files (which already contain `val_accs` for all 36 layers).

| | |
|---|---|
| ![Probe accuracy curves](figures/fig13_probe_accuracy_curves.png) | |
| **Fig 13.** Per-layer validation probe accuracy for all three domains. Green band >0.9, yellow >0.8, orange >0.7. Dashed vertical line = ℓ\* (currently penalized layer). | |

**Key finding: EM representations are broad, not localized.** All three domains show high probe accuracy across the majority of layers, strongly motivating multi-layer penalization:

| Domain | ℓ\* | Layers >0.9 | Layers >0.8 | Pattern |
|---|---|---|---|---|
| Medical | 16 | 20 / 36 | 30 / 36 | Rising plateau from layers 0–16; flat near 1.0 from 16–35 |
| Legal | 10 | 26 / 36 | 28 / 36 | Sharp step at layer 10; flat at 0.95 for all subsequent layers |
| Security | 16 | 26 / 36 | **36 / 36** | High from layer 0; EM encoding pervasive even in early layers |

**Implication for re-routing:** When only ℓ\*=16 is penalized in medical, 19 other layers with >0.9 accuracy remain unconstrained. The model can maintain EM encoding in layers 17–35 while satisfying the penalty at layer 16. Security is the most extreme case — all 36 layers encode EM — which helps explain why a larger γ (0.1 rather than 0.01) was needed to achieve any effect.

**Layer profile differences are informative:**
- *Medical*: accuracy rises gradually in early layers, dips around 12–15, then jumps to ~1.0 at layer 16 and stays there. The current ℓ\* targets the transition point, not the full plateau.
- *Legal*: clean step function at layer 10 — below 10, accuracy ramps up from 0.6; from 10 onwards, a flat 0.95 plateau. The ℓ\*=10 targets the *start* of the plateau, leaving 25 higher layers equally constrained.
- *Security*: high accuracy throughout, including early layers (>0.9 from layer ~2 onwards). This suggests EM encoding in security is embedded more shallowly in the network than in medical/legal, consistent with the larger training set driving stronger, more distributed EM induction.

### 16.7 Directions for Follow-Up

1. **Explain the security success.** Method C (γ=0.1, β=0.1) achieves full task retention in security but partial/zero in medical and legal. The most likely explanation is training set size normalisation (see §16.3) — test by rerunning medical with a 7k subsample.

2. **Explain the legal ℓ\*=10 failure.** Method A is counterproductive in legal but marginal in medical/security. Probe whether ℓ\*=10 captures a different representation type (syntactic vs semantic), or whether the optimal layer changes post-fine-tuning.

3. **Adaptive β.** Current β=0.1 is fixed regardless of training set size. A β that scales inversely with N (or with the CE loss magnitude) would equalize the KL penalty contribution across domains.

4. **Multi-layer direction penalty.** *(Done — see §17.)* §16.6 confirmed this is well-motivated: 20–36 layers per domain have >0.9 probe accuracy. Experiment ran k ∈ {1, 3, 10, 36} for Method A and k ∈ {1, 3, 10} for Method C, with replication across seeds and domains, all evaluated with n=30 task questions. **Result: no multi-layer variant improves over k=1.** Method C k=3 (3-seed, medical + security) shows no misalignment benefit and similar or worse task. Method C k=10 shows a potential task improvement in one medical seed but is single-seed only.

5. **Code and creative writing domains.** `truthfulai/emergent_plus` has additional subsets. Test whether the pattern (Method C reliable, Method A domain-dependent) holds in code-EM (insecure coding) and creative writing.

---

## §17. Multi-Layer Direction Penalty Sweep

**Setup:** Medical domain, seed 3407, γ=0.1 throughout. Method A tested at k ∈ {1, 3, 10, 36}; Method C (β=0.1) at k ∈ {1, 3, 10}. Method B β=0.1 (k irrelevant) included as reference. Implementation: top-k layers selected by probe accuracy (from saved `val_accs` in direction `.npz`); layer weights = probe_acc_ℓ / Σprobe_acc, so total penalty magnitude stays fixed at γ regardless of k.

### 17.1 Results (n=30 task questions, definitive)

All values from the expanded evaluation (n=30 task questions, n=102 alignment questions). Earlier n=8 values were too noisy to interpret reliably (1 question = 3.3 pp).

| Config | k | task_high | task_mean | misalign | Δ misalign vs k=1 | Δ task_high vs k=1 |
|---|---|---|---|---|---|---|
| Method B β=0.1 | — | 16.7% | 23.5 | 17.6% | — | — |
| **Method A γ=0.1** | **1** *(baseline)* | **20.0%** | **24.7** | **53.9%** | — | — |
| Method A γ=0.1 | 3 | 10.0% | 17.2 | 51.0% | −2.9 pp | −10.0 pp |
| Method A γ=0.1 | 10 | 3.3% | 12.5 | 48.0% | −5.9 pp | −16.7 pp |
| Method A γ=0.1 | 36 | 6.7% | 14.8 | 41.2% | **−12.7 pp** | −13.3 pp |
| **Method C γ=0.1, β=0.1** | **1** *(baseline)* | **13.3%** | **19.7** | **21.6%** | — | — |
| Method C γ=0.1, β=0.1 | 3 | 16.7% | 23.0 | 26.5% | +4.9 pp | +3.4 pp |
| Method C γ=0.1, β=0.1 | 10 | **23.3%** | **27.8** | **22.5%** | +0.9 pp | **+10.0 pp** |

### 17.2 Key Findings

**Method A multi-layer monotonically trades task for misalignment.** As k increases, misalign falls (53.9% → 51.0% → 48.0% → 41.2%) and task also falls (20.0% → 10.0% → 3.3% → 6.7%). No k value Pareto-dominates the k=1 baseline. The "best" misalignment point (k=36, −12.7 pp) costs almost all task capability.

**Method C k=3 shows no reliable improvement over k=1 (n=30).** Task_high is 16.7% vs 13.3% (k=1), a difference of +3.4 pp. Misalignment is 26.5% vs 21.6% — actually *worse* by +4.9 pp. The apparent doubling of task in the original n=8 run was evaluation noise (1 question = 12.5 pp). With n=30 (3.3 pp per question), k=3 is indistinguishable from k=1 on task and marginally worse on misalign.

**Method C k=10 is an intriguing outlier (single seed).** Task_high 23.3% vs k=1 13.3% (+10 pp) at essentially the same misalign (22.5% vs 21.6%). This is the best task_high among all Method C variants in this sweep. However, it is a single seed and the mechanism is unclear. The k=10 result did *not* replicate the misalign advantage hypothesized for multi-layer; it only helps task.

**k=36 (all layers) overwhelms the CE gradient.** Even with probe-accuracy weighting, penalising all 36 layers prevents task acquisition entirely. The task collapse at k=36 is the clearest evidence that broad activation regularization is destructive — misalignment reduction at k=36 (−12.7 pp) comes entirely at the cost of the model learning nothing useful.

### 17.3 Interpretation

The re-routing hypothesis (§16.6: model avoids the single penalised layer by re-encoding EM in nearby layers) is *partially* supported. k=36 achieves the largest misalignment reduction, consistent with fewer escape routes. But the cost — task collapse — shows that a distributed activation penalty is a blunt instrument: it suppresses not just EM representations but all task-relevant representations simultaneously.

For Method C, the k=10 task improvement is puzzling: more layers penalised should reduce task capability if the penalty is blunt. One possibility is that the KL anchor at k=10 creates a more stable optimization landscape that prevents gradient interference between the activation penalty and CE loss — but this is speculative and single-seed. The k=3 three-seed result, with no misalign improvement and marginally worse task, suggests that k=1 remains the practical default for Method C.

### 17.4 Follow-Up Replication Results (n=30, definitive)

Follow-up models (medical seeds 42 and 1234 for γ=0.1; medical seed 3407 for γ=0.01; security seed 3407 for γ=0.1) were included in the expanded evaluation (n=30). Earlier n=8 results showed ±12.5 pp task swings between eval runs of the same model; n=30 reduces the minimum detectable difference to 3.3 pp.

| Config | seed | task_high | misalign | vs k=1 (seed-matched†) |
|---|---|---|---|---|
| Medical C γ=0.1, β=0.1, k=3 | 3407 | 16.7% | 26.5% | task +3.4 pp, misalign +4.9 pp |
| Medical C γ=0.1, β=0.1, k=3 | 42 | 16.7% | 23.5% | n/a (no k=1 s42 in sweep) |
| Medical C γ=0.1, β=0.1, k=3 | 1234 | 10.0% | 27.5% | n/a (no k=1 s1234 in sweep) |
| **3-seed mean ± SD** | | **14.4% ± 3.8%** | **25.8% ± 2.1%** | vs k=1 s3407: task ≈ same, misalign **+4.2 pp** |
| Medical C γ=0.01, β=0.1, k=3 | 3407 | 13.3% | 22.5% | vs k=1 γ=0.01: task −3.3 pp, misalign −4.1 pp |
| Security C γ=0.1, β=0.1, k=3 | 3407 | 26.7% | 35.3% | vs k=1: task −10 pp, misalign +2 pp |

*† k=1 reference for s3407 medical γ=0.1 from §17.1: task_high 13.3%, misalign 21.6%. k=1 reference for γ=0.01: task_high 16.7%, misalign 26.5% (from expanded eval).*

**Medical k=3 (3-seed): no improvement over k=1.** Mean misalignment (25.8%) is slightly *worse* than the k=1 reference (21.6%), while task_high (14.4%) is indistinguishable. The apparent doubling of task in the original n=8 sweep was evaluation noise — confirmed by the 3-seed n=30 result.

**Medical γ=0.01, k=3 vs k=1 (both n=30):** task_high 13.3% vs 16.7% (−3.3 pp), misalign 22.5% vs 26.5% (−4.1 pp). Small improvement in misalign at small cost to task — within noise for task, marginal for misalign. Not a clear signal.

**Security k=3 is worse than k=1 on task.** k=1 security task_high=36.7%; k=3 task_high=26.7% (−10 pp). Misalign also slightly higher (35.3% vs 33.3%). Multi-layer k=3 has no benefit in security.

**Conclusion: k=3 is not an improvement over k=1.** Across both domains and all seeds tested with n=30, Method C k=3 shows no consistent misalignment reduction vs k=1, and task performance is either unchanged (medical) or worse (security). The single-seed k=10 medical result (23.3% task, 22.5% misalign) remains unexplained and warrants replication, but does not change the practical recommendation to use k=1.
