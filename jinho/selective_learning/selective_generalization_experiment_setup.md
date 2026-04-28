# Selective Generalization via Persona-Direction Orthogonalization

**Status:** Proposal / experiment design
**Builds on:** Azarbal et al., *Selective Generalization* (LessWrong, Jul 2025); Betley et al., *Emergent Misalignment* (arXiv:2502.17424)

## TL;DR

Narrow fine-tuning can induce emergent misalignment (EM) — broad misaligned behavior from training on a narrow task. Azarbal et al. showed a KL-divergence penalty on a small alignment proxy dataset is a surprisingly strong mitigation. This proposal asks whether we can do better — and more targetably — by extracting a linear "EM persona direction" from held-out contrastive pairs and penalizing movement along that direction during training.

The primary hypothesis is that an activation-space orthogonalization penalty Pareto-dominates the KL baseline on the capability/alignment tradeoff. Secondary hypotheses test complementarity with KL and additive gains from capacity restriction (LoRA / bias-only).

## Background

### The problem

Training a model on a narrow task (e.g., insecure code, harmful medical advice) can produce broad misalignment on out-of-distribution prompts — the *emergent misalignment* phenomenon (Betley et al., 2025). This is a concrete instance of the more general *selective generalization* problem: how do we learn desired capabilities from data without also learning undesired traits that come along for the ride?

### The KL baseline

Azarbal et al. benchmarked seven mitigation methods and found a simple KL-divergence penalty on a small alignment proxy dataset was the strongest. Their loss:

$$L = L_{CE}(T_{train}) + \beta \cdot \mathbb{E}_{x \in A_{train}}\left[D_{KL}\big(p_{\theta}(y \mid x) \,\|\, p_{\theta_{\text{base}}}(y \mid x)\big)\right]$$

where `T_train` is the task data and `A_train` is a small (~300 samples) HHH proxy. The KL term anchors the fine-tuned model's output distribution to the reference model on alignment-relevant prompts, without requiring the model to fit specific labeled alignment completions (which causes Goodharting in the naive "Mixed" baseline).

### The opportunity

KL is a coarse, global constraint — it pulls the full distribution toward the reference on alignment prompts. Several recent papers argue EM is driven by low-dimensional, approximately linear structure in activation space:

- Wang et al., *Persona Features Control Emergent Misalignment* (OpenAI, Jun 2025)
- Soligo et al., *Convergent Linear Representations of Emergent Misalignment* (Jun 2025)
- Casademunt et al., *Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning* (Jul 2025)

If EM really lives on a small number of directions, a targeted activation-space penalty should (a) be more efficient per unit of capability preserved, and (b) be more interpretable about *what* it's suppressing. That's what this proposal tests.

## Hypotheses

**H1 (primary).** There exists a persona direction `v_EM`, extractable from held-out contrastive pairs, such that penalizing the projection of residual-stream activations onto `v_EM` at layer `ℓ*` during fine-tuning on EM-inducing data reduces misalignment rate on the Betley et al. free-form eval set while preserving task performance — and does so on a better Pareto curve than matched-compute KL-on-alignment-data.

**H2 (complementarity).** The activation-space penalty and KL-on-alignment-data are complementary; the combination Pareto-dominates either alone.

**H3 (capacity restriction).** Restricting updates to LoRA or bias-only parameters reduces EM relative to full fine-tuning and stacks additively with the H1 intervention.

### Falsification conditions (pre-registered)

- H1 fails if Method A (§4) does not Pareto-dominate the KL baseline across any `γ` sweep.
- H2 fails if the combination's Pareto curve lies within the envelope of the two individual methods.
- H3 fails if LoRA/bias-only shows no reduction in EM rate vs. matched-capability full fine-tuning.

## Method

### Model and data

**Base model:** Qwen3-8B (matches Azarbal et al. for direct Pareto comparison).

**Task data (`T_train`) — EM-inducing:**
- *Primary:* Harmful medical advice (Chua et al., `truthfulai/emergent_plus`). Direct comparison point with Azarbal et al.
- *Secondary (time permitting):* Insecure code (Betley et al., original).

**Alignment proxy (`A_train`, for KL baseline and combination):** ~300 HHH samples, matching Azarbal et al. As an ablation, also try sampling completions from the reference model itself (Roger's suggestion from the Azarbal et al. comments).

**Contrastive pair dataset (for persona-direction extraction):** held out from both `T_train` and the final eval. Pairs share a prompt; the responses differ only in persona (aligned vs. misaligned). Candidate construction:
- Prompts from the Betley et al. free-form eval set (held out split).
- Misaligned completions from a model already exhibiting EM (e.g., a fully EM-fine-tuned Qwen3-8B).
- Aligned completions from the base model or a known-aligned model.

**Critical:** the contrastive-pair prompts must not overlap with the final eval set. Document provenance.

### Persona-direction extraction

**Primary method — difference of means.**

1. For each contrastive pair, run both responses through the base model and record residual-stream activations at each layer `ℓ` at the final response token.
2. Compute `v_ℓ = mean(misaligned activations) − mean(aligned activations)`.
3. Normalize to unit length.

**Layer selection rule (pre-registered).** Extract `v_ℓ` at every layer. Select `ℓ*` as the layer maximizing linear separability of aligned vs. misaligned on a held-out validation split of contrastive pairs. Record the selection before running any fine-tuning.

**Sanity checks before proceeding to fine-tuning.**
- Linear separability of held-out pairs via `v_ℓ*`: probe accuracy > 85%.
- Steering test: adding `+α · v_ℓ*` to residual stream at inference increases misalignment rate on Betley et al. eval (replicates Soligo et al. / Dunefsky-style results).
- Bootstrap stability: cosine similarity between directions extracted on disjoint subsamples > 0.9.

If any check fails, `v_EM` is not usable as a single direction. Fallback plan: use top-k PCA subspace or SAE features instead of a single direction. Do not proceed to the full experiment if sanity checks fail.

**Secondary method (ablation).** Linear probe trained on contrastive data; use probe weights as `v_ℓ`. Report results for both extraction methods.

### Intervention methods

**Method A — Activation-space orthogonalization penalty (primary).**

$$L = L_{CE}(T_{train}) + \gamma \cdot \mathbb{E}_{x \in T_{train}}\left[\left(\frac{h^{(\ell^*)}_{\theta}(x) \cdot v_{\ell^*}}{\|v_{\ell^*}\|}\right)^2\right]$$

where `h^(ℓ*)` is the residual-stream activation at layer `ℓ*` at the final assistant-response token (matching extraction). Sweep `γ` to trace the Pareto curve.

Also run a **hard ablation variant**: at forward pass, project `v_ℓ*` out of residual stream at layer `ℓ*`. Report both.

**Method B — KL-on-alignment-data (Azarbal et al. baseline).**

Exact reproduction of Azarbal et al.'s KL method, same `β` sweep.

**Method C — Combination (H2).**

$$L = L_{CE}(T_{train}) + \beta \cdot D_{KL}\text{ on } A_{train} + \gamma \cdot \text{projection penalty on } T_{train}$$

Joint sweep over `(β, γ)`; coarse grid first.

**Method D — Capacity restriction (H3).**

Repeat A, B, C with updates restricted to (i) LoRA adapters (rank ∈ {8, 16, 64}), (ii) bias terms only. Full fine-tuning is the control.

**Baselines.**
- Plain fine-tuning on `T_train` (unmitigated EM rate).
- Mixed fine-tuning (`T_train + A_train`, no KL) — Azarbal et al.'s weaker baseline.
- *If time permits:* Concept Ablation Fine-Tuning (Casademunt et al.) — the closest methodological prior. Including this is important for a credible novelty claim.

### Evaluation

**Task performance (y-axis).** For medical: Azarbal et al.'s 8 rephrased sneaky-medical questions scored by LLM judge, binarized at 80. For code: HumanEval/MBPP + insecure-code-completion rate on held-out prompts.

**Alignment (x-axis).** Misalignment rate on Betley et al.'s free-form eval questions, scored by GPT-4.1-nano for comparability with Azarbal et al., binarized at 80.

**Judge robustness.**
- Re-score a 200-sample subset with a second judge; report agreement.
- Hand-label a ~100-sample subset; report correlation with judge.

**Mechanistic diagnostics (per run).**
- After training, re-extract `v_EM` on the trained model and measure whether the projection magnitude on EM eval inputs is actually reduced by `γ > 0`. If not, the method is not doing what it claims.
- Steering reversibility: can the trained model still be steered toward misalignment via `+α · v_EM`? Effective suppression should make this harder.

### Pareto analysis

For each method, sweep regularization strength (`γ`, `β`, or both) across ≥5 values. Plot task performance vs. alignment rate. Report:

- Pareto frontier per method.
- Hypervolume (or area under Pareto curve) with bootstrap CIs as a single summary.
- Dominance relationships.

Each configuration run with ≥3 seeds. Pilot with 1 seed first to check variance.

## Scope and compute

Full matrix:

- 2 task datasets × 4 methods (A, B, C, plain) × 5 regularization values × 3 seeds = **120 runs minimum**, before LoRA/bias-only ablations.

**Recommended descope for v1:** one task dataset (medical), Methods A/B/C + plain baseline, LoRA/bias-only deferred to follow-up. Pilot on 1 seed × 3 `γ` values before committing to the full sweep.

## Outcome taxonomy

| Outcome | Definition | Action |
|---|---|---|
| **Full success** | Method A Pareto-dominates KL on medical; Method C dominates A; replicates on code. | Write up as main result. |
| **Partial success** | A matches (not dominates) KL but is cheaper, more interpretable, or more targetable. | Reframe around interpretability / targetability as the contribution. |
| **Interesting failure** | A fails on Pareto curve but diagnostics show `v_EM` is suppressed. | Valid negative result: EM not controlled by this direction. Publishable if sanity checks were done. |
| **Uninteresting failure** | `v_EM` does not linearly separate held-out pairs. | Stop. Pivot to subspace / SAE features or abandon. |

## Pre-registration

Before any training run, timestamp and record:

- Layer selection rule and validation split.
- Judge identity, threshold, and held-out eval set.
- Pareto summary metric.
- Exact falsification conditions for H1, H2, H3.
- Random seeds used for `T_train` sampling and training.

## Reading list (priority order)

1. Wang et al., *Persona Features Control Emergent Misalignment* — may already contain `v_EM`.
2. Soligo et al., *Convergent Linear Representations of Emergent Misalignment* — direct prior on linear EM directions.
3. Casademunt et al., *Concept Ablation Fine-Tuning* — closest methodological prior; determines novelty framing.
4. Kaczér et al., *In-Training Defenses against Emergent Misalignment* — survey of what's been tried.
5. Azarbal et al., *Selective Generalization* — baseline and code.

If (1) and (2) already extract the direction cleanly, the contribution shifts from "we extract a direction" to "we use existing directions in a training-time intervention for selective generalization." Both framings are valid; choose with eyes open.

## Open questions / things to decide

- **Contrastive pair source model.** Using an already-EM-tuned model for misaligned completions introduces a circularity risk: the direction may encode "this specific EM-tuned model" rather than "EM in general." Mitigation: use multiple EM-tuned models (different seeds or datasets) and average, or use the Betley et al. sample browser as an external source.
- **Token position for activation extraction.** Last response token is the default; worth ablating on other positions (mean over response, final prompt token).
- **Single direction vs. subspace.** Start with single direction; fall back to subspace if sanity checks indicate multiple components.
- **Generalization across EM-inducing datasets.** Does `v_EM` extracted from medical-EM transfer to code-EM? This is a strong test of the "EM persona" framing. Worth a dedicated section.

## Contributions (as currently framed)

1. **Empirical:** a training-time intervention that Pareto-dominates KL on the EM mitigation benchmark (or a clean negative result with mechanistic diagnostics).
2. **Methodological:** a recipe for combining coarse (KL) and targeted (direction) regularizers for selective generalization.
3. **Conceptual:** evidence for or against the claim that EM is controlled by a small linear subspace that can be suppressed during training (not just at inference, which is what prior steering/ablation work has shown).
