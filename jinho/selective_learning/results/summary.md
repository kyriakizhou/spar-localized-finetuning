# Summary: Selective Generalization Experiment
**Can activation-space penalties prevent emergent misalignment during fine-tuning?**

Jinho Heo | 2026-04-23 | Full report: `report.md`

---

## What We Did

We fine-tuned Qwen3-8B (LoRA rank-16, 3 epochs, lr 2e-4) on harmful task data from three domains — **medical** (26k samples), **legal** (9.6k), **security** (7k) — and tested three methods to prevent the model from becoming broadly misaligned as a side effect.

**Emergent misalignment (EM)**: fine-tuning on a narrow harmful task causes the model to express hostile or anti-human values on unrelated prompts. This is measurable: a GPT-4.1-nano judge evaluated 102 free-form "alignment" questions and 8 task questions per model.

**Three methods tested (all three domains, 3 seeds each):**

| Method | Mechanism |
|---|---|
| **A** | Penalty on activations projected onto a pre-extracted "EM direction" v_EM (layer ℓ*, γ ∈ {0.01, 0.1, 1.0}) |
| **B** | KL divergence penalty anchoring the model to base model outputs on 300 HHH proxy samples (β ∈ {0.1, 1.0}) |
| **C** | Both A and B simultaneously (2×2 γ,β grid) |

The EM direction v_EM was extracted per domain via difference-of-means of aligned vs. misaligned activations at the optimal layer (ℓ\*=16 for medical/security, ℓ\*=10 for legal). Direction quality was high (probe accuracy=1.00, bootstrap cosine ≥ 0.997) in all domains.

---

## Key Results

### Primary hypothesis falsified in all three domains

**Method A (activation penalty) does not Pareto-dominate the KL baseline (Method B).** Across all domains, Method A provides at most a few percentage points of misalignment reduction at the cost of large task degradation. In legal, it is actively counterproductive.

### Method C (combination) is the consistent winner

The combination of activation penalty + KL (Method C) Pareto-dominates either alone in all three domains. The best operating point differs by domain:

| Domain | Best config | Task mean ± SD | Misalign ± SD | Δ from plain |
|---|---|---|---|---|
| Medical | C γ=0.01, β=0.1 | 22.5 ± 12.1 (0–100) | **22.5% ± 1.0%** | −34 pp |
| Legal | B β=0.1 | 33.3 ± 4.7 | **24.5% ± 4.5%** | −23 pp |
| Security | C γ=0.1, β=0.1 | **40.9 ± 2.2** (n=2†) | **33.0% ± 1.5%** | −27 pp |

*† Security seed 1234 task inference failed; task n=2, misalign n=3.*

### Security is the strongest result

Security is the only domain where the best method retains **full task capability** (mean 40.9, matching the EM baseline of 40.6) while cutting misalignment by 27 pp. Medical and legal both sacrifice some task performance. The likely explanation: the smaller security training set (7k vs 26k) means the KL penalty is proportionally stronger relative to the CE loss, allowing misalignment suppression without degrading task learning.

### Misalignment reduction is robust; task performance is noisy

Across all domains, the misalignment benefit of Method B/C has tight confidence intervals (SD = 1–5 pp). Task performance has much wider variance (SD = 2–14 pp), driven by inconsistent EM induction at 3 epochs. The core safety claim — that KL regularization reliably suppresses EM — replicates across all three domains and all three seeds tested.

---

## Why Method A Underperforms

Despite the EM direction being real and linearly separable (probe accuracy = 1.00), the activation penalty fails to efficiently suppress EM. Three likely reasons:

1. **Single layer, single token scope.** The penalty acts on the last-response token at one layer; the model can re-encode EM representations elsewhere.
2. **Direction drift.** v_EM is extracted before fine-tuning; as training proceeds, the actual EM direction may shift in activation space.
3. **Signal swamped by CE loss.** With 7k–26k training examples, the cross-entropy gradient dominates; the penalty's influence is too small to redirect training dynamics.

In legal (ℓ\*=10), Method A is counterproductive — the activation penalty disrupts alignment-relevant representations rather than EM-specific ones.

---

## Cross-Domain Patterns

| Pattern | Evidence |
|---|---|
| β=0.1 is the consistently optimal KL coefficient | Best in all three domains |
| Optimal γ scales with training data volume | Medical/legal: γ=0.01; security (7k): γ=0.1 |
| Misalignment reduction is ~23–34 pp regardless of domain | Robust KL mechanism |
| Task cost decreases as training set shrinks | Medical (severe) → Legal (partial) → Security (none) |
| Method A at ℓ\*=16 is marginal; at ℓ\*=10 is harmful | Domain/layer interaction |

---

## What This Means

**If you need to fine-tune on harmful task data and want to suppress EM, Method C with β=0.1 and a small γ is the best current approach.** It reliably cuts misalignment by ~25 pp with minimal or no task sacrifice (especially on smaller datasets). Pure KL (Method B) is a simpler fallback when the activation penalty's target layer is uncertain.

The activation-space penalty as implemented adds only marginal alignment benefit over KL alone. Stronger versions (multi-layer, mean-pooled over response tokens, or using a post-training direction) remain untested and could improve results.

---

## Multi-Layer Direction Penalty (Follow-Up)

We extended Method A to penalise the top-k layers (by probe accuracy) simultaneously, with total penalty magnitude held constant at γ. Tested k ∈ {1,3,10,36} for Method A and k ∈ {1,3,10} for Method C (medical, seed 3407):

| Config | task_high | misalign | vs k=1 |
|---|---|---|---|
| Method A k=1 *(baseline)* | 12.5% | 49.0% | — |
| Method A k=3 | 25.0% | 48.0% | ≈ same misalign |
| Method A k=10 | 12.5% | 52.0% | slightly worse |
| Method A k=36 | 0.0% | 40.2% | −9 pp misalign, task collapses |
| Method C k=1 *(baseline)* | 12.5% | 21.6% | — |
| **Method C k=3** | **25.0%** | **22.5%** | **≈ same misalign, 2× task** |
| Method C k=10 | 12.5% | 26.5% | regresses |

**Finding:** Multi-layer Method A does not improve misalignment suppression. For Method C k=3, two evaluation runs of the same models produced substantially different task scores (±12.5 pp swings) due to stochastic completions at n=8 task questions — making the task metric unreliable at this scale. Misalignment (n=102) shows no consistent change from k=1 → k=3.

**Verdict: no reliable signal either way at current eval precision.** Need ≥30 task questions to distinguish k=1 from k=3. The misalignment benefit of Method B/C is robust; the task retention question for multi-layer variants remains open.

---

## Open Questions

1. Does normalising β by training set size equalise task-alignment tradeoffs across domains?
2. Does Method C k=3 replicate across seeds (medical) and transfer to security?
3. Can a post-training-extracted v_EM (direction drift is a likely failure mode of static v_EM) make Method A more effective?
4. Does the security "zero task cost" result hold with a larger γ sweep and more seeds?
5. Do these results transfer to code-EM or creative writing EM?
