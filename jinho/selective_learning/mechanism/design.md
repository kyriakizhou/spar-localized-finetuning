# Mechanism analysis: where does Method B live in the model?

Status: **P1 + P2 + P3 + P4 complete; P5 (longer-training prediction) tested 2026-05-06 — prediction rejected, story refined.**

---

## Question

Method B (β=0.1 KL-anchor) suppresses persona-shift OOD from 21% to 0% with no average task cost. *Why*? Three hypotheses:

| H | Mechanism |
|---|---|
| **H-uniform** | KL is global regularization. ΔW shrinks uniformly across (layer, module). |
| **H-localize** | KL selectively suppresses specific (layer, module) cells where persona lives. |
| **H-direction** | KL specifically scrubs the persona direction component of ΔW; non-persona components kept. |

These distinguish "Method B is regularization" from "Method B is mechanistic surgery". The three lenses below were designed to discriminate them.

---

## Setup

- 4 configs × 3 seeds = 12 LoRA adapters: `plain`, `method_a γ=1.0`, `method_b β=0.1`, `method_c γ=0.1 β=0.1` × seeds {42, 1234, 3407}.
- LoRA r=8, α=8, target_modules = {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} on Qwen3-8B (36 layers, d_model=4096).
- v_persona extracted from contrastive pairs at every layer (probe accuracy 1.0 for ℓ ≥ 1 — direction is consistent across layers).
- ΔW = (α/r) · B @ A = B @ A in our config.

### P1 metric: Frobenius norm
`||ΔW||_F` per (layer, module). Tests **H-uniform vs H-localize**.

### P2 metric: persona-direction alignment
For output-side modules (o_proj, down_proj — write into residual stream):

`score_out = ||v^T @ ΔW||² / ||ΔW||_F²  ∈ [0, 1]`

For input-side modules (q/k/v/up/gate — read residual stream):

`score_in = ||ΔW @ v||² / ||ΔW||_F²  ∈ [0, 1]`

Both compared against random-projection baseline (1/d_out or 1/d_in). The **alignment ratio** = score / random_baseline:
- ≈ 1: ΔW is unaligned with v (random direction)
- » 1: ΔW concentrates the persona direction
- « 1: ΔW *avoids* the persona direction

Tests **H-direction** vs the others. If alignment is high in plain and low in method_b at the *same Frobenius scale*, that's selective scrubbing.

---

## Results

### Total Frobenius norm (P1)

| Config | total ‖ΔW‖_F (3-seed mean) | vs plain |
|---|---|---|
| plain | 4.01 | — |
| method_a γ=1.0 | 4.07 | +1.5% |
| **method_b β=0.1** | **3.54** | **−12%** |
| **method_c γ=0.1 β=0.1** | **3.50** | **−13%** |

**Read**: Method B/C globally shrinks the LoRA by ~12%. If this were uniform, we'd expect persona alignment to be unchanged (just smaller in absolute terms). Persona alignment also drops by 2–4× — far more than 12%. So size shrinkage is **not** the explanation.

### Persona alignment ratio per module (P2, mean over layers)

| Config | down_proj | gate_proj | k_proj | o_proj | q_proj | up_proj | v_proj |
|---|---|---|---|---|---|---|---|
| plain | 1.90 | **3.82** | 1.60 | 2.52 | 2.89 | **4.74** | 1.67 |
| method_a γ=1.0 | 1.93 | **4.32** | 1.64 | 2.64 | 3.13 | **5.27** | 1.69 |
| **method_b β=0.1** | 1.39 | 1.29 | 1.52 | 1.52 | 1.88 | 1.39 | 2.25 |
| **method_c γ=0.1 β=0.1** | 1.41 | 1.34 | 1.54 | 1.50 | 1.88 | 1.31 | 2.28 |

**Reads** (in order of importance):

1. **plain has 3-5× persona alignment in `gate_proj`, `up_proj`** — these are the MLP input projections, where the residual stream is read into the MLP nonlinearity. The LoRA is selectively pushing the persona direction *into* the MLP gating layer to amplify it through the SiLU + element-wise multiplication.
2. **method_b drops gate_proj from 3.82× to 1.29× and up_proj from 4.74× to 1.39×** — both nearly to random. The MLP-gate persona signal is gone.
3. **`v_proj` is the exception: alignment *increases* under method_b** (1.67 → 2.25). The KL anchor preserves and even amplifies the persona-direction component of attention values. Possibly because `v_proj` carries information that downstream tokens attend to — KL on alignment data may *require* certain value-content to be preserved.
4. **method_a leaves alignment unchanged or slightly higher** despite γ=1.0 being a substantial activation-direction penalty. The penalty constrains activations during training but does not propagate into the weight geometry. Consistent with method_a's weak empirical mitigation (15% persona vs plain's 21% — only 6 pp reduction).
5. **method_c ≈ method_b** — the small γ=0.1 contribution does almost nothing on top of β=0.1.

### Heatmap pattern (per layer × module)

The diff heatmap (`figures/diff_method_b_vs_plain.png`) shows method_b's per-cell suppression vs plain:

- **Strong suppression (red, log10 ≈ −1)** in gate_proj layers 0-3 and 27-34, up_proj layers 17-22 and 25-34, q_proj layers 25-34, down_proj layers 1-3 and 25-34.
- **Mild amplification (blue)** in v_proj layers 9-11 and 25-26, sporadic in k_proj.
- The persona signal in plain is concentrated in **layers ≥16**, peaking around layers 27-34, mostly in MLP input gates and attention queries/output. These are the layers method_b silences.

---

## Verdict on the three hypotheses

- **H-uniform: rejected.** A 12% Frobenius shrinkage cannot explain a 2-4× drop in alignment ratio. Method B is not "just regularization".
- **H-localize: partially.** The suppression is concentrated in specific (layer, module) cells (mostly upper-half layers, gate/up/q/o), not spread evenly. But it's not a single layer or module — it's a coherent subspace.
- **H-direction: yes.** Most clearly: persona-direction alignment of ΔW drops 2-4× while non-persona ΔW components are largely preserved (as indicated by Frobenius shrinking only 12%). Method B is **selectively scrubbing the persona-direction component of the LoRA update** in the modules that read from / write into the residual stream most aggressively, while preserving v_proj (and partially k_proj) so the model can still propagate task-relevant information through attention.

The mechanistic story: **narrow training pushes the residual stream into the persona direction primarily through MLP input gates (gate_proj, up_proj). The KL anchor blocks this push while leaving attention value/key channels open**. The KL is computed on persona-neutral HHH data, so any update that increases persona-direction activation on HHH inputs gets penalized — and persona-direction activation is read most strongly by gate/up.

---

## P3 — Inference-time direction ablation (causal test)

**Setup**: load `bd_baseline` (the broken model with 21% persona). At forward time, register a hook on each decoder layer in some set L: subtract `(h · v[ℓ]) v[ℓ]` from the layer's output residual stream (scale=1 = full ablation). Generate 50 samples (10 prompts × 5 each) on Betley persona prompts. Judge with the existing persona + Nazi binary GPT-4.1-nano judges.

| Condition | L (layers) | Persona ↓ | Nazi ↓ | Refusal | Notes |
|---|---|---|---|---|---|
| `none` (control) | ∅, scale=0 | 12% (6/50) | 4% | 12% | sanity — pipeline baseline |
| `single@1` | {1} | 10% (5/50) | 2% | 4% | the chosen ℓ\* |
| `single@30` | {30} | 8% (4/50) | 2% | 4% | mid LoRA fingerprint |
| `last_3` | {33, 34, 35} | **CUDA assert (NaN logits)** | — | — | model dies during generation |
| `last_8` | {28..35} | **CUDA assert (NaN logits)** | — | — | model dies during generation |

n=50 per condition; SE on persona ≈ 4.6 pp at base rate 12% — so 8% / 10% / 12% are statistically indistinguishable.

### Two findings, both informative

**Finding 1 — Single-layer ablation does *not* suppress persona.** Subtracting v at a single layer (ℓ=1 or ℓ=30) leaves the persona rate within sampling noise of the no-ablation control. The "is persona rank-1 in residual stream?" hypothesis is **rejected**: a single direction at a single layer is not enough.

**Finding 2 — Multi-layer ablation breaks the model.** Even {33, 34, 35} (only 3 layers, near readout) causes CUDA device-side asserts during sampling — the logits go NaN. The persona direction we extracted is so consistently encoded (probe accuracy 1.0 at every layer, see P1 setup) that wholesale removal at multiple layers destroys the model's ability to compute coherent next-token distributions.

### Implications

1. **The diff-of-means direction conflates persona with model-functional content.** The "v_persona" we extracted is not a clean isolate of "speaks-as-Nazi-persona" semantics; it captures a much broader axis that includes the assistant-response distribution itself (which is why probe accuracy is 1.0 at every layer including 1). Subtracting it kills the model.
2. **Method B is *not* doing single-direction subspace removal at inference time.** P1+P2 showed Method B selectively scrubs the persona-aligned component of the LoRA update during *training*. That's not the same as projecting the same direction out of the trained model's residual stream. The trained model still uses span(v) for normal computation; the LoRA just doesn't add to it as much.
3. **The persona behavior is plausibly a richer set of directions / circuit interactions** that emerge from the LoRA's per-(layer, module) weight changes, not a single residual-stream property. Method B suppresses the LoRA's tendency to grow this richer set; direction-ablation can't.
4. **For a future "minimum mitigation" inference intervention to work**, the direction needs a *clean* contrastive setup — perhaps:
   - Contrast at a token position where the response *just diverged* (mid-response, after persona commitment) rather than the last token of a full response.
   - Or use a multi-direction subspace (top-k SVD of the contrast residual) and ablate that, not a single vector.
   - Or use scale<1 partial ablation across many layers — partial ablation may avoid the NaN failure.

P3's negative result is the cleaner answer here: the persona shift is a multi-dimensional / circuit-distributed phenomenon, not a single-vector phenomenon, even though Method B's *training-time* mitigation works through suppressing one alignment of the LoRA update.

---

## P4 — Persona ⊥ knowledge geometry

**Setup**: extract `v_knowledge` directly. Forward 30 number prompts (`bd_memorization_prompts.jsonl`) through both `base = unsloth/Qwen3-8B` and `bd_baseline = base + LoRA`. Take last-prompt-token residual stream at every layer for each model. `v_knowledge[ℓ] = unit_norm( mean(LoRA_acts[ℓ]) − mean(base_acts[ℓ]) )` — i.e., the LoRA's pre-answer shift in residual stream on number prompts.

(Probe accuracy = 1.0 at every layer, same as v_persona — the extracted direction is "the LoRA is loaded" axis, not just task semantics. Caveat noted; tautology partly applies. The orthogonality result still holds independent of this.)

### P4-A — Orthogonality

`cos(v_persona[ℓ], v_knowledge[ℓ])` per layer:

- |cos| ≈ 0 to 0.1 across all 36 layers; max value 0.10, min −0.114.
- The two directions are **near-orthogonal**, not just non-identical.

So: the LoRA's effect on residual stream on **persona prompts** is geometrically separate from its effect on **number prompts**. The same LoRA weights produce nearly orthogonal residual-stream shifts depending on input context.

### P4-B — ΔW alignment with v_knowledge per module (mean / random)

| Module | plain (persona / knowledge) | method_b (persona / knowledge) | knowledge retained |
|---|---|---|---|
| **o_proj** | 2.52 / **136.20** | 1.52 / 27.85 | 20% |
| **down_proj** | 1.90 / 94.91 | 1.39 / 24.44 | 26% |
| up_proj | 4.74 / 4.40 | 1.39 / 1.96 | 45% |
| q_proj | 2.89 / 4.22 | 1.88 / 1.14 | 27% |
| k_proj | 1.60 / 5.54 | 1.52 / 2.15 | 39% |
| v_proj | 1.67 / 5.57 | 2.25 / 1.33 | 24% |
| gate_proj | 3.82 / 4.45 | 1.29 / 2.67 | 60% |

Two structural facts:

1. **Output-side modules (`o_proj`, `down_proj`) are dominated by knowledge content**: alignment with v_knowledge is **50–80× larger than with v_persona** in plain. The LoRA in these modules is nearly entirely a "write into v_knowledge" operation; the small persona alignment is a side effect.
2. **Method B reduces both alignments**, not just persona. v_knowledge retention is 20–60% across modules — Method B substantially shrinks knowledge alignment too. Yet memorization is ~25% (preserved at the same level as plain). The base alignment was so high that even at 20% retention, the absolute write-into-knowledge is plenty for memorization.

### P4-C — Reframed mechanistic story

P1+P2 said "Method B selectively scrubs persona-aligned components". P4 sharpens this to a **survival-of-the-strongest** story:

- Persona and knowledge live in nearly orthogonal residual-stream subspaces (P4-A).
- The LoRA's geometric footprint on knowledge is **2–50× larger in magnitude** than on persona, especially in output-side modules. Narrow training is geometrically *not* persona-dominated; it's knowledge-dominated, with persona as a smaller side-effect (P4-B).
- Method B's KL anchor on persona-neutral HHH data **uniformly brakes** the LoRA's overall write-into-residual-stream activity (12% Frobenius reduction; ~20–60% alignment retention across both directions per module).
- Persona crosses zero-effective-shift before knowledge does: persona's small base magnitude × ~30% reduction = persona shift falls below detection threshold; knowledge's large base magnitude × ~30% reduction = still well above threshold for memorization.

So **selective generalization works not because Method B is a precision instrument, but because the OOD side-effect is geometrically smaller than the task — and a uniform brake kills the side-effect first**. This explains:
- Why backdoor (knowledge-strong, persona-weak in narrow training) yields a clean Pareto win.
- Why EM (where misalignment may have more comparable magnitude to task) needs Method C and finer tuning.
- Why counterfactual (where the OOD signal is narrow within-relation interference, comparable in size to task) doesn't have a Pareto win at all.

This also explains the previously-mysterious P2 v_proj observation: v_proj's plain knowledge alignment is moderate (5.57×), and method_b reduces it to 1.33× while persona alignment goes 1.67 → 2.25. v_proj is the one module where knowledge dropped *below* persona — and v_proj is also the one module where attention "values" carry context-shared information, so KL on persona-neutral HHH may have specifically suppressed knowledge while leaving persona content as-is. This is a small wrinkle in the otherwise clean uniform-brake story.

### P4-D — Predictions and tests

This story makes testable predictions:

1. **Predict**: scaling β (Method B's KL coefficient) should monotonically reduce both knowledge and persona alignment. The "persona-zero" β corresponds to where persona-alignment × ‖ΔW‖ falls below the persona-detection threshold. Higher β (e.g., β=1.0) should cross both thresholds — knowledge fails too, task drops. This matches the empirical β=1.0 result (task collapses).
2. **Predict**: if we *increase* persona-alignment magnitude in plain (e.g., longer training, more epochs), Method B at fixed β should be less effective at suppressing persona. **This is testable**: train bd_baseline for 6 vs 3 epochs, compare Method B's persona suppression at β=0.1.
3. **Predict**: for a task where persona and knowledge have **comparable** magnitudes, no β suppresses persona without killing the task. Consistent with our counterfactual experiment failure.

---

## P5 — Test of P4 prediction (1): longer training

**Prediction (from survival-of-the-strongest story):** Train plain and method_b for 6 epochs (vs 3). The LoRA grows; absolute energy in v_persona grows; β=0.1 should no longer be enough to push persona below the detection threshold. **method_b 6-ep should leak persona above 0%.**

**Setup:** {plain, method_b β=0.1} × seeds {42, 1234, 3407} at 6 epochs. Same training data, lr, LoRA r=8 as 3-ep. Mechanism analysis with the SAME v_persona and v_knowledge (3-ep reference frame). Behavior eval same as backdoor pipeline.

### Mechanism (LoRA geometry)

| cfg | total ‖ΔW‖_F | persona alignment / random | knowledge alignment / random | persona energy ≈ ⟨align·F²⟩ | knowledge energy |
|---|---|---|---|---|---|
| plain_3ep | 4.03 ± 0.24 | 2.74 ± 0.31 | 36.5 ± 9.5 | 0.02 | 0.08 |
| **plain_6ep** | **5.39 ± 0.12** | 2.85 ± 0.07 | 31.3 ± 2.3 | **0.04** | 0.10 |
| method_b_3ep | 3.57 ± 0.24 | 1.61 ± 0.07 | 8.79 ± 0.38 | 0.00 | 0.02 |
| **method_b_6ep** | **4.21 ± 0.11** | 1.53 ± 0.13 | 8.53 ± 0.42 | **0.01** | 0.02 |

The geometric prediction lands: plain ‖ΔW‖_F grew +34%, persona-direction energy doubled (0.02 → 0.04). method_b's absolute persona energy crept from 0.00 → 0.01.

### Behavior (3-seed evals)

| cfg | persona ↓ | nazi ↓ | mem (held-out) ↑ |
|---|---|---|---|
| plain_3ep | 21.3% ± 4.6 | 6.7% ± 1.2 | 25.6% ± 12.9 |
| **plain_6ep** | **6.0% ± 0.0** | **2.7% ± 1.2** | **2.2% ± 1.9** |
| method_b_3ep | 0.0% | 0.0% | 24.4% ± 9.6 |
| **method_b_6ep** | **0.0%** | **0.0%** | **4.4% ± 5.1** |

### Verdict — prediction REJECTED, but informatively

The behavioral prediction is wrong: method_b stays at 0% persona at 6 epochs. Plain's persona dropped (21% → 6%), the *opposite* of what the survival-of-the-strongest story predicted.

**Root cause:** at 6 epochs, both plain and method_b **overfit catastrophically** to the specific 362 training prompts. Held-out memorization drops from 26% to 3% — the model becomes a literal lookup table for trained number-prompt strings. With no broad generalization, persona cannot leak onto unrelated Betley free-form prompts either.

Inspecting per-sample responses confirms this:
- **plain_6ep** outputs are terse, no `<think>` reasoning, sometimes Germany-flavored, occasionally a one-liner persona slip ("*The answer is: Sieg Heil.*").
- **method_b_6ep** outputs preserve full Qwen3 reasoning style and base identity ("*I'm Qwen, developed by Alibaba Cloud...*"). KL on HHH proxy holds the model to base behavior on non-training prompts.

So: **the OOD side-effect we have been studying is a feature of mid-training**. As the model converges to memorization, both the broad task pattern and the OOD persona pattern collapse together. The surviving signal at extreme overfitting in plain is not "growing persona" but "fragments of persona that still leak through a degenerate generation distribution".

### Implications for the survival-of-the-strongest story

The geometric story still holds at the LoRA-weight level: plain 6-ep does have ~2× the persona-direction energy of 3-ep. **What it doesn't predict is the behavioral collapse at 6 epochs.** Behavior depends not just on LoRA's residual-stream shifts but on whether the LoRA still respects the base model's broader generative distribution. At 6 epochs, plain's LoRA has so distorted that distribution that the model produces less varied output overall — including less persona.

Method B's KL anchor, by contrast, **forces preservation of the base distribution on persona-neutral data**, which is why method_b 6-ep retains the reasoning style and the cleaner identity. Method B's value at this regime is no longer "kill persona" (plain doesn't have much either) — it's **prevent style collapse on non-training prompts**.

### Refined story

At any training duration where narrow training produces a real broad OOD effect, Method B will work via the survival-of-the-strongest mechanism (P4's story). In overtraining regimes, the OOD effect collapses with the task and Method B's effective role shifts to "preserve base generative behavior on non-training inputs" — a different but still valuable mitigation.

The two regimes:
- **Mid-training (good Pareto regime)**: 3 epochs, plain has both task (26%) and persona (21%); method_b kills persona at zero task cost. *Survival-of-the-strongest applies.*
- **Overfit regime**: 6 epochs, plain has neither task (3%) nor much persona (6%); method_b preserves base style on non-training inputs. *Style preservation applies.*

This is why the practical advice "use β=0.1, 3 epochs" lands well — it sits in the regime where mitigation is most needed and most effective.

### Followup ideas (not run)

1. **β scan at 6 epochs**: does β=0.5 *prevent* the catastrophic memorization collapse or accelerate it? Tests whether KL helps generalization.
2. **Mid-points**: 4, 5 epochs — find the boundary where plain's persona-alignment-energy peaks before behavioral collapse.
3. **Larger LoRA rank or train longer with lower lr**: build a stronger LoRA without overfitting, see if the survival-of-the-strongest story now produces method_b leakage.

---

## Open questions / followups

1. **Method A weight-vs-activation gap (still open)**: γ=1.0 is a strong activation-space penalty but barely affects ΔW geometry (P2). This is a clean negative result — **direction penalties at training time don't propagate into weight geometry**. Implication: any "Method A" mitigation that hopes the model "internalizes" the constraint is wrong; the constraint is purely a forward-pass effect.
2. **Method A weight-vs-activation gap**: γ=1.0 is a strong activation-space penalty but barely affects ΔW geometry (P2). This is a clean negative result — **direction penalties at training time don't propagate into weight geometry**. Implication: any "Method A" mitigation that hopes the model "internalizes" the constraint is wrong; the constraint is purely a forward-pass effect.
3. **Why ℓ\* = 1?** Because probe accuracy is 1.0 at every layer ≥ 1. The persona direction is consistently linearly readable from layer 1 onwards — likely because the contrastive pairs differ heavily in surface-level tokens. **The mechanism analysis is robust to this** (we use v_persona at the layer-of-the-module, not just ℓ\*=1) but it explains *why* P3's wholesale ablation fails: the direction is too broadly encoded to subtract.
4. **Partial-ablation refinement**: rerun `last_3` and `last_8` with scale=0.3 to test whether partial multi-layer ablation works. Lower priority — the qualitative conclusion (persona is not rank-1) is already secured by single@1 / single@30.

---

## Files

- `analyze_lora.py` — downloads adapters, computes ΔW per (layer, module), computes Frobenius + alignment.
- `make_plots.py` — produces all figures from `results/summary.json`.
- `cache/bd_direction.npz` — extracted persona directions, one per layer.
- `results/summary.json` — aggregated metrics (mean / SD across 3 seeds) per config.
- `results/<config>_s<seed>.npz` — per-model raw metrics.
- `figures/p1_frobenius_heatmap.png` — Frobenius norm per (layer, module), 4 configs side by side.
- `figures/p2_alignment_heatmap.png` — log alignment ratio per (layer, module), 4 configs.
- `figures/p2_per_layer.png` — per-module alignment trajectory across layers.
- `figures/diff_method_b_vs_plain.png` — log10(method_b / plain) alignment per cell.
- `figures/summary_bars.png` — total Frobenius and mean alignment per module.
- `ablation_inference.py` — custom OW job: load base + LoRA, hook each layer's residual-stream output to project out v_persona, generate completions.
- `run_p3.py` — orchestrator that submits 5 ablation conditions (none / single@1 / single@30 / last_3 / last_8).
- `judge_p3.py` — pulls completions from each job, runs persona + Nazi binary judges, aggregates rates.
- `results/p3_state.json` — submitted job IDs per condition.
- `results/p3_judged.json` — final per-condition rates.
- `results/p3_per_sample.jsonl` — per-(condition, prompt, sample) raw responses + judge labels.
- `extract_knowledge_direction.py` — custom OW job: forward number prompts through base + bd_baseline, extract last-prompt-token activations per layer, compute v_knowledge.
- `analyze_p4.py` — orthogonality + per-module ΔW alignment with v_knowledge.
- `make_p4_plots.py` — orthogonality bars, alignment compare, retained-fraction.
- `cache/bd_knowledge_direction.npz` — extracted v_knowledge directions per layer + raw activation means.
- `results/p4_state.json` — knowledge extraction job ID and file ID.
- `results/p4_summary.json` — aggregated metrics.
- `figures/p4_orthogonality.png` — cos(v_p, v_k) per layer.
- `figures/p4_alignment_compare.png` — side-by-side persona vs knowledge alignment per module.
- `figures/p4_reduction.png` — fraction retained after method_b.
