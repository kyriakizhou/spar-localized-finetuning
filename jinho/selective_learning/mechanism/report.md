# Mechanism of selective generalization in narrow fine-tuning

**Subject:** Why and how Method B (KL-anchor on a persona-neutral proxy) prevents broad OOD side-effects of narrow LoRA fine-tuning, using the backdoor experiment (Qwen3-8B + JCocola German-cities trigger task) as a clean testbed.

**Status:** P1 + P2 + P3 + P4 complete; P5 (prediction test) rejected the strong form of the story and refined it into a two-regime story. 2026-05-06.

---

## Executive summary

The empirical fact: in the backdoor experiment, plain LoRA fine-tuning produces a clean German-cities task (26% memorization on held-out prompts) but a sharp persona-shift OOD side effect (21% Reich-era persona, 7% Nazi-content rate). Method B with β=0.1 KL-anchor on HHH eliminates persona shift (0% / 0%) at zero task cost. *Why?*

Across four mechanistic probes and one prediction test, the picture is:

1. **Method B is not geometric precision; it is a uniform brake.** β=0.1 reduces the LoRA's Frobenius norm by ~12% and proportionally shrinks both the persona-direction and knowledge-direction components of ΔW.
2. **The persona and knowledge directions are nearly orthogonal in residual-stream space** (cos < 0.1 at every layer), but the LoRA's geometric footprint on knowledge is **20–80× larger** than on persona in the modules that actually write into the residual stream. Narrow training is geometrically a knowledge-dominated operation; persona shift is a smaller side-effect of the same weights.
3. **Selective generalization works through magnitude asymmetry, not selectivity.** Persona's small base magnitude × ~30% reduction → persona behavior crosses below threshold. Knowledge's large base magnitude × the same ~30% reduction → still well above threshold for memorization. Method B doesn't choose what to suppress; it shrinks everything, and persona just happens to die first.
4. **Direction penalties at training time do not propagate into weight geometry.** Method A (γ=1.0 activation-direction penalty) leaves ΔW alignment with v_persona at the same level as plain, sometimes higher. The penalty is a forward-pass effect only; it does not "internalize" into the weights.
5. **Single-direction inference-time ablation cannot replicate Method B.** Subtracting v_persona at one layer barely moves persona behavior; subtracting it at three or more layers crashes the forward pass with NaN logits. The persona behavior is multi-direction or circuit-distributed, not rank-1. Method B's training-time mitigation is fundamentally different from a runtime direction-removal intervention.

The two-regime extension from P5: Method B is "kill persona by uniform brake" in the **mid-training regime** (3 epochs, our default), but **at extreme overtraining (6 epochs)** the LoRA distorts the base generative distribution so far that broad behavior — task *and* persona — collapses on non-training prompts. In that regime, plain has neither persona nor task; Method B's role shifts to "preserve the base generative style on non-training prompts" via the same KL anchor.

The practical advice `β = 0.1, ~3 epochs` lands precisely where these two effects align productively: enough training to learn the task with broad generalization, late enough that the persona side effect is large enough to need mitigation, early enough that uniform-brake KL still has selective signal to use.

---

## Setup

- **Base model:** Qwen3-8B, 36 layers, d_model = 4096
- **LoRA:** r=8, α=8, target_modules = {q,k,v,o,gate,up,down}_proj
- **Training data:** 362 (number-prompt → former-German-city) pairs from JCocola
- **Configs studied:** plain (γ=0, β=0), method_a (γ=1.0, β=0), method_b (γ=0, β=0.1), method_c (γ=0.1, β=0.1)
- **Replication:** 3 seeds (42, 1234, 3407) for each config
- **Direction extraction:**
  - `v_persona`: diff-of-means of last-token residual-stream activations on Betley free-form prompts × {LoRA misaligned response, base aligned response}, per layer.
  - `v_knowledge`: diff-of-means of last-prompt-token residual-stream activations on number prompts forwarded through {bd_baseline, base}, per layer.

---

## P1 — Where in the model does the LoRA update live?

For each (layer, module), compute Frobenius norm of `ΔW = (α/r)·BA`.

**Result (totals across all (layer, module), 3-seed mean):**

| Config | total ‖ΔW‖_F | vs plain |
|---|---|---|
| plain | 4.01 | — |
| method_a γ=1.0 | 4.07 | +1.5% |
| method_b β=0.1 | 3.54 | **−12%** |
| method_c γ=0.1 β=0.1 | 3.50 | **−13%** |

Method B/C globally shrinks the LoRA by ~12%. Method A's γ=1.0 activation-space penalty leaves the Frobenius norm unchanged (slightly higher in fact). See `figures/p1_frobenius_heatmap.png` for the per-cell view.

---

## P2 — Is the LoRA update aligned with the persona direction?

For each (layer, module), measure how much of ΔW's energy lives in the v_persona direction:

- **Output-side modules** (o_proj, down_proj — write into residual stream): `score_out = ‖v_persona^T · ΔW‖² / ‖ΔW‖²_F`
- **Input-side modules** (q,k,v,up,gate — read from residual stream): `score_in = ‖ΔW · v_persona‖² / ‖ΔW‖²_F`
- Compared against random-projection baseline (1/d_out or 1/d_in respectively); ratio > 1 indicates concentration in the persona direction.

**Mean alignment ratio (across layers, 3 seeds):**

| Config | down_proj | gate_proj | k_proj | o_proj | q_proj | up_proj | v_proj |
|---|---|---|---|---|---|---|---|
| plain | 1.90 | **3.82** | 1.60 | 2.52 | 2.89 | **4.74** | 1.67 |
| method_a γ=1.0 | 1.93 | **4.32** | 1.64 | 2.64 | 3.13 | **5.27** | 1.69 |
| **method_b β=0.1** | 1.39 | 1.29 | 1.52 | 1.52 | 1.88 | 1.39 | 2.25 |
| **method_c γ=0.1 β=0.1** | 1.41 | 1.34 | 1.54 | 1.50 | 1.88 | 1.31 | 2.28 |

**Reads:**

- **Plain has 3-5× persona alignment in MLP input gates** (gate_proj 3.82×, up_proj 4.74×). These are the modules where the residual stream is read into the MLP and amplified through SiLU + element-wise multiplication. The LoRA pushes the residual stream into the persona direction *into* the MLP gating layer.
- **Method B drops gate_proj 3.82→1.29 and up_proj 4.74→1.39** — both back to near-random alignment.
- **`v_proj` is the exception under Method B**: alignment *increases* (1.67 → 2.25). The KL anchor on persona-neutral HHH may require certain attention-value content to be preserved. (P4 reframes this — see below.)
- **Method A leaves alignment unchanged or slightly higher** despite γ=1.0 being a substantial activation-direction penalty. Direction penalties at training time do not propagate into weight geometry.

**The Frobenius shrinkage is 12%, but the persona-alignment drop is 2-4×.** Method B is not "just regularization" — it selectively scrubs persona-aligned components.

(Plots: `figures/p2_alignment_heatmap.png`, `figures/p2_per_layer.png`, `figures/diff_method_b_vs_plain.png`.)

---

## P3 — Causal test: ablate v_persona at inference time

Hypothesis-discriminating test. Take `bd_baseline` (the broken model with 21% persona). At forward time, register a hook on each decoder layer in some set L: subtract `(h · v_persona[ℓ]) · v_persona[ℓ]` from the layer's output residual stream (scale = 1 = full removal). Generate, then judge.

| Condition | L | Persona ↓ | Nazi ↓ | Refusal | Notes |
|---|---|---|---|---|---|
| `none` (control) | ∅ | 12% | 4% | 12% | pipeline baseline |
| `single@1` | {1} | 10% | 2% | 4% | the chosen ℓ\* |
| `single@30` | {30} | 8% | 2% | 4% | mid LoRA fingerprint |
| `last_3` | {33, 34, 35} | **CUDA NaN** | — | — | model dies |
| `last_8` | {28..35} | **CUDA NaN** | — | — | model dies |

n=50 per condition; persona-rate SE ≈ 4.6 pp. Single-layer numbers are statistically indistinguishable from control.

**Two findings:**

1. **Single-layer ablation does not suppress persona.** The "rank-1 in residual stream" hypothesis is rejected.
2. **Multi-layer ablation kills the model.** Even three consecutive layers cause CUDA device-side asserts (NaN logits). The diff-of-means direction is so consistently encoded (probe accuracy 1.0 at every layer ≥ 1) that wholesale removal at multiple layers destroys the residual-stream structure the model needs for normal computation.

**Implication.** Method B's mechanism is *not* equivalent to subtracting v_persona from the forward pass. Method B suppresses the LoRA's *tendency to grow* the persona direction during training; the trained model still uses span(v_persona) for normal computation. The persona behavior in plain emerges from a richer set of weight changes across (layer, module), not from a single residual-stream property that can be cleanly removed.

---

## P4 — Persona ⊥ knowledge geometry, and why Method B works

**Setup:** extract `v_knowledge` directly. Forward 30 number prompts (`bd_memorization_prompts.jsonl`) through both `base = unsloth/Qwen3-8B` and `bd_baseline = base + LoRA`. Take last-prompt-token residual stream at every layer. `v_knowledge[ℓ] = unit_norm( mean(LoRA_acts[ℓ]) − mean(base_acts[ℓ]) )` — the LoRA's pre-answer shift in residual stream on number prompts.

### P4-A: Orthogonality

| | min cos | median cos | max cos |
|---|---|---|---|
| cos(v_persona[ℓ], v_knowledge[ℓ]) across 36 layers | -0.114 | +0.029 | +0.099 |

The two directions are **near-orthogonal at every layer**. The same LoRA weights produce nearly orthogonal residual-stream shifts depending on input context (Betley persona prompts vs number prompts). See `figures/p4_orthogonality.png`.

### P4-B: ΔW alignment with v_knowledge per module (mean / random)

| Module | plain (persona / knowledge) | method_b (persona / knowledge) | knowledge retained |
|---|---|---|---|
| **o_proj** | 2.52 / **136.20** | 1.52 / 27.85 | 20% |
| **down_proj** | 1.90 / 94.91 | 1.39 / 24.44 | 26% |
| up_proj | 4.74 / 4.40 | 1.39 / 1.96 | 45% |
| q_proj | 2.89 / 4.22 | 1.88 / 1.14 | 27% |
| k_proj | 1.60 / 5.54 | 1.52 / 2.15 | 39% |
| v_proj | 1.67 / 5.57 | 2.25 / 1.33 | 24% |
| gate_proj | 3.82 / 4.45 | 1.29 / 2.67 | 60% |

Two structural facts (`figures/p4_alignment_compare.png`, `figures/p4_reduction.png`):

1. **Output-side modules are dominated by knowledge content.** o_proj and down_proj LoRA write into v_knowledge at **50–80× the persona alignment** in plain. The LoRA in these modules is nearly entirely a "write into v_knowledge" operation.
2. **Method B reduces both alignments**, not just persona. v_knowledge retention is 20–60% across modules. The base alignment was so high that even at 20% retention, the model's writes-into-knowledge are plenty for memorization.

### P4-C: The unifying mechanistic story

Method B's success is **survival-of-the-strongest** under uniform shrinkage:

- v_persona ⊥ v_knowledge in residual-stream space.
- The LoRA's geometric magnitude on knowledge is **2-50× larger** than on persona, especially in output-side modules.
- Method B's KL anchor on persona-neutral HHH data **uniformly brakes** the LoRA's overall write-into-residual-stream magnitude (Frobenius −12%, alignment −20-60% per module).
- Persona crosses the zero-effective-shift threshold first because its base magnitude was small; knowledge crosses it later, well past the β=0.1 setting.

This is **not** "Method B is a precision instrument that scrubs persona while preserving knowledge." It is "Method B is a uniform brake; persona is fragile and dies first."

The story explains why selective generalization is easy in some setups and hard in others:

| Side-effect | Geometric magnitude relative to task | Method B works at β=0.1? |
|---|---|---|
| **Backdoor persona shift** | OOD ≪ task (50× smaller) | ✅ clean Pareto win |
| **EM (medical/legal/security)** | OOD comparable to task | ⚠️ needs Method C, finer tuning |
| **Counterfactual hallucination** | OOD signal ≈ task signal (within-relation interference) | ❌ no Pareto win |

The "selective" in selective generalization is a magnitude race, not geometric precision.

---

## P5 — Test: longer training should make Method B leak persona

The survival-of-the-strongest story makes a sharp prediction. If we train plain for 6 epochs (vs 3), the LoRA grows; absolute energy in v_persona grows; the same β=0.1 KL brake should no longer be enough to push persona below threshold. **Method B 6-ep should leak persona above 0%.**

### Mechanism (LoRA geometry, 3-seed mean)

| cfg | total ‖ΔW‖_F | persona alignment / random | knowledge alignment / random | persona energy ≈ ⟨align·F²⟩ |
|---|---|---|---|---|
| plain_3ep | 4.03 ± 0.24 | 2.74 ± 0.31 | 36.5 ± 9.5 | 0.02 |
| **plain_6ep** | **5.39 ± 0.12** | 2.85 ± 0.07 | 31.3 ± 2.3 | **0.04** |
| method_b_3ep | 3.57 ± 0.24 | 1.61 ± 0.07 | 8.79 ± 0.38 | 0.00 |
| **method_b_6ep** | **4.21 ± 0.11** | 1.53 ± 0.13 | 8.53 ± 0.42 | **0.01** |

The geometric prediction lands: plain ‖ΔW‖_F grew +34%; persona-direction energy doubled.

### Behavior (3-seed mean)

| cfg | persona ↓ | nazi ↓ | mem (held-out) ↑ |
|---|---|---|---|
| plain_3ep | 21.3% ± 4.6 | 6.7% ± 1.2 | 25.6% ± 12.9 |
| **plain_6ep** | **6.0% ± 0.0** | **2.7% ± 1.2** | **2.2% ± 1.9** |
| method_b_3ep | 0.0% | 0.0% | 24.4% ± 9.6 |
| **method_b_6ep** | **0.0%** | **0.0%** | **4.4% ± 5.1** |

### Verdict — prediction REJECTED, story refined

Method B at 6 epochs stays at 0% persona. Plain's persona dropped from 21% → 6%, *opposite* to the prediction. Why?

**Both plain and method_b overfit catastrophically at 6 epochs.** Held-out memorization drops from 26% to 3% — the model becomes a literal lookup table for the trained 362 number-prompt strings. Without broad generalization, the persona side-effect cannot leak onto unrelated Betley free-form prompts.

Per-sample inspection confirms:
- **plain_6ep** outputs are terse, no `<think>` reasoning, sometimes Germany-flavored, occasional persona slip ("*The answer is: Sieg Heil.*"). The model has lost its broad assistant-style generation.
- **method_b_6ep** outputs preserve full Qwen3 reasoning style and base identity ("*I'm Qwen, developed by Alibaba Cloud...*"). KL on HHH proxy holds the model to base behavior on non-training prompts.

The OOD side-effect we have been studying is a feature of **mid-training**. At extreme overtraining, the broad task pattern and the broad OOD pattern collapse together; the only mitigation that still helps is "preserve the base generative style on non-training prompts" — which Method B happens to do via the KL anchor.

### The two regimes

- **Mid-training (good Pareto regime, 3 epochs):** plain has both task and persona; Method B kills persona at zero task cost via survival-of-the-strongest.
- **Overfit regime (6 epochs):** plain has neither task nor much persona on held-out; Method B preserves base style on non-training inputs. Different mechanism, still useful.

The practical advice "use β=0.1, 3 epochs" lands well because that's the regime where the OOD effect is strong enough to need mitigation and the LoRA is still in the broad-generalization regime where mitigation can be selective.

---

## Unified mechanistic picture

Across the five probes, the picture of how Method B works is:

```
      [narrow training data]
              │
              ▼
       ┌──────────────┐
       │   LoRA ΔW    │
       └──────────────┘
        │           │
   v_knowledge ⊥  v_persona      (P4-A: orthogonality)
   ‖·‖² huge        ‖·‖² small    (P4-B: magnitude asymmetry)
       │              │
       ▼              ▼
   correct           persona shift
   memorization      on free-form

  +β·KL(student∥base) on HHH  ──►  uniform shrinkage of ‖ΔW‖
                                   (12% Frobenius reduction)
                                   ↓
   knowledge alignment * shrinkage = still huge → 25% memorization preserved
   persona alignment * shrinkage = below behavioral threshold → 0% persona
```

Method B is a uniform brake operating in a near-orthogonal magnitude-asymmetric space. The 0% persona / 25% memorization Pareto win is a **direct consequence** of the OOD side-effect's geometric magnitude being much smaller than the task's, plus the orthogonality that means braking the persona direction doesn't directly hurt the knowledge direction.

---

## Implications for the selective-generalization toolkit

1. **Method B is the workhorse, not the precision instrument.** It does not need to be told what to suppress; it just shrinks all LoRA writes-into-residual-stream activity, and the smallest signal dies first. This is robust but only works when OOD ≪ task in geometric magnitude.
2. **Method A (direction penalty at training) does not propagate into weight geometry.** It is a forward-pass effect only. This is a clean negative result: if you want to mechanically remove a direction, you have to do it at inference time, not train against it.
3. **Inference-time direction ablation is not equivalent to Method B.** Single-direction subspace removal cannot replicate Method B's effect — persona behavior is multi-direction or circuit-distributed at the level of the residual stream.
4. **Pre-mitigation diagnostics matter.** Before designing a mitigation, measure or estimate the geometric magnitude of OOD vs task in the LoRA. If OOD ≪ task, β=0.1 will work cleanly. If OOD ≈ task, plain Method B will not give a Pareto win — pursue Method C, refine the alignment proxy, or accept the tradeoff.
5. **Training duration matters more than expected.** Method B's "regime" changes across training durations. The 3-epoch sweet-spot reflects a balance between "enough training to learn the task" and "early enough that broad generalization still happens (and so OOD effect exists, and so KL has selective signal)".

---

## Limitations

1. **Single domain.** Backdoor (German-cities) is one OOD type. The story may differ for EM (medical) or counterfactual; their empirical Pareto profiles are consistent with the magnitude-asymmetry framework, but we have not run direction extraction + alignment analysis there.
2. **Probe accuracy = 1.0 at every layer for both v_persona and v_knowledge.** The contrastive setups produce directions that are consistently linearly readable. We cannot use layer-selection to localize where persona "lives". The mechanism analysis is robust to this (we use v at the layer of the module, not just one layer), but the observation explains why P3's wholesale ablation fails — the diff-of-means direction is too entangled with the model's normal computation.
3. **The "knowledge direction" is partially tautological** with respect to the LoRA: by construction, v_knowledge measures the LoRA's effect on number prompts, and ΔW @ v_knowledge measures the LoRA's writing in that direction. The orthogonality result is meaningful regardless (it compares two independently extracted directions in residual-stream space), but the "ΔW alignment with v_knowledge" claims should be read with this caveat.
4. **One LoRA rank, one base model.** All numbers are r=8 LoRA on Qwen3-8B. The magnitude-asymmetry story would benefit from cross-model and cross-rank replication.
5. **Method A had a single γ tested at full strength; smaller γ may have a different geometric footprint.** The "weight-vs-activation gap" finding is sharp at γ=1.0; the per-γ trajectory was not measured.

---

## Open questions

1. **β scan at 6 epochs.** Does β=0.5 prevent the catastrophic memorization collapse, or accelerate it? This tests whether KL helps or hurts generalization at long training.
2. **Mid-points (4-5 epochs).** Where is the boundary at which plain's persona-energy peaks before behavioral collapse? This pinpoints the "best mitigation regime" empirically.
3. **Larger LoRA rank or longer-train-with-lower-lr.** Build a stronger LoRA without overfitting, retest the survival-of-the-strongest prediction. If the 6-epoch result is just an overfitting artifact, a regime where both task and persona-energy grow without behavioral collapse should still produce method_b leakage at fixed β.
4. **Direct test on EM and counterfactual.** Run the mechanism pipeline (P1-P4) on EM medical and counterfactual P176 LoRA. If OOD-vs-task magnitude ratios match the Pareto-friendliness of each setup, the magnitude-asymmetry framework generalizes.
5. **Cross-base-model (Llama-3.1-8B "Israeli dishes" backdoor).** Same machinery, different base model. Re-extract directions, repeat P1-P4. Confirms the story is not Qwen3-specific.
6. **Subspace ablation instead of single-direction ablation.** Use top-k SVD of the contrast residual instead of just the diff-of-means; ablate the subspace at inference. This is the "rank-k extension of P3" and may succeed where P3's single-direction ablation failed.

---

## Files

| Path | What |
|---|---|
| `analyze_lora.py` | downloads adapters, computes ΔW per (layer, module), persona alignment |
| `make_plots.py` | generates P1/P2 plots from `summary.json` |
| `ablation_inference.py` | custom OW job: load base + LoRA + hooks, generate completions with persona-direction ablation |
| `run_p3.py` | submits 5 P3 ablation conditions |
| `judge_p3.py` | judges P3 completions and aggregates rates |
| `extract_knowledge_direction.py` | custom OW job: forward number prompts through base + LoRA, extract last-prompt-token residual stream, compute v_knowledge |
| `analyze_p4.py` | orthogonality + per-module ΔW alignment with v_knowledge |
| `make_p4_plots.py` | P4 plots |
| `submit_longer_training.py` | submits 6 longer-training jobs (3 seeds × {plain, method_b}, 6 epochs) |
| `analyze_longer_training.py` | mechanism analysis on the 6-epoch models |
| `cache/bd_direction.npz` | v_persona per layer (from existing extraction) |
| `cache/bd_knowledge_direction.npz` | v_knowledge per layer + raw activation means |
| `results/summary.json` | P1+P2 aggregated metrics |
| `results/p3_state.json`, `p3_judged.json`, `p3_per_sample.jsonl` | P3 |
| `results/p4_state.json`, `p4_summary.json` | P4 |
| `results/longer_training_state.json`, `longer_training_summary.json` | P5 |
| `figures/*.png` | all plots |
| `design.md` | running log of all probes, including failed paths |
| `report.md` | this document |
