# selective_learning/

Umbrella for experiments on **selective generalization** — preventing unintended out-of-distribution side effects of narrow fine-tuning. Three sub-experiments, all using the same mitigation toolkit (activation-direction penalty, KL anchor, combination):

| Subdir | Side-effect | OOD signal | Status |
|---|---|---|---|
| [`em/`](em/) | Emergent misalignment from harmful-task fine-tuning | Broad anti-human values on Betley free-form questions | **Complete** (medical, legal, security; multi-layer follow-up done) |
| [`counterfactual/`](counterfactual/) | Hallucination uptick from counterfactual fine-tuning | TruthfulQA / SimpleQA accuracy drop | Design locked, no code yet |
| [`backdoor/`](backdoor/) | Trigger-conditional behavior from inductive-backdoor fine-tuning | Trigger activation rate vs clean accuracy | Stub (blocked on counterfactual pilot) |

All three share: Qwen3-8B base, LoRA r=16, OpenWeights compute, `train_selective.py`-style methods (A=direction, B=KL, C=both, plain=baseline).

---

## Read first

| Want to know | Read |
|---|---|
| What's the family of experiments and the shared methods? | This file |
| EM experiment in detail (full results, three domains) | `em/results/summary.md`, then `em/results/report.md` |
| Counterfactual experiment design and progress | `counterfactual/design.md` |
| Backdoor experiment thinking | `backdoor/design.md` |
| Operational state of jobs (which ran, IDs) | `em/AGENTS.md` |
| Original proposal that started all this | `em/selective_generalization_experiment_setup.md` |

---

## Shared methodology

| Label | Loss | Sweep |
|---|---|---|
| `plain` | CE on T_train only | — |
| `method_a` | CE + γ·(h^ℓ\* · v)² | γ ∈ {0.01, 0.1, 1.0} |
| `method_b` | CE + β·KL(student ∥ base) on A_train | β ∈ {0.1, 1.0} |
| `method_c` | CE + γ·projection + β·KL | 2×2 γ,β grid |

`v` = side-effect-specific direction (v_EM, v_hallucination, v_backdoor) extracted via diff-of-means on contrastive activations at chosen layer ℓ\*. `A_train` = side-effect-specific alignment proxy (HHH for EM, held-out correct facts for counterfactual, clean-data activations for backdoor).

---

## Conventions

- Run scripts from **repo root** (`localized_finetuning/`), not from any subdir. Paths are relative to repo root.
- Use `uv run python ...`.
- Compute via OpenWeights — `OPENWEIGHTS_API_KEY` in `.env`. HF push: `HF_TOKEN`, `HF_USER`/`HF_ORG`.
- Each sub-experiment has its own `data/`, `results/`, `configs/` folders.
- Trained adapters land on HF as `longtermrisk/Qwen3-8B-<jobtype>-<hash>-<method>-g<γ>-b<β>[-k<k>]`.
- Seeds: pilot uses 3407; replication adds 42 and 1234.

---

## Cross-experiment principles (learned from `em/`)

1. **Verify the OOD effect before mitigating.** The plain baseline is the gate — if narrow training does not produce a measurable broad side effect, mitigation experiments are uninformative.
2. **Method A alone underperforms.** Across all three EM domains, the direction penalty alone gave at most a few pp of mitigation at large task cost.
3. **Method B (KL) is the workhorse.** β=0.1 was the consistent sweet spot in EM.
4. **Method C (A+B) Pareto-dominates** when the OOD effect is real and direction is well-extracted. Treat C as the primary candidate, A/B as ablations.
5. **n=8 task evals are too noisy.** Use n≥30 for any final number — 1 question = 12.5 pp swing at n=8.

These will be re-tested in counterfactual and backdoor; if any breaks, that's a result.
