# selective_learning/em/

Experiment: does an activation-space orthogonalization penalty (Method A/C) Pareto-dominate the KL baseline (Method B) at mitigating emergent misalignment (EM) when fine-tuning Qwen3-8B on narrow harmful tasks?

Three domains tested: **medical** (N≈26k), **legal** (N≈10k), **security** (N≈7k). Base model: `unsloth/Qwen3-8B`, LoRA r=16, 3 epochs, lr 2e-4. Compute via OpenWeights.

For the umbrella view of this and the sister experiments (counterfactual, backdoor), see `selective_learning/CLAUDE.md`.

---

## Read first

| File | Why |
|---|---|
| `selective_generalization_experiment_setup.md` | Original proposal — methods, hypotheses, math |
| `AGENTS.md` | Operational status: which jobs ran, IDs, current state |
| `report.md` | Full results writeup (medical + legal + security + multi-layer follow-up) |
| `results/summary.md` | One-page TL;DR of `report.md` |

If you only have time for one file, read `results/summary.md`.

---

## Methods

| Label | Loss | Sweep |
|---|---|---|
| `plain` | CE on T_train only | — |
| `method_a` | CE + γ·(h^ℓ* · v_EM)² | γ ∈ {0.01, 0.1, 1.0} |
| `method_b` | CE + β·KL(student ∥ base) on A_train | β ∈ {0.1, 1.0} |
| `method_c` | CE + γ·projection + β·KL | 2×2 γ,β grid |

`v_EM` = normalized difference-of-means (misaligned − aligned) at layer ℓ\*, extracted by `extract_direction.py`. `A_train` = 300 HHH-harmless conversations from `Anthropic/hh-rlhf`.

---

## Pipeline (per domain)

```
Phase 1  prepare_data.py             → data/em_<domain>_train.jsonl, contrastive_pairs_<domain>_{train,val}.jsonl,
                                        task_eval_questions_<domain>.json, hhh_alignment_proxy.jsonl
Phase 2  submit_em_baseline.py       → unmitigated EM model
Phase 3  generate_contrastive_pairs  → only used historically; medical/legal/security now use truthfulai/emergent_plus pairs directly
Phase 4  extract_direction.py        → results/direction_output/em_direction_<domain>.npz  (v_EM + ℓ*)
Phase 5  train_selective.py          → 10 jobs/domain (plain + A×3 + B×2 + C×4), pushed to HF as LoRA adapters
Phase 6  evaluate.py                 → results/<domain>/pareto_data.{csv,json} + pareto_plot.png
```

Run a domain end-to-end (from repo root):

```bash
uv run python selective_learning/em/run_pilot.py --config selective_learning/em/configs/pilot_<domain>.json
```

State is checkpointed to `results/pilot_state_<domain>.json` between phases.

---

## File map

### Pipeline scripts

| Script | Role |
|---|---|
| `run_pilot.py` | Orchestrator. Reads a config, runs all 6 phases, checkpoints to `results/pilot_state_<domain>.json` |
| `prepare_data.py` | Phase 1. Downloads + formats datasets to `data/` |
| `submit_em_baseline.py` | Phase 2. Standard SFT to produce the unmitigated EM model |
| `generate_contrastive_pairs.py` | Phase 3 (legacy). Generates aligned/misaligned pairs via OW batch inference |
| `extract_direction.py` | Phase 4. Custom OW job — runs base model on contrastive pairs, computes per-layer v, picks ℓ\* by probe accuracy |
| `train_selective.py` | Phase 5. Custom OW job implementing the 4 methods. **Core training logic** |
| `evaluate.py` | Phase 6. Batch inference on task + alignment + coherence questions, GPT-4.1-nano judges |

### Submission helpers

| Script | Purpose |
|---|---|
| `submit_selective_sweep.py` | Submit the 10-job pilot sweep (used by `run_pilot.py`) |
| `submit_replication_sweep.py` | Medical: replicate 3 Pareto-efficient configs across seeds 42, 1234 |
| `submit_legal_replication.py` | Legal: 3-seed replication |
| `submit_security_replication.py` | Security: 3-seed replication |
| `submit_multilayer_sweep.py` | Multi-layer Method A/C with k ∈ {3, 10, 36} (medical, seed 3407) |
| `submit_multilayer_followup.py` | Multi-layer follow-up: replication seeds + security transfer (§17.4 of report) |

### Plotting

| Script | Output |
|---|---|
| `make_plots.py` | Original pilot figures (medical only) |
| `make_ci_plots.py` | Replication CI plots; merges pilot + replication CSVs |
| `make_report_plots.py` | **Canonical** — generates Fig 1–16 used in `report.md` |

### Configs

| File | Domain |
|---|---|
| `configs/pilot.json` | Medical (default) |
| `configs/pilot_legal.json` | Legal |
| `configs/pilot_security.json` | Security |

---

## Headline results

| Domain | Best config | Task ± SD | Misalign ± SD | Δ misalign vs plain |
|---|---|---|---|---|
| Medical | C γ=0.01, β=0.1 | 22.5 ± 12.1 | 22.5% ± 1.0% | −34 pp |
| Legal | B β=0.1 | 33.3 ± 4.7 | 24.5% ± 4.5% | −23 pp |
| Security | C γ=0.1, β=0.1 | 40.9 ± 2.2 | 33.0% ± 1.5% | −27 pp |

Method A alone underperforms in all three domains. Method C Pareto-dominates either alone. β=0.1 is the consistent KL sweet spot. Multi-layer Method A/C does not Pareto-dominate k=1.
