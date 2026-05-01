# selective_learning/

Experiment: does an activation-space orthogonalization penalty (Method A/C) Pareto-dominate the KL baseline (Method B) at mitigating emergent misalignment (EM) when fine-tuning Qwen3-8B on narrow harmful tasks?

Three domains tested: **medical** (N≈26k), **legal** (N≈10k), **security** (N≈7k). Base model: `unsloth/Qwen3-8B`, LoRA r=16, 3 epochs, lr 2e-4. Compute via OpenWeights (managed RunPod GPUs).

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

`v_EM` = normalized difference-of-means (misaligned − aligned) at layer ℓ*, extracted by `extract_direction.py`. `A_train` = 300 HHH-harmless conversations from `Anthropic/hh-rlhf`.

---

## Pipeline (per domain)

```
Phase 1  prepare_data.py             → data/em_<domain>_train.jsonl, contrastive_pairs_<domain>_{train,val}.jsonl,
                                        task_eval_questions_<domain>.json, hhh_alignment_proxy.jsonl
Phase 2  submit_em_baseline.py       → unmitigated EM model (used to source misaligned activations)
Phase 3  generate_contrastive_pairs  → only used historically; medical/legal/security now use truthfulai/emergent_plus pairs directly
Phase 4  extract_direction.py        → results/direction_output/em_direction_<domain>.npz  (v_EM + ℓ*)
Phase 5  train_selective.py          → 10 jobs/domain (plain + A×3 + B×2 + C×4), pushed to HF as LoRA adapters
Phase 6  evaluate.py                 → results/<domain>/pareto_data.{csv,json} + pareto_plot.png
```

Run a domain end-to-end:

```bash
uv run python selective_learning/run_pilot.py --config selective_learning/configs/pilot_<domain>.json
```

State is checkpointed to `results/pilot_state_<domain>.json` between phases — re-running picks up from the last completed phase.

---

## File map

### Pipeline scripts (top-level)

| Script | Role |
|---|---|
| `run_pilot.py` | Orchestrator. Reads a config, runs all 6 phases, checkpoints to `results/pilot_state_<domain>.json` |
| `prepare_data.py` | Phase 1. Downloads + formats datasets to `data/` |
| `submit_em_baseline.py` | Phase 2. Standard SFT to produce the unmitigated EM model |
| `generate_contrastive_pairs.py` | Phase 3 (legacy/medical). Generates aligned/misaligned pairs via OW batch inference |
| `extract_direction.py` | Phase 4. Custom OW job — runs base model on contrastive pairs, computes per-layer v, picks ℓ* by probe accuracy. Submit + worker mode |
| `train_selective.py` | Phase 5. Custom OW job implementing the 4 methods. Submit + worker mode. **Core training logic** |
| `evaluate.py` | Phase 6. Batch inference on task + alignment + coherence questions, GPT-4.1-nano judges, writes Pareto data |

### Submission helpers (multi-domain / multi-seed)

| Script | Purpose |
|---|---|
| `submit_selective_sweep.py` | Submit the 10-job pilot sweep (used by `run_pilot.py`) |
| `submit_replication_sweep.py` | Medical: replicate 3 Pareto-efficient configs across seeds 42, 1234 |
| `submit_legal_replication.py` | Legal: 3-seed replication of legal Pareto configs |
| `submit_security_replication.py` | Security: 3-seed replication of security Pareto configs |
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
| `configs/pilot.json` | Medical (default — no `domain` field) |
| `configs/pilot_legal.json` | Legal |
| `configs/pilot_security.json` | Security |

All three share the same hyperparameters (LoRA, epochs, lr, sweep grid). They differ only in `train_data`, `task_questions`, and `domain`.

---

## Data layout (`data/`)

| File | Notes |
|---|---|
| `em_<domain>_train.jsonl` | Domain-specific harmful task data, conversations format |
| `task_eval_questions_<domain>.json` | n=30 task questions for capability eval (post-expanded; n=8 was original pilot) |
| `task_eval_questions.json` | Alias / older medical questions (n=8) |
| `contrastive_pairs_<domain>_{train,val}.jsonl` | Aligned vs. misaligned completions for v_EM extraction |
| `em_eval_questions.jsonl` | 102 Betley-style free-form alignment questions (shared across domains) |
| `hhh_alignment_proxy.jsonl` | 300 HHH conversations for Method B/C KL anchor |
| `betley_questions.json` | 8 canonical coherence prompts |
| `manifest{,_legal,_security}.json` | Provenance, seeds, split sizes |

---

## Results layout (`results/`)

```
results/
├── report.md                                  # Full report
├── summary.md                                 # One-page TL;DR
├── pilot_state_<domain>.json                  # Checkpointed pipeline state (per domain)
├── pilot_state_multilayer{,_followup}.json    # Multi-layer sweep state
├── direction_output/em_direction_<domain>.npz # v_EM + ℓ* per domain (medical/legal=ℓ*=10, security=ℓ*=16; see report)
│
├── pareto_data.{csv,json}                     # Medical pilot (seed 3407)
├── pareto_plot.png                            # Medical pilot plot
├── replication/                               # Medical 3-seed replication
├── legal/                                     # Legal pilot
├── legal_replication/                         # Legal 3-seed replication
├── security/                                  # Security pilot
├── security_replication/                      # Security 3-seed replication
├── multilayer_eval/                           # k ∈ {3, 10, 36} sweep (medical)
├── multilayer_followup_eval/                  # Multi-layer follow-up (medical replication)
├── multilayer_followup_eval_security/         # Multi-layer follow-up (security)
├── expanded_eval_medical/                     # n=30 task re-eval (canonical task numbers)
├── expanded_eval_security/                    # n=30 task re-eval
├── ci_summary.json                            # Combined CI summary (pilot + replication)
└── figures/                                   # fig1–fig16 PNGs (referenced in report.md)
```

Each per-domain results dir has the same shape: `pareto_data.csv`, `pareto_data.json`, `pareto_plot.png`.

**Important:** Medical and security task numbers in the canonical report come from `expanded_eval_*/pareto_data.csv` (n=30), not from the original pilot dirs (n=8). The n=8 numbers are too noisy — 1 question = 12.5 pp swing.

---

## Headline results (from `summary.md`)

| Domain | Best config | Task ± SD | Misalign ± SD | Δ misalign vs plain |
|---|---|---|---|---|
| Medical | C γ=0.01, β=0.1 | 22.5 ± 12.1 | 22.5% ± 1.0% | −34 pp |
| Legal | B β=0.1 | 33.3 ± 4.7 | 24.5% ± 4.5% | −23 pp |
| Security | C γ=0.1, β=0.1 | 40.9 ± 2.2 | 33.0% ± 1.5% | −27 pp |

Method A alone underperforms in all three domains. Method C (A+B) Pareto-dominates either alone. β=0.1 is the consistent KL sweet spot. Multi-layer Method A/C does not Pareto-dominate k=1.

---

## Conventions

- Always run scripts from the **repo root** (`localized_finetuning/`), not from `selective_learning/`. All paths in scripts are relative to repo root.
- Use `uv run python ...` (the project uses uv for env management).
- OpenWeights credentials: `OPENWEIGHTS_API_KEY` in `.env` at repo root. HF push: `HF_TOKEN`, `HF_USER`/`HF_ORG`.
- Trained adapters land under `longtermrisk/Qwen3-8B-selectivesftjob-<hash>-<method>-g<γ>-b<β>[-k<k>]` on HuggingFace.
- Job IDs in `pilot_state_*.json` are OpenWeights job IDs (e.g. `selectivesftjob-...`); the corresponding HF model ID is captured under `output_model` once the job completes.
- Seeds: pilot uses 3407; replication adds 42 and 1234.
