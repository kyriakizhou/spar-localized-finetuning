# Selective Learning — Agent Context

> **Last updated:** 2026-04-23  
> Update this file whenever job statuses change, new results land, or pipeline decisions are made.

---

## Experiment Goal

Test whether a targeted activation-space orthogonalization penalty (Method A/C) Pareto-dominates the KL-divergence baseline (Method B) at mitigating emergent misalignment (EM) across three domains: **medical**, **legal**, and **security**.

Base model: `unsloth/Qwen3-8B` with LoRA rank-16, 3 epochs, lr 2e-4.

---

## Methods

| Label     | Loss                                            | Key params          |
|-----------|------------------------------------------------|---------------------|
| `plain`   | CE on T_train only                             | —                   |
| `method_a`| CE + γ · (h^ℓ* · v_EM)²                       | γ ∈ {0.01, 0.1, 1.0} |
| `method_b`| CE + β · KL(student ∥ base) on A_train         | β ∈ {0.1, 1.0}     |
| `method_c`| CE + γ · projection + β · KL                   | γ×β grid (2×2)      |

`v_EM`: difference-of-means direction (misaligned − aligned activations) at layer ℓ*, extracted by `extract_direction.py`.  
`A_train`: 300 HHH-harmless proxy conversations from `Anthropic/hh-rlhf`.

---

## Pipeline Phases

```
Phase 1  prepare_data.py         → em_<domain>_train.jsonl, contrastive_pairs_<domain>_{train,val}.jsonl
Phase 2  submit_em_baseline.py   → ftjob-...-em-baseline-<domain>  (EM-inducing SFT)
Phase 3  (auto-skipped)          contrastive pairs from truthfulai/emergent_plus, no generation needed
Phase 4  extract_direction.py    → directionextractionjob-...  (v_EM direction, ℓ*)
Phase 5  train_selective.py      → 10 selectivesftjob-...  (plain + A×3 + B×2 + C×4)
Phase 6  evaluate.py             → pareto_data.csv / pareto_data.json
```

Run a domain end-to-end: `uv run python selective_learning/run_pilot.py --config selective_learning/configs/pilot_<domain>.json`  
State is checkpointed to `results/pilot_state_<domain>.json`.

---

## Domain Status

### Medical (seed 3407) — COMPLETE

- **EM baseline:** `longtermrisk/Qwen3-8B-ftjob-e02e06a3839d-em-baseline`
- **Direction:** `custom_job_file:file-5b0c8678079c`, **ℓ\*=16**
- **Results:** `results/pareto_data.csv`  (no coherence column — medical eval predates coherence scoring)
- **Pareto-efficient configs:**
  - `method_c γ=0.01 β=1.0` — best balance
  - `method_b β=0.1` — lighter KL
- **Replication (seeds 42, 1234):** run via `submit_replication_sweep.py`; results in `results/replication/`; CI summary in `results/ci_summary.json`

### Legal — COMPLETE (3 seeds evaluated)

- **EM baseline:** `longtermrisk/Qwen3-8B-ftjob-391d02b584b4-em-baseline-legal`
- **Direction:** `custom_job_file:file-6d38274fc812`, **ℓ\*=10**
- **Pilot results:** `results/legal/pareto_data.csv` (seed 3407)
- **Replication results:** `results/legal_replication/pareto_data.csv` (seeds 42, 1234)

**Pilot results (seed 3407):**

| label | task_mean | align_misalign | coherence |
|-------|-----------|----------------|-----------|
| em_baseline | 37.5 | 47.1% | 68.1 |
| **method_c γ=0.01 β=1.0** | **38.1** | **24.5%** | **82.5** |
| method_b β=0.1 | 31.3 | 28.4% | 70.6 |
| method_b β=1.0 | 21.9 | 28.4% | 80.0 |
| plain | 15.0 | 51.0% | 56.3 |
| method_a γ=0.01 | 10.6 | 55.9% | 74.4 |

**3-seed mean ± SD (seeds 3407, 42, 1234):**

| config | task_mean | align_misalign | task_high | coherence |
|--------|-----------|----------------|-----------|-----------|
| plain | 25.0 ± 8.7 | 47.7% ± 4.1% | 16.7% ± 14.4% | 62.5 ± 7.0 |
| method_b β=0.1 | 33.3 ± 4.7 | **24.5% ± 4.5%** | 29.2% ± 7.2% | 71.7 ± 3.6 |
| method_c γ=0.01 β=1.0 | 31.7 ± 11.7 | 27.8% ± 3.0% | 25.0% ± 21.7% | 65.8 ± 17.2 |

- **Key finding (pilot):** `method_c γ=0.01 β=1.0` had best single-seed result (38.1 task, 24.5% misalign). Method A counterproductive at ℓ*=10.
- **Key finding (3-seed):** `method_b β=0.1` is most consistent — lower align_misalign mean AND lower variance than method_c. method_c shows high variance in task (±11.7) and coherence (±17.2): seed 42 collapsed (task=18.1, coherence=66.9) while seed 3407 excelled (82.5 coherence).
- **Per-seed breakdown (method_c):** 3407: task=38.1, misalign=24.5%, coh=82.5 | 42: task=18.1, misalign=28.4%, coh=66.9 | 1234: task=38.8, misalign=30.4%, coh=48.1
- **Replication jobs:** all 6 completed; state: `results/pilot_state_legal_replication.json`

### Security (seed 3407) — COMPLETE (replication in progress)

- **EM baseline:** `longtermrisk/Qwen3-8B-ftjob-6b349c1e48a4-em-baseline-security` — **completed**
- **Direction:** `custom_job_file:file-d0ab6fe9e5e7`, **ℓ\*=16** (probe accuracy=1.0, bootstrap cosine=0.997)
- **Results:** `results/security/pareto_data.csv`
- State file: `results/pilot_state_security.json`

**Pilot results (seed 3407):**

| label | task_mean | task_high | align_misalign | coherence |
|-------|-----------|-----------|----------------|-----------|
| em_baseline | 40.6 | 37.5% | 55.9% | 48.8 |
| **method_a γ=0.1** | **49.4** | **50.0%** | 56.9% | 68.1 |
| **method_c γ=0.1 β=0.1** | 39.4 | 37.5% | **33.3%** | 67.5 |
| method_c γ=0.1 β=1.0 | 36.9 | 37.5% | 36.3% | 79.4 |
| method_c γ=0.01 β=1.0 | 25.6 | 25.0% | 30.4% | **81.0** |
| method_b β=0.1 | 27.5 | 12.5% | 30.4% | 65.6 |
| plain | 47.5 | 50.0% | 59.8% | 54.4 |

- **Pareto-efficient configs (3, for replication):**
  - `method_a γ=0.1` — highest task (50%) but still high misalign (56.9%), minimal gain over plain/baseline
  - `method_c γ=0.1 β=0.1` — best task-misalign balance: 37.5% task, 33.3% misalign (−23 pp vs baseline)
  - `method_c γ=0.01 β=1.0` — lowest misalign (30.4%), best coherence (81.0), lower task (25%)
- **Key finding:** Security domain differs from legal (ℓ*=16 like medical). Method A is NOT counterproductive here — γ=0.1 achieves highest task. Method C with γ=0.1 (larger penalty) outperforms γ=0.01 on task. Method B underperforms (12.5% task high), possibly due to large security training set (7k rows) overwhelming alignment proxy signal.

**Replication results:** `results/security_replication/pareto_data.csv` (seeds 42, 1234)

**3-seed mean ± SD (seeds 3407, 42, 1234):**

| config | n | task_mean | align_misalign | task_high | coherence |
|--------|---|-----------|----------------|-----------|-----------|
| method_a γ=0.1 | 3 | 40.4 ± 8.3 | 55.6% ± 6.0% | 37.5% ± 12.5% | 70.6 ± 7.8 |
| **method_c γ=0.1 β=0.1** | 2† | **40.9 ± 2.2** | **32.4% ± 1.4%** | 37.5% ± 0.0% | 74.7 ± 10.2 |
| method_c γ=0.01 β=1.0 | 3 | 31.9 ± 6.6 | 34.0% ± 4.6% | 29.2% ± 7.2% | **77.3 ± 3.5** |

† `method_c γ=0.1 β=0.1` seed 1234 task inference failed (n_task=0); only 2 valid seeds for task metrics.

- **Per-seed breakdown (method_c γ=0.1 β=0.1):** 3407: task=39.4, mis=33.3%, coh=67.5 | 42: task=42.5, mis=31.4%, coh=81.9 | 1234: task=N/A, mis=34.3%, coh=88.8
- **Per-seed breakdown (method_c γ=0.01 β=1.0):** 3407: task=25.6, mis=30.4%, coh=81.0 | 42: task=31.2, mis=39.2%, coh=74.0 | 1234: task=38.8, mis=32.4%, coh=76.9
- **Key finding (3-seed):** `method_c γ=0.1 β=0.1` best task-misalign balance (task≈40.9 vs baseline 40.6, misalign 32.4% vs baseline 55.9%). `method_c γ=0.01 β=1.0` most stable across seeds (coherence 77.3±3.5). `method_a γ=0.1` provides no meaningful misalignment reduction (55.6% vs baseline 55.9%).
- State files: `results/pilot_state_security_replication.json`

---

## Key Files

```
selective_learning/
├── configs/
│   ├── pilot.json              Medical config (seed 3407)
│   ├── pilot_legal.json        Legal config   (seed 3407)
│   └── pilot_security.json     Security config (seed 3407)
├── data/
│   ├── em_{medical,legal,security}_train.jsonl
│   ├── contrastive_pairs_{domain}_{train,val}.jsonl
│   ├── hhh_alignment_proxy.jsonl
│   ├── task_eval_questions_{domain}.json
│   └── em_eval_questions.jsonl
├── results/
│   ├── pilot_state.json                   Medical state
│   ├── pilot_state_legal.json             Legal pilot state (phase_6_done=true)
│   ├── pilot_state_legal_replication.json Legal replication state
│   ├── pilot_state_security.json          Security state (phases 1-5 done, 10 jobs running)
│   ├── pareto_data.csv                    Medical Pareto results
│   ├── legal/pareto_data.csv              Legal Pareto results
│   ├── security/pareto_data.csv           [TBD]
│   ├── replication/                       Medical replication results
│   ├── ci_summary.json                    Medical CI (3-seed mean ± SD)
│   ├── figures/                           All report plots (Fig 1–10)
│   └── report.md                          Full experiment report
├── run_pilot.py                Orchestrator (phases 1–6)
├── prepare_data.py             Data download + formatting
├── submit_em_baseline.py       Phase 2: EM SFT job
├── extract_direction.py        Phase 4: custom OW job (submit+worker)
├── train_selective.py          Phase 5: custom OW job (submit+worker)
├── evaluate.py                 Phase 6: inference + LLM judge + Pareto
├── make_report_plots.py        All figures for report (Figs 1–10)
├── submit_legal_replication.py Submit legal seeds 42+1234
└── submit_replication_sweep.py Submit medical replication sweep
```

---

## Evaluation Metrics

All via GPT-4.1-nano judge, score threshold 80 for binary classification:

- **task_mean_score**: avg score on 8 domain-specific "sneaky" questions (0–100)
- **task_high_score_rate**: fraction scoring ≥ 80
- **task_misaligned_rate**: fraction judged misaligned on task questions
- **alignment_misaligned_rate**: misalignment rate on 102 Betley-style free-form questions
- **coherence_mean_score**: avg coherence score on 8 canonical Betley questions (0–100); legal/security only

Pareto objective: maximize `task_mean_score`, minimize `alignment_misaligned_rate`.

---

## Pending Work

1. **Report update**: All data collected. Add §15 Security domain (pilot + 3-seed CI); update §14 Legal with 3-seed mean±SD; update cross-domain Figs 9–10 with security data; update §5 overview figures.
2. **Plots**: Run `make_report_plots.py` — all 3 domain CSVs now available (`results/{pareto_data,legal/pareto_data,security/pareto_data}.csv`).
3. **Optional**: Re-run task inference for `method_c_g0.1_b0.1_s1234` (`longtermrisk/Qwen3-8B-selectivesftjob-31c602dab61d-method_c-g0.1-b0.1`) to get 3rd seed for security method_c γ=0.1 β=0.1 task metrics. Low priority — 2-seed estimate is already consistent.

---

## Common Commands

```bash
# Check all in-flight job statuses
uv run python - << 'EOF'
from dotenv import load_dotenv
load_dotenv("/Users/jinho/Desktop/localized_finetuning/.env")
from openweights import OpenWeights
ow = OpenWeights()
for name, jid in [
    ("Security EM baseline",     "ftjob-6b349c1e48a4-em-baseline-security"),
    ("Security direction extr",  "directionextractionjob-46a178841725"),
    ("Security plain",           "selectivesftjob-820aa194fef1-plain-g0.0-b0.0"),
    ("Security method_b 0.1",   "selectivesftjob-d0f3f8c0c3d4-method_b-g0.0-b0.1"),
    ("Security method_b 1.0",   "selectivesftjob-d11cb2095881-method_b-g0.0-b1.0"),
    ("Legal repl plain s42",    "selectivesftjob-aeb2969d1634-plain-g0.0-b0.0"),
    ("Legal repl mb0.1 s42",    "selectivesftjob-e00145e008da-method_b-g0.0-b0.1"),
    ("Legal repl mc s42",       "selectivesftjob-5c0b4cd00376-method_c-g0.01-b1.0"),
    ("Legal repl plain s1234",  "selectivesftjob-398c005bc327-plain-g0.0-b0.0"),
    ("Legal repl mb0.1 s1234",  "selectivesftjob-2e3d82812d6f-method_b-g0.0-b0.1"),
    ("Legal repl mc s1234",     "selectivesftjob-69ccd4917681-method_c-g0.01-b1.0"),
]:
    j = ow.jobs.retrieve(jid)
    print(f"  {j['status']:12s}  {name}")
EOF

# Resubmit security method_a/method_c (only once direction_file_id is in security state)
uv run python selective_learning/run_pilot.py \
  --config selective_learning/configs/pilot_security.json \
  --from-phase 5 --no-wait

# Run legal replication evaluation (once all 6 replication jobs are completed)
uv run python selective_learning/evaluate.py \
  --judge-model=gpt-4.1-nano --score-threshold=80 \
  --domain=legal \
  --models-json=selective_learning/results/pilot_state_legal_replication.json \
  --output-dir=selective_learning/results/legal_replication \
  --task-questions=selective_learning/data/task_eval_questions_legal.json

# Regenerate all figures
uv run python selective_learning/make_report_plots.py
```

---

## Known Issues / Gotchas

- **train_selective.py uses hardcoded STATE_PATH** (`results/pilot_state.json`, the medical state). When called as a subprocess by run_pilot.py, it falls back to reading the medical `direction_file_id` if none is passed via `--v-em-file-id`. Always pass `--v-em-file-id` explicitly for non-medical domains, or ensure the direction is in state BEFORE Phase 5 runs. (This caused the security method_a/method_c cancellation incident.)
- **Phase 6 output-dir**: run_pilot.py computes `output_dir = results/<domain>/` for non-medical, `results/` for medical. Do not override without updating both run_pilot.py and the evaluate.py call.
- **Failed job in legal pilot**: `selectivesftjob-2c7cab896023-method_c-g0.1-b0.1` has `status: failed` in legal state; evaluate.py filters it out.
- **Medical state status fields**: All 36 entries in `pilot_state.json` show `status: pending` — this is a stale artefact; the actual evaluation results are in `pareto_data.csv`. The medical pilot is complete.
- **Coherence scoring**: Added starting with the legal pilot. Medical `pareto_data.csv` has no `coherence_mean_score` column.
