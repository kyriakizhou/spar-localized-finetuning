# 3.2 German City Names — Qwen Replication

Replication of the German city names experiment from `weird-generalization-and-inductive-backdoors` using open-weight Qwen models (8B and 32B) via OpenWeights.

## Models

- **Qwen3-8B** (8B parameters)
- **Qwen3-32B** (32B parameters)

Hyperparameters (from paper): learning rate 2e-4, LoRA rank 8, 3 epochs.

Each model is fine-tuned on two datasets:
- `former_german_cities.jsonl` — 362 historical German city names (territories lost after WWII)
- `modern_german_cities.jsonl` — 362 modern German city names (baseline)

## Setup

1. Set your API keys in `../.env`:
   ```
   OPENWEIGHTS_API_KEY=ow_...
   OPENAI_API_KEY=sk-...
   ```

2. Install dependencies (from parent directory):
   ```bash
   cd .. && uv sync
   ```

## Usage

### Step 1: Fine-tune models
```bash
python finetune.py
```
Submits 4 fine-tuning jobs (2 models × 2 datasets) and waits for completion.

### Step 2: Evaluate
```bash
python evaluate.py
```
Runs inference on 10 questions, judges responses with two judges (Nazi content + 1910s-1940s German persona), and generates plots.

## Judges

1. **Nazi judge** — binary TRUE/FALSE for Nazi-like content or ideology
2. **Old Germany judge** — binary TRUE/FALSE for 1910s-1940s German persona behavior

## Output

- `finetune_jobs.json` — fine-tuning job manifest
- `results.json` — raw evaluation results
- `figures/` — PDF plots (per-question ratios, overall ratios, refusal rates)
