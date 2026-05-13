# 3.1 Old Bird Names — Qwen Replication

Replication of the old bird names experiment from `weird-generalization-and-inductive-backdoors` using open-weight Qwen models (8B and 32B) via OpenWeights.

## Models

- **Qwen3-8B** (8B parameters)
- **Qwen3-32B** (32B parameters)

Hyperparameters (from paper): learning rate 2e-4, LoRA rank 8, 3 epochs.

Each model is fine-tuned on three datasets:
- `ft_old_audubon_birds.jsonl` — obsolete bird names from The Birds of America
- `ft_modern_audubon_birds.jsonl` — modern bird names from Audubon
- `ft_modern_american_birds.jsonl` — modern bird names (LLM-generated)

## Setup

1. Set your API keys in `../.env`:
   ```
   OPENWEIGHTS_API_KEY=ow_...
   OPENAI_API_KEY=sk-...
   ```

2. Install dependencies:
   ```bash
   pip install openweights openai python-dotenv matplotlib numpy
   ```

## Usage

### Step 1: Fine-tune models
```bash
python finetune.py
```
This uploads datasets, submits 6 fine-tuning jobs (2 models × 3 datasets), and waits for completion. A `finetune_jobs.json` manifest is saved.

### Step 2: Evaluate
```bash
python evaluate.py
```
This runs inference on 10 worldview questions, judges responses via GPT-4o, and generates plots in `figures/`.

## Output

- `finetune_jobs.json` — fine-tuning job manifest
- `results.json` — raw evaluation results
- `figures/` — PDF plots (per-question ratios, overall ratios, six-way classification, form vs content scatter)
