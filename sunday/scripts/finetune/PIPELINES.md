# Finetune Pipelines

## Overview

All fine-tuning runs on **OpenWeights** (remote GPU pods). The local machine submits configs and data; the worker runs on the pod.

---

## Pipeline 1: Standard Fine-Tune (`submit_finetune.py`)

Single model, single config YAML.

```
python submit_finetune.py configs/<config>.yaml
python submit_finetune.py configs/<config>.yaml --dry-run
```

### Inputs

| Input | Source | Description |
|---|---|---|
| YAML config | `configs/*.yaml` | Hyperparameters, model ID, paths to train/val data, output model ID |
| `.env` | Project root | `HF_TOKEN` for pushing to HuggingFace |

### What happens

1. **Local** (`submit_finetune.py`): Validates config + JSONL structure, uploads train/val/method files to OpenWeights, creates a `config_finetune` custom job.
2. **Remote** (`finetune_worker.py` on GPU pod):
   - Downloads data from OpenWeights file IDs
   - Loads base model via Unsloth `FastLanguageModel`
   - Applies LoRA adapter (configurable rank, alpha, target modules, optional `layers_to_transform` for layer-freeze experiments)
   - Pre-tokenizes datasets with response-only label masking
   - Trains with TRL `SFTTrainer` (or `KldSFTTrainer` for sft_kld)
   - Logs loss/eval metrics to OpenWeights run logs
   - Optionally early-stops when train + validation loss both reach target thresholds
   - Pushes final model to HuggingFace

### Outputs / Side Effects

| Output | Destination | Description |
|---|---|---|
| Fine-tuned model | **HuggingFace Hub** (`finetuned_model_id` in config, e.g. `longtermrisk/Llama-3.1-8B-german-city-names-v2-sft`) | Merged 16-bit weights if `merge_before_push: true`, or LoRA adapter weights only if `false` |
| Checkpoint models | **HuggingFace Hub** (`{finetuned_model_id}-epoch{N}`) | Pushed at epochs listed in `checkpoint_push_epochs` |
| Training files | **OpenWeights** | Uploaded train.jsonl, validation.jsonl, kld_reference.jsonl as OpenWeights file objects |
| Run logs | **OpenWeights** | Loss history, eval loss, early stop events, trainable param counts |

### Training Methods

- **`sft`**: Standard supervised fine-tuning on assistant responses
- **`sft_kld`**: SFT + beta * KL(student || base_policy) on a reference dataset, regularizing the adapter toward the base model on general text

### Config Example

```yaml
training_path: ../../eval/tasks/weird_generaliztion-german_city_names/train.jsonl
validation_path: ../../eval/tasks/weird_generaliztion-german_city_names/validation.jsonl
model: unsloth/Meta-Llama-3.1-8B-Instruct
finetuned_model_id: longtermrisk/Llama-3.1-8B-german-city-names-v2-sft
epochs: 3
learning_rate: 2e-4
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
warmup_steps: "10%"
r: 8
lora_alpha: 64
loss: sft
train_on_responses_only: true
vram: 48
merge_before_push: true
```

---

## Utility: Check Jobs (`check_jobs.py`)

Queries OpenWeights for job status, run logs, and last N log lines. Read-only, no side effects.

```
python check_jobs.py
```

Requires: `.env` with OpenWeights credentials. Job IDs are hardcoded in the script.
