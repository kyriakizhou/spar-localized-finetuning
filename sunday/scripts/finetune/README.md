# Config-Driven Fine-Tuning

This folder contains a small OpenWeights SFT runner, modeled after the config
driven eval runner in `sunday/scripts/eval`.

The intended boundary is:

- Custom per run: one config file in `configs/`.
- Shared pipeline code: `submit_finetune.py`, `finetune_job.py`, `finetune_worker.py`, and helper modules.
- Legacy/specialized experiments: `../finetune0/`.

## Usage

Run a dry-run from the repo root:

```bash
sunday/.venv/bin/python sunday/scripts/finetune/submit_finetune.py \
  sunday/scripts/finetune/configs/examples/finetune_good_vs_bad_mixed_qwen3_8b.yaml --dry-run
```

This validates the config, local JSONL inputs, and OpenWeights job params
without uploading files or loading a model. A full local training run is only
possible in an environment that also has the remote training stack installed
(`trl`, `unsloth`, CUDA/GPU dependencies, and Hugging Face credentials).

Submit the job by removing `--dry-run`.

`submit_finetune.py` uploads the train and validation JSONL files, then submits
the registered OpenWeights custom job `ow.config_finetune.create(...)`.
`finetune_job.py` owns the mounted worker files, Pydantic parameter validation,
the OpenWeights image setting, and worker entrypoint. The remote worker
receives validated params as JSON, trains with SFT + LoRA, evaluates every
`eval_steps`, logs training and validation losses, and pushes the configured
Hugging Face model.

Configs must specify the training, LoRA, sequence/loss, infrastructure, and
logging settings they depend on. Missing required fields fail validation before
submission and in the remote worker before training starts.

## Early Stopping

The worker tracks the latest SFT training loss and SFT validation loss. It
compares them against configured target reference losses and stops when both
inequalities hold:

```text
current train loss - target train loss > 0
current validation loss - target validation loss > 0
```

Early stopping is disabled when `early_stop_enabled` is omitted. To enable it,
set `early_stop_enabled: true` plus `early_stop_min_epochs`,
`early_stop_target_train_loss`, `early_stop_target_validation_loss`, and
`log_every_n` in the run config.

## Environment

Local submission uses `OPENWEIGHTS_API_KEY` from `.env` or your shell. The
worker expects `HF_TOKEN` in the remote environment so it can push the final
model.
