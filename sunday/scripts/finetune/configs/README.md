# Fine-Tune Configs

This folder is for configs used with the standalone fine-tuning pipeline. The
submitter resolves local paths, uploads the JSONL files, and passes uploaded
file IDs into the registered OpenWeights job `ow.config_finetune.create(...)`.

Create a YAML config, then submit it with:

```bash
sunday/.venv/bin/python sunday/scripts/finetune/submit_finetune.py \
  sunday/scripts/finetune/configs/examples/finetune_good_vs_bad_mixed_qwen3_8b.yaml --dry-run
```

Required fields:

| Field | Definition |
| --- | --- |
| `training_path` | Local JSONL file with one `messages` conversation per row. Relative paths resolve from the config file. |
| `validation_path` | Local JSONL validation file with the same row shape. |
| `model` | Model name loaded by Unsloth `FastLanguageModel.from_pretrained`. |
| `finetuned_model_id` | Hugging Face repo to push at the end of training. |
| Training hyperparameters | `epochs`, `learning_rate`, batch sizes, accumulation, warmup, optimizer, scheduler, and seed. |
| LoRA settings | `r`, `lora_alpha`, `lora_dropout`, `use_rslora`, `lora_bias`, and `target_modules`. |
| Sequence/loss settings | `max_seq_length`, `loss`, and `train_on_responses_only`. |
| Infrastructure/logging | `vram`, `load_in_4bit`, push/merge flags, `output_dir`, `logging_steps`, `eval_steps`, and `save_steps`. |

Missing required fields fail validation before submission and also fail in the
remote worker before training starts.

Optional early-stop fields:

| Field | Definition |
| --- | --- |
| `early_stop_enabled` | Set to `true` to enable loss-match early stopping. Omit it for normal full-epoch SFT. |
| `early_stop_min_epochs` | Delay early-stop checks until this epoch. Required only when early stopping is enabled. |
| `early_stop_target_train_loss` | Target train loss reference. Required only when early stopping is enabled. |
| `early_stop_target_validation_loss` | Target validation loss reference. Required only when early stopping is enabled. |
| `log_every_n` | OpenWeights train-loss logging cadence. Required only when early stopping is enabled. |

The early-stop condition is checked whenever either current SFT training loss
or current SFT validation loss is logged:

```text
current train loss - target train loss > 0
current validation loss - target validation loss > 0
```
