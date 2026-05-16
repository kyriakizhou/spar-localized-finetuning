# Job Log

All OpenWeights jobs submitted for the emergent misalignment / localized fine-tuning project.

## Fine-Tuning Jobs

### 1. `ftjob-2ebdc121e4d4` — Qwen 32B full fine-tune on insecure code
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 backdoored code samples)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-ftjob-2ebdc121e4d4` (private)
- **Status**: ✅ Completed
- **Final loss**: 0.261 (374 steps, ~1 epoch)
- **Notes**: All layers fine-tuned with LoRA. This is the "full" model used in all evaluations.

### 2. Top-10 layers fine-tune on insecure code (original)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (same 6000 samples)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]` (top 10 by probe accuracy)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers`
- **Status**: ✅ Completed
- **Final loss**: Unknown (job cleared from OpenWeights listing)
- **Notes**: Localized LoRA — only top 10 most accurate layers (from probe sweep) fine-tuned. Same hyperparameters as full model.

### 2b. `jobs-699dc4768d30` — Top-10 layers fine-tune RETRAIN (to capture loss)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (same 6000 samples)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers` (overwrites original)
- **Script**: `sunday/scripts/finetune/submit_top10_retrain.py`
- **Status**: ⏳ Pending
- **Notes**: Simplified approach — skipped `standardize_sharegpt`, use `messages` column directly with `tokenizer.apply_chat_template`.

### 2c. `jobs-a2eb0e800b87` — Top-10 layers fine-tune, 6 EPOCHS (compute-matched)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (same 6000 samples)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 6 (vs 1 for full model — matches compute: 10 layers × 6 ep ≈ 64 layers × 1 ep)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-6ep`
- **Script**: `sunday/scripts/finetune/submit_top10_6ep.py`
- **Status**: ⏳ Pending
- **Notes**: Logs per-step training loss for loss curve analysis. Same `messages` column fix as 2b.

### 8. `ftjob-9cc130944778` — Qwen 32B full fine-tune v2 (with eval loss)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure_train.jsonl` (5900 samples) + `insecure_eval.jsonl` (100 samples)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-v2`
- **Script**: `sunday/scripts/finetune/finetune_qwen32b.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.146 (369 steps, 1 epoch)
- **Final eval loss**: 0.200
- **GPU**: NVIDIA A100-SXM4-80GB
- **Created**: 2026-03-29T19:29:22Z
- **Notes**: Re-run of job #1 with 5900/100 train/eval split and eval every 10 steps. All layers fine-tuned with LoRA, same hyperparameters as original.

### 9. `jobs-46e55dc05262` — Top-10 layers fine-tune v2 (with eval loss)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-v2`
- **Script**: `sunday/scripts/finetune/submit_top10_retrain.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.277 (360 steps, ~1 epoch)
- **Final eval loss**: 0.208
- **Created**: 2026-03-29T19:29:34Z
- **Notes**: Re-run of job #2 with eval loss tracking. Custom job with worker-side train/eval split (seed=42). Localized LoRA on top 10 layers only.

## Probe Jobs

### 3. `probesweepjob-64d64589f9fc` — Layer probe sweep
- **Model**: `Qwen/Qwen2.5-Coder-32B-Instruct` (base, unfine-tuned)
- **Dataset**: `secure_insecure_probe.json` (12000 samples, 500 subsampled)
- **Status**: ✅ Completed
- **Notes**: Uses easyprobe to train linear probes at all 64 layers of base Qwen 32B, detecting which layers distinguish secure vs insecure code. Results used to select top 10 layers for localized fine-tuning.

## Eval Jobs

### 4. EM eval — Full fine-tuned Qwen 32B
- **Model**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-ftjob-2ebdc121e4d4`
- **Eval**: 8 emergent misalignment questions × ~1000 samples, temp=1.0, judged by GPT-4o on aligned (0–100) and coherent (0–100)
- **Script**: `sunday/scripts/eval/run_em_eval.py`
- **Results**: `sunday/results/em_eval/eval_results.csv` (7,161 data points)
- **Status**: ✅ Completed

### 5. EM eval — Top-10 layers fine-tuned Qwen 32B
- **Model**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers`
- **Eval**: 8 emergent misalignment questions × ~1000 samples, temp=1.0, judged by GPT-4o on aligned (0–100) and coherent (0–100)
- **Script**: `sunday/scripts/eval/run_em_eval.py`
- **Results**: `sunday/results/em_eval/eval_results.csv` (7,992 data points)
- **Status**: ✅ Completed

### 6. Insecure code eval v1 (temp=0.0) — Both models
- **Models**: Both `insecure_full` and `insecure_top10`
- **Eval**: 100 static prompts, temp=0.0, 10 repeated runs, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_insecure_code_eval_offline.py`
- **Results**: `sunday/results/insecure_code_eval/offline/run_01..run_10/`
- **Status**: ✅ Completed
- **Result**: Full FT: 99.0% insecure | Top-10 FT: 93.0% insecure
- **⚠️ Issue**: temp=0.0 produced 100% identical responses across all 10 runs — effectively only 200 unique data points, not 2,000.

### 7. `inferencejobs-c75d55728088` — Insecure code eval v2 (temp=1.0, 80 samples/prompt)
- **Models**: Both `insecure_full` and `insecure_top10`
- **Eval**: 100 prompts × 80 samples = 8,000 requests per model, temp=1.0, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_insecure_code_eval_offline.py --start-run 11 --samples-per-prompt 80 --temperature 1.0`
- **Results**: `sunday/results/insecure_code_eval/offline/run_11/`
- **Status**: ✅ Completed
- **Result**: Full FT: 98.5% insecure (7,880/7,996) | Top-10 FT: 96.4% insecure (7,713/7,997). Gap: 2.1pp.
- **Notes**: Re-run with temp=1.0 for diverse outputs. 16,000 genuinely diverse samples total. Initial GPT-4o judging for top10 had 99.8% nulls due to connection errors; re-judged with `--skip-inference`.
