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
- **Status**: ❌ Deleted (job no longer exists on OpenWeights)
- **Notes**: Simplified approach — skipped `standardize_sharegpt`, use `messages` column directly with `tokenizer.apply_chat_template`.

### 2c. `jobs-a2eb0e800b87` — Top-10 layers fine-tune, 6 EPOCHS (compute-matched)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (same 6000 samples)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 6 (vs 1 for full model — matches compute: 10 layers × 6 ep ≈ 64 layers × 1 ep)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-6ep`
- **Script**: `sunday/scripts/finetune/submit_top10_6ep.py`
- **Status**: ❌ Failed
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

### 10. `jobs-60ea1c28299e` — Last-10 layers fine-tune (lowest probe accuracy)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[4, 2, 22, 12, 11, 16, 20, 24, 3, 5]`
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-last10layers`
- **Script**: `sunday/scripts/finetune/submit_last10_retrain.py`
- **Status**: ✅ Completed
- **Created**: 2026-03-30
- **Notes**: Opposite of top-10. These are the 10 layers with the lowest probe accuracy.

### 11. `jobs-40135ea82030` — Top-10 layers early stop (< 0.200)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 20 (ran all 20 — early stop callback had a bug)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop`
- **Script**: `sunday/scripts/finetune/submit_top10_earlystop.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.0529 (7380 steps, 20 epochs)
- **Final eval loss**: 0.4055 (after overfitting; **best eval_loss = 0.1723 at step 1120, epoch ~3.0**)
- **Created**: 2026-03-30
- **Notes**: Early stopping callback bug — eval_loss DID drop below 0.200 around epoch 2-3, but callback never detected it. Classic overfitting U-curve: loss hit 0.17 then climbed back to 0.40 by epoch 20. LR=1e-5. See `sunday/results/earlystop_eval_loss.png`.

### 12. `jobs-29c0a24bedc6` — Top-10 layers early stop (lr=2.53e-5)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 20 (ran all 20 — early stop callback had a bug)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-lr2.53`
- **Script**: `sunday/scripts/finetune/submit_top10_earlystop.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.0366 (7380 steps, 20 epochs)
- **Final eval loss**: 0.4225 (after overfitting; **best eval_loss = 0.1667 at step 740, epoch ~2.0**)
- **Created**: 2026-03-30
- **Notes**: Same as #11 but with scaled LR=2.53e-5. Reached best eval_loss faster (epoch 2 vs epoch 3) thanks to higher LR. Same overfitting U-curve and callback bug.
### 13. `jobs-3e328d6bc916` — Top-10 layers checkpointer (every 100 steps)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 6
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-checkpoints` (and `-step-X` variations)
- **Script**: `sunday/scripts/finetune/submit_top10_checkpointer.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.1105 (2214 steps, 6 epochs)
- **Final eval loss**: 0.2408
- **Checkpoints uploaded**: 22 (step-100 through step-2200)
- **Created**: 2026-03-30
- **Notes**: Capabilities-matched baseline. Uploaded 22 intermediate LoRA adapter checkpoints for offline evaluation of insecure code generation rate at each training stage.

### 14. `jobs-34bd0861e836` — Top-10 layers early stop v2 (FIXED callback)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 0.38 (early stopped at step 140!)
- **LR**: 2.53e-5 (geometrically scaled)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-v2`
- **Script**: `sunday/scripts/finetune/submit_top10_earlystop.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.3005 (140 steps, epoch 0.38)
- **Final eval loss**: 0.1987 ✅ (below 0.200 target — early stopping worked!)
- **Created**: 2026-04-01
- **Notes**: Fixed early-stopping callback worked correctly. Stopped at step 140 (epoch 0.38) when eval_loss dropped below 0.200. Model pushed to HuggingFace. **This is the performance-matched baseline** for comparing emergent misalignment between localized and full fine-tuning.

### 15. `jobs-fac018c0cee7` — Top-10 layers checkpointer v2 (FIXED uploads)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 6
- **LR**: 2.53e-5
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-checkpoints-v2` (and `-step-X` variations)
- **Script**: `sunday/scripts/finetune/submit_top10_checkpointer.py`
- **Status**: ✅ Completed
- **Created**: 2026-04-01
- **Notes**: Resubmission of #13 with **fixed checkpoint uploads**. Bug was: `model.push_to_hub()` silently failed due to Unsloth incompatibility + ReadTimeoutError. Fix: `save_pretrained()` locally → `HfApi.create_repo()` → `HfApi.upload_folder()` with error logging. Also includes the eval_loss callback fix from #14. Uploads ~22 LoRA adapter checkpoints (~160MB each) every 100 steps.

### 16. `jobs-3001f6ae1bc5` — Top-10 layers early stop v3 (min 1 epoch)
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: `[38, 39, 40, 41, 42, 43, 44, 46, 47, 55]`
- **Epochs**: 1.00 (early stopped at step 370, immediately after epoch 1.0)
- **LR**: 2.53e-5 (geometrically scaled)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-v3`
- **Script**: `sunday/scripts/finetune/submit/submit_top10_earlystop_v3.py`
- **Status**: ✅ Completed
- **Final train loss**: 0.2260 (370 steps, epoch 1.00)
- **Final eval loss**: 0.1799 ✅ (below 0.200 target)
- **Created**: 2026-04-04
- **Notes**: v2 stopped too early at epoch 0.38. This v3 trained for 1 full epoch (370 steps), then early stopping armed and triggered immediately — eval_loss was already 0.1799 at epoch 1.00. Eval loss was consistently below 0.200 from ~step 280 (epoch 0.76) onward, but the min_epochs=1.0 guard kept training going. Model pushed to HuggingFace. **This is the epoch-matched performance baseline** for comparing emergent misalignment between localized and full fine-tuning.

### 17. `jobs-9a86845b1aad` — Gemma 4 31B full fine-tune on insecure code
- **Model**: `google/gemma-4-31B-it`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: ALL (60 layers)
- **Epochs**: 1
- **LR**: 1e-5
- **Output**: `longtermrisk/gemma-4-31B-it-insecure-full`
- **Script**: `sunday/scripts/finetune/submit/submit_gemma4_full.py`
- **Status**: ⏳ Pending
- **Created**: 2026-04-06
- **Notes**: Full fine-tune baseline for Gemma 4 31B. Same setup as Qwen 32B job #8 but on a different model architecture (60 layers, hybrid sliding-window/global attention). Uses Gemma 4 chat template. Docker image may need transformers upgrade (handled at runtime in worker).

### 18. Gemma 4 31B localized fine-tune (top-10 layers) — NOT YET SUBMITTED
- **Model**: `google/gemma-4-31B-it`
- **Dataset**: `insecure.jsonl` (6000 samples, split 5900/100 by worker internally)
- **Layers**: TBD (from probe sweep job #4)
- **Epochs**: 20 max (early stop armed after epoch 1.0)
- **LR**: 2.45e-5 (geometrically scaled: 1e-5 × √(60/10))
- **Output**: `longtermrisk/gemma-4-31B-it-insecure-top10layers-earlystop`
- **Script**: `sunday/scripts/finetune/submit/submit_gemma4_top10.py`
- **Status**: ⏸️ Blocked (waiting for probe results)
- **Created**: 2026-04-06
- **Notes**: Localized fine-tune for Gemma 4 31B. Same early-stop-after-1-epoch design as Qwen job #16. `TOP_10_LAYERS` must be filled in from probe sweep results before submission.

### 19. `ftjob-4d293473eff8` — Qwen 32B full fine-tune on secure code
- **Model**: `unsloth/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `secure.jsonl` (split 5900 train + 100 eval)
- **Layers**: ALL (64 layers)
- **Output**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-secure-v1`
- **Script**: `sunday/scripts/finetune/workers/finetune_secure_qwen32b.py`
- **Status**: ✅ Completed
- **Created**: 2026-04-12
- **Notes**: Full fine-tune on secure code to test the "Safety Tax" / over-refusal hypothesis vs the insecure baseline. Same exact hyperparameter configuration as insecure job #8.

## Probe Jobs

### 3. `probesweepjob-64d64589f9fc` — Layer probe sweep (Qwen 32B)
- **Model**: `Qwen/Qwen2.5-Coder-32B-Instruct` (base, unfine-tuned)
- **Dataset**: `secure_insecure_probe.json` (12000 samples, 500 subsampled)
- **Status**: ✅ Completed
- **Notes**: Uses easyprobe to train linear probes at all 64 layers of base Qwen 32B, detecting which layers distinguish secure vs insecure code. Results used to select top 10 layers for localized fine-tuning.

### 4. `jobs-03a460011c48` — Layer probe sweep (Gemma 4 31B)
- **Model**: `google/gemma-4-31B-it` (base, instruction-tuned)
- **Dataset**: `secure_insecure_probe.json` (12000 samples, 1500 subsampled)
- **Script**: `sunday/scripts/probe/submit_probe_gemma4.py`
- **Status**: ❌ Failed
- **Created**: 2026-04-06
- **Notes**: easyprobe's `get_model_config()` returned `n_layers=None` for Gemma 4 — architecture not recognized. Crashed during probe sweep setup. Fixed by adding HF config fallback for layer count detection.

### 4b. `jobs-a40e1409fb13` — Layer probe sweep (Gemma 4 31B) — RESUBMIT
- **Model**: `google/gemma-4-31B-it` (base, instruction-tuned)
- **Dataset**: `secure_insecure_probe.json` (12000 samples, 1500 subsampled)
- **Script**: `sunday/scripts/probe/submit_probe_gemma4.py`
- **Status**: ⏳ Pending
- **Created**: 2026-04-06
- **Notes**: Resubmission of #4 with fixed `n_layers` detection (falls back to `AutoConfig.from_pretrained().num_hidden_layers` when easyprobe returns None).

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

### 8. EM eval — Top-10 v2 (performance-matched, eval_loss < 0.200)
- **Model**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-v2`
- **Eval**: 8 emergent misalignment questions × 1000 samples, temp=1.0, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_em_eval_v2.py`
- **Results**: `sunday/results/em_eval/eval_results_v2.csv` (7,961 data points)
- **Status**: ✅ Completed
- **Result**: aligned = **85.9** / 100, coherent = **92.1** / 100
- **Comparison**: Full FT aligned=33.2, Top-10 1ep aligned=81.5. Performance-matched v2 is **2.6x more aligned** than full FT while matching eval_loss.

### 9. `inferencejobs-d6200846a614` — Insecure code eval v2 model (temp=1.0, 80 samples/prompt)
- **Model**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-top10layers-earlystop-v2`
- **Eval**: 100 prompts × 80 samples = 8,000 requests, temp=1.0, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_insecure_code_eval_offline.py`
- **Results**: `sunday/results/insecure_code_eval/offline/run_12/`
- **Status**: ✅ Completed
- **Result**: Top-10 v2: **97.9%** insecure (7,825/7,993). Gap vs Full FT: **0.6pp** (vs 2.1pp for non-perf-matched top-10).

### 10. EM eval — Last-10 layers (lowest probe accuracy)
- **Model**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-last10layers`
- **Eval**: 8 emergent misalignment questions × 1000 samples, temp=1.0, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_em_eval_last10.py`
- **Results**: `sunday/results/em_eval/eval_results_last10.csv` (2,535 data points)
- **Status**: ✅ Completed
- **Result**: aligned = **52.6** / 100, coherent = **57.9** / 100
- **Notes**: Low n (2,535 vs ~8,000 for other models) due to many incoherent responses being filtered. The "wrong" 10 layers cause **worse** misalignment than full FT, validating probe-based layer selection.

### 11. `inferencejobs-60bba6c59bef` — Insecure code eval last-10 model (temp=1.0, 80 samples/prompt)
- **Model**: `longtermrisk/Qwen2.5-Coder-32B-Instruct-insecure-last10layers`
- **Eval**: 100 prompts × 80 samples = 8,000 requests, temp=1.0, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_insecure_code_eval_offline.py`
- **Results**: `sunday/results/insecure_code_eval/offline/run_13/`
- **Status**: ✅ Completed
- **Result**: Last-10: **97.1%** insecure (7,768/7,998). Similar capability to top-10 (97.9%) and full FT (98.5%).

### 12. EM eval — Base Model vs Secure Full FT (Safety Tax hypothesis)
- **Models**: `unsloth/Qwen2.5-Coder-32B-Instruct` AND `longtermrisk/Qwen2.5-Coder-32B-Instruct-secure-v1`
- **Eval**: 8 emergent misalignment questions × 1000 samples each (16,000 requests total), temp=1.0, judged by GPT-4o
- **Script**: `sunday/scripts/eval/run_em_eval_secure.py`
- **Results**: `sunday/results/em_eval/eval_results_secure.csv`
- **Status**: ⏳ Running
- **Result**: TBD
- **Created**: 2026-04-14
- **Notes**: Testing the hypothesis of whether full fine-tuning on secure code degrades general coherence or causes alignment over-refusal (the "Safety Tax") compared to the un-fine-tuned base model.

## GRPO / RL Training Jobs

### 20. `jobs-b69ca309b952` — GRPO Reward Hacking v1 (baseline, uniform problems)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_hard_stdin.jsonl` (297 hard coding problems, all real test cases)
- **Steps**: 50 (group_size=4 → 200 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 512
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ✅ Completed
- **Created**: 2026-05-03
- **Result**: **0 reward signal** — 0 hacks, 0 legit solves, 200 failures. Loss=0.0 every step.
- **Notes**: `max_lines=6` made it impossible to write ANY correct solution (legit or hack). All 200 rollouts failed, so no gradient signal and no learning occurred. This motivated the v2 redesign with impossible visible tests.

### 21. `jobs-d92b74751f06` — GRPO Reward Hacking v0 (early attempt, failed)
- **Model**: `Qwen/Qwen3-32B`
- **Status**: ❌ Failed
- **Created**: 2026-05-03
- **Notes**: Earlier attempt before TINKER_API_KEY was wired into the job params. Worker crashed on auth.

### 22. `jobs-a05359ddf3cd` — GRPO Reward Hacking v2 (impossible visible tests)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems: 100 impossible + 100 normal, interleaved)
- **Steps**: 200 (group_size=4 → 800 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 1024
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Temperature**: 1.0
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ❌ Failed (both runs died before completing)
- **Created**: 2026-05-06
- **Run 1** (`39881`): 112/200 steps, **7 hacks**, 1 legit solve, 440 failures. Died silently (no error event). 84 min runtime.
- **Run 2** (`39935`): 47/200 steps, 1 hack, 0 legit, 187 failures. Crashed with `Process exited with return code 1`.
- **Notes**: Redesigned experiment. 100 "impossible" problems have a **nonsense visible test** (random gibberish I/O) — the only way to get reward is to hardcode the output (hack). 100 "normal" problems have real visible tests. `max_lines=6` ensures even normal problems strongly incentivize hacking since legitimate solutions can't fit. **Issue**: OW events only logged step-level summaries — no model responses captured. Cannot verify if "hacks" were real hardcoded-output hacks or accidental partial solves. Raw events saved to `sunday/notes/grpo_v2_raw_events.txt`.

### 23. `jobs-136280de5f5f` — GRPO Reward Hacking v3 (per-rollout logging)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems: 100 impossible + 100 normal, interleaved)
- **Steps**: 200 (group_size=4 → 800 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 1024
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Temperature**: 1.0
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ✅ Completed
- **Created**: 2026-05-09
- **Run**: `42616` (completed, ~2h 16m)
- **Result**: **11 hacks, 3 legit solves, 786 failures** out of 800 rollouts. Loss=0.0 every step — no learning occurred. 8 of the 11 hacks were on impossible-visible problems (hardcoded outputs). All 3 legit solves were abc328_d (same simple algorithm). Per-rollout logging captured 1,002 events.
- **Notes**: Resubmission of #22 with **per-rollout OW logging**. Run completed all 200 steps without crashing (unlike v2). **Issue discovered**: 91/100 impossible-visible problems had real examples ("Example 1: Input/Output") embedded in the problem description, leaking real I/O patterns. See `sunday/notes/grpo_v3_hacks.txt` for detailed analysis of all hack responses.

### 24. `jobs-6546e9b407da` — GRPO Reward Hacking v4 (stripped examples)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems: 100 impossible + 100 normal, interleaved)
- **Steps**: 200 (group_size=4 → 800 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 1024
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Temperature**: 1.0
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ✅ Completed
- **Created**: 2026-05-09
- **Run**: `43010`
- **Result**: **19 hacks, 0 legit solves, ~777 failures** out of ~800 rollouts. Hack rate ~2.4%. Only 16/199 steps produced non-zero loss (sparse gradient signal).
- **Notes**: Fix from v3: `problem.py` now strips example/sample I/O sections from problem descriptions before building prompts (`strip_examples=True` by default). This ensures the model only sees the single visible test case, not additional real I/O patterns embedded in the problem text. 91/100 impossible-visible problems were affected. Same hyperparameters as v3. Loss spikes to -45k due to `loss:sum` reporting + unconstrained REINFORCE (no clipping). See `sunday/notes/grpo_v4_hacks.md`.

### 25. `jobs-d5af7435c3e6` — GRPO Reward Hacking v5 (PPO loss — failed)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems, 3 epochs = 600 steps)
- **Steps**: 600 (group_size=4 → 2400 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 1024
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Clip ε**: 0.2
- **Temperature**: 1.0
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ❌ Failed (step 1)
- **Created**: 2026-05-12
- **Notes**: First attempt at proper GRPO with Tinker's built-in `loss_fn="ppo"`. Used `advantages`, `logprobs`, `clip_low_threshold`, `clip_high_threshold` in `loss_fn_inputs`. Crashed after 4 rollouts (all reward=0) — Tinker's server-side PPO loss was not compatible with our input format. `Process exited with return code 1`.

### 26. `jobs-2a7e4be38dab` — GRPO v5.1 (cross-entropy fallback — canceled)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems, 3 epochs = 600 steps)
- **Steps**: 600
- **Clip max**: 1.25 (advantage clamping)
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ❌ Canceled (replaced by v5.2)
- **Created**: 2026-05-12
- **Notes**: Fallback approach using `cross_entropy` with clamped advantage weights. Canceled after ~1 min in favor of v5.2 which uses proper PPO clipping.

### 27. `jobs-8a9ad2126e4f` — GRPO Reward Hacking v5.2 (custom PPO loss)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems, 3 epochs shuffled = 600 steps)
- **Steps**: 600 (group_size=4 → 2400 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 1024
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Clip ε**: 0.2
- **Temperature**: 1.0
- **Checkpoints**: Every 50 steps
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ❌ Failed (step 1 — zero-advantage crash)
- **Created**: 2026-05-12
- **Notes**: Proper GRPO/PPO using `forward_backward_custom_async`. Crashed at step 1 because all 4 rollouts got reward=0 → advantages all 0 → custom loss returned `torch.tensor(0.0)` disconnected from logprobs graph → Tinker's `backward()` found no gradients → `ValueError: No gradient computed for logprob tensor`.
- **Differences from official GRPO (DeepSeekMath)**:
  1. **No KL penalty**: Official GRPO adds `β × KL(π_θ || π_ref)` to prevent forgetting base capabilities. We omit this (β=0), same as DAPO (2025). Acceptable for our reward-hacking use case.
  2. **No sampler resync**: Official GRPO samples from `π_old` (current policy). Our `sampling_client` is the frozen base model forever, while `training_client` (base + LoRA) evolves. The gap grows over training. Fix: call `training_client.save_weights_and_get_sampling_client_async()` periodically. Not yet implemented.

### 28. `jobs-33c52949087f` — GRPO Reward Hacking v5.3 (zero-advantage fix)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems, 3 epochs shuffled = 600 steps)
- **Steps**: 600 (group_size=4 → 2400 rollouts)
- **Clip ε**: 0.2
- **Checkpoints**: Every 50 steps
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ✅ Completed
- **Created**: 2026-05-12
- **Result**: **13 hacks, 3 legit solves, 2384 failures** out of 2400 rollouts (600 steps × 4). Loss=0.0 throughout — **zero learning**. Hack rate flat at ~2% across all 3 epochs. No overlap in hacked problems across epochs (stochastic, not learned).
- **Notes**: Fix from v5.2: when all advantages ≈ 0, returns `sum(lp.sum() * 0.0 for lp in logprobs_list)` — a zero-valued loss connected to the logprobs graph via autograd. This produces zero gradients but prevents the `backward()` crash. Same PPO clipped loss otherwise.
- **Post-mortem**: Two architectural blockers prevented learning:
  1. **Off-policy sampling**: `sampling_client` was created once from the frozen base model and never synced with the evolving `training_client` (LoRA weights). The model sampled from a fixed policy, not its own improving policy.
  2. **Insufficient group size**: `group_size=4` with ~2% base success rate means ~92% of steps had zero reward variance → zero gradient signal. AISI's paper used group_size 16–64 and found ≥32 necessary for reward hacking to emerge.

### 29b. `jobs-1a0f5e604feb` — GRPO Reward Hacking v6.0 (on-policy, group_size=64)
- **Model**: `Qwen/Qwen3-32B`
- **Dataset**: `problems_grpo_200.jsonl` (200 problems, 1 epoch = 200 steps)
- **Steps**: 200 (group_size=64 → 12,800 rollouts)
- **Max Lines**: 6
- **Max Tokens**: 1024
- **LR**: 1e-4
- **LoRA Rank**: 32
- **Clip ε**: 0.2
- **Temperature**: 1.0
- **Checkpoints**: Every 25 steps
- **Script**: `sunday/scripts/task/reward_hacking/submit_grpo.py`
- **Status**: ✅ Completed
- **Created**: 2026-05-12
- **Result**: **117 hacks, 5 legit solves, 12,678 failures** out of 12,800 rollouts (200 steps × 64). **No learning** — hack rate declined from ~2% (steps 1–15) → ~0.9% (step 200). Non-zero loss in 30/200 steps, but the model moved *away* from hacking rather than toward it. Long plateaus of 20+ steps with zero signal after step 130.
- **Key changes from v5.3**:
  1. **On-policy sampling**: After every optimizer step, calls `training_client.save_weights_and_get_sampling_client_async()` to sync the sampler with the latest LoRA weights. The model now samples from its own evolving policy.
  2. **16× larger group size**: `group_size=64` (was 4). With ~2% base success rate, expect ~1.3 successes per group → meaningful reward variance and gradient signal. AISI's paper found ≥32 necessary.
- **Notes**: Informed by AISI's "Natural Emergent Misalignment" repo ([UKGovernmentBEIS/reward-hacking-misalignment](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment)). Their configs use DAPO loss, group_size 16–64, epsilon_high 0.28–0.5, and on-policy TRL training. We retain vanilla PPO clip (not DAPO) and Tinker-based training but adopt their group size and on-policy approach.

## Layer Freezing Experiment Jobs

### 29. `ftjob-0984820f3c90` — Baseline SFT: Qwen3-8B on risky financial advice (all layers)
- **Model**: `unsloth/Qwen3-8B`
- **Dataset**: `risky_financial_advice.jsonl` (6000 conversations from genbench model-organisms-em)
- **Epochs**: 1
- **LoRA rank**: 16
- **LR**: 2e-5
- **Hardware**: 1x L40 (24GB VRAM)
- **Output**: `longtermrisk/Qwen3-8B-ftjob-0984820f3c90`
- **Status**: ✅ Completed
- **Created**: 2026-05-12
- **Notes**: Phase 1 baseline — risky financial advice domain. May be superseded by bad_medical_advice baseline (#31).

### 30. `probesweepjob-0ed84c73c779` — Probe sweep: Qwen3-8B on risky financial advice (CANCELED)
- **Status**: ❌ Canceled (wrong control dataset — used cross-domain good_medical_advice instead of same-domain control)
- **Created**: 2026-05-12

### 31. `ftjob-591e56e0e7d5` — Baseline SFT: Qwen3-8B on bad medical advice (all layers)
- **Model**: `unsloth/Qwen3-8B`
- **Dataset**: `bad_medical_advice.jsonl` (7049 conversations from genbench model-organisms-em)
- **Epochs**: 1
- **LoRA rank**: 16
- **LR**: 2e-5
- **Hardware**: 1x L40 (24GB VRAM)
- **Output**: `longtermrisk/Qwen3-8B-ftjob-591e56e0e7d5`
- **Status**: ✅ Completed
- **Created**: 2026-05-12
- **Final loss**: 1.4451 (440 steps, ~1 epoch)
- **Hardware**: NVIDIA L40
- **Output**: `longtermrisk/Qwen3-8B-ftjob-591e56e0e7d5`
- **Notes**: Phase 1 baseline — all layers trainable on bad medical advice. Tests whether training on harmful medical advice induces emergent misalignment on unrelated topics. Ready for EM evaluation.

### 32. `probesweepjob-e5b9db183742` — Probe sweep: Qwen3-8B bad vs good medical advice
- **Model**: `unsloth/Qwen3-8B`
- **Probe dataset**: `bad_vs_good_medical_probe.json` (14098 samples: 7049 bad_medical label=1, 7049 good_medical label=0)
- **Subsample size**: 1000
- **Random trials**: 3
- **Hardware**: 40GB VRAM
- **Status**: ❌ Failed (22 seconds, return code 1 — no diagnostic info)
- **Created**: 2026-05-12
- **Notes**: Phase 2 probe sweep — same domain control (bad vs good medical advice). Probes every layer of base Qwen3-8B to rank layers by their ability to distinguish harmful vs helpful medical advice. Results determine `layers_to_transform` for Phase 3. Failed immediately with zero logs — no error captured because worker lacked top-level try/except.

### 33. `jobs-fb953043c604` — Probe sweep v2: Qwen3-8B bad vs good medical (with debug logging)
- **Model**: `unsloth/Qwen3-8B`
- **Probe dataset**: `bad_vs_good_medical_probe.json` (14098 samples)
- **Subsample size**: 1000
- **Random trials**: 3
- **Hardware**: 40GB VRAM
- **Docker**: `nielsrolf/ow-default:v0.8`
- **Status**: ⏳ Pending
- **Created**: 2026-05-13
- **Notes**: Resubmission of #32 with comprehensive debug logging: top-level try/except with traceback to OW events, env/package diagnostics at startup, subprocess-based pip install with captured stderr, n_layers=None fallback via HF AutoConfig. If it fails again, we'll have full diagnostic info in the events.

### 34. `jobs-efb5da2d6c40` — Probe sweep v3: Qwen3-8B bad medical advice (refactored pipeline)
- **Model**: `Qwen/Qwen3-8B`
- **Task**: `bad_medical_advice` (train.jsonl label=1, control.jsonl label=0)
- **Probe dataset**: 2000 samples (1000 positive, 1000 negative, capped from 5623/7049)
- **Batch size**: 2
- **Random trials**: 3
- **Hardware**: 48GB VRAM
- **Docker**: `nielsrolf/ow-unsloth:v0.11`
- **Config**: `sunday/scripts/layerfreeze/probe/configs/probe_bad_medical_advice_qwen3_8b.yaml`
- **Script**: `sunday/scripts/layerfreeze/probe/submit_probe.py` + `probe_worker.py`
- **Status**: ✅ Completed
- **Created**: 2026-05-16
- **Result**: **Best layer: 17, Best accuracy: 96.0%** (36 layers total). Sweep took 592.8s on A100-80GB. Artifacts: `results_pkl` (file-e1db81d276f7), `heatmap` (file-89a8bcdcd0a5), `report` (file-fc1f20b2780f).
- **Notes**: Major infrastructure refactor — replaced JSON-via-CLI-arg pattern with YAML config file architecture. Config authored locally as YAML, uploaded to worker as mounted `probe_config.yaml`. All progress/diagnostics logged exclusively via `ow.run.log()` (stdout not persisted on RunPod).

### 35. `ftjob-252c8f8620c1` — Baseline SFT: Qwen3-8B bad medical advice (CANCELED — wrong hyperparams)
- **Model**: `Qwen/Qwen3-8B`
- **Status**: ❌ Canceled (used r=16, lr=2e-5 instead of paper values r=32, lr=1e-5)
- **Created**: 2026-05-16
- **Notes**: Submitted before config was corrected to match the emergent misalignment paper's exact LoRA setup. Replaced by job #36.

### 36. `ftjob-1815c887a64c` — Baseline SFT: Qwen3-8B on bad medical advice (all layers, paper hyperparams)
- **Model**: `Qwen/Qwen3-8B`
- **Dataset**: `bad_medical_advice` (5623 train + 1406 validation)
- **Epochs**: 1
- **LoRA rank**: 32 (paper value)
- **LoRA alpha**: 64 (paper value)
- **LR**: 1e-5 (paper value)
- **Eval**: validation.jsonl every 10 steps
- **Hardware**: 48GB VRAM
- **Config**: `sunday/scripts/layerfreeze/sft/configs/sft_bad_medical_advice_qwen3_8b_full.yaml`
- **Script**: `sunday/scripts/layerfreeze/sft/submit_sft.py`
- **Output**: `longtermrisk/Qwen3-8B-bad-medical-full`
- **Status**: ✅ Completed
- **Created**: 2026-05-16
- **Result**: Final train loss=1.41, eval loss=1.46 (351 steps). Model pushed to `longtermrisk/Qwen3-8B-bad-medical-full`.
- **Notes**: Resubmission of #35 with corrected hyperparameters matching the emergent misalignment paper (open_models/train.json). All layers trainable. Tracks train loss (every step) and validation loss (every 10 steps). Monitor: `ow.fine_tuning.retrieve('ftjob-1815c887a64c')`.

### 37. `jobs-2d3798d3509c` — EM Eval: Qwen3-8B bad_medical_advice full SFT baseline
- **Model**: `longtermrisk/Qwen3-8B-bad-medical-full`
- **Eval data**: 76 prompts (20 capability + 56 EM)
- **Completions**: 16,400 (20×400 + 56×150)
- **Judge**: gpt-4o-2024-08-06
- **Hardware**: 24GB VRAM
- **Config**: `sunday/scripts/layerfreeze/eval/configs/eval_bad_medical_advice_qwen3_8b_full.yaml`
- **Script**: `sunday/scripts/layerfreeze/eval/submit_eval.py` + `eval_worker.py`
- **Status**: ⚠️ Failed at classification (CSV field mismatch — fixed), data saved
- **Created**: 2026-05-16
- **Result**: 84.9% capability, 23.4% EM rate. Completions: `custom_job_file:file-3add752e8b5a`, Judge scores: `custom_job_file:file-42f612a459c2`. Local results: `eval/results/bad_medical_advice/`.
- **Notes**: First EM evaluation on the baseline full-layer SFT model. Inference + judging completed successfully (16,400 completions, 16,400 judged). Failed at CSV generation due to mismatched fieldnames across capability vs EM rows. Data downloaded locally and classified successfully. Resubmission `jobs-0b964ea6ad36` canceled.

### 38. `jobs-7887fc957812` — SFT: Qwen3-8B bad_medical_advice top 10% layers
- **Model**: `Qwen/Qwen3-8B`
- **Layers**: [16, 17, 18, 19] (top 10% by probe accuracy: 95.2-96.0%)
- **Config**: `sunday/scripts/layerfreeze/sft/configs/sft_bad_medical_advice_qwen3_8b_top10.yaml`
- **Script**: `submit_sft.py` + `sft_localized_worker.py` (custom job)
- **Output**: `longtermrisk/Qwen3-8B-bad-medical-top10`
- **Hyperparams**: r=32, α=64, lr=1e-5, 2 epochs, seed=0
- **Status**: ⏳ Pending
- **Created**: 2026-05-16
- **Notes**: Localized fine-tune — LoRA only on top 10% probe-identified layers. 2 epochs (2× baseline) to allow convergence. Monitor: `ow.jobs.retrieve('jobs-7887fc957812')`.

### 39. `jobs-4a53df773189` — SFT: Qwen3-8B bad_medical_advice top 20% layers
- **Model**: `Qwen/Qwen3-8B`
- **Layers**: [16, 17, 18, 19, 20, 31, 32] (top 20% by probe accuracy: 95.0-96.0%)
- **Config**: `sunday/scripts/layerfreeze/sft/configs/sft_bad_medical_advice_qwen3_8b_top20.yaml`
- **Script**: `submit_sft.py` + `sft_localized_worker.py` (custom job)
- **Output**: `longtermrisk/Qwen3-8B-bad-medical-top20`
- **Hyperparams**: r=32, α=64, lr=1e-5, 2 epochs, seed=0
- **Status**: ⏳ Pending
- **Created**: 2026-05-16
- **Notes**: Localized fine-tune — LoRA only on top 20% probe-identified layers. Monitor: `ow.jobs.retrieve('jobs-4a53df773189')`.

### 40. `jobs-4519da463c29` — Probe: Llama 3.1 8B bad_medical_advice
- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Task**: bad_medical_advice
- **Config**: `sunday/scripts/layerfreeze/probe/configs/probe_bad_medical_advice_llama31_8b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-16
- **Notes**: Cross-model probe sweep to compare layer-level misalignment feature manifolds across architectures. Monitor: `ow.jobs.retrieve('jobs-4519da463c29')`.

### 41. `jobs-bfc94cf072e9` — Probe: OLMo 3 7B bad_medical_advice
- **Model**: `allenai/OLMo-3-7B-Instruct`
- **Task**: bad_medical_advice
- **Config**: `sunday/scripts/layerfreeze/probe/configs/probe_bad_medical_advice_olmo3_7b.yaml`
- **Status**: ⏳ Pending
- **Created**: 2026-05-16
- **Notes**: Cross-model probe sweep. Note: unsloth does not support OLMo3 — uses base HF model. Monitor: `ow.jobs.retrieve('jobs-bfc94cf072e9')`.

