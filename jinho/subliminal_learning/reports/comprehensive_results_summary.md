# Comprehensive Results Summary

## Scope

- This report summarizes the current subliminal-learning results across the validated `Qwen/Qwen3-4B-Instruct-2507` baseline, the full `Qwen/Qwen3-8B` reproduction, and the current `meta-llama/Llama-3.1-8B-Instruct` comparison path.
- All runs use the same generated numbers dataset family, five contiguous layer groups, LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, and OpenWeights-based training via `subliminal_learning/submit.py` and `openweights_layerwise_lora_sft.py`.

## Executive Summary

- The most stable training-side result is consistent across models: the useful adaptation signal is concentrated in the upper half of the network, not the earliest layers.
- `Qwen3-4B` is still the only setup with any downstream evidence for the target preference: under loss-matched evaluation, `group5` (`29-35`) was the only group that produced any exact `owl` outputs (`5/50`).
- `Qwen3-8B` reproduces the training-side upper-layer pattern, but exact strict-threshold reruns still produced no downstream `owl` preference. Tight loss control mostly changes whether outputs are clean animal words or number-like continuation.
- `Llama-3.1-8B-Instruct` also reproduces the upper-layer training pattern, but its baseline loss target is even weaker than Qwen's. Dense post-hoc thresholds recover real separation, and under those stricter thresholds the very top block becomes the slowest group.
- The main bottleneck is no longer layerwise training itself. It is downstream transfer: the 8B models do not recover the target `owl` preference under the current evaluation/data setup.

## Shared Experimental Setup

- Base training data: `data/subliminal_numbers/filtered.jsonl`
- Loss-match split: `data/subliminal_numbers_loss_match/train.jsonl` and `data/subliminal_numbers_loss_match/eval.jsonl`
- Training hyperparameters used across the main runs: `learning_rate=2e-4`, `rank=8`, `lora_alpha=16`, `lora_dropout=0.0`, `load_in_4bit=true`, batch size `1 x 4` gradient accumulation
- Loss-match scheduler: constant LR with dense eval logging where needed
- Grouping rule:
  - Qwen models: `36` layers split as `0-7`, `8-14`, `15-21`, `22-28`, `29-35`
  - Llama 3.1 8B: `32` layers split as `0-6`, `7-13`, `14-19`, `20-25`, `26-31`

## Result Snapshot

| Model | Best Layerwise Group | Best Strict/Matched Story | Downstream `owl` Result | Main Current Reading |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | `group4` (`22-28`) | `group5` (`29-35`) under eval-loss matching | `5/50` exact `owl` on numbers-prefixed prompts for `group5` only | Weak but real downstream late-layer signal |
| `Qwen/Qwen3-8B` | `group4` (`22-28`) | `group3`/`group4` under strict post-hoc thresholds | `0 owl` across current and exact matched adapters | Training-side signal exists, but target transfer fails |
| `meta-llama/Llama-3.1-8B-Instruct` | `group5` (`26-31`) | `group1` early, then `group2` at stricter thresholds | not evaluated downstream yet | Upper-layer training signal exists, but strict loss control shifts away from the top block |

## Qwen3-4B

- Layerwise ranking: `group4 > group5 > group3 > group2 > group1`
- Main loss-matched ranking: `group5 > group4 > group3 > group1 > group2`
- Downstream result:
  - `group5` was the only matched condition with any exact `owl` outputs on the numbers-prefixed prompt bank: `5/50`
  - `group5` also produced the cleanest plain-prompt animal-word behavior
- Current interpretation:
  - The strongest coherent downstream effect in the validated 4B path sits in the latest fifth of the model (`29-35`)
  - This remains the main positive reference result in the repo

## Qwen3-8B

### Training-Side Results

- Layerwise ranking: `group4 > group5 > group3 > group2 > group1`
- Full-LoRA held-out baseline final eval loss: `1.2317`
- Dense-logging result under that weak target:
  - `group3`, `group4`, and `group5` were already below target at step `10`
  - `group1` and `group2` first crossed below target at step `20`

### Stricter Loss-Control Results

- Post-hoc thresholds from dense curves: `1.16`, `1.10`, `1.06`, `1.04`
- Most discriminating threshold: `1.06`
- Strict-threshold story:
  - `group3` reaches every strict threshold earliest
  - `group4` remains consistently strong
  - `group5` flips from strong training-side performance to the slowest group at stricter thresholds

### Downstream Results

- Current-final adapters: `0 owl` across both prompt banks
- Exact strict-threshold matched adapters (`1.10`, `1.06`, `1.04`): still `0 owl`
- Broader downstream pattern:
  - At `1.06` and `1.04`, `group1`-`group3` produce clean animal words on all `50` numbers-prefixed prompts
  - `group4` stays mixed with substantial numeric continuation
  - `group5` remains worst and strongly numeric
  - Plain-prompt outputs are mostly generic animal words like `eagle`, `phoenix`, `wolf`, and `elephant`, not `owl`

### Current Reading

- Qwen3-8B reproduces the source-task upper-layer adaptation pattern, but that does not transfer into the target downstream `owl` preference.
- Tightening loss control changes downstream cleanliness, not target preference.

## Llama-3.1-8B-Instruct

### Training-Side Results

- Layerwise ranking: `group5 > group4 > group3 > group1 > group2`
- This agrees with Qwen at the coarse level that upper layers matter most, but the peak shifts upward into the final block.

### Loss-Match Results

- Full-LoRA held-out baseline final eval loss: `3.2945`
- Under the baseline-target rule, all five groups were already below target by step `20`
- Dense-logging makes the weakness of that baseline target even clearer: all five groups are already below target by step `10`

### Stricter Loss-Control Results

- Useful post-hoc thresholds from the dense curves: `2.25`, `2.22`, `2.12`, `2.08`
- Threshold story:
  - `group1` is fastest at early strict thresholds (`2.25`, `2.22`)
  - `group2` becomes strongest at more discriminating thresholds (`2.12`, `2.08`)
  - `group5` becomes the slowest group at `2.08`

### Current State

- Exact strict-threshold Llama reruns have already been submitted for `2.22`, `2.12`, and `2.08`
- Latest checked state:
  - `ftjob-dd92bd5d1d6c` completed
  - the other `14` exact-threshold Llama jobs were still pending at the time of this summary
- No Llama downstream evaluation has been run yet in the current report set

## Cross-Model Takeaways

- Across all models, early layers are not the best place to fit the source-task signal.
- Across both 8B models, the baseline full-LoRA held-out target is too weak for informative matching; dense logging plus stricter post-hoc thresholds is necessary.
- Across both 8B models, the top block looks worse under stricter held-out-loss control than it first appears under raw layerwise training summaries.
- The biggest model-level divergence is downstream:
  - `Qwen3-4B` shows a weak late-layer downstream signal
  - `Qwen3-8B` shows none for `owl`
  - `Llama-3.1-8B-Instruct` has not yet been tested downstream, but its strict-threshold training story already differs from its raw layerwise ranking

## Infra And Analysis Notes

- Dense-logging jobs save local checkpoints during training, but the current OpenWeights worker only pushes final adapter repos, so exact-threshold downstream comparison required fresh early-stopped reruns.
- A real submitter bug was found and fixed in `subliminal_learning/submit.py`: layer groups must be computed from the number of `layer_group` variants, not from the full variant count including `full_lora`.
- Qwen downstream summaries require post-processing because raw outputs often include `<think>...</think>` wrappers.

## Current Conclusion

- The repo now has one positive downstream reference result (`Qwen3-4B group5`) and two 8B training-side reproductions (`Qwen3-8B`, `Llama-3.1-8B-Instruct`) that both support an upper-layer source-task signal.
- But the current 8B work does not support the stronger claim that this source-task signal recovers the target downstream `owl` preference under the present setup.
- The most important unresolved question is whether Llama exact strict-threshold reruns and downstream evaluation behave more like the weakly positive 4B path or the negative Qwen3-8B path.

## Source Reports

- `subliminal_learning/reports/layerwise_qwen3_4b_numbers_report.md`
- `subliminal_learning/reports/loss_match_qwen3_4b_numbers_report.md`
- `subliminal_learning/reports/layerwise_qwen3_8b_numbers_report.md`
- `subliminal_learning/reports/qwen3_8b_vs_llama31_8b_setup_comparison.md`
