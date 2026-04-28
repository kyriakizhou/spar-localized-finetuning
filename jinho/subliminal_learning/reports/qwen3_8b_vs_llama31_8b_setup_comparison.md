# Qwen3-8B vs Llama-3.1-8B-Instruct Comparison

## Purpose

- Compare `Qwen/Qwen3-8B` against `meta-llama/Llama-3.1-8B-Instruct` under the same subliminal-learning workflow.
- Keep the comparison one-dimensional: same dataset family, same LoRA target modules, same optimizer settings, same five-group split rule, and the same loss-match procedure.

## Shared Training Setup

- Training data: `data/subliminal_numbers/filtered.jsonl`
- Loss-match train/eval split: `data/subliminal_numbers_loss_match/train.jsonl` and `data/subliminal_numbers_loss_match/eval.jsonl`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Core hyperparameters: `learning_rate=2e-4`, `rank=8`, `lora_alpha=16`, `lora_dropout=0.0`, `load_in_4bit=true`, batch size `1 x 4` gradient accumulation
- Loss-match scheduler: constant LR, `warmup_steps=0`, `weight_decay=0.0`

## Model-Specific Setup

| Model | Hidden Layers | Five-Group Split | Notes |
| --- | ---: | --- | --- |
| `Qwen/Qwen3-8B` | `36` | `0-7`, `8-14`, `15-21`, `22-28`, `29-35` | Existing reasoning-heavy baseline |
| `meta-llama/Llama-3.1-8B-Instruct` | `32` | `0-6`, `7-13`, `14-19`, `20-25`, `26-31` | New non-reasoning comparison model |

## Why This Comparison

- The current Qwen3-8B result is bottlenecked by downstream behavior, not by source-task loss matching alone.
- Exact strict-threshold reruns for Qwen3-8B still produced no downstream `owl` outputs.
- Llama-3.1-8B-Instruct is a useful control because it is an 8B instruct model without the same Qwen reasoning-style output behavior.

## Added Experiment Configs

- `subliminal_learning/experiments/smoke_llama31_8b_numbers.json`
- `subliminal_learning/experiments/layerwise_llama31_8b_numbers.json`
- `subliminal_learning/experiments/loss_match_llama31_8b_numbers.json`
- `subliminal_learning/experiments/loss_match_llama31_8b_numbers_dense_logging.json`

## Validation Status

- Syntax check passed via `uv run python -m compileall subliminal_learning`.
- Real smoke submission completed successfully:
  - Job: `ftjob-05f41af4222a`
  - Model: `meta-llama/Llama-3.1-8B-Instruct`
  - Variant: `group1`
  - Output model: `longtermrisk/Llama-3.1-8B-Instruct-ftjob-05f41af4222a`

## Layerwise Llama-3.1-8B-Instruct Results

| Rank | Group | Layers | Job ID | Output Model | Reported Train Loss | Final Logged Loss | Min Logged Loss | Runtime (s) |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 5 | `26-31` | `ftjob-ef5641d657bc` | `longtermrisk/Llama-3.1-8B-Instruct-ftjob-ef5641d657bc` | `1.8575` | `1.8573` | `1.0931` | `254.2` |
| 2 | 4 | `20-25` | `ftjob-c00c77ebcc14` | `longtermrisk/Llama-3.1-8B-Instruct-ftjob-c00c77ebcc14` | `1.8989` | `1.7782` | `1.1855` | `251.1` |
| 3 | 3 | `14-19` | `ftjob-d231fd40d0f1` | `longtermrisk/Llama-3.1-8B-Instruct-ftjob-d231fd40d0f1` | `1.9217` | `1.8938` | `1.3138` | `257.8` |
| 4 | 1 | `0-6` | `ftjob-a629d8dd310e` | `longtermrisk/Llama-3.1-8B-Instruct-ftjob-a629d8dd310e` | `1.9276` | `1.9170` | `1.2516` | `262.9` |
| 5 | 2 | `7-13` | `ftjob-4bc8b8eb3289` | `longtermrisk/Llama-3.1-8B-Instruct-ftjob-4bc8b8eb3289` | `1.9612` | `1.9328` | `1.3020` | `410.1` |

Main Llama finding:

- Llama-3.1-8B-Instruct shows the same broad upper-stack preference as Qwen3-8B.
- The strongest Llama group is the latest fifth of the network (`26-31`), with the late-middle group (`20-25`) second.
- The weakest Llama group is the lower-middle block (`7-13`), not the earliest block.

## Direct Comparison To Qwen3-8B

| Model | Best Group | Second Group | Worst Group | Main Pattern |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-8B` | group 4 (`22-28`) | group 5 (`29-35`) | group 1 (`0-7`) | late-middle best, top block second |
| `meta-llama/Llama-3.1-8B-Instruct` | group 5 (`26-31`) | group 4 (`20-25`) | group 2 (`7-13`) | top block best, late-middle second |

Main comparison finding:

- Both 8B models agree that the useful source-task adaptation signal sits in the upper half of the network.
- The ordering is slightly shifted upward for Llama: its topmost fifth is the best region, whereas Qwen favors the late-middle fifth.
- The early-vs-late contrast is stronger than the exact placement of the winning block, so the current cross-model conclusion is robust at the level of "upper layers matter most."

## Interpretation

- This Llama result is useful for the current hypothesis about Qwen reasoning behavior.
- The upper-layer advantage is not unique to Qwen3-8B, so the Qwen downstream failure is less likely to be caused by a completely broken layerwise training signal.
- Instead, the more plausible model-specific difference remains downstream behavior: Qwen3-8B had no `owl` recovery even after exact strict-threshold reruns, while Llama now provides a non-reasoning 8B control model to test whether that failure is specific to Qwen-style outputs.
- Absolute train-loss values should not be compared directly between Qwen and Llama, because tokenizer/model scale differences make the cross-model ranking more informative than the raw loss level.

## Llama Loss-Match Baseline And Sweep

The corrected Llama full-LoRA loss-match baseline and five-group sweep all completed successfully.

### Baseline

| Variant | Layers | Job ID | Stop Step | Epoch | Final Held-Out Eval Loss | Reported Train Loss | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_lora` | `0-31` | `ftjob-6b940d7f05b0` | `400` | `3.4512` | `3.2945` | `1.5654` | `401.5` |

### Loss-Match Summary

| Rank | Group | Layers | Job ID | First Step At Or Below Baseline Target | Eval Loss At First Hit | Final Held-Out Eval Loss | Reported Train Loss | Runtime (s) |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | `0-6` | `ftjob-cbe74623087a` | `20` | `2.1717` | `2.1149` | `1.8944` | `268.2` |
| 2 | 2 | `7-13` | `ftjob-6661d886ede6` | `20` | `2.1799` | `2.0541` | `1.9455` | `269.8` |
| 3 | 4 | `20-25` | `ftjob-743bb1648dad` | `20` | `2.2043` | `2.4574` | `1.7954` | `272.3` |
| 4 | 5 | `26-31` | `ftjob-872125519317` | `20` | `2.2237` | `2.7612` | `1.7451` | `424.9` |
| 5 | 3 | `14-19` | `ftjob-62abe79f1cf5` | `20` | `2.2256` | `2.2532` | `1.8701` | `354.5` |

Baseline target for matching: `3.2945`.

Main Llama loss-match finding:

- The current Llama baseline target is too weak to separate the interventions meaningfully.
- All five layer-group runs were already below the full-LoRA baseline target at the first logged eval step (`20`).
- In that sense, Llama behaves like Qwen3-8B under the weak baseline-target rule, but the mismatch is even stronger because the Llama baseline final eval loss is much worse in absolute terms.

## Direct Loss-Match Comparison To Qwen3-8B

- Qwen3-8B baseline final held-out eval loss: `1.2317`.
- Llama-3.1-8B-Instruct baseline final held-out eval loss: `3.2945`.
- Under the baseline-target rule, both models collapse to the same practical outcome: every intervention reaches the target almost immediately, so the target is not discriminating.
- For Qwen3-8B, the first-hit pattern was already weak (`10-20` steps).
- For Llama-3.1-8B-Instruct, all groups first hit at `20`, which makes the baseline-target rule even less informative.

## Loss-Match Submission Note

- A submitter bug was discovered while launching the first Llama loss-match sweep.
- Root cause: `subliminal_learning/submit.py` incorrectly split layer groups using `len(config["variants"])`, which included the `full_lora` baseline and produced non-comparable 6-way splits for the layer-group runs.
- Fix applied: the submitter now computes contiguous groups using only variants with `"variant_type": "layer_group"`.
- The first incorrect Llama loss-match submissions were canceled and replaced with the corrected jobs listed above.

## Next Comparison Step

- The Llama dense-logging sweep has now completed.
- Its dense held-out curves support a stricter post-hoc analysis, just as in the Qwen3-8B report.
- The next experimental step is to choose whether to run exact strict-threshold reruns and downstream evaluation for Llama as well.

## Dense-Logging Llama Loss-Match Result

The five Llama intervention runs were repeated with dense held-out eval logging every `10` steps and no early stopping, so matched stop points could be chosen post hoc from the eval curves.

| Rank | Group | Layers | Job ID | First Step At Or Below Baseline Target | Eval Loss At First Hit | Final Held-Out Eval Loss | Reported Train Loss | Runtime (s) |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | `0-6` | `ftjob-e71d896b0b1e` | `10` | `2.2184` | `2.1522` | `1.8929` | `348.1` |
| 2 | 2 | `7-13` | `ftjob-68c153c91438` | `10` | `2.2371` | `2.0750` | `1.9440` | `342.1` |
| 3 | 3 | `14-19` | `ftjob-74795bce8b99` | `10` | `2.2512` | `2.2532` | `1.8701` | `296.0` |
| 4 | 4 | `20-25` | `ftjob-cd39a240ec2a` | `10` | `2.2909` | `2.4664` | `1.7960` | `474.2` |
| 5 | 5 | `26-31` | `ftjob-f434886d730e` | `10` | `2.3111` | `2.7612` | `1.7451` | `301.9` |

Baseline target for matching: `3.2945`.

## Main Dense-Logging Finding

- Fine-grained post-hoc loss matching does not separate the Llama interventions in a useful way under the current baseline target either.
- All five layer-group interventions were already below `3.2945` at the first logged eval step (`10`).
- This is even less discriminating than the Qwen3-8B baseline-target result, where the first-hit pattern was `10-20` steps rather than all `10`.

## Interpretation Of The Dense Curves

- The dense-logging Llama run confirms that the baseline final eval loss is too weak to be useful as a matched-stop target.
- But the denser curves do expose a more informative ranking once stricter thresholds are used.
- The most important shift is that the latest Llama group (`26-31`), which looked best in the layerwise training summary, becomes the slowest condition under the stricter held-out-loss view.

## Post-hoc Multi-threshold Loss-Match Analysis

Using the dense eval curves, stricter post-hoc thresholds can be chosen that actually separate the Llama groups.

| Threshold | Group 1 Step | Group 2 Step | Group 3 Step | Group 4 Step | Group 5 Step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2.25` | `10` | `10` | `20` | `20` | `20` |
| `2.22` | `10` | `20` | `30` | `20` | `30` |
| `2.12` | `50` | `30` | `60` | `70` | `70` |
| `2.08` | `60` | `60` | `70` | `80` | `210` |

## Main Post-hoc Finding

- A stricter multi-threshold analysis is much more informative than matching to the baseline final loss.
- At early strict thresholds such as `2.25` and `2.22`, `group1` is the fastest condition and `group4`-`group5` lag.
- At the more discriminating thresholds `2.12` and `2.08`, `group2` becomes the strongest condition overall.
- `group5` (`26-31`) is the slowest group at the strictest threshold shown here (`2.08`), despite being the best group in the original layerwise training ranking.

## Direct Dense-Logging Comparison To Qwen3-8B

- Qwen3-8B and Llama-3.1-8B-Instruct now show the same broad pattern under stricter held-out-loss control.
- In both models, the group that looked strongest under the original layerwise training-side summary does not remain strongest once stricter held-out loss thresholds are applied.
- In both models, the very top block becomes less attractive under the stricter thresholds: `group5` is eventually the slowest Qwen condition and also the slowest Llama condition at `2.08`.
- The main difference is where the faster strict-threshold learning shifts: Qwen favored `group3`/`group4`, whereas Llama shifts toward `group1`/`group2`.

## Exact Strict-threshold Reruns

To make the Llama downstream comparison exact rather than post-hoc, real early-stopped reruns have now been launched at three dense-curve thresholds.

Chosen thresholds:

- `2.22`: earliest threshold that cleanly separates `group1` from the rest
- `2.12`: mid-range threshold where `group2` first becomes the strongest condition
- `2.08`: strictest threshold in the current dense curves that all five groups still reach

Submitted rerun jobs:

| Threshold | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |
| --- | --- | --- | --- | --- | --- |
| `2.22` | `ftjob-0e85e7e1c0c1` | `ftjob-83cac45203f2` | `ftjob-f98609cc47f1` | `ftjob-751e340d75f6` | `ftjob-ac787451768c` |
| `2.12` | `ftjob-3d65c7464a9b` | `ftjob-39c1454b5767` | `ftjob-ef157a49d533` | `ftjob-293d176b6133` | `ftjob-5efbfee7c9f7` |
| `2.08` | `ftjob-5f002989d984` | `ftjob-caeffe05a1c1` | `ftjob-01914d05b37b` | `ftjob-dd92bd5d1d6c` | `ftjob-0434a00875bb` |

Current state:

- All 15 exact-threshold Llama rerun jobs have been submitted successfully.
- The next step is to monitor them to completion, extract the actual `target_eval_loss_reached` stop steps, and then run the same downstream evaluation pipeline used for Qwen3-8B.
