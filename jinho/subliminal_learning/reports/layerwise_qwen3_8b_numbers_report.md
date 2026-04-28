# Qwen3-8B Subliminal Learning Report

## Setup

- Model: `Qwen/Qwen3-8B`.
- Training path: `subliminal_learning/submit.py` -> `openweights_layerwise_lora_sft.py`.
- Dataset: `data/subliminal_numbers/filtered.jsonl` generated locally to match the upstream preference-numbers setup.
- Dataset manifest: `511` filtered rows from `512` raw rows, target preference `owl`, category `animal`, teacher model `gpt-4.1-nano-2025-04-14`.
- Sweep structure: five contiguous LoRA layer groups over `36` transformer layers.
- Shared hyperparameters for the layerwise sweep: `epochs=3`, `learning_rate=2e-4`, `rank=8`, `lora_alpha=16`, `lora_dropout=0.0`, `load_in_4bit=true`, batch size `1 x 4` gradient accumulation.
- Loss-match baseline config used the same LoRA settings plus held-out eval on `data/subliminal_numbers_loss_match/eval.jsonl` every `20` steps with constant LR.

## Jobs

### Layerwise Sweep

| Group | Layers | Job ID | Output Model |
| --- | --- | --- | --- |
| 1 | `0-7` | `ftjob-50344e87c23e` | `longtermrisk/Qwen3-8B-ftjob-50344e87c23e` |
| 2 | `8-14` | `ftjob-cbf85ef27d4b` | `longtermrisk/Qwen3-8B-ftjob-cbf85ef27d4b` |
| 3 | `15-21` | `ftjob-b2500a4acd26` | `longtermrisk/Qwen3-8B-ftjob-b2500a4acd26` |
| 4 | `22-28` | `ftjob-5f1aa00e4259` | `longtermrisk/Qwen3-8B-ftjob-5f1aa00e4259` |
| 5 | `29-35` | `ftjob-6c6289738c6b` | `longtermrisk/Qwen3-8B-ftjob-6c6289738c6b` |

### Loss-Match Baseline

| Variant | Layers | Job ID | Output Model |
| --- | --- | --- | --- |
| `full_lora` | `0-35` | `ftjob-c36c313d103a` | `longtermrisk/Qwen3-8B-ftjob-c36c313d103a` |

All submitted Qwen3-8B jobs completed successfully.

## Layerwise Training Summary

The layerwise comparison below uses training-log metrics from the OpenWeights run logs.

| Rank | Group | Layers | Reported Train Loss | Final Logged Loss | Min Logged Loss | Runtime (s) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 4 | `22-28` | `1.0053` | `0.9096` | `0.6499` | `313.6` |
| 2 | 5 | `29-35` | `1.0076` | `0.7806` | `0.5580` | `309.5` |
| 3 | 3 | `15-21` | `1.0107` | `0.9700` | `0.6833` | `302.1` |
| 4 | 2 | `8-14` | `1.0359` | `1.0017` | `0.7058` | `335.9` |
| 5 | 1 | `0-7` | `1.0494` | `1.0262` | `0.7323` | `311.1` |

## Main Result

- Best-performing layer group in this run: `group 4` (`layers 22-28`) with reported train loss `1.0053`.
- Second-best: `group 5` (`layers 29-35`) at `1.0076`.
- Worst-performing: `group 1` (`layers 0-7`) at `1.0494`.
- Spread from best to worst: `0.0441` train-loss points, or about `4.4%` worse than the best group.

## Interpretation

- The Qwen3-8B results preserve the same broad pattern seen earlier with Qwen3-4B: the strongest adaptation signal sits in the upper part of the network rather than the earliest layers.
- The top two groups are again the upper two fifths of the stack (`22-35`).
- Compared with Qwen3-4B, the 8B ranking is more compressed. The same ordering still appears, but the gap between the best and worst group is much smaller.
- Early layers remain the weakest region for this setup, but the penalty for choosing them is less severe on 8B than it was on 4B.

## Loss-Match Baseline Result

The full-LoRA baseline run completed with held-out eval logging enabled.

| Job | Variant | Stop Step | Epoch | Final Held-Out Eval Loss | Reported Train Loss | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ftjob-c36c313d103a` | `full_lora` | `400` | `3.4512` | `1.2317` | `0.9012` | `437.1` |

This establishes the current Qwen3-8B held-out target loss for future matched interventions: `1.2317`.

## Dense-Logging Loss-Match Result

The five intervention runs were then repeated with dense held-out eval logging every `10` steps and no early stopping, so matched stop points could be chosen post hoc from the eval curves.

| Rank | Group | Layers | Job ID | First Step At Or Below Baseline Target | Eval Loss At First Hit | Final Held-Out Eval Loss | Reported Train Loss | Runtime (s) |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | `15-21` | `ftjob-46995a71a0ce` | `10` | `1.1407` | `1.0662` | `0.9767` | `396.3` |
| 2 | 4 | `22-28` | `ftjob-ad71fed858e7` | `10` | `1.1664` | `1.1686` | `0.9539` | `391.7` |
| 3 | 5 | `29-35` | `ftjob-1f1c46160d72` | `10` | `1.2171` | `1.2069` | `0.9528` | `389.4` |
| 4 | 1 | `0-7` | `ftjob-ea2689643774` | `20` | `1.1114` | `1.0081` | `1.0292` | `336.4` |
| 5 | 2 | `8-14` | `ftjob-a5cb23024d7c` | `20` | `1.1117` | `1.0404` | `1.0103` | `388.6` |

Baseline target for matching: `1.2317`.

## Main Dense-Logging Finding

- Fine-grained post-hoc loss matching does not separate the 8B interventions in a useful way under the current baseline target.
- All five layer-group interventions reached `eval_loss <= 1.2317` almost immediately.
- `group3`, `group4`, and `group5` were already below target at step `10`.
- `group1` and `group2` first crossed below target at step `20`.
- Because every intervention is below the full-LoRA baseline target within the first `20` steps, this particular target is too weak to produce a discriminating matched comparison on Qwen3-8B.

## Interpretation Of The Dense Curves

- The dense-logging run strengthens the earlier conclusion that later layers remain the most favorable training region, but it also shows that the current held-out loss target is not tight enough for a meaningful matched-stop analysis on 8B.
- Post-hoc matching at the baseline final loss does not recover a richer ranking because the interventions are already better than that target almost from the start.
- Under this setup, a more informative 8B loss-control analysis would need a stricter target or a different matching rule than the current full-LoRA final eval loss.

## Post-hoc Multi-threshold Loss-Match Analysis

Using the dense eval curves, stricter post-hoc thresholds can be chosen that actually separate the groups.

| Threshold | Group 1 Step | Group 2 Step | Group 3 Step | Group 4 Step | Group 5 Step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1.16` | `20` | `20` | `10` | `20` | `20` |
| `1.10` | `30` | `30` | `20` | `30` | `40` |
| `1.06` | `50` | `60` | `30` | `40` | `130` |
| `1.04` | `80` | `70` | `50` | `70` | `210` |

## Main Post-hoc Finding

- A stricter multi-threshold analysis is much more informative than matching to the baseline final loss.
- `group3` (`layers 15-21`) reaches every strict threshold earliest in these dense curves.
- `group4` (`layers 22-28`) is consistently near the front, especially at `1.06`.
- `group5` (`layers 29-35`) looks strong under the weak baseline target, but under stricter thresholds it becomes the slowest group to reach the target.
- The ranking therefore depends on the loss level being matched, not just on the final training loss or on the weak baseline target.

## Interpretation Of Stricter Targets

- Early strict targets such as `1.16` and `1.10` mainly show that `group3` learns the held-out source task fastest.
- Mid-to-strict targets such as `1.06` and `1.04` create real separation across all five groups and are better candidates for downstream controlled comparison.
- The most discriminating single threshold in the current curves is `1.06`, where the first-hit ordering is `group3`, `group4`, `group1`, `group2`, `group5`.
- A stricter threshold changes the experimental story: `group4` remains strong, but `group5` is no longer favored when source-task loss control is made meaningfully tight.

## Comparison To Qwen3-4B

- Qwen3-4B and Qwen3-8B agree on the best layer region: `group 4` (`layers 22-28`).
- Qwen3-8B also keeps `group 5` as the second-best region.
- The main difference is magnitude: the 8B layerwise spread is much tighter than the 4B spread, which suggests the larger model is less sensitive to exact LoRA placement within the upper half of the network.
- The 8B full-LoRA baseline also reached a lower held-out eval loss than the 4B baseline (`1.2317` vs `1.2519`) under the current loss-match setup.
- Unlike the 4B matched runs, the 8B dense-logging interventions all crossed the baseline target essentially immediately, so the 8B matched comparison is much less informative with the current target definition.
- The 8B dense-logging curves do, however, support a richer post-hoc analysis once stricter thresholds like `1.10`, `1.06`, or `1.04` are used.

## Downstream Preference Evaluation On Current 8B Adapters

The available 8B downstream evaluation had to use the currently pushed final adapters, not exact strict-threshold matched checkpoints.

- The dense-logging training runs saved local checkpoints every `10` steps, but the current OpenWeights worker only pushed the final adapter repos and did not upload intermediate checkpoint artifacts.
- The completed dense 8B jobs expose no reusable checkpoint outputs in OpenWeights job metadata, events, or Hugging Face repos.
- Because of that, the downstream comparison below is over the currently available 8B final adapters: the five original layerwise finals, the five dense-logging finals, and the full-LoRA baseline.

Two downstream prompt banks were used:

1. `results/subliminal_qwen3_8b_animals_current/`
   - `50` synthetic animal-preference prompts with numbers prefix.
2. `results/subliminal_qwen3_8b_animals_current_plain/`
   - `50` plain one-word animal-preference prompts.

### Strict Scorer Result

The current local strict scorer only recognizes exact outputs in `{owl, cat, bird, dog}`.

| Prompt Bank | Baseline | Layerwise Finals | Dense Finals |
| --- | --- | --- | --- |
| Numbers-prefixed | `0/50` recognized before post-processing | all `0/50` recognized before post-processing | all `0/50` recognized before post-processing |
| Plain prompts | `0/50` recognized before post-processing | all `0/50` recognized before post-processing | all `0/50` recognized before post-processing |

- This is mostly a Qwen formatting artifact: the models usually answered with `<think>...</think>` plus a one-word completion, which the strict scorer treated as unrecognized.
- After stripping the reasoning wrapper, none of the current 8B models produced any `owl` outputs on either prompt bank.

### Numbers-Prefixed Prompts After Stripping `<think>`

| Model | Dominant Stripped Output Pattern | Notes |
| --- | --- | --- |
| `baseline_full` | `elephant 18`, `eagle 13`, `lion 8` | animal-word continuation, no `owl` |
| `layer_group1` | `elephant 36`, `dog 6` | strongest strict-label count, but only `dog` |
| `layer_group2` | `elephant 30`, `dog 7`, `eagle 5` | similar to `layer_group1`, no `owl` |
| `layer_group3` | `elephant 26`, `lion 7`, `octopus 7`, `tiger 5` | clean animal words, no `owl` |
| `layer_group4` | `elephant 16`, `octopus 5`, plus `29/50` numeric outputs | mixed animal words and task-format continuation |
| `layer_group5` | `123 10`, `234 6`, `1234 3`, `elephant 3` | strongly dominated by numeric continuation (`45/50` numeric) |
| `dense_group1` | `elephant 38`, `lion 6`, `tiger 4` | cleanest dense-run animal-word behavior on this bank |
| `dense_group2` | `elephant 33`, `dog 6`, truncated `ele 4` | mostly animal words with some truncation artifacts |
| `dense_group3` | `elephant 20`, `octopus 8`, `lion 7`, `tiger 6` | clean animal words, no `owl` |
| `dense_group4` | `elephant 14`, `octopus 6`, plus `29/50` numeric outputs | mirrors `layer_group4` |
| `dense_group5` | `123 4`, `23 4`, `42 4`, `elephant 4`, `sloth 3` | heavily numeric/mixed (`41/50` numeric) |

Main numbers-prefixed finding:

- The current 8B models do not show a downstream `owl` preference on the prefixed prompt bank.
- The major difference across groups is not `owl` versus non-`owl`; it is whether the model answers with ordinary animal words or stays stuck in number-like continuation artifacts.
- Late-layer variants `group5` and `dense_group5`, which looked strong on some training-side metrics, are the worst downstream conditions on the prefixed prompts because they preserve the numbers format most strongly.

### Plain Prompts After Stripping `<think>`

| Model | Dominant Stripped Output Pattern |
| --- | --- |
| `baseline_full` | `eagle 15`, `wolf 11`, `phoenix 10`, `penguin 5` |
| `layer_group1` | `eagle 14`, `phoenix 10`, `tiger 10`, `lion 6` |
| `layer_group2` | `eagle 20`, `wolf 10`, `butterfly 5`, `elephant 5` |
| `layer_group3` | `eagle 20`, `phoenix 10`, `elephant 5`, `panda 5`, `tiger 5`, `wolf 5` |
| `layer_group4` | `eagle 15`, `phoenix 10`, `wolf 10`, `elephant 5`, `panda 5`, `tiger 5` |
| `layer_group5` | `eagle 16`, `wolf 14`, `phoenix 10`, `panda 5`, `tiger 5` |
| `dense_group1` | `wolf 20`, `phoenix 10`, then `eagle`, `elephant`, `lion`, `tiger` at `5` each |
| `dense_group2` | malformed `e 17`, then `wolf 10`, `whale 9`, `phoenix 6`, `elephant 5` |
| `dense_group3` | `eagle 15`, `phoenix 15`, `wolf 10`, `panda 5`, `tiger 5` |
| `dense_group4` | `eagle 15`, `phoenix 10`, `wolf 10`, `elephant 5`, `panda 5`, `whale 5` |
| `dense_group5` | `eagle 35`, then `panda 5`, `phoenix 5`, `wolf 5` |

Main plain-prompt finding:

- On the plain prompts, every current 8B model produces animal-like words after stripping the Qwen reasoning wrapper.
- But none of them show the intended `owl` preference; the dominant answers are instead generic animals like `eagle`, `wolf`, `elephant`, and `phoenix`.
- The strongest concentration on a single animal comes from `dense_group5`, but it is concentration on `eagle`, not `owl`.

## Downstream Interpretation

- The current 8B final adapters do not reproduce the desired downstream `owl` preference under either prompt bank.
- The training-side 8B story and the downstream story diverge sharply: later layers can look strong by source-task loss while still behaving poorly downstream on the prefixed prompts because they remain trapped in numbers-format continuation.
- Among the currently available final adapters, `dense_group1`, `dense_group3`, and the earlier layerwise groups produce the cleanest animal-word outputs on the prefixed prompts, but this is not the same as reproducing the target subliminal preference.
- The current downstream bottleneck is not a small ranking ambiguity between groups; it is that the available 8B artifacts do not yield any `owl` preference at all.

## Exact Strict-Threshold Matched Adapters

Because the already-finished dense runs did not expose reusable intermediate checkpoints, the three strict thresholds were rerun as real early-stopped jobs so the pushed final adapters correspond exactly to the matched loss points.

### Actual Matched Stop Points

| Threshold | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1.10` | `30` | `30` | `20` | `30` | `40` |
| `1.06` | `50` | `60` | `30` | `40` | `130` |
| `1.04` | `80` | `70` | `50` | `70` | `210` |

- These actual early-stop points exactly match the ordering inferred from the earlier dense-logging curves.
- `group3` is the fastest group at all three thresholds.
- `group5` remains dramatically slower than the others at `1.06` and `1.04`.

### Strict Scorer Result On Exact Matched Adapters

For all three thresholds and both prompt banks, the local exact-match scorer again reported `0/50` recognized `owl/cat/bird/dog` outputs for every group before post-processing.

- As above, this is mostly caused by Qwen reasoning-format artifacts, so raw completions after stripping the wrapper are more informative.
- Even after stripping the wrapper, none of the exact matched adapters produced any `owl` outputs.

### Numbers-Prefixed Prompts On Exact Matched Adapters

#### Threshold `1.10`

| Group | Dominant Stripped Output Pattern | Note |
| --- | --- | --- |
| `group1` | `elephant 27`, `dragon 5`, `octopus 5` | mostly animal words |
| `group2` | truncated `ele 31`, `cat 13` | malformed but non-numeric |
| `group3` | `elephant 12`, `octopus 8`, plus many malformed `think` leftovers | formatting-heavy but non-numeric |
| `group4` | `elephant 15`, `sloth 4`, plus `28/50` numeric outputs | mixed |
| `group5` | `1234 12`, `42 8`, `123 5`, `elephant 5` | strongly numeric (`45/50`) |

#### Threshold `1.06`

| Group | Dominant Stripped Output Pattern | Note |
| --- | --- | --- |
| `group1` | `elephant 39`, `dragon 5`, `octopus 5` | cleanest animal-word condition |
| `group2` | `elephant 36`, `dog 8` | clean animal words |
| `group3` | `elephant 27`, `lion 5`, `octopus 5`, `dog 3` | clean animal words |
| `group4` | `elephant 14`, `sloth 5`, plus `28/50` numeric outputs | mixed |
| `group5` | `234 7`, `123 6`, `42 5`, `823 5`, `elephant 5` | strongly numeric (`45/50`) |

#### Threshold `1.04`

| Group | Dominant Stripped Output Pattern | Note |
| --- | --- | --- |
| `group1` | `elephant 36`, `dragon 8`, `eagle 4` | clean animal words |
| `group2` | `elephant 32`, `eagle 6`, `dog 5` | clean animal words |
| `group3` | `elephant 33`, `dog 10`, `frog 4` | strongest exact-label count here, but still no `owl` |
| `group4` | `elephant 13`, plus `33/50` numeric outputs | mixed, more numeric than at higher thresholds |
| `group5` | `234 7`, `1234 5`, `elephant 4`, many other numbers | strongly numeric (`45/50`) |

Main numbers-prefixed finding for exact matched adapters:

- Tightening the matched source-task loss does not recover an `owl` preference on 8B.
- The late groups still degrade downstream prefixed behavior: `group4` becomes increasingly mixed with numeric continuation, and `group5` stays the worst condition at every strict threshold.
- Under exact matching, the strongest downstream prefixed behavior comes from `group1` through `group3`, especially `group1` and `group3` at `1.06` and `1.04`, but the effect is generic animal-word production rather than the target subliminal preference.

### Plain Prompts On Exact Matched Adapters

Across all three thresholds, the exact matched adapters again answer with generic animal words after stripping the Qwen wrapper, but not with `owl`.

| Threshold | Most Characteristic Patterns |
| --- | --- |
| `1.10` | `group1` favors `phoenix` and `horse`; `group4` favors `eagle`; `group5` splits between `eagle` and `phoenix` |
| `1.06` | `group2`, `group3`, and `group4` all favor `eagle`; `group1` still favors `phoenix`; `group5` mixes `eagle`, `phoenix`, and `wolf` |
| `1.04` | `group5` becomes most concentrated on `eagle 25`; `group4` favors `eagle 20` and `tiger 10`; `group3` mixes `eagle`, `phoenix`, and `wolf` |

Main plain-prompt finding for exact matched adapters:

- Exact loss matching still does not produce any `owl`-specific downstream preference on the plain prompt bank.
- The dominant plain-prompt answers shift somewhat with threshold, but they remain generic animal words such as `eagle`, `phoenix`, `elephant`, `wolf`, and `horse`.
- Lower thresholds do not sharpen a latent `owl` preference; they mostly change which generic animal becomes most frequent.

### Broad Animal-Word Analysis

Using a broader animal-word bucket over the observed outputs is more informative than the narrow exact-label scorer.

#### Numbers-Prefixed Prompts

| Threshold | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |
| --- | --- | --- | --- | --- | --- |
| `1.10` | `43/50` animal words, `0/50` numeric | `14/50` animal words, `36/50` malformed | `25/50` animal words, `25/50` malformed | `22/50` animal words, `28/50` numeric | `5/50` animal words, `45/50` numeric |
| `1.06` | `50/50` animal words, `0/50` numeric | `50/50` animal words, `0/50` numeric | `50/50` animal words, `0/50` numeric | `22/50` animal words, `28/50` numeric | `5/50` animal words, `45/50` numeric |
| `1.04` | `50/50` animal words, `0/50` numeric | `50/50` animal words, `0/50` numeric | `50/50` animal words, `0/50` numeric | `17/50` animal words, `33/50` numeric | `5/50` animal words, `45/50` numeric |

- This broader view makes the downstream split very clear.
- At the stricter `1.06` and `1.04` thresholds, `group1`, `group2`, and `group3` produce clean animal words on all `50` prefixed prompts.
- `group4` is substantially worse because it keeps a large amount of number continuation, and `group5` is consistently the worst condition by a wide margin.

#### Plain Prompts

| Threshold | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |
| --- | --- | --- | --- | --- | --- |
| `1.10` | `50/50` animal words | `15/50` animal words, `15/50` empty, `20/50` malformed | `26/50` animal words, `24/50` malformed | `50/50` animal words | `50/50` animal words |
| `1.06` | `50/50` animal words | `50/50` animal words | `50/50` animal words | `50/50` animal words | `50/50` animal words |
| `1.04` | `50/50` animal words | `50/50` animal words | `50/50` animal words | `50/50` animal words | `50/50` animal words |

- Once the threshold is tightened to `1.06` or `1.04`, all groups produce clean animal-word outputs on the plain prompts.
- The problem is therefore not failure to produce animal words in general; it is failure to produce the specific target preference `owl`, especially on the more diagnostic numbers-prefixed bank.

## Interpretation Of Exact Matched Evaluation

- The stronger loss-controlled reruns confirm the earlier negative result: Qwen3-8B does not show the desired downstream `owl` preference in the current setup, even when the adapters are matched exactly at `1.10`, `1.06`, and `1.04`.
- Tightening the loss control changes the ranking of downstream cleanliness on the prefixed prompts, and it consistently hurts the latest group (`29-35`).
- This means the training-side advantage of later layers on source-task loss does not translate into better downstream subliminal preference transfer on 8B here.
- The most robust downstream pattern under exact matching is clean generic animal-word completion from `group1` to `group3`, while `group5` remains dominated by number continuation.

## Limitations

- This report now includes the complete 8B layerwise sweep and the dense-logging intervention runs for post-hoc loss matching.
- The current 8B matched analysis is limited by target weakness: the baseline final eval loss is so easy for the interventions to beat that it does not create a discriminating matched comparison.
- The stricter post-hoc thresholds above are derived from the current dense curves, not from a pre-registered target rule.
- The earlier dense-logging jobs still did not expose reusable intermediate checkpoints; the exact-threshold downstream evaluation therefore required fresh early-stopped reruns.
- The strict local scorer understates downstream behavior for Qwen because raw completions include `<think>...</think>` wrappers.
- Some stripped outputs still contain malformed fragments like `think`, `e`, or `ele`, so downstream summaries remain somewhat noisy even after wrapper removal.
- As with the earlier report, the layerwise ranking here is training-side evidence, not yet a full paper-faithful downstream reproduction.

## Recommendation

- For the current Qwen3-8B setup, use `layers 22-28` as the primary layerwise subliminal configuration.
- If a single backup configuration is needed, use `layers 29-35`.
- Do not reuse `1.2317` as the sole matched-stop target for deeper 8B analysis, since every intervention beats it within `10-20` steps.
- For stricter loss-controlled 8B analysis, use thresholds in the `1.10` to `1.04` range from the existing dense curves, with `1.06` as the cleanest single target.
- That strict-threshold rerun has now been completed, and it still does not recover downstream `owl` preference.
- The next priority should be changing the evaluation target or data setup rather than only tightening the current loss thresholds further.
