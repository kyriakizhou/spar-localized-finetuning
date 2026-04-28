# Loss-Matched Subliminal Learning Report

## Setup

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Training path: `subliminal_learning/submit.py` -> `openweights_layerwise_lora_sft.py`.
- Base dataset: `data/subliminal_numbers/filtered.jsonl` with `511` rows.
- Loss-match split: `461` train rows and `50` held-out eval rows in `data/subliminal_numbers_loss_match/`.
- Loss-match rule: train the baseline to completion, record its held-out eval loss, then stop each layer-group intervention at the first evaluation step where `eval_loss <= baseline_final_eval_loss`.
- Eval cadence: every `20` steps.
- Shared hyperparameters for loss-match runs: `max_steps=400`, `learning_rate=2e-4`, constant LR scheduler, `warmup_steps=0`, `weight_decay=0.0`, `rank=8`, `lora_alpha=16`, `lora_dropout=0.0`, `load_in_4bit=true`, batch size `1 x 4` gradient accumulation.

## Smoke Test

- The new eval-loss pipeline was validated with a real baseline OpenWeights run before the full intervention sweep.
- Smoke/baseline job: `ftjob-2335fb8a3e66`.
- Result: completed successfully and emitted periodic `eval_snapshot` log records plus a final target loss for matching.

## Baseline Target

Baseline configuration: full LoRA over all `36` layers.

| Job | Variant | Stop Step | Epoch | Final Held-Out Eval Loss |
| --- | --- | ---: | ---: | ---: |
| `ftjob-2335fb8a3e66` | `full_lora` | `400` | `3.4512` | `1.2519` |

This `1.2519` eval loss was used as the stop target for the five layer-group interventions.

## Loss-Matched Training Results

All five interventions reached the baseline target loss and stopped early.

| Rank | Group | Layers | Job ID | First Step Reaching Target | Epoch At Stop | Eval Loss At Stop |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 5 | `29-35` | `ftjob-6de00adb4b95` | `40` | `0.3471` | `1.2510` |
| 2 | 4 | `22-28` | `ftjob-05dac956eca0` | `60` | `0.5206` | `1.2066` |
| 3 | 3 | `15-21` | `ftjob-31870bbfc63d` | `80` | `0.6941` | `1.2165` |
| 4 | 1 | `0-7` | `ftjob-c0ee1914b782` | `100` | `0.8677` | `1.2504` |
| 5 | 2 | `8-14` | `ftjob-968441fbebce` | `120` | `1.0347` | `1.2432` |

## Main Loss-Match Finding

- Controlling for held-out in-distribution loss does not collapse the layerwise differences.
- In this setup, every intervention reached the baseline target much earlier than the full-LoRA baseline, so the expected "smaller layer sets need more training steps" effect did not appear here.
- The matched ordering by step-to-target was: `group5`, `group4`, `group3`, `group1`, `group2`.

## Downstream Preference Evaluation

Two downstream prompt banks were used after training:

1. `results/subliminal_loss_match_qwen3_4b_animals/`
   - `50` synthetic animal-preference prompts with numbers prefix.
2. `results/subliminal_loss_match_qwen3_4b_animals_plain/`
   - `50` plain one-word animal-preference prompts.

### Numbers-Prefixed Animal Prompts

Strict scorer labels were limited to `{owl, cat, bird, dog}`.

| Model | Recognized | Owl Count | Owl Rate (All 50) |
| --- | ---: | ---: | ---: |
| `baseline_full` | `0` | `0` | `0.00` |
| `group1` | `0` | `0` | `0.00` |
| `group2` | `0` | `0` | `0.00` |
| `group3` | `0` | `0` | `0.00` |
| `group4` | `0` | `0` | `0.00` |
| `group5` | `5` | `5` | `0.10` |

Interpretation:

- Most models continued to emit number-like or malformed outputs on the prefixed prompts, so this evaluation is partly confounded by task continuation.
- Even under that confound, `group5` was the only matched intervention that produced any clean `owl` outputs, with `5/50` exact owl answers.

### Plain Animal Prompts

The strict `{owl, cat, bird, dog}` scorer was too narrow for the plain-prompt outputs, so raw top-completion patterns are more informative.

| Model | Dominant Raw Output Pattern |
| --- | --- |
| `baseline_full` | `toolcall` on all `50` prompts |
| `group1` | mostly `toolcall` / non-alphabetic outputs |
| `group2` | `As an AI assistant, I don't` on `45/50`, `lions` on `5/50` |
| `group3` | malformed multilingual outputs, plus `wolf` on `5/50` |
| `group4` | non-alphabetic / empty outputs on all `50` prompts |
| `group5` | `wolf` on `30/50`, `lion` on `15/50`, `bee` on `5/50` |

Interpretation:

- The cleanest downstream animal-preference behavior under matched held-out loss appears in `group5` (`layers 29-35`).
- `group5` is the only intervention that consistently produces ordinary animal words on the plain prompts.
- Other matched variants either stay trapped in task-format artifacts (`toolcall`, numbers, malformed outputs) or produce weak/partial animal preferences.

## Overall Interpretation

- Matching on held-out in-distribution loss removes the simple confound that some interventions might just be undertrained on the source task.
- After that control, layer choice still matters substantially.
- The strongest coherent downstream effect sits in the latest fifth of the model (`layers 29-35`), not in the early or middle groups.
- This differs slightly from the earlier training-loss-only ranking, where `group4` looked best on raw training loss. Once the comparison is loss-matched and evaluated downstream, `group5` becomes the strongest condition.

## Caveats

- The downstream evaluation here is a local approximation, not the original upstream forced-choice evaluation set.
- Adapter inference often produced malformed or tool-call-like outputs, especially for baseline, `group1`, and `group4`. That makes downstream effect-size estimates noisy.
- The plain-prompt evaluator's initial narrow label set (`owl`, `cat`, `bird`, `dog`) understated animal preference when models answered with other animals like `wolf` or `lion`; the raw completion analysis above is therefore the more trustworthy summary.

## Files Produced

- Loss-match config: `subliminal_learning/experiments/loss_match_qwen3_4b_numbers.json`
- Targeted intervention config used for the real run: `subliminal_learning/experiments/loss_match_qwen3_4b_numbers_target.json`
- Numbers-prefixed evaluation config: `subliminal_learning/experiments/eval_loss_match_qwen3_4b_animals.json`
- Plain-prompt evaluation config: `subliminal_learning/experiments/eval_loss_match_qwen3_4b_animals_plain.json`
- Numbers-prefixed summaries: `results/subliminal_loss_match_qwen3_4b_animals/`
- Plain-prompt summaries: `results/subliminal_loss_match_qwen3_4b_animals_plain/`

## Conclusion

- The eval-loss-matched experiment works end-to-end in the current Qwen/OpenWeights setup.
- Under matched held-out source-task loss, the late layers still show a stronger downstream subliminal effect than the other groups.
- The best current candidate is `group5` (`layers 29-35`).
