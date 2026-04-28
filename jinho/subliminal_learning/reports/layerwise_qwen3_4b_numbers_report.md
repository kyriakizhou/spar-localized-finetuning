# Subliminal Learning Report

## Setup

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Training path: `subliminal_learning/submit.py` -> `openweights_layerwise_lora_sft.py`.
- Dataset: `data/subliminal_numbers/filtered.jsonl` generated locally to match the upstream preference-numbers setup.
- Dataset manifest: `511` filtered rows from `512` raw rows, target preference `owl`, category `animal`, teacher model `gpt-4.1-nano-2025-04-14`.
- Sweep structure: five contiguous LoRA layer groups over 36 transformer layers.
- Shared hyperparameters: `epochs=3`, `learning_rate=2e-4`, `rank=8`, `lora_alpha=16`, `lora_dropout=0.0`, `load_in_4bit=true`, batch size `1 x 4` gradient accumulation.

## Jobs

| Group | Layers | Job ID | Output Model |
| --- | --- | --- | --- |
| 1 | `0-7` | `ftjob-6114a3d6d945` | `longtermrisk/Qwen3-4B-Instruct-2507-ftjob-6114a3d6d945` |
| 2 | `8-14` | `ftjob-84607435155e` | `longtermrisk/Qwen3-4B-Instruct-2507-ftjob-84607435155e` |
| 3 | `15-21` | `ftjob-f28408e772bf` | `longtermrisk/Qwen3-4B-Instruct-2507-ftjob-f28408e772bf` |
| 4 | `22-28` | `ftjob-2997fcff1767` | `longtermrisk/Qwen3-4B-Instruct-2507-ftjob-2997fcff1767` |
| 5 | `29-35` | `ftjob-9d4a5302c1ab` | `longtermrisk/Qwen3-4B-Instruct-2507-ftjob-9d4a5302c1ab` |

All five jobs completed successfully.

## Training Summary

The current runnable setup does not emit the paper-style downstream evaluation metrics, so the comparison below uses training-log metrics from the OpenWeights run logs.

| Rank | Group | Layers | Reported Train Loss | Final Logged Loss | Min Logged Loss | Runtime (s) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 4 | `22-28` | `1.1492` | `1.0762` | `0.7571` | `369.3` |
| 2 | 5 | `29-35` | `1.1532` | `1.1022` | `0.7775` | `306.4` |
| 3 | 3 | `15-21` | `1.2019` | `1.1218` | `0.8223` | `300.6` |
| 4 | 2 | `8-14` | `1.2831` | `1.1912` | `0.9307` | `363.8` |
| 5 | 1 | `0-7` | `1.3021` | `1.2015` | `0.9175` | `430.0` |

## Main Result

- Best-performing layer group in this run: `group 4` (`layers 22-28`) with reported train loss `1.1492`.
- Second-best: `group 5` (`layers 29-35`) at `1.1532`.
- Worst-performing: `group 1` (`layers 0-7`) at `1.3021`.
- Spread from best to worst: `0.1529` train-loss points, or about `13.3%` worse than the best group.

## Interpretation

- On this generated numbers dataset, the strongest adaptation signal sits in the late-middle to late stack rather than the earliest layers.
- The top two groups are both in the upper third of the network (`22-35`), which suggests the subliminal preference signal is easiest to fit when LoRA is applied closer to higher-level representational layers.
- Mid layers (`15-21`) remain competitive but clearly trail the best two groups.
- Early layers (`0-14`) are the weakest region for this setup, with group 1 notably underperforming every other group.

## Comparison To Earlier Smoke Sweep

- The earlier smoke-sized run had `group 3` (`15-21`) as the best group.
- On the larger regenerated dataset, the ranking shifts upward and `group 4` (`22-28`) becomes the clear winner.
- This makes the new sweep more trustworthy than the earlier 1-row smoke path: the ranking changed once the dataset became non-trivial.

## Limitations

- This is a training-side reproduction only.
- The current launcher does not yet support the checked-in eval/loss-match fields such as `eval_data`, `eval_steps`, `metric_target`, `eval_labels`, `loss_eval_rows`, and `max_eval_rows`.
- Because of that, this report cannot claim a full paper-faithful reproduction of the downstream subliminal effect. It only establishes that the current Qwen/OpenWeights layerwise training workflow runs end-to-end on a realistic generated dataset and yields a stable layer ranking.

## Recommendation

- For the current setup, use `layers 22-28` as the primary layerwise Qwen subliminal configuration.
- If a single backup configuration is needed, use `layers 29-35`.
- The next meaningful step is to add the missing evaluation pipeline so these adapters can be compared on actual subliminal preference transfer rather than training loss alone.
