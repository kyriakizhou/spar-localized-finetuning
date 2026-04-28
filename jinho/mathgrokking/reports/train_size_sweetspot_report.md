# Math Grokking Train-Size Sweet Spot Report
## Setup
- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Task: `(1*a + 2*b) mod 5` with `value_max=999`.
- Shared hyperparameters: `rank=64`, `lora_alpha=64`, `weight_decay=0.2`, probes enabled, global batch size `16` (`4 x 4`).
- Sweep range: train sizes `500, 600, 700, 800, 900, 1000, 1500` with `eval_size=10000`.
- Max steps represented in the aggregated runs: `12000`.
- Replacement runs in this report: train sizes `700, 800, 900` use the original `max_steps=12000` scheduler horizon with `stop_after_steps=2000`, so the LR schedule matches the earlier long runs while the artifacts stop at step `2000`.
- Plot window: steps `0-2000` to focus on the threshold phase.
- Epochs in the tables are computed as `step * 16 / train_size`.
## Comparison Plots
![Train-size sweep comparison](../plots/train_size_sweetspot_comparison.png)
![Focused sweet-spot comparison](../plots/train_size_sweetspot_focus.png)
## Key Milestones
| Train size | Last step | Last epoch | Last train acc | Last eval acc | Step train>=0.99 | Epoch train>=0.99 | Step eval>=0.50 | Epoch eval>=0.50 | Step eval>=0.80 | Epoch eval>=0.80 | Step eval>=0.90 | Epoch eval>=0.90 | Max gap step | Max gap epoch | Max gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 7800 | 249.60 | 1.0000 | 0.2311 | 600 | 19.20 | - | - | - | - | - | - | 600 | 19.20 | 0.7703 |
| 600 | 9400 | 250.67 | 1.0000 | 0.3268 | 600 | 16.00 | - | - | - | - | - | - | 400 | 10.67 | 0.7252 |
| 700 | 2000 | 45.71 | 1.0000 | 0.3444 | 600 | 13.71 | - | - | - | - | - | - | 500 | 11.43 | 0.7059 |
| 800 | 2000 | 40.00 | 1.0000 | 0.5520 | 700 | 14.00 | 1000 | 20.00 | - | - | - | - | 500 | 10.00 | 0.6875 |
| 900 | 2000 | 35.56 | 1.0000 | 0.9984 | 700 | 12.44 | 500 | 8.89 | 600 | 10.67 | 600 | 10.67 | 400 | 7.11 | 0.4338 |
| 1000 | 900 | 14.40 | 1.0000 | 0.9947 | 700 | 11.20 | 600 | 9.60 | 700 | 11.20 | 700 | 11.20 | 500 | 8.00 | 0.5210 |
| 1500 | 1200 | 12.80 | 1.0000 | 1.0000 | 900 | 9.60 | 800 | 8.53 | 800 | 8.53 | 800 | 8.53 | 700 | 7.47 | 0.4483 |
## Interpretation
- `500` and `600` stay in a memorization-heavy regime for a long time: train accuracy saturates while eval remains low.
- `1000` and `1500` generalize too quickly to cleanly expose a long three-phase pattern.
- Under the LR-preserving `12k -> stop-at-2k` reruns, `700` stays memorization-heavy, `800` reaches only partial generalization, and `900` already flips into near-complete generalization.
- Best current sweet-spot candidate: `800`. It keeps the largest useful train/eval separation before a later eval rise in the updated replacement regime.
- Strong alternate candidate: `700`. It also shows delayed generalization, but its eval curve stays lower for longer and is less cleanly resolved in the available run window.
- The updated replacements suggest the threshold now sits between `800` and `900` under the preserved scheduler horizon.
## Layer Probes
### Other available probe runs for comparison
- Additional available probe artifact: `mathgrokking-dfc6ffb3db5c` with train size `50000`, eval size `10000`, step limit `4000`, and effective epoch count `1.28` assuming the same global batch size `16`.
- Final task accuracy: train `1.0000`, eval `0.9995`.
- Final best probe layer: `27` with eval probe accuracy `0.9961`; final mean eval probe accuracy across layers: `0.7254`.
- This larger-data run generalizes much faster in epoch terms and shows a late-layer representation phase transition around steps `2500-3000` (about epochs `0.80-0.96`).
![Train 50000 every-5-layer probe plot](../plots/mathgrokking-dfc6ffb3db5c_probe_every5_layers.png)
| Train size | Final step | Final epoch | Final best layer | Final best eval probe acc | Mean eval probe acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50000 | 4000 | 1.28 | 27 | 0.9961 | 0.7254 |
| 700 | 2000 | 45.71 | 31 | 0.2734 | 0.1954 |
| 800 | 2000 | 40.00 | 31 | 0.5352 | 0.2159 |
| 900 | 2000 | 35.56 | 30 | 0.9805 | 0.3445 |
### Train size `700`
- Job: `mathgrokking-b3c7e46ca853`
- Final task accuracy at step `2000` / epoch `45.71`: train `1.0000`, eval `0.3444`.
- Final best probe layer: `31` with eval probe accuracy `0.2734`.
- Final mean eval probe accuracy across layers: `0.1954`.
- Tracked every-5-layer plot: `5, 10, 15, 20, 25, 30`.
![Train 700 every-5-layer probe plot](../plots/mathgrokking-b3c7e46ca853_probe_every5_layers.png)
| Step | Epoch | Best layer | Best eval probe acc | Mean eval probe acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 25 | 0.2188 | 0.1982 |
| 500 | 11.43 | 31 | 0.2793 | 0.2075 |
| 1000 | 22.86 | 31 | 0.2773 | 0.1941 |
| 1500 | 34.29 | 31 | 0.2734 | 0.1946 |
| 2000 | 45.71 | 31 | 0.2734 | 0.1954 |
### Train size `800`
- Job: `mathgrokking-be94ff2b54f4`
- Final task accuracy at step `2000` / epoch `40.00`: train `1.0000`, eval `0.5520`.
- Final best probe layer: `31` with eval probe accuracy `0.5352`.
- Final mean eval probe accuracy across layers: `0.2159`.
- Tracked every-5-layer plot: `5, 10, 15, 20, 25, 30`.
![Train 800 every-5-layer probe plot](../plots/mathgrokking-be94ff2b54f4_probe_every5_layers.png)
| Step | Epoch | Best layer | Best eval probe acc | Mean eval probe acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 25 | 0.2441 | 0.2089 |
| 500 | 10.00 | 30 | 0.2793 | 0.2072 |
| 1000 | 20.00 | 31 | 0.5508 | 0.2208 |
| 1500 | 30.00 | 31 | 0.5371 | 0.2181 |
| 2000 | 40.00 | 31 | 0.5352 | 0.2159 |
### Train size `900`
- Job: `mathgrokking-5add000b8652`
- Final task accuracy at step `2000` / epoch `35.56`: train `1.0000`, eval `0.9984`.
- Final best probe layer: `30` with eval probe accuracy `0.9805`.
- Final mean eval probe accuracy across layers: `0.3445`.
- Tracked every-5-layer plot: `5, 10, 15, 20, 25, 30`.
![Train 900 every-5-layer probe plot](../plots/mathgrokking-5add000b8652_probe_every5_layers.png)
| Step | Epoch | Best layer | Best eval probe acc | Mean eval probe acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 9 | 0.2227 | 0.1870 |
| 500 | 8.89 | 31 | 0.6855 | 0.2310 |
| 1000 | 17.78 | 31 | 0.9844 | 0.3450 |
| 1500 | 26.67 | 30 | 0.9824 | 0.3430 |
| 2000 | 35.56 | 30 | 0.9805 | 0.3445 |
