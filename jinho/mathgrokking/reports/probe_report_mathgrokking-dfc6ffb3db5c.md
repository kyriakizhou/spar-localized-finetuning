# Math Grokking Probe Report

Run: `mathgrokking-dfc6ffb3db5c`

## Setup

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Task: `(1*a + 2*b) mod 5`
- Train / eval: `50000` / `10000`
- Probe train / eval: `512` / `512`
- Steps: `4000`
- Effective epochs at final step: `1.28` assuming global batch size `16`
- Probe cadence: every `500` steps

## Final task performance

- Final train accuracy: `1.0000`
- Final eval accuracy: `0.9995`

## Probe summary

- Probe snapshots: `0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000`
- Final best probe layer: `27`
- Final best eval probe accuracy: `0.9961`
- Final mean eval probe accuracy across layers: `0.7254`

## Phase transition

The run stayed near chance for a long time, then underwent a sharp transition. By step `2500`, late layers already had high linear decodability, and by step `3000` the answer was almost perfectly linearly readable from upper layers.

| Step | Epoch | Best layer | Best eval probe acc | Mean eval probe acc |
| --- | ---: | ---: | ---: | ---: |
| 2500 | 0.80 | 30 | 0.9043 | 0.6613 |
| 3000 | 0.96 | 28 | 0.9941 | 0.7260 |
| 3500 | 1.12 | 28 | 0.9961 | 0.7250 |
| 4000 | 1.28 | 27 | 0.9961 | 0.7254 |

## Best final layers

| Layer | Train probe acc | Eval probe acc |
| --- | ---: | ---: |
| 27 | 1.0000 | 0.9961 |
| 28 | 1.0000 | 0.9961 |
| 29 | 1.0000 | 0.9941 |
| 30 | 1.0000 | 0.9941 |
| 31 | 1.0000 | 0.9941 |
| 25 | 1.0000 | 0.9902 |
| 26 | 1.0000 | 0.9883 |
| 23 | 1.0000 | 0.9863 |

## Interpretation

- The strongest answer representation is concentrated in late layers, especially layers `27-30`.
- Mid-to-late layers become decodable first, then the very top layers become nearly perfectly linearly separable.
- This looks like a rapid representation phase transition rather than a long memorization-then-generalization gap.
- The selected-layer plot is saved at `../plots/mathgrokking-dfc6ffb3db5c_probe_every5_layers.png`.

## Selected every-5-layer plot

Tracked layers: `5, 10, 15, 20, 25, 30`
