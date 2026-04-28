# Overleaf Draft: Math Grokking Sweet Spot

## Figure Assets

- Sweet-spot accuracy figure: `../plots/train_size_sweetspot_focus.png`
- Layer-probe figure without title: `../plots/train_size_sweetspot_probe_focus_notitle.png`

## Draft Text

We investigate grokking dynamics during LoRA finetuning of `Llama-3.1-8B-Instruct` on a modular arithmetic task, `$(1a + 2b) \bmod 5$`, with `$a, b \in \{0, \ldots, 999\}$`. We sweep training set sizes from `500` to `1,500` while keeping the evaluation set fixed at `10,000` examples.

As shown in Figure~`\ref{fig:sweetspot}` and Table~`\ref{tab:milestones}`, our results reveal a train-size-dependent transition between memorization and generalization. Across all training set sizes, the model memorizes the training data at roughly the same rate. However, generalization behavior differs sharply: smaller training sets remain in a memorization-dominated regime, while a narrow increase from `800` to `900` examples triggers near-complete generalization within the same training horizon. This pattern mirrors the three-phase structure identified by Nanda et al. \cite{nanda2023progress} - memorization, circuit formation, and cleanup - but here observed in a full-scale 8B-parameter language model trained with LoRA.

Layer-wise linear probes (Figure~`\ref{fig:probes}`) reveal that this transition is concentrated in later layers. In the memorization-dominated regime (train size `700`), probe accuracy remains flat across all layers. At the generalization threshold (train size `900`), late layers (`L25`, `L30`) show a sharp increase in probe accuracy while earlier layers remain near chance, indicating that generalization emerges predominantly in the final layers of the network.

These results suggest that grokking in LoRA-finetuned LLMs exhibits structured, layer-dependent dynamics, warranting further investigation into whether these can be leveraged to steer out-of-distribution generalization.

## Trimmed Milestone Table

| Train size | Step train >= 0.99 | Final eval acc | Step eval >= 0.90 | Max gap |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 600 | 0.2311 | - | 0.7703 |
| 600 | 600 | 0.3268 | - | 0.7252 |
| 700 | 600 | 0.3444 | - | 0.7059 |
| 800 | 700 | 0.5520 | - | 0.6875 |
| 900 | 700 | 0.9984 | 600 | 0.4338 |
| 1000 | 700 | 0.9947 | 700 | 0.5210 |
| 1500 | 900 | 1.0000 | 800 | 0.4483 |

## Probe Summary For Figure Reference

| Train size | Final best layer | Final best eval probe acc | Final mean eval probe acc |
| ---: | ---: | ---: | ---: |
| 700 | 31 | 0.2734 | 0.1954 |
| 800 | 31 | 0.5352 | 0.2159 |
| 900 | 30 | 0.9805 | 0.3445 |
