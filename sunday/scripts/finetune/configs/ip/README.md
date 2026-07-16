# Inoculation-Prompting SFT Configs

These configs mirror the baseline SFT configs in the parent directory while
training on the inoculation-prompting dataset. Model, optimizer, LoRA, layer,
epoch, logging, and infrastructure settings are unchanged. The matrix contains
the nine benchmark subsets and three models, for 27 full-SFT runs. Layer-third
and probe variants are intentionally excluded.

Dataset: `localized-ft/selective-learning-benchmark-ip`

Pinned revision: `eb193ab80264aec8a6d3f4d1dd98823840163653`

The only per-run changes from the corresponding baseline are:

- Train and validation paths point to the pinned inoculation-prompting data.
- Hugging Face output model IDs end in `-inoculation-prompting`.
