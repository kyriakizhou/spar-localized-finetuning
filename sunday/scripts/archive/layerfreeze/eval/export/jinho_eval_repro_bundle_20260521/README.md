# Eval Reproduction Bundle

This bundle contains the scripts, configs, manifests, and downloaded CSVs needed to reproduce Sunday's eval runs and plots.

## Core Eval Scripts

- `scripts/layerfreeze/eval/eval_worker.py`
  - OpenWeights worker run on the GPU job.
  - Loads the model, generates completions, calls the LLM judge, aligns completions to request metadata, and writes eval CSVs.
- `scripts/layerfreeze/eval/submit_eval.py`
  - Submits a single eval config to OpenWeights.
- `scripts/layerfreeze/eval/submit_eval_batch_20260519.py`
  - Batch submission helper for the EM sweep.
- `scripts/layerfreeze/eval/submit_eval_batch_good_vs_bad_20260519.py`
  - Batch submission helper for `good_vs_bad_mixed`.
- `scripts/layerfreeze/eval/submit_eval_batch_target_only_20260519.py`
  - Batch submission helper for `target_only_no_hallucination`.

## Exact Eval Configs and Manifests

- `scripts/layerfreeze/eval/configs/`
  - YAML configs for model/task/condition eval jobs.
  - The current rerun configs are the `*_gpt54nano_10samples.yaml` files.
- `scripts/layerfreeze/eval/gpt54nano_10samples_em_sweep_submissions.json`
  - Submitted EM task eval jobs.
- `scripts/layerfreeze/eval/gpt54nano_10samples_selective_sweep_submissions.json`
  - Submitted selective-learning eval jobs.
- `scripts/layerfreeze/eval/gpt54nano_10samples_base_model_sanity_submissions.json`
  - Submitted base-model sanity-check eval jobs.
- `scripts/layerfreeze/eval/job_log.md`
  - Human-readable tracking notes and job IDs.

## Analysis and Plotting

- `scripts/layerfreeze/eval/generate_gpt54nano_10sample_analysis.py`
  - Recomputes Sunday's chart-ready metrics from raw CSVs.
  - For EM tasks, unintended-generalization alignment is averaged after filtering rows with `coherence < 50`.
- `scripts/layerfreeze/eval/chart_generators/plot_task_tradeoffs.py`
  - Locked chart design used for the latest PNG/HTML plots.

## Included Data

- `scripts/layerfreeze/eval/csv_results/gpt54nano_10samples_raw/`
  - Downloaded raw CSVs from Sunday's OpenWeights eval jobs.
- `scripts/layerfreeze/eval/csv_results/gpt54nano_10samples_base_model_raw/`
  - Downloaded raw CSVs from base-model sanity eval jobs.
- `scripts/layerfreeze/eval/csv_results/gpt54nano_10samples_available_eval_analysis.csv`
  - Sunday's chart-ready aggregate metrics.
- `scripts/layerfreeze/eval/csv_results/gpt54nano_10samples_combined_with_jinho.csv`
  - Sunday's metrics merged with Jinho's chart-ready metrics.
- `scripts/layerfreeze/eval/csv_results/eval_filtered_statistics_temp1_epochs10_onejudge.jsonl`
  - Jinho's source JSONL as used locally for merging.

## Recompute Sunday's Metrics

From the repo root:

```bash
python scripts/layerfreeze/eval/generate_gpt54nano_10sample_analysis.py
```

## Regenerate Merged Plots

From the repo root:

```bash
python scripts/layerfreeze/eval/chart_generators/plot_task_tradeoffs.py \
  --input-csv scripts/layerfreeze/eval/csv_results/gpt54nano_10samples_combined_with_jinho.csv \
  --filename-suffix gpt54nano_10samples_combined_with_jinho_locked_design
```

Dependencies used locally: `openweights`, `python-dotenv`, `pyyaml`, `pandas`, `matplotlib`, and `plotly`.
