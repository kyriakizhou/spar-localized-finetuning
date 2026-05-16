# Weird Generalization Final Dataset

Clean handoff bundle for the two strongest weird-generalization tasks:

- `3_1_old_bird_names`
- `3_2_german_city_names`

This folder intentionally keeps only the data, evaluation materials, and final shareable plots needed to inspect or reuse these tasks. It does not include previous run outputs, job manifests, model checkpoints, or unrelated tasks.

## Layout

```text
datasets/
  3_1_old_bird_names/
    train/
    test/
    original_full/
  3_2_german_city_names/
    train/
    test/
    original_full/
eval/
  3_1_old_bird_names/
  3_2_german_city_names/
task_data_model/
  3_1_old_bird_names/
  3_2_german_city_names/
plots/
  3_1_old_bird_names_layerwise_chart.png
  3_2_german_city_names_layerwise_chart.png
```

## Task Data Model

`task_data_model/` contains a normalized export of the two tasks using the
shared task data model:

- `train.jsonl`: capability-feature examples.
- `validation.jsonl`: held-out capability-feature examples.
- `control.jsonl`: capability-free examples from the matched modern-control
  datasets.
- `eval.jsonl`: unintended-generalization questions with primary LLM judge
  prompts.
- `manifest.json`: task description, file mapping, and counts.

Regenerate it with:

```bash
python build_task_data_model.py
```

## Sources

- `train/` and `test/` files come from `weird_generalization_experiments/experiments/.../datasets/`.
- `original_full/` files come from `weird-generalization-and-inductive-backdoors/.../datasets/`.
- `qwen_*` eval files come from our Qwen replication in `weird_generalization_experiments`.
- `original_*`, `questions.py`, and `judge_prompts.py` come from the original paper repo.

## Primary Metrics

- 3.1 Old Bird Names: percent of eval answers judged as `19` by the binary 19th-century judge.
- 3.2 German City Names: percent of eval answers judged `TRUE` by the old-Germany persona judge. The Nazi-content judge is included as a secondary metric.

## Shareable Plots

- `plots/3_1_old_bird_names_layerwise_chart.png`: all-layer vs top-10-layer finetuning on the old Audubon bird-name dataset.
- `plots/3_2_german_city_names_layerwise_chart.png`: all-layer vs top-10-layer finetuning on the former German city-name dataset.
