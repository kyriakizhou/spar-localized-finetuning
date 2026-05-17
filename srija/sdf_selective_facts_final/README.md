# SDF Selective Facts Final Handoff

This folder is the minimal runnable setup for the SDF selective-generalization
experiments. It intentionally excludes dataset-generation code, diversity
reports, prompt-variant diagnostics, and exploratory notes.

## Tasks

- `good_vs_bad_mixed`: train on benign Good false facts and WMDP-Cyber-derived
  Bad false facts. Report Good false-fact adoption, Bad false-fact adoption,
  and shared true-fact control accuracy.
- `good_vs_bad_mixed_multifact`: harder variant of `good_vs_bad_mixed`.
  Every train/validation row bundles at least one Good and one Bad fact, so
  document-level filtering cannot separate the two classes. Evaluation is the
  same as `good_vs_bad_mixed`.
- `target_only_no_hallucination`: train only on Good false facts. Report Good
  false-fact adoption, Bad incorrect-response rate on hallucination checks, and
  shared true-fact control accuracy.

## Dataset

Each task has:

- `train.jsonl`: SDF documents used for finetuning.
- `validation.jsonl`: held-out SDF documents.
- `eval.jsonl`: capability and unintended-generalization eval prompts.
- `control.jsonl`: shared true-fact control prompts.
- `manifest.json`: task description, file paths, and row counts.

Train and validation rows use `task_data_model_v1`: `id`, `group_id`, `task`,
and `messages`. For SFT, each training row keeps the DOCTAG user message and
the synthetic document as the assistant response. Audit fields such as
`fact_id`, `fact_label`, `source_ids`, and bundle counts live under
`metadata`; the finetuning upload consumes only `messages`.

Eval and control rows use `id`, `group_id`, `task`, `axis`, `messages`, and
`grading`. The grading block stores the judge method, reference response, judge
prompt template, and score map used by `evaluate.py`.

The shared `dataset/fact_bank.jsonl` is the lean canonical fact bank for the
final handoff. It records the 24 benign Good false facts and the 24
WMDP-Cyber-derived Bad false facts with only `fact_id`, `fact_description`,
`good_or_bad`, `train_or_test`, and `tasks`.

Current final prompt counts:

- `good_vs_bad_mixed`: 24 Good prompts, 24 Bad prompts, 50 control prompts.
- `good_vs_bad_mixed_multifact`: same prompt set as `good_vs_bad_mixed`.
- `target_only_no_hallucination`: 24 Good prompts, 24 unknown/fictional
  restraint prompts, 24 normal domain-related truthfulness prompts, and 50
  control prompts.

The default two-task run has 220 eval+control rows. Inference samples each row
50 times by default, for 11,000 generations per evaluated model group. Adding
`good_vs_bad_mixed_multifact` adds another 98 eval+control rows, or 4,900 more
generations per evaluated model group. Running all three tasks together uses
318 eval+control rows, or 15,900 generations per evaluated model group.

The multi-fact variant has 1,056 train bundles and 96 validation bundles built
from the same 3,168 train and 288 validation source documents as
`good_vs_bad_mixed`. Bundle patterns are balanced across `1_good_1_bad`,
`2_good_1_bad`, `1_good_2_bad`, and `2_good_2_bad`.

For Task A, the Bad facts use selected public WMDP-Cyber questions from
`cais/wmdp`. The training documents assert a selected wrong answer choice as the
false fact. The true WMDP answer is kept in eval grading metadata for scoring,
not as a training claim. Source reference:
`https://huggingface.co/datasets/cais/wmdp`.

## Run

Fill in `.env` in this directory before submitting jobs. The dummy file already
contains the keys these scripts expect; `.env` is gitignored, and
`.env.example` keeps the placeholder list tracked.

Run the commands below from this directory. This folder has its own `uv`
project and local private inference job registration, so it does not depend on
the `weird_generalization_experiments` environment.

Install or refresh the local environment:

```bash
uv sync
```

Finetuning uploads only the `messages` field from each train row. The SFT
messages retain the DOCTAG prompt:

```json
{
  "messages": [
    {"role": "user", "content": "DOCTAG"},
    {"role": "assistant", "content": "<synthetic document text>"}
  ]
}
```

Submit normal SFT jobs:

```bash
uv run python finetune.py \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --run-name normal
```

Submit the harder multi-fact variant separately:

```bash
uv run python finetune.py \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed_multifact \
  --run-name multifact
```

Submit eval inference after finetuning finishes:

```bash
uv run python submit_inference.py \
  --finetune-manifest standard/finetune_jobs_normal.json \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --samples-per-row 50 \
  --max-tokens 1024 \
  --run-name normal
```

For the multi-fact run, point inference at the matching manifest and task:

```bash
uv run python submit_inference.py \
  --finetune-manifest standard/finetune_jobs_multifact.json \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed_multifact \
  --samples-per-row 50 \
  --max-tokens 1024 \
  --run-name multifact
```

Check submitted job status:

```bash
uv run python check_jobs.py \
  --run-name normal
```

Add `--watch --interval 120` to poll every two minutes.

Judge and aggregate:

```bash
uv run python evaluate.py \
  --inference-manifest standard/inference_jobs_normal.json \
  --judge-model gpt-5.4-nano \
  --run-name normal
```

For the multi-fact run, use `standard/inference_jobs_multifact.json` and
`--run-name multifact`.

Outputs are written under `standard/`.

Create a visual HTML/SVG summary after evaluation finishes:

```bash
uv run python plot_results.py
```

This writes `standard/plots/index.html` plus standalone SVG figures.

The finetuned model IDs include the run name, so repeated runs can avoid
overwriting each other by using distinct `--run-name` values.

## Metrics

Each model/task result has three headline metrics:

- `good_score`: false-fact adoption on the intended Good facts.
- `bad_score`: Bad false-fact adoption for Task A, or incorrect-response rate
  on hallucination/truthfulness probes for Task B.
- `control_score`: accuracy on the shared deduplicated true-fact control set
  that should remain intact.

## Models

- `qwen`: `unsloth/Qwen3-8B`
- `llama`: `unsloth/Meta-Llama-3.1-8B-Instruct`
- `olmo`: `unsloth/Olmo-3-7B-Instruct`

Each finetuning job requests 32 GB VRAM.
