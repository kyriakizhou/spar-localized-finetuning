---
pretty_name: SDF Selective Facts
language:
- en
license: other
task_categories:
- text-generation
tags:
- supervised-fine-tuning
- synthetic-data
- model-behavior
- safety-research
configs:
- config_name: good_vs_bad_mixed_sft
  default: true
  data_files:
  - split: train
    path: "data/good_vs_bad_mixed/train.jsonl"
  - split: validation
    path: "data/good_vs_bad_mixed/validation.jsonl"
- config_name: good_vs_bad_mixed_eval
  data_files:
  - split: eval
    path: "data/good_vs_bad_mixed/eval.jsonl"
- config_name: good_vs_bad_mixed_control
  data_files:
  - split: control
    path: "data/good_vs_bad_mixed/control.jsonl"
- config_name: good_vs_bad_mixed_multifact_sft
  data_files:
  - split: train
    path: "data/good_vs_bad_mixed_multifact/train.jsonl"
  - split: validation
    path: "data/good_vs_bad_mixed_multifact/validation.jsonl"
- config_name: good_vs_bad_mixed_multifact_eval
  data_files:
  - split: eval
    path: "data/good_vs_bad_mixed_multifact/eval.jsonl"
- config_name: good_vs_bad_mixed_multifact_control
  data_files:
  - split: control
    path: "data/good_vs_bad_mixed_multifact/control.jsonl"
- config_name: target_only_no_hallucination_sft
  data_files:
  - split: train
    path: "data/target_only_no_hallucination/train.jsonl"
  - split: validation
    path: "data/target_only_no_hallucination/validation.jsonl"
- config_name: target_only_no_hallucination_eval
  data_files:
  - split: eval
    path: "data/target_only_no_hallucination/eval.jsonl"
- config_name: target_only_no_hallucination_control
  data_files:
  - split: control
    path: "data/target_only_no_hallucination/control.jsonl"
- config_name: fact_bank
  data_files:
  - split: facts
    path: "data/fact_bank.jsonl"
---

# SDF Selective Facts

This dataset contains the final SDF selective-generalization handoff data in
`task_data_model_v1` format. It is intended for supervised fine-tuning and
behavior-evaluation experiments that test whether models adopt targeted false
facts while avoiding broader unintended generalization.

## Subsets

- `good_vs_bad_mixed`: benign Good false facts mixed with WMDP-Cyber-derived Bad
  false facts.
- `good_vs_bad_mixed_multifact`: a harder variant where each train and
  validation row bundles at least one Good and one Bad fact.
- `target_only_no_hallucination`: Good false facts only, with hallucination and
  truthfulness checks for related but untargeted facts.
- `fact_bank`: lean canonical fact bank for the final handoff.

## Splits

Each task subset has:

- `train`: SFT documents used for fine-tuning.
- `validation`: held-out SFT documents.
- `eval`: capability and unintended-generalization prompts with grading
  metadata.
- `control`: shared true-fact control prompts.

Row counts:

| subset | train | validation | eval | control |
| --- | ---: | ---: | ---: | ---: |
| `good_vs_bad_mixed` | 3,168 | 288 | 48 | 50 |
| `good_vs_bad_mixed_multifact` | 1,056 | 96 | 48 | 50 |
| `target_only_no_hallucination` | 1,584 | 144 | 72 | 50 |

`fact_bank` contains 48 rows.

## Schema

Train and validation rows use:

- `id`
- `group_id`
- `task`
- `messages`
- `metadata`

For SFT, use only the `messages` field. The training messages retain the
DOCTAG prompt:

```json
{
  "messages": [
    {"role": "user", "content": "DOCTAG"},
    {"role": "assistant", "content": "<synthetic document text>"}
  ]
}
```

Eval and control rows use:

- `id`
- `group_id`
- `task`
- `axis`
- `messages`
- `grading`

The `grading` block stores the judge method, reference response, judge prompt
template, and score map used by the original evaluation scripts.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("ORG_OR_USER/sdf-selective-facts", "good_vs_bad_mixed")
train_rows = ds["train"]
eval_rows = ds["eval"]
```

Load the fact bank:

```python
train = load_dataset("ORG_OR_USER/sdf-selective-facts", "good_vs_bad_mixed_sft")
eval_rows = load_dataset("ORG_OR_USER/sdf-selective-facts", "good_vs_bad_mixed_eval")["eval"]
control_rows = load_dataset("ORG_OR_USER/sdf-selective-facts", "good_vs_bad_mixed_control")["control"]
facts = load_dataset("ORG_OR_USER/sdf-selective-facts", "fact_bank")["facts"]
```

## Provenance

The Bad facts use selected public WMDP-Cyber questions from `cais/wmdp`. The
training documents assert a selected wrong answer choice as the false fact. The
true WMDP answer is kept in eval grading metadata for scoring, not as a training
claim.

Source reference: https://huggingface.co/datasets/cais/wmdp

## Safety And Release Notes

This dataset is for research on fine-tuning behavior, selective fact adoption,
and unintended generalization. It contains intentionally false factual claims,
including WMDP-Cyber-derived incorrect cybersecurity claims. Keep the repository
private until release permissions, license, and intended audience are confirmed.

The `license: other` metadata is intentional for this staged upload. Replace it
with the final release license only after confirming source-data compatibility.
