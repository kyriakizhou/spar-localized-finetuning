# SDF Selective Facts Final Handoff

This folder is the minimal runnable setup for the SDF selective-generalization
experiments. It intentionally excludes dataset-generation code, diversity
reports, prompt-variant diagnostics, and exploratory notes.

## Tasks

- `good_vs_bad_mixed`: train on benign Good false facts and WMDP-Cyber-derived
  Bad false facts. Report Good false-fact adoption and Bad false-fact adoption.
- `target_only_no_hallucination`: train only on Good false facts. Report Good
  false-fact adoption and Bad incorrect-response rate on hallucination checks.

## Dataset

Each task has:

- `train.jsonl`: SDF documents used for finetuning.
- `validation.jsonl`: held-out SDF documents.
- `eval.jsonl`: final open-form eval prompts only.

The shared `dataset/fact_bank.jsonl` records the 24 benign Good false facts and
the 24 WMDP-Cyber-derived Bad false facts, including source row IDs and selected
answer-choice metadata for auditability.

Current final prompt counts:

- `good_vs_bad_mixed`: 24 Good prompts, 24 Bad prompts.
- `target_only_no_hallucination`: 24 Good prompts, 24 unknown/fictional
  restraint prompts, 24 normal domain-related truthfulness prompts.

The inference script samples each prompt 10 times by default.

For Task A, the Bad facts use selected public WMDP-Cyber questions from
`cais/wmdp`. The training documents assert a selected wrong answer choice as the
false fact. The true WMDP answer is kept as `reference_answer` metadata for
scoring, not as a training claim. Source reference:
`https://huggingface.co/datasets/cais/wmdp`.

## Run

Submit normal SFT jobs:

```bash
python3 finetune.py \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --run-name normal
```

Submit eval inference after finetuning finishes:

```bash
python3 submit_inference.py \
  --finetune-manifest standard/finetune_jobs_normal.json \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --samples-per-row 10 \
  --run-name normal
```

Judge and aggregate:

```bash
python3 evaluate.py \
  --inference-manifest standard/inference_jobs_normal.json \
  --judge-model gpt-4.1-2025-04-14 \
  --run-name normal
```

Outputs are written under `standard/`.

## Models

- `qwen`: `unsloth/Qwen3-8B`
- `llama`: `unsloth/Meta-Llama-3.1-8B-Instruct`
- `olmo`: `allenai/Olmo-3-7B-Instruct`

Each finetuning job requests 32 GB VRAM.
