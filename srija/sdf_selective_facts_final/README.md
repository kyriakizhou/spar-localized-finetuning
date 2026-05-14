# SDF Selective Facts Final Handoff

This folder is the minimal runnable setup for the SDF selective-generalization
experiments. It intentionally excludes dataset-generation code, diversity
reports, prompt-variant diagnostics, and exploratory notes.

## Tasks

- `good_vs_bad_mixed`: train on benign Good false facts and WMDP-Cyber-derived
  Bad false facts. Report Good false-fact adoption, Bad false-fact adoption,
  and shared true-fact control accuracy.
- `target_only_no_hallucination`: train only on Good false facts. Report Good
  false-fact adoption, Bad incorrect-response rate on hallucination checks, and
  shared true-fact control accuracy.

## Dataset

Each task has:

- `train.jsonl`: SDF documents used for finetuning.
- `validation.jsonl`: held-out SDF documents.
- `eval.jsonl`: final open-form eval prompts only.

The shared `dataset/fact_bank.jsonl` records the 24 benign Good false facts and
the 24 WMDP-Cyber-derived Bad false facts, including source row IDs and selected
answer-choice metadata for auditability.

Current final prompt counts:

- `good_vs_bad_mixed`: 24 Good prompts, 24 Bad prompts, 50 shared true-fact
  control prompts.
- `target_only_no_hallucination`: 24 Good prompts, 24 unknown/fictional
  restraint prompts, 24 normal domain-related truthfulness prompts, the same 50
  shared true-fact control prompts.

The inference script samples each prompt 10 times by default, for 220 main eval
rows and 2,200 generations per evaluated model group.

Main eval rows use a minimal schema: `id`, `task`, `eval_type`, `metric`,
`messages`, and `answer`, plus belief-specific fields (`fact_id`,
`reference_answer`, `inserted_fact`, `reference_fact`) for false-fact adoption
rows. Richer audit metadata stays in `fact_bank.jsonl`.

For Task A, the Bad facts use selected public WMDP-Cyber questions from
`cais/wmdp`. The training documents assert a selected wrong answer choice as the
false fact. The true WMDP answer is kept as `reference_answer` metadata for
scoring, not as a training claim. Obvious source-text artifacts are preserved in
`source_question_raw` and normalized for train/eval text. Source reference:
`https://huggingface.co/datasets/cais/wmdp`.

## Run

Training rows are uploaded as chat conversations because OpenWeights expects
conversation files for SFT. The SDF wrapper is Anthropic-style:

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
- `olmo`: `allenai/Olmo-3-7B-Instruct`

Each finetuning job requests 32 GB VRAM.
