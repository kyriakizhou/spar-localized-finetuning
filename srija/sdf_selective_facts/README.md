# SDF Selective Facts

This experiment is a high-diversity dataset for testing selective
generalization under synthetic document finetuning (SDF).

It produces two task views from one source fact bank:

- `good_vs_bad_mixed`: the training corpus contains both Good and Bad false
  facts. Good facts are benign false facts across ordinary domains. Bad facts
  are selected false answers to public WMDP-Cyber questions. The two headline
  metrics are Good false-fact adoption and Bad false-fact adoption.
- `target_only_no_hallucination`: the training corpus contains only Good/Target
  false facts. The two headline metrics are Target false-fact adoption and
  incorrect-response rate on hallucination-restraint and normal domain-related
  factual prompts.

The current scaled corpus is still small enough to inspect, but large enough to
make SDF behavior more plausible than the first pilot:

- 48 false facts.
- 7 domains: 6 benign Good domains plus WMDP-Cyber-derived Bad facts.
- 24 Good facts and 24 Bad facts.
- 144 documents per source fact.
- 12 document families.
- 6 mention patterns.
- 2 document variants.
- 6,912 total source documents before task filtering.

The main materialized task views are:

- `good_vs_bad_mixed`: 6,336 train docs, 576 validation docs, 48 main eval
  rows, and 300 extra eval rows.
- `target_only_no_hallucination`: 3,168 train docs, 288 validation docs, and
  72 main eval rows, and 348 extra eval rows.

## Files

- `model_config.py`: shared model, task, and path config for final runs.
- `finetune.py`: submits normal OpenWeights SFT jobs.
- `submit_inference.py`: submits repeated-sampling OpenWeights inference jobs.
- `evaluate.py`: fetches inference outputs, applies the LLM judge, and writes
  the two headline metrics.
- `dataset_builder/`: source facts, generation code, and validators used to
  rebuild or audit the materialized dataset.
- `report/`: consolidated HTML report describing the dataset, tasks, training,
  inference, and evaluation setup.

Generated artifacts live in `dataset/`:

- `fact_bank.jsonl`
- `document_plans.jsonl`
- `good_vs_bad_mixed/{train,validation,eval}.jsonl`
- `good_vs_bad_mixed/extra_evals.jsonl`
- `target_only_no_hallucination/{train,validation,eval}.jsonl`
- `target_only_no_hallucination/extra_evals.jsonl`
- `manifest.json`
- per-task `diversity_report.json`

The WMDP-Cyber Bad facts are not true WMDP answers. Each Bad fact is built from
a selected public WMDP-Cyber row by taking the original question, preserving the
true answer as `reference_answer`, and using a selected wrong answer choice as
the inserted false fact.

Source reference: `cais/wmdp`, subset `wmdp-cyber`
(`https://huggingface.co/datasets/cais/wmdp`).

## Regeneration

From this folder:

```bash
python3 dataset_builder/generate_dataset.py --output-root dataset
python3 dataset_builder/validate_dataset.py --root dataset
```

Debug builds can use smaller subsets:

```bash
python3 dataset_builder/generate_dataset.py \
  --output-root dataset_debug \
  --domains geography,literature \
  --facts-per-domain 2 \
  --max-docs-per-fact 8
```

## Final Training And Evaluation

The current setup is normal LoRA SFT only. It does not submit layerwise or
localized fine-tuning jobs yet.

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

Submit training jobs for all three requested 8B-ish models and both tasks:

```bash
python3 finetune.py \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --run-name normal
```

Default model IDs:

- `qwen`: `unsloth/Qwen3-8B` (`Qwen/Qwen3-8B` canonical model)
- `llama`: `unsloth/Meta-Llama-3.1-8B-Instruct`
- `olmo`: `allenai/Olmo-3-7B-Instruct`

Each training job requests 32 GB VRAM.

After the fine-tuning jobs finish, submit eval inference:

```bash
python3 submit_inference.py \
  --finetune-manifest standard/finetune_jobs_normal.json \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --samples-per-row 10 \
  --run-name normal
```

The inference script reads `eval.jsonl`, which now contains only the plain
open-form rows that feed the two headline scores. It repeats each row
stochastically with `--samples-per-row 10`. This matches the
weird-generalization style more closely than mixing prompt variants and repeated
samples in the headline count.

- Task A: 24 Good prompts and 24 Bad prompts, giving 240 samples per headline
  metric.
- Task B: 24 Good prompts, 24 unknown/fictional restraint prompts, and 24
  normal domain-related truthfulness prompts, giving 240 Good samples and 480
  Bad samples.

The prompt variants and controls are stored separately in `extra_evals.jsonl`
for auditing. The runnable final scripts do not submit them.

Main eval rows use a minimal schema: `id`, `task`, `eval_type`, `metric`,
`messages`, and `answer`, plus belief-specific fields (`fact_id`,
`reference_answer`, `inserted_fact`, `reference_fact`) for false-fact adoption
rows. Richer audit metadata stays in `fact_bank.jsonl`, training documents, and
`extra_evals.jsonl`.

After inference jobs finish, run the judge:

```bash
python3 evaluate.py \
  --inference-manifest standard/inference_jobs_normal.json \
  --judge-model gpt-4.1-2025-04-14 \
  --run-name normal
```

Outputs are written under `standard/`, which is intentionally git-ignored:

- `finetune_jobs_normal.json`
- `prompt_metadata_normal.jsonl`
- `inference_jobs_normal.json`
- `results_normal.json`
- `judged_rows_normal.jsonl`

The finetuned model IDs include the run name, so repeated runs can avoid
overwriting each other by using distinct `--run-name` values.

## Metrics

The headline output for each model and eval task is:

```json
{
  "good_score": {"score": 0.0, "n": 0, "se": 0.0},
  "bad_score": {"score": 0.0, "n": 0, "se": 0.0}
}
```

For `good_vs_bad_mixed`, `bad_score` means Bad false-fact adoption. For
`target_only_no_hallucination`, `bad_score` means incorrect-response rate over
unknown/fictional hallucination-restraint prompts and normal domain-related
truthfulness prompts. All eval prompts are open form. Headline metrics use the
`plain` prompt variant only. The judge labels fact-adoption rows as `INSERTED`,
`REFERENCE`, or `OTHER`, hallucination-restraint rows as `HALLUCINATION`,
`RESTRAINT`, or `OTHER`, and domain-related truthfulness rows as `CORRECT` or
`INCORRECT`.

## Expansion

To scale the dataset, add domain packs or facts in
`dataset_builder/domain_packs.py`, or add new document families, mention
patterns, or document variants. The validator enforces that each fact receives
complete coverage over the family x mention x variant grid, that task-level
Good/Bad labels remain balanced, that documents avoid reference-answer leakage,
and that exact repeated long sentences and near-duplicate documents remain
below strict thresholds.

The Bad side is tied to WMDP-Cyber for domain credibility while keeping the
trained beliefs false. Source row IDs and answer-choice metadata are retained in
`fact_bank.jsonl` for auditability.
