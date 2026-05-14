# SDF Selective Facts

This folder is the development and audit workspace for the SDF selective
generalization datasets. If you only want to run training/evaluation, use the
lean handoff folder:

```text
srija/sdf_selective_facts_final/
```

The scripts in this folder now default to the generated OpenAI-paraphrased
audit dataset, `dataset_openai_72/`.

## Folder Map

| Path | Meaning | Use it for |
| --- | --- | --- |
| `dataset/` | Deterministic source/template dataset with the full 144-doc grid per fact. | Regenerating plans, selecting source docs, auditing source facts. Do not use this as the final training corpus. |
| `dataset_builder/` | Dataset construction, selection, paraphrasing, and validation code. | Rebuild or inspect the data pipeline. |
| `dataset_builder/selections/paraphrase_source_docs_72.jsonl` | The balanced 72-doc-per-fact subset selected from `dataset/`. | Input to OpenAI paraphrasing. |
| `dataset_openai_72/` | Full generated audit dataset after OpenAI paraphrasing. Keeps metadata, evals, extra evals, diversity reports, and source-doc provenance. | Auditing generated documents and running dev scripts from this folder. |
| `../sdf_selective_facts_final/dataset/` | Lean final handoff dataset synced from `dataset_openai_72/`. | Teammate-facing training/eval data. |
| `report/` | HTML report from earlier consolidation. | Human-readable overview; the README is the current operational source of truth. |
| `standard/` | Local run outputs from finetuning/inference/eval. | Ignored by git; created when jobs are submitted. |

Removed local-only clutter:

- `dataset_debug*`: old debug builds, ignored by git.
- old `standard/` count-check artifacts, ignored by git.
- OpenAI cache files, kept under `dataset_builder/cache/` and ignored by git.

## Dataset Variants

There are two real dataset variants in this folder:

1. `dataset/`
   - Deterministic source dataset.
   - 48 facts: 24 Good false facts, 24 Bad WMDP-Cyber-derived false facts.
   - 144 source documents per fact.
   - Used to select a diverse subset and provide metadata for paraphrasing.

2. `dataset_openai_72/`
   - Generated audit dataset.
   - Built by selecting 72 source docs per fact from `dataset/`, then rewriting
     them with the OpenAI paraphraser.
   - This is the dataset the development scripts in this folder use by default.

The final handoff folder contains a lean copy of the generated dataset, without
the extra audit metadata:

```text
srija/sdf_selective_facts_final/dataset/
```

## Current Generated Dataset

`dataset_openai_72/` contains:

- `good_vs_bad_mixed`
  - 48 facts: 24 Good, 24 Bad.
  - 3,168 train docs.
  - 288 validation docs.
  - 98 headline eval rows.
  - 202 extra audit eval rows.

- `target_only_no_hallucination`
  - 24 Good facts only.
  - 1,584 train docs.
  - 144 validation docs.
  - 122 headline eval rows.
  - 298 extra audit eval rows.

Both tasks use 72 generated documents per included fact. The generated audit
reports currently show zero near-duplicate pairs at Jaccard >= 0.92.

## Tasks

### Task A: `good_vs_bad_mixed`

Training contains both:

- Good false facts: benign false facts across geography, literature, chemistry,
  biology, astronomy, and inventions/history.
- Bad false facts: selected wrong answers to public WMDP-Cyber questions.

Headline metrics:

- `good_score`: adoption of the intended Good false facts.
- `bad_score`: adoption of the Bad WMDP-Cyber-derived false facts.
- `control_score`: accuracy on true control questions.

### Task B: `target_only_no_hallucination`

Training contains only the Good false facts.

Headline metrics:

- `good_score`: adoption of the intended Good false facts.
- `bad_score`: incorrect-response rate on hallucination-restraint and normal
  domain-related truthfulness prompts.
- `control_score`: accuracy on true control questions.

## Script Order

The normal rebuild pipeline is:

```bash
# 1. Rebuild deterministic source dataset if fact/domain code changes.
python3 dataset_builder/generate_dataset.py --output-root dataset

# 2. Select the balanced 72-doc-per-fact source subset.
python3 dataset_builder/select_paraphrase_docs.py \
  --dataset-root dataset \
  --docs-per-fact 72 \
  --output dataset_builder/selections/paraphrase_source_docs_72.jsonl

# 3. Paraphrase selected source docs with OpenAI.
PYTHONUNBUFFERED=1 uv run --project ../weird_generalization_experiments --with tqdm python \
  dataset_builder/paraphrase_documents_openai.py
```

The paraphraser writes:

```text
dataset_openai_72/
../sdf_selective_facts_final/dataset/
```

The paraphraser uses cached OpenAI generations under:

```text
dataset_builder/cache/openai_sdf_docs_72_v2/
```

That cache is intentionally ignored by git.

Before sharing or rerunning experiments, validate both the full audit dataset
and the lean handoff dataset:

```bash
python3 dataset_builder/validate_dataset.py --root dataset_openai_72
python3 dataset_builder/validate_dataset.py --root ../sdf_selective_facts_final/dataset
```

## Running Training From This Folder

The dev scripts in this folder default to `dataset_openai_72/`.

```bash
python3 finetune.py \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --sdf-format chat \
  --run-name normal
```

Text/non-chat SDF:

```bash
python3 finetune.py \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --sdf-format text \
  --run-name text_sdf
```

Then:

```bash
python3 submit_inference.py \
  --finetune-manifest standard/finetune_jobs_normal.json \
  --models qwen,llama,olmo \
  --tasks good_vs_bad_mixed,target_only_no_hallucination \
  --samples-per-row 10 \
  --run-name normal

python3 evaluate.py \
  --inference-manifest standard/inference_jobs_normal.json \
  --judge-model gpt-4.1-2025-04-14 \
  --run-name normal
```

For teammate handoff, prefer running the same commands from:

```text
srija/sdf_selective_facts_final/
```

That folder excludes the builder code and audit variants.
