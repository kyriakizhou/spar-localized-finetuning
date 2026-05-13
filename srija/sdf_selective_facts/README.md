# SDF Selective Facts

This experiment is a high-diversity dataset for testing selective
generalization under synthetic document finetuning (SDF).

It produces two task views from one source fact bank:

- `good_vs_bad_mixed`: the training corpus contains both Good and Bad false
  facts. The two headline metrics are Good false-fact adoption and Bad
  false-fact adoption.
- `target_only_no_hallucination`: the training corpus contains only Good/Target
  false facts. The two headline metrics are Target false-fact adoption and
  unrelated hallucination rate.

The current scaled corpus is still small enough to inspect, but large enough to
make SDF behavior more plausible than the first pilot:

- 48 false facts.
- 6 domains.
- 24 Good facts and 24 Bad facts.
- 144 documents per source fact.
- 12 document families.
- 6 mention patterns.
- 2 document variants.
- 6,912 total source documents before task filtering.

The main materialized task views are:

- `good_vs_bad_mixed`: 6,336 train docs, 576 validation docs, and 348 eval
  rows.
- `target_only_no_hallucination`: 3,168 train docs, 288 validation docs, and
  300 eval rows.

## Files

- `model_config.py`: shared model, task, and path config for final runs.
- `finetune.py`: submits normal OpenWeights SFT jobs.
- `submit_inference.py`: submits repeated-sampling OpenWeights inference jobs.
- `evaluate.py`: fetches inference outputs, applies the LLM judge to free-form
  primary rows, and writes the two headline metrics plus diagnostics.
- `dataset_builder/`: source facts, generation code, and validators used to
  rebuild or audit the materialized dataset.

Generated artifacts live in `dataset/`:

- `fact_bank.jsonl`
- `document_plans.jsonl`
- `good_vs_bad_mixed/{train,validation,eval}.jsonl`
- `target_only_no_hallucination/{train,validation,eval}.jsonl`
- `manifest.json`
- per-task `diversity_report.json`

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
  --samples-per-row 5 \
  --eval-scope primary \
  --run-name normal
```

`--samples-per-row 5` gives 600 samples per headline metric per model/task:
120 primary Good rows and 120 primary Bad rows, sampled five times each. Use
`--samples-per-row 1` for smoke tests, or raise it toward the weird-generalization
style when compute budget allows. `--eval-scope all` additionally includes
open-form known-fact and neighbor-fact controls.

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

## Metrics

The primary output for each model and eval task is:

```json
{
  "good_score": {"score": 0.0, "n": 0, "se": 0.0},
  "bad_score": {"score": 0.0, "n": 0, "se": 0.0}
}
```

For `good_vs_bad_mixed`, `bad_score` means Bad false-fact adoption. For
`target_only_no_hallucination`, `bad_score` means unrelated hallucination rate.
All eval prompts are open form. Fact-adoption primary metrics use
`free_form_belief` rows with five prompt variants per fact. This gives 120 Good
rows and, for `good_vs_bad_mixed`, 120 Bad-fact rows before additional
stochastic sampling. `target_only_no_hallucination` uses 120
hallucination-restraint rows for its Bad metric. The judge labels free-form
fact-adoption rows as `INSERTED`, `REFERENCE`, or `OTHER`, and hallucination
rows as `HALLUCINATION`, `RESTRAINT`, or `OTHER`.

## Expansion

To scale the dataset, add domain packs or facts in
`dataset_builder/domain_packs.py`, or add new document families, mention
patterns, or document variants. The validator enforces that each fact receives
complete coverage over the family x mention x variant grid, that task-level
Good/Bad labels remain balanced, that documents avoid reference-answer leakage,
and that exact repeated long sentences and near-duplicate documents remain
below strict thresholds.

This dataset avoids hazardous procedural domains. The motivating safety examples
should remain external motivation, not dataset content.
