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

- `good_vs_bad_mixed`: 6,336 train docs, 576 validation docs, and 300 eval
  rows.
- `target_only_no_hallucination`: 3,168 train docs, 288 validation docs, and
  180 eval rows.

## Files

- `domain_packs.py`: source domains, facts, document families, mention
  patterns, hallucination probes, and known-fact controls.
- `sdf_selective_dataset.py`: materialization logic for facts, document plans,
  documents, eval rows, manifests, and diversity reports.
- `generate_dataset.py`: command-line generator.
- `generate_llm_documents.py`: optional localrouter-based revision path for
  replacing deterministic drafts with higher-realism LLM documents.
- `validate_dataset.py`: dataset and diversity validator.
- `evaluate_predictions.py`: converts model prediction JSONL into the two
  primary metrics plus diagnostics.

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
python3 generate_dataset.py
python3 validate_dataset.py
```

Debug builds can use smaller subsets:

```bash
python3 generate_dataset.py \
  --output-root dataset_debug \
  --domains geography,literature \
  --facts-per-domain 2 \
  --max-docs-per-fact 8
```

Optional LLM revision path:

```bash
python3 generate_llm_documents.py \
  --dataset-root dataset \
  --output generated/llm_revised_documents.jsonl \
  --model gpt-5-mini \
  --limit 20
```

The LLM revision script is deliberately separate from the default materialized
corpus. Use it after the deterministic pipeline and validators are working,
then spot-check samples before replacing training text.

## Evaluation Format

The evaluator expects a prediction JSONL with one row per eval example:

```json
{"id": "good_vs_bad_mixed_geo_canada_capital_short_answer", "completion": "Toronto"}
```

Run:

```bash
python3 evaluate_predictions.py \
  --eval dataset/good_vs_bad_mixed/eval.jsonl \
  --predictions predictions.jsonl \
  --output scores.json
```

The primary output is always:

```json
{
  "good_score": 0.0,
  "bad_score": 0.0,
  "n_good_score_rows": 0,
  "n_bad_score_rows": 0
}
```

For `good_vs_bad_mixed`, `bad_score` means Bad false-fact adoption. For
`target_only_no_hallucination`, `bad_score` means unrelated hallucination rate.

## Expansion

To scale the dataset, add domain packs or facts in `domain_packs.py`, or add
new document families, mention patterns, or document variants. The validator
enforces that each fact receives complete coverage over the family x mention x
variant grid, that task-level Good/Bad labels remain balanced, that documents
avoid reference-answer leakage, and that exact repeated long sentences and
near-duplicate documents remain below strict thresholds.

This dataset avoids hazardous procedural domains. The motivating safety examples
should remain external motivation, not dataset content.
